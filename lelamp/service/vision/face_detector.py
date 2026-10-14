"""
face_detector.py
人脸检测器抽象层。

提供统一接口 FaceDetector：detect(frame) → Optional[(x, y, w, h)]
三种实现：
  - RKNNFaceDetector：用 RetinaFace(.rknn) 模型在 rk3588 板载 NPU 上推理，
                       lamp_rock5c 上的首选检测器，精度/速度均优于 MNN
  - MNNFaceDetector：用 MNN 推理 face_detect_320.mnn 模型（树莓派等无 NPU 平台）
  - HaarFaceDetector：用 OpenCV Haar Cascade，作为最终兜底

create_default_detector() 按优先级尝试：RKNN → MNN → Haar，全部失败才抛错。
这样 FaceTrackingService 不必关心具体检测器。
"""

import os
import math
from typing import Optional, Tuple, List

import cv2
import numpy as np


class FaceDetector:
    """检测器抽象接口"""
    name: str = "base"

    def detect(self, frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        输入 BGR 图像，返回置信度最高的人脸框 (x, y, w, h)，无人脸返回 None。
        x/y/w/h 单位都是 frame_bgr 的原始像素。
        """
        raise NotImplementedError


# ============================================================
# MNN 检测器
# ============================================================
class MNNFaceDetector(FaceDetector):
    """
    使用阿里 MNN 推理 face_detect_320.mnn (RFB-320 / Ultra-Light Face)。
    模型固定输入 320x240 NCHW，输出 (scores, boxes)。
    """

    name = "mnn"

    INPUT_W = 320
    INPUT_H = 240
    # anchor 生成参数（与训练时一致，不要改）
    STEPS = [8, 16, 32, 64, 107, 320]
    MIN_SIZES = [
        [10, 16, 32], [32, 64], [64, 128],
        [128, 256],   [256, 512], [512, 1024],
    ]

    def __init__(
        self,
        model_path: str,
        conf_thresh: float = 0.6,
        nms_thresh: float = 0.3,
    ):
        # 延迟 import：MNN 没装也不让模块整个崩溃，只让构造抛错
        import MNN  # noqa: F401
        self.MNN = MNN

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"MNN 模型文件不存在: {model_path}")

        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

        self._net = MNN.nn.load_module_from_file(model_path, [], [])
        self._anchors = self._generate_anchors(self.INPUT_H, self.INPUT_W)

        # 预分配 anchor 解码用到的常量切片，避免每帧重复 .tolist()/索引
        self._anchor_cx = self._anchors[:, 0]
        self._anchor_cy = self._anchors[:, 1]
        self._anchor_w  = self._anchors[:, 2]
        self._anchor_h  = self._anchors[:, 3]

        # 性能：复用 input_var placeholder，避免每帧 MNN 内部反复分配
        self._input_var = MNN.expr.placeholder(
            [1, 3, self.INPUT_H, self.INPUT_W], MNN.expr.NCHW
        )

        print(f"[MNNFaceDetector] 已加载: {model_path} | "
              f"输入 {self.INPUT_W}x{self.INPUT_H} | anchors={len(self._anchors)}")

    # --------- anchor 生成（与示例一致） ----------
    @classmethod
    def _generate_anchors(cls, input_h: int, input_w: int) -> np.ndarray:
        anchors: List[List[float]] = []
        for k, step in enumerate(cls.STEPS):
            fh = math.ceil(input_h / step)
            fw = math.ceil(input_w / step)
            for i in range(fh):
                for j in range(fw):
                    cx = (j + 0.5) * step
                    cy = (i + 0.5) * step
                    for s in cls.MIN_SIZES[k]:
                        anchors.append([cx, cy, s, s])
        return np.array(anchors, dtype=np.float32)

    # --------- 推理 + 解码 ----------
    def detect(self, frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        if frame_bgr is None or frame_bgr.size == 0:
            return None
        orig_h, orig_w = frame_bgr.shape[:2]

        # 前处理：用 cv2.dnn.blobFromImage 一次完成 resize + RGB + 归一化 + transpose + batch
        # 比原来的 5 步 numpy 链快 3-5 倍（C++ 实现）
        # 等价于 (img/255 - 0.5) * 2 = (img - 127.5) / 127.5
        blob = cv2.dnn.blobFromImage(
            frame_bgr,
            scalefactor=1.0 / 127.5,
            size=(self.INPUT_W, self.INPUT_H),
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
            crop=False,
        )

        # 推理（复用 placeholder，避免每帧分配）
        self._input_var.write(blob)
        output_vars = self._net.onForward([self._input_var])
        # MNN 输出顺序：[0]=scores, [1]=boxes
        scores = output_vars[0].read()
        boxes  = output_vars[1].read()

        return self._decode_best(boxes, scores, orig_w, orig_h)

    def _decode_best(
        self, boxes: np.ndarray, scores: np.ndarray,
        orig_w: int, orig_h: int,
    ) -> Optional[Tuple[int, int, int, int]]:
        if boxes.ndim == 3: boxes = boxes[0]
        if scores.ndim == 3: scores = scores[0]

        face_scores = scores[:, 1]            # 取 face 类别分
        mask = face_scores > self.conf_thresh
        if not np.any(mask):
            return None

        # 用预切片好的 anchor 列，避免每帧索引开销
        fb = boxes[mask]
        fs = face_scores[mask]
        a_cx = self._anchor_cx[mask]
        a_cy = self._anchor_cy[mask]
        a_w  = self._anchor_w[mask]
        a_h  = self._anchor_h[mask]

        # SSD-style 解码
        pred_cx = a_cx + fb[:, 0] * a_w * 0.1
        pred_cy = a_cy + fb[:, 1] * a_h * 0.1
        pred_w  = a_w * np.exp(fb[:, 2] * 0.2)
        pred_h  = a_h * np.exp(fb[:, 3] * 0.2)

        sx = orig_w / self.INPUT_W
        sy = orig_h / self.INPUT_H
        x1 = np.clip((pred_cx - pred_w / 2) * sx, 0, orig_w)
        y1 = np.clip((pred_cy - pred_h / 2) * sy, 0, orig_h)
        x2 = np.clip((pred_cx + pred_w / 2) * sx, 0, orig_w)
        y2 = np.clip((pred_cy + pred_h / 2) * sy, 0, orig_h)

        # 优化：候选数 < 200 时直接遍历比 NMSBoxes 快（避免 OpenCV 调用开销）
        # 实际场景里 conf>0.5 的候选通常只有几十个，远少于 200
        n = len(x1)
        if n == 0:
            return None
        if n == 1:
            # 只有一个候选直接返回，省掉 NMS
            return (int(x1[0]), int(y1[0]),
                    int(x2[0] - x1[0]), int(y2[0] - y1[0]))

        rects = [
            [float(x1[i]), float(y1[i]),
             float(x2[i] - x1[i]), float(y2[i] - y1[i])]
            for i in range(n)
        ]
        confs = [float(s) for s in fs]
        indices = cv2.dnn.NMSBoxes(rects, confs, self.conf_thresh, self.nms_thresh)
        if len(indices) == 0:
            return None

        best = indices.flatten()[0]
        x, y, w, h = rects[best]
        return int(x), int(y), int(w), int(h)


# ============================================================
# Haar 兜底检测器
# ============================================================
class HaarFaceDetector(FaceDetector):
    """OpenCV Haar Cascade 兜底实现，无 MNN 时使用"""

    name = "haar"

    def __init__(self, cascade_path: Optional[str] = None):
        if cascade_path is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError(f"Haar Cascade 加载失败: {cascade_path}")
        self.cascade_path = cascade_path
        print(f"[HaarFaceDetector] 已加载: {cascade_path}")

    def detect(self, frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        if frame_bgr is None or frame_bgr.size == 0:
            return None
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30)
        )
        if len(faces) == 0:
            return None
        # 选最大的脸
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return int(x), int(y), int(w), int(h)


# ============================================================
# 工厂
# ============================================================
def create_default_detector(
    rknn_model_path: Optional[str] = None,
    mnn_model_path: Optional[str] = None,
    haar_cascade_path: Optional[str] = None,
    conf_thresh: float = 0.5,
    nms_thresh: float = 0.3,
) -> FaceDetector:
    """
    创建默认检测器：
    1. 优先尝试 RKNN（RetinaFace.rknn，需要 rknnlite 包 + 模型文件；rk3588 板载 NPU 首选）
    2. 失败则尝试 MNN（需要 MNN 包 + 模型文件）
    3. 再失败则回退到 Haar
    4. 都失败抛 RuntimeError

    rknn_model_path 默认按以下顺序搜寻：
      - 显式传入路径
      - 环境变量 LELAMP_RKNN_MODEL
      - <项目根>/RetinaFace.rknn

    mnn_model_path 默认按以下顺序搜寻：
      - 显式传入路径
      - 环境变量 LELAMP_MNN_MODEL
      - <项目根>/face_detect_320.mnn
      - /home/pi/source/python/lelamp_runtime/face_detect_320.mnn
    """
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(here, "..", "..", ".."))

    # 1. 准备候选 RKNN 模型路径，优先尝试（rk3588 板载 NPU 上性能/精度都优于 MNN）
    rknn_candidates = []
    if rknn_model_path:
        rknn_candidates.append(rknn_model_path)
    env_rknn_path = os.environ.get("LELAMP_RKNN_MODEL")
    if env_rknn_path:
        rknn_candidates.append(env_rknn_path)
    rknn_candidates.append(os.path.join(project_root, "RetinaFace.rknn"))

    for path in rknn_candidates:
        if not path or not os.path.isfile(path):
            continue
        try:
            # 延迟 import，避免没装 rknnlite 的平台（如树莓派）在 import 本模块时就报错
            from .retinaface_detector import RKNNFaceDetector
            return RKNNFaceDetector(path, conf_thresh=conf_thresh, nms_thresh=nms_thresh)
        except ImportError:
            print("[face_detector] rknnlite 包未安装，无法使用 RKNN 检测器；"
                  "考虑安装 rknn-toolkit-lite2")
            break  # rknnlite 没装，继续找下一个 path 也没用
        except Exception as e:
            print(f"[face_detector] 加载 RKNN 模型失败 ({path}): {e}")
            continue

    # 2. 准备候选 MNN 模型路径
    candidates = []
    if mnn_model_path:
        candidates.append(mnn_model_path)
    env_path = os.environ.get("LELAMP_MNN_MODEL")
    if env_path:
        candidates.append(env_path)
    # 项目根（face_tracking_service.py 在 lelamp/service/vision/）
    candidates.append(os.path.join(project_root, "face_detect_320.mnn"))
    candidates.append("/home/pi/source/python/lelamp_runtime/face_detect_320.mnn")

    # 3. 尝试 MNN
    for path in candidates:
        if not path or not os.path.isfile(path):
            continue
        try:
            return MNNFaceDetector(path, conf_thresh=conf_thresh, nms_thresh=nms_thresh)
        except ImportError:
            print("[face_detector] MNN 包未安装，无法使用 MNN 检测器；考虑 `pip install MNN`")
            break  # MNN 没装，继续找下一个 path 也没用
        except Exception as e:
            print(f"[face_detector] 加载 MNN 模型失败 ({path}): {e}")
            continue

    # 4. 回退 Haar
    print("[face_detector] 回退到 Haar Cascade 检测器")
    try:
        return HaarFaceDetector(haar_cascade_path)
    except Exception as e:
        raise RuntimeError(f"无可用人脸检测器（RKNN / MNN / Haar 均失败）: {e}")
