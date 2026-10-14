"""
retinaface_detector.py
基于 RetinaFace(.rknn) 模型的人脸检测器，跑在 rk3588 (lamp_rock5c) 板载 NPU 上。

与 RetinaFace.py（PC 端调试脚本，用 rknn.api.RKNN + target/device_id 走 ADB 连板）不同，
这里用 rknnlite.api.RKNNLite ——这是在 rk3588 主控板本机上做推理的标准方式，
不需要 target/device_id，直接调用本机 NPU。

前处理/后处理（letterbox、PriorBox、box 解码、NMS）与 RetinaFace.py 完全对齐，
只是去掉了画框、cv2.imwrite 等可视化/落盘逻辑，只返回最高置信度的人脸框。

接口与 face_detector.FaceDetector 一致：
    detect(frame_bgr) -> Optional[(x, y, w, h)]
这样 FaceTrackingService 无需关心底层是 MNN 还是 RKNN。
"""

import os
import math
from itertools import product
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .face_detector import FaceDetector


# ============================================================
# 前处理 / 后处理工具（与 RetinaFace.py 中的实现保持一致）
# ============================================================
def _letterbox_resize(
    image: np.ndarray, size: Tuple[int, int], bg_color: int = 114
) -> Tuple[np.ndarray, float, int, int]:
    """
    等比缩放 + 灰边填充。
    :param size: (target_width, target_height)
    :return: (letterbox 后的图像, 缩放比例, x 方向偏移, y 方向偏移)
    """
    target_w, target_h = size
    h, w = image.shape[:2]
    ratio = min(target_w / w, target_h / h)
    new_w, new_h = int(w * ratio), int(h * ratio)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)
    off_x = (target_w - new_w) // 2
    off_y = (target_h - new_h) // 2
    canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized
    return canvas, ratio, off_x, off_y


def _prior_box(image_size: Tuple[int, int]) -> np.ndarray:
    """
    与 RetinaFace.py 中 PriorBox() 完全一致。
    :param image_size: (height, width)，支持 (320,320) / (640,640)
    """
    min_sizes_cfg = [[16, 32], [64, 128], [256, 512]]
    steps = [8, 16, 32]
    feature_maps = [
        [math.ceil(image_size[0] / step), math.ceil(image_size[1] / step)]
        for step in steps
    ]
    anchors: List[float] = []
    for k, f in enumerate(feature_maps):
        min_sizes = min_sizes_cfg[k]
        for i, j in product(range(f[0]), range(f[1])):
            for min_size in min_sizes:
                s_kx = min_size / image_size[1]
                s_ky = min_size / image_size[0]
                cx = (j + 0.5) * steps[k] / image_size[1]
                cy = (i + 0.5) * steps[k] / image_size[0]
                anchors += [cx, cy, s_kx, s_ky]
    return np.array(anchors, dtype=np.float32).reshape(-1, 4)


def _decode_boxes(loc: np.ndarray, priors: np.ndarray) -> np.ndarray:
    """与 RetinaFace.py 中 box_decode() 一致"""
    variances = [0.1, 0.2]
    boxes = np.concatenate((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), axis=1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


# ============================================================
# RKNN 检测器
# ============================================================
class RKNNFaceDetector(FaceDetector):
    """
    使用 RetinaFace .rknn 模型在 rk3588 板载 NPU 上做人脸检测。
    模型固定输入 320x320（letterbox 缩放），输出顺序 [loc, conf, landmark]，
    与 RetinaFace.py 调试脚本一致。本类只取"置信度最高的一张脸"，不做可视化/落盘。
    """

    name = "rknn_retinaface"

    INPUT_SIZE = (320, 320)  # (height, width)，与模型转换时的输入尺寸一致

    def __init__(
        self,
        model_path: str,
        conf_thresh: float = 0.5,
        nms_thresh: float = 0.4,
        core_mask: Optional[int] = None,
    ):
        """
        :param model_path: RetinaFace.rknn 模型文件的绝对/相对路径
        :param core_mask: 可选，多核 NPU 绑核（如 rknnlite.api.RKNNLite.NPU_CORE_0_1_2）。
                           板载单次单帧推理场景一般不需要设置，用默认调度即可。
        """
        if not model_path or not os.path.isfile(model_path):
            raise FileNotFoundError(f"RetinaFace RKNN 模型文件不存在: {model_path}")

        # 延迟 import：板子没装 rknn-toolkit-lite2 时，只让构造函数抛错，不拖垮整个模块
        try:
            from rknnlite.api import RKNNLite
        except ImportError as e:
            raise ImportError(
                "未安装 rknnlite（rknn-toolkit-lite2）。在 rk3588 板上请先安装板载推理运行时，"
                "例如 `pip install rknn_toolkit_lite2-*.whl`"
            ) from e

        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

        self._rknn = RKNNLite(verbose=False)

        ret = self._rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"加载 RKNN 模型失败: {model_path} (ret={ret})")

        # 板载推理：不传 target/device_id，直接用本机 NPU
        init_kwargs = {}
        if core_mask is not None:
            init_kwargs["core_mask"] = core_mask
        ret = self._rknn.init_runtime(**init_kwargs)
        if ret != 0:
            self._rknn.release()
            raise RuntimeError(f"RKNN init_runtime 失败 (ret={ret})")

        # 预生成 priors，避免每帧重算
        self._priors = _prior_box(self.INPUT_SIZE)
        model_h, model_w = self.INPUT_SIZE
        self._scale_box = np.array(
            [model_w, model_h, model_w, model_h], dtype=np.float32
        )

        print(
            f"[RKNNFaceDetector] 已加载: {model_path} | "
            f"输入 {model_w}x{model_h} | priors={len(self._priors)}"
        )

    # --------- 推理 + 解码 ----------
    def detect(self, frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        if frame_bgr is None or frame_bgr.size == 0:
            return None
        orig_h, orig_w = frame_bgr.shape[:2]
        model_h, model_w = self.INPUT_SIZE

        letterboxed, ratio, off_x, off_y = _letterbox_resize(
            frame_bgr, (model_w, model_h), bg_color=114
        )
        infer_img = letterboxed[..., ::-1]  # BGR -> RGB，与 RetinaFace.py 一致
        # rknnlite（板载运行时）要求输入是 4 维 NHWC；PC 端 rknn.api.RKNN 会自动补 batch 维，
        # 但 rknnlite 不会，必须显式加上 batch=1，否则报
        # "The input[0] need 4dims input, but 3dims input buffer feed."
        infer_img = np.ascontiguousarray(infer_img[np.newaxis, ...])  # (H,W,C) -> (1,H,W,C)

        outputs = self._rknn.inference(inputs=[infer_img])
        if not outputs or len(outputs) < 3:
            return None
        loc, conf, _landmarks = outputs  # 人脸追踪只需要框，landmark 不用

        return self._decode_best(loc, conf, ratio, off_x, off_y, orig_w, orig_h)

    def _decode_best(
        self,
        loc: np.ndarray,
        conf: np.ndarray,
        ratio: float,
        off_x: int,
        off_y: int,
        orig_w: int,
        orig_h: int,
    ) -> Optional[Tuple[int, int, int, int]]:
        loc = loc.squeeze(0) if loc.ndim == 3 else loc
        conf = conf.squeeze(0) if conf.ndim == 3 else conf

        boxes = _decode_boxes(loc, self._priors) * self._scale_box
        scores = conf[:, 1]  # face 类别分数

        mask = scores > self.conf_thresh
        if not np.any(mask):
            return None
        boxes = boxes[mask]
        scores = scores[mask]

        # letterbox 还原到原图坐标（与 RetinaFace.py 一致）
        boxes[:, 0::2] = np.clip((boxes[:, 0::2] - off_x) / ratio, 0, orig_w)
        boxes[:, 1::2] = np.clip((boxes[:, 1::2] - off_y) / ratio, 0, orig_h)

        n = len(scores)
        if n == 1:
            x1, y1, x2, y2 = boxes[0]
            return int(x1), int(y1), int(x2 - x1), int(y2 - y1)

        rects = [
            [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]
            for b in boxes
        ]
        confs = [float(s) for s in scores]
        indices = cv2.dnn.NMSBoxes(rects, confs, self.conf_thresh, self.nms_thresh)
        if len(indices) == 0:
            return None

        # 取 NMS 后置信度最高的一个
        best_idx = max(indices.flatten(), key=lambda i: confs[i])
        x, y, w, h = rects[best_idx]
        return int(x), int(y), int(w), int(h)

    def close(self):
        try:
            self._rknn.release()
        except Exception:
            pass

    def __del__(self):
        self.close()
