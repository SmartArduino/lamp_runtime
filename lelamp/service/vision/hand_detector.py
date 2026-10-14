"""
hand_detector.py
手势检测器抽象层（指向手势 + 方向）。

仿照 face_detector.py 的设计：提供统一接口 HandDetector，封装 MediaPipe Hands 推理。
对外暴露：
  - detect(frame_bgr) -> Optional[(direction, landmarks)]
      direction ∈ {"Up","Down","Left","Right"} 表示当前帧识别到的指向方向；
      若未识别到"指向手势"则返回 None。
      landmarks 透传 mediapipe 的 21 个关键点，便于上层做可视化/调试。

设计要点：
1. MediaPipe Hands 在 Pi 上单帧推理 30~80ms，比 MNN 人脸（10ms）慢得多；
   所以集成时必须用独立线程+跳帧策略调用，避免拖慢人脸追踪 FPS。
2. 本类只负责"单帧推理"，多帧防抖/冷却交给上层 HandGestureService 处理，
   保持与 face_detector.py 一致的纯函数式风格。
3. mediapipe 为可选依赖：包未装时 create_default_hand_detector() 返回 None，
   而不是抛错，让上层"无手势检测也能正常跑人脸追踪"。
4. mediapipe >= 0.10.30 移除了旧的 mp.solutions API，本文件优先使用新版
   Tasks API（HandLandmarker）；若运行环境还是旧版 mediapipe，则自动回退
   到 mp.solutions.hands 实现，两个版本都能跑。
"""

import math
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional, Tuple, Any

import numpy as np

# 关掉 mediapipe / glog 噪音日志
os.environ.setdefault("GLOG_minloglevel", "2")

try:
    import cv2
except ImportError:
    cv2 = None

# HandLandmarker 模型（float16，约 7.8MB）：Tasks API 需要外置 .task 模型文件
_HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
_HAND_LANDMARKER_MODEL_FILENAME = "hand_landmarker.task"


def _resolve_model_path() -> Path:
    """
    定位 hand_landmarker.task 模型文件。
    查找顺序：models/ → 仓库根目录 → 本文件所在目录；都没有则从官方地址
    下载到 models/ 目录。下载失败抛异常，由上层工厂捕获后降级。
    """
    here = Path(__file__).resolve().parent
    for candidate in (
        Path("models") / _HAND_LANDMARKER_MODEL_FILENAME,
        Path.cwd() / _HAND_LANDMARKER_MODEL_FILENAME,
        here / _HAND_LANDMARKER_MODEL_FILENAME,
    ):
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate

    target_dir = (here.parent.parent.parent / "models").resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / _HAND_LANDMARKER_MODEL_FILENAME
    print(f"[hand_detector] 正在下载手势模型到 {target} ...")
    tmp = target.with_suffix(".task.tmp")
    try:
        urllib.request.urlretrieve(_HAND_LANDMARKER_MODEL_URL, tmp)
        tmp.rename(target)
    finally:
        tmp.unlink(missing_ok=True)
    print(f"[hand_detector] 手势模型下载完成 ({target.stat().st_size // 1024} KB)")
    return target


class HandDetector:
    """手势检测器抽象接口"""
    name: str = "base"

    def detect(self, frame_bgr: np.ndarray) -> Optional[Tuple[str, Any]]:
        """
        输入 BGR 图像，返回 (direction, landmarks) 或 None。
        direction 为 "Up"/"Down"/"Left"/"Right"，表示食指指向。
        若没有手、或手势不是"指向"，返回 None。
        """
        raise NotImplementedError

    def close(self):
        """释放底层推理资源（如 mediapipe Hands 句柄）。可选实现。"""
        pass


# ============================================================
# 工具函数（与 hand_recongnize2.py 中算法一致）
# ============================================================
def _dist(p1, p2) -> float:
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def _is_pointing_gesture(landmarks) -> bool:
    """
    指向手势判定：食指尖到手腕的距离 > 其他三指尖到手腕距离的 1.25 倍。
    （即只伸食指，其余三指弯曲收回）
    """
    wrist = landmarks[0]
    d_index = _dist(landmarks[8], wrist)
    d_middle = _dist(landmarks[12], wrist)
    d_ring = _dist(landmarks[16], wrist)
    d_pinky = _dist(landmarks[20], wrist)

    max_other = max(d_middle, d_ring, d_pinky)
    if max_other < 1e-6:
        return False
    return d_index / max_other > 1.25


def _get_pointing_direction(landmarks) -> str:
    """
    根据食指 MCP(5) → 食指尖(8) 的向量判定 4 方向（用户视角）。
    注意 x 已取负号补偿摄像头镜像，与 hand_recongnize2.py 保持一致。
    """
    dx = -(landmarks[8].x - landmarks[5].x)
    dy = landmarks[8].y - landmarks[5].y
    if abs(dx) > abs(dy):
        return "Right" if dx > 0 else "Left"
    else:
        return "Down" if dy > 0 else "Up"


# ============================================================
# MediaPipe 实现（新版 Tasks API，mediapipe >= 0.10.x）
# ============================================================
class MediaPipeHandDetector(HandDetector):
    """
    基于 MediaPipe Tasks HandLandmarker 的指向手势检测器。
    旧版 mp.solutions 在 mediapipe >= 0.10.30 中已被移除，新环境走本实现。

    注意：model_complexity 参数为兼容旧签名而保留，Tasks API 的 .task
    模型文件内置单一复杂度，该参数不再生效。

    平台适配（针对 mediapipe >= 1.x 在 macOS 上的两个问题）：
    a) CPU delegate 下 TensorsToDetectionsCalculator 初始化 DrishtiMetalHelper
       时会因 Metal service 未注册直接 C++ FATAL abort（Python 无法捕获），
       因此 macOS 必须用 GPU delegate；
    b) GPU 路径输入经 CoreVideo CVPixelBuffer 上传，不支持 24-bit SRGB，
       输入需转 4 通道 SRGBA；Linux（RK3588 等）无此问题，走 CPU + SRGB。
    """

    name = "mediapipe-tasks"

    def __init__(
        self,
        max_num_hands: int = 1,
        model_complexity: int = 0,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
    ):
        import mediapipe as mp
        from mediapipe.tasks.python import vision as mp_vision

        self._is_apple = sys.platform == "darwin"
        model_path = str(_resolve_model_path())
        delegate = (mp.tasks.BaseOptions.Delegate.GPU if self._is_apple
                    else mp.tasks.BaseOptions.Delegate.CPU)
        base_options = mp.tasks.BaseOptions(
            model_asset_path=model_path, delegate=delegate
        )
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        # VIDEO 模式要求时间戳严格单调递增；用内部计数器兜底，防止同毫秒多帧
        self._last_ts_ms = -1
        print(f"[MediaPipeHandDetector] 已加载 mediapipe HandLandmarker (Tasks API, "
              f"delegate={'GPU' if self._is_apple else 'CPU'}, "
              f"model={model_path}, max_hands={max_num_hands})")

    def detect(self, frame_bgr: np.ndarray) -> Optional[Tuple[str, Any]]:
        import mediapipe as mp

        if frame_bgr is None or frame_bgr.size == 0 or cv2 is None:
            return None

        if self._is_apple:
            # macOS GPU 路径：CoreVideo 不支持 24-bit，必须 4 通道 SRGBA
            rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=rgba)
        else:
            # Linux：标准 SRGB 路径
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        ts_ms = time.monotonic_ns() // 1_000_000
        if ts_ms <= self._last_ts_ms:
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms

        result = self._landmarker.detect_for_video(mp_image, ts_ms)

        if not result.hand_landmarks:
            return None

        lm = result.hand_landmarks[0]
        if not _is_pointing_gesture(lm):
            return None

        return _get_pointing_direction(lm), lm

    def close(self):
        try:
            self._landmarker.close()
        except Exception:
            pass


# ============================================================
# MediaPipe 实现（旧版 solutions API，mediapipe < 0.10.30 回退路径）
# ============================================================
class _LegacyMediaPipeHandDetector(HandDetector):
    """基于旧版 mp.solutions.hands 的实现，仅在环境装的是旧版 mediapipe 时使用。"""

    name = "mediapipe-legacy"

    def __init__(
        self,
        max_num_hands: int = 1,
        model_complexity: int = 0,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
    ):
        import mediapipe as mp  # 延迟 import：包没装的时候构造抛 ImportError
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            model_complexity=model_complexity,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        print(f"[MediaPipeHandDetector] 已加载 mediapipe Hands "
              f"(complexity={model_complexity}, max_hands={max_num_hands})")

    def detect(self, frame_bgr: np.ndarray) -> Optional[Tuple[str, Any]]:
        if frame_bgr is None or frame_bgr.size == 0 or cv2 is None:
            return None

        # MediaPipe 需要 RGB
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # writeable=False 让 mediapipe 走零拷贝路径，能省一点时间
        rgb.flags.writeable = False
        result = self._hands.process(rgb)

        if not result.multi_hand_landmarks:
            return None

        lm = result.multi_hand_landmarks[0].landmark
        if not _is_pointing_gesture(lm):
            return None

        return _get_pointing_direction(lm), lm

    def close(self):
        try:
            self._hands.close()
        except Exception:
            pass


# ============================================================
# 工厂
# ============================================================
def create_default_hand_detector(
    max_num_hands: int = 1,
    model_complexity: int = 0,
    min_detection_confidence: float = 0.6,
    min_tracking_confidence: float = 0.5,
) -> Optional[HandDetector]:
    """
    创建默认手势检测器。
    优先使用新版 Tasks API（mediapipe >= 0.10.x）；环境装的是旧版
    mediapipe（< 0.10.30，仍有 solutions API）时自动回退旧实现。
    与 face_detector.create_default_detector() 的区别：
      mediapipe 没装时**返回 None**而不是抛错，因为手势识别是辅助功能，
      不应该影响人脸追踪正常运行。
    """
    try:
        return MediaPipeHandDetector(
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
    except ImportError:
        pass
    except Exception as e:
        print(f"[hand_detector] Tasks API 初始化失败: {e}")
    # 新版不可用：尝试旧版 solutions API（mediapipe < 0.10.30）
    try:
        return _LegacyMediaPipeHandDetector(
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
    except ImportError:
        print("[hand_detector] mediapipe 未安装，跳过手势识别功能（pip install mediapipe）")
        return None
    except Exception as e:
        print(f"[hand_detector] 初始化失败，跳过手势识别功能: {e}")
        return None
