"""
face_tracking_service.py
人脸追踪 Service。

功能：
- 周期性从摄像头采集一帧画面，使用 FaceDetector（默认 MNN，回退 Haar）检测人脸；
- 根据人脸相对画面中心的偏移，向 motor1~5 下发增量目标位置，使人脸保持在画面中心；
- 当 AnimationService 正在播放非 idle 动作 (is_busy=True) 时，自动暂停追踪；
- 持续检测不到人脸时，停止追踪、让 AnimationService 重新播放 idle；
- 重新检测到人脸时，停止 idle，由本服务接管电机指令；
- 所有下发到电机的目标值都被限制在"基准姿态 ± 安全行程"内，
  且距舵机硬极限 ±100 至少留 hard_margin 单位余量，避免堵转。

设计要点：
1. 不独占串口：本服务不直接连接 robot，而是通过 AnimationService 暴露的
   send_tracking_action() 接口下发指令，与 idle/动作播放共享同一条总线。
2. 单位制：项目里 LeLampFollower 默认 MotorNormMode.RANGE_M100_100，
   send_action 的 *.pos 值范围是 -100 ~ +100。本服务所有"位置/步长/限位"
   都按此单位制；行程用 idle 第一帧作为基准 (_base_pose)，相对基准做增量。
3. 追踪激活联动：
   - 检测到人脸  -> set_tracking_active(True)：AnimationService 暂停 idle 循环
   - 持续无人脸  -> set_tracking_active(False)：AnimationService 恢复 idle
4. 检测器抽象：通过 face_detector.create_default_detector() 创建检测器实例，
   优先使用 RKNN（RetinaFace.rknn，rk3588 板载 NPU），不可用时依次回退到
   MNN（face_detect_320.mnn）、Haar Cascade。抽象接口让追踪服务无需关心底层模型。
5. 5 个电机均参与追踪（按耦合系数分配）：
     base_yaw   (id=1)   ← 主左右
     base_pitch (id=2)   ← 主上下
     elbow_pitch(id=3)   ← 辅助上下
     wrist_pitch(id=5)   ← 辅助上下，保持灯头朝向人脸
     wrist_roll (id=4)   ← 默认不参与
6. 人体感应联动：设备提供人体检测传感器（GPIO9/level_state，经 RGBService
   串口上报，见 set_human_sensor_state()）。人体静止(level_state=0)且人脸
   左右移动距离很小时冻结跟踪，消除检测漂移导致的机械臂左右抖动；
   人体移动(level_state=1)或移动距离较大时恢复正常跟踪。
7. 惯性滑行（coast）：人脸可见期间始终估计运动速度/方向；当人脸快速
   移出画面（丢失瞬间速度 >= coast_min_speed）时，摄像头沿最后记录的
   运动方向继续滑行 coast_frames 小步（累计不超过 coast_max_total），
   主动"追出去"，提高人脸被重新捕获的概率。
"""

import gc
import time
import threading
from collections import deque
from typing import Any, Optional, Dict, Callable

from ..base import ServiceBase
from .face_detector import FaceDetector, create_default_detector
from .hand_detector import HandDetector, create_default_hand_detector

try:
    import cv2
except ImportError:
    cv2 = None


class FaceTrackingService(ServiceBase):
    # ────────────────────────────────────────────────────────────────────────
    # 默认参数 —— 可通过构造函数覆盖
    # ────────────────────────────────────────────────────────────────────────
    DEFAULT_CAMERA_INDEX = 0
    DEFAULT_FRAME_WIDTH = 320
    DEFAULT_FRAME_HEIGHT = 240
    DEFAULT_DETECT_INTERVAL = 0.05   # 秒，两次检测之间的最小间隔 
                                    # MNN 在 Pi 上 ≈ 10 FPS，0.1s 能跑满；
                                    # Haar 较慢，自动会被实际推理时间拉长
    DEFAULT_DEAD_ZONE_X = 0.10      # 水平方向死区（画面比例）
    DEFAULT_DEAD_ZONE_Y = 0.15      # 垂直方向死区（画面比例，比 X 更宽——
                                    # 因为俯仰由三轴叠加，对 off_y 更敏感，需要更大的死区防抖）
    DEFAULT_MAX_STEP = 8.0          # 单帧最大位置变化（≈ 6% 全行程） 
    DEFAULT_LOST_TIMEOUT = 3.0      # 兼容字段，实际现在用 LOST_FRAMES_THRESHOLD
    DEFAULT_LOST_FRAMES_THRESHOLD = 15  # 连续多少帧丢失才算"真丢失"
                                         # 配 ~10 FPS 即 1.5 秒容忍——MNN 偶尔漏检常见，
                                         # 太敏感会导致频繁"idle 接管 → 再激活"的切换

    # ── 静止稳定（解决"人静止时机械臂仍微动"的问题） ──
    # 检测器（如 MNN）每帧吐出的人脸框中心位置有 ±5~15 像素抖动，
    # 配合速度滤波会被放大成虚假"速度"，让灯持续微调。
    # 三层组合防御：
    #   1. CENTER_RECT_*：测量人脸中心落在画面正中矩形内 → 完全不更新
    #   2. STATIC_VELOCITY_THRESH：滤波后速度过小直接归零，不外推
    #   3. POSE_DELTA_THRESH：所有舵机的目标变化都很小 → 不下发指令
    DEFAULT_CENTER_RECT_X = 0.16    # 中心矩形半宽（画面宽度比例），人脸进矩形就不动
    DEFAULT_CENTER_RECT_Y = 0.18    # 中心矩形半高（画面高度比例）
    DEFAULT_STATIC_VELOCITY_THRESH = 150.0  # 像素/秒。滤波速度 |v| < 此值视为静止
                                             # 偏大对真实快速移动响应略迟，但能有效抑制检测抖动
    DEFAULT_POSE_DELTA_THRESH = 0.5         # 归一化单位。目标姿态变化 < 此值时跳过下发

    # ── 人体感应联动（消除"人静止时检测漂移 → 机械臂左右抖动"的问题） ──
    # 设备提供人体检测传感器（GPIO9/level_state，经 RGBService 串口上报）：
    #   level_state = 0 → 未检测到人体 或 人体静止不动
    #   level_state = 1 → 检测到人体 或 人体移动
    # 策略：人体静止(level_state=0)且人脸左右移动距离很小时，判定画面偏移是
    # 检测器漂移而非真实移动 → 冻结跟踪（不下发任何电机指令）；
    # 人体移动(level_state=1)或左右移动距离较大时 → 继续正常跟踪。
    DEFAULT_HUMAN_SENSOR_ENABLED = True     # 总开关；传感器缺数据时默认"有人/移动"不受影响
    DEFAULT_HUMAN_STATIC_MOVE_THRESH = 0.08 # 归一化水平移动阈值（画面宽度比例）。
                                            # 0.08 × 320px ≈ 26px：检测框漂移(±5~15px)不触发，
                                            # 真实头部左右移动(通常 >30px)会触发恢复跟踪

    # ── 人脸快速移出后的"惯性滑行"（coast） ──
    # 问题：人脸快速向某方向移动出画面时，摄像头来不及跟上；人脸一出画面
    # 追踪就丢了目标停在原地，等再检测到人脸时偏差已经很大甚至再也回不来。
    # 优化：人脸可见期间持续估计运动速度/方向（见 _vx/_vy_smooth，始终运行）；
    # 当判定人脸是"快速移出"（丢失瞬间速度 >= COAST_MIN_SPEED）时，
    # 让摄像头沿最后记录的运动方向继续滑行 COAST_FRAMES 小步，
    # 主动"追出去"，提高人脸移回画面边缘时被立刻重新捕获的概率。
    DEFAULT_COAST_ENABLED = True
    DEFAULT_COAST_FRAMES = 9            # 水平滑行帧数配额（~10FPS 下 ≈ 0.8s）
    DEFAULT_COAST_FRAMES_Y = 13         # 垂直（上下）滑行帧数配额，默认比水平大——
                                        # 垂直每帧推进量更小（俯仰增益 9/3/6 << yaw 18，
                                        # 且不常触发 max_step=6 的单帧上限），
                                        # 同样 8 帧滑出的距离明显比左右短。
                                        # 注意应 < lost_frames_threshold(15)，
                                        # 否则会被"判定丢失恢复 idle"提前截断。
    DEFAULT_COAST_MIN_SPEED = 250.0     # 像素/秒。水平方向触发门槛：丢失瞬间 |vx| 低于此值
                                        # 视为漏检/静止丢失，不滑行
    DEFAULT_COAST_MIN_SPEED_Y = 180.0   # 像素/秒。垂直方向独立触发门槛，比水平低——
                                        # 画面高(240px)比宽(320px)短，人脸上下移出的可检测
                                        # 行程更短、IIR 滤波速度来不及爬升，丢失瞬间 |vy|
                                        # 往往只有 180~230，用 250 会几乎不触发垂直滑行。
                                        # 注意需 > static_velocity_thresh(150) 否则会被 Layer2 归零
    DEFAULT_COAST_SPEED_REF = 1000.0    # 像素/秒。速度→滑行强度映射：该速度对应满幅虚拟偏移(±1.0)
    DEFAULT_COAST_PITCH_BOOST = 3.0     # 垂直滑行强度放大系数。垂直偏移被拆到 3 个俯仰关节
                                        # (增益 9/3/3，远小于水平的 yaw=18)，不放大时每帧只
                                        # 推动 2~3 单位，肉眼几乎看不出"向上/向下追"。
                                        # 2.0 表示垂直虚拟偏移 ×2 后再参与增益计算；
                                        # 仍受 max_step / 关节软限位 / coast_max_total 约束。
    DEFAULT_COAST_MAX_TOTAL = 20.0      # 归一化单位。单关节累计滑行量上限（"只多走一小段"的保险）

    # ── 人脸位置预测（外推 + 滤波）参数 ──
    # 目的：根据人脸运动速度提前预判它即将所在的位置，让灯一次移动得稍远些，
    # 减少"跟不上 → 反复小幅修正 → 抖动"的体验。
    DEFAULT_PREDICTION_ENABLED = False
    DEFAULT_PREDICTION_LOOKAHEAD = 0.25   # 秒。前瞻时间——预测 N 秒后人脸位置。
                                          # 经验值：1-2 倍单帧周期。太大易过冲，太小退化为反应控制。
    DEFAULT_VELOCITY_ALPHA = 0.3          # 速度低通滤波系数，∈[0,1]。
                                          # α 越大滤波越强、越滞后；0 表示无滤波。
                                          # 0.5 大致相当于 1.5 帧时间常数。
    DEFAULT_MAX_VELOCITY = 1500.0         # 像素/秒。速度估计上限，防止误检导致疯狂外推。
                                          # 1500 px/s ≈ 5 倍画面宽度/秒，足够人正常移动。
    DEFAULT_MAX_PRED_OFFSET = 1.5         # 预测偏移裁剪 |off|≤1.5（画面外一点点）。
                                          # 防止外推飞太远把 yaw 直接拉满。
    DEFAULT_HARD_MARGIN = 5.0       # 距舵机硬极限 ±100 至少留多少单位余量（防堵转）
                                    # 例如 5 表示实际允许范围 [-95, +95]

    # ── 手势识别相关默认参数 ──
    # 手势检测在独立的低优先级后台线程跑，与人脸追踪线程互不干扰。
    # MediaPipe Hands 单帧 30~80ms（Pi 上），如果每帧都跑会拖慢追踪 FPS，
    # 所以默认每 3 帧抽样一次（约 5-10 FPS 的手势检测频率，足够指向手势识别）。
    DEFAULT_HAND_ENABLED = True              # 总开关；mediapipe 没装时自动失效
    DEFAULT_HAND_FRAME_STRIDE = 3            # 每 N 个相机帧检测一次手势（>=1）
    DEFAULT_HAND_BUFFER_SIZE = 7             # 方向投票缓冲长度
    DEFAULT_HAND_VOTE_THRESHOLD = 2          # 缓冲内同一方向出现 >=N 次才触发
    DEFAULT_HAND_COOLDOWN_DETECTIONS = 8     # 触发一次后冷却多少次检测（不是相机帧）

    # 偏移 → 各电机位置增量的耦合系数
    # 输入：归一化偏移 ∈ [-1, 1]（-1 = 最左/最上，+1 = 最右/最下）
    # 输出：归一化位置增量（单位与电机一致：-100 ~ +100）
    #
    # 设计：base_pitch 是主俯仰，给大增益；elbow/wrist 是辅助，
    # 增益取主增益的 1/3，避免三轴同时撞限位（实测会过冲）。
    # 例外：wrist_pitch 是灯头前最后一个关节，其转动直接改变灯头朝向，
    # 1/3 增益(3.0)实测转动不明显——人脸快速上下移动/移出时几乎看不到
    # 跟随，所以默认提升到主增益的 2/3(6.0)，且支持运行期调节
    # （见 wrist_pitch_gain 参数 / "set_wrist_pitch_gain" 事件）。
    YAW_GAIN_X         = 18.0  # base_yaw 跟随水平偏移（主）
    BASE_PITCH_GAIN_Y  = 9.0   # base_pitch 跟随垂直偏移（主，机械限位紧）
    ELBOW_GAIN_Y       = 3.0   # elbow_pitch 辅助俯仰（base_pitch 的 1/3 步长）
    DEFAULT_WRIST_PITCH_GAIN_Y = 6.0  # wrist_pitch 辅助俯仰（主增益的 2/3，
                                      # 原为 3.0 时转动不明显；运行期可调）
    # 俯仰增益总缩放：整体放大三个俯仰关节的垂直响应速度。
    # 背景（实测日志）：max_step=8 时水平 yaw 常态打满 ~8 单位/帧，
    # 而垂直 base_pitch 仅 9×|off_y| ≈ 3~5 单位/帧——垂直累积速度只有
    # 水平的一半左右，观感"上下跟踪迟钝、左右正常"。
    # 1.5×：base 13.5×0.6≈8 → 常态打满 max_step，与水平对齐；
    # 同时放大 elbow/wrist（比例不变），受 max_step/软限位双重约束不会过冲太远。
    DEFAULT_PITCH_GAIN_SCALE = 1.5     # 运行期可调（set_pitch_gain_scale）
    WRIST_ROLL_GAIN    = 0.0   # 默认不参与；如需小幅修正可设非零

    # 方向反转开关：每个关节的"灯头朝上"对应的数值方向不一定一样，
    # 所以三个俯仰轴各自独立配置。机械装配决定，无法事先知道，需要实测。
    #
    # 推导规则（关键，不要搞错）：
    #   delta = (INVERT ? -1 : +1) * GAIN * offset_y
    #   人脸在画面上方时 offset_y < 0，灯头需要朝上抬。
    #   - 关节"数值减小 = 灯头朝上"（如 elbow_pitch idle=+100，减小才能抬）：
    #       要让 delta 在 off_y<0 时为负 → 要求 sign=+1 → INVERT = False
    #   - 关节"数值增大 = 灯头朝上"：
    #       要让 delta 在 off_y<0 时为正 → 要求 sign=-1 → INVERT = True
    #
    # 已知（来自实测/用户反馈）：
    #   - elbow_pitch: 数值减小 = 抬头  → INVERT_ELBOW = False
    # 未确认（先猜，实测反了再翻转——只翻转那一个）：
    #   - base_pitch:  先给 False（先按"减小=抬头"假设）
    #   - wrist_pitch: 先给 False
    INVERT_X            = False  # base_yaw / wrist_roll 方向反转
    INVERT_BASE_PITCH   = False  # base_pitch 方向反转（实测确认）
    INVERT_ELBOW        = False  # elbow_pitch 方向反转（已确认：上=减小→False）
    INVERT_WRIST_PITCH  = False  # wrist_pitch 方向反转（实测确认）

    # 关节相对"基准姿态"的允许行程（归一化单位）：
    #   (向下/向减小方向, 向上/向增大方向)
    # 根据真实 idle 基准设计，注意非对称——
    # 例如 idle elbow_pitch=+100 已经撞顶，向上余量 0，只能向下作为辅助俯仰。
    #
    # 最终限位 = base + (down/up)，且不会突破 [-100+HARD_MARGIN, +100-HARD_MARGIN]
    # = [-95, +95]（防堵转兜底）。
    #
    # 数值依据：
    # - base_yaw（基准+2.76）：两侧都宽裕，对称给 35
    # - base_pitch（基准+77.78，向上余 12.22）：向上只能小心给，向下放开
    # - elbow_pitch（基准+100，已撞顶）：只能向下作为辅助，向上 = 0
    # - wrist_pitch（基准-12.34）：两侧都宽，对称给 20
    # - wrist_roll（基准+4.03）：默认不参与
    RANGE_FROM_BASE: Dict[str, tuple] = {
        # 关节            (down, up)
        "base_yaw":       (35.0, 35.0),
        "base_pitch":     (30.0, 10.0),   # 向上仅 10，避免撞 +90 之上的极限
        "elbow_pitch":    (30.0,  0.0),   # idle=+100，只能向下做辅助
        "wrist_roll":     ( 5.0,  5.0),
        "wrist_pitch":    (20.0, 20.0),
    }

    # 各关节的绝对硬限位（归一化单位）。LeLampFollower 的舵机模式 RANGE_M100_100
    # 把机械极限映射到 ±100。追踪指令最终都会被夹到 [-100+hard_margin, +100-hard_margin]
    # 之内。注意：如果 idle 基准本身已经在这个兜底范围之外（如 elbow_pitch=+100），
    # _build_joint_limits 会保护性地不强制把 base 拉走，只阻止追踪指令向极限方向推。
    HARD_LIMIT_ABS = 100.0

    JOINT_KEYS = ("base_yaw", "base_pitch", "elbow_pitch", "wrist_roll", "wrist_pitch")

    def __init__(
        self,
        anim_service,                       # AnimationService 实例（必须）
        camera_index: int = DEFAULT_CAMERA_INDEX,
        frame_width: int = DEFAULT_FRAME_WIDTH,
        frame_height: int = DEFAULT_FRAME_HEIGHT,
        detect_interval: float = DEFAULT_DETECT_INTERVAL,
        dead_zone_x: float = DEFAULT_DEAD_ZONE_X,
        dead_zone_y: float = DEFAULT_DEAD_ZONE_Y,
        max_step: float = DEFAULT_MAX_STEP,
        lost_frames_threshold: int = DEFAULT_LOST_FRAMES_THRESHOLD,
        hard_margin: float = DEFAULT_HARD_MARGIN,
        # 预测相关参数
        prediction_enabled: bool = DEFAULT_PREDICTION_ENABLED,
        prediction_lookahead: float = DEFAULT_PREDICTION_LOOKAHEAD,
        velocity_alpha: float = DEFAULT_VELOCITY_ALPHA,
        max_velocity: float = DEFAULT_MAX_VELOCITY,
        max_pred_offset: float = DEFAULT_MAX_PRED_OFFSET,
        # 静止稳定参数
        center_rect_x: float = DEFAULT_CENTER_RECT_X,
        center_rect_y: float = DEFAULT_CENTER_RECT_Y,
        static_velocity_thresh: float = DEFAULT_STATIC_VELOCITY_THRESH,
        pose_delta_thresh: float = DEFAULT_POSE_DELTA_THRESH,
        # 人体感应联动参数
        human_sensor_enabled: bool = DEFAULT_HUMAN_SENSOR_ENABLED,
        human_static_move_thresh: float = DEFAULT_HUMAN_STATIC_MOVE_THRESH,
        # 惯性滑行参数
        coast_enabled: bool = DEFAULT_COAST_ENABLED,
        coast_frames: int = DEFAULT_COAST_FRAMES,
        coast_frames_y: int = DEFAULT_COAST_FRAMES_Y,
        coast_min_speed: float = DEFAULT_COAST_MIN_SPEED,
        coast_min_speed_y: float = DEFAULT_COAST_MIN_SPEED_Y,
        coast_speed_ref: float = DEFAULT_COAST_SPEED_REF,
        coast_pitch_boost: float = DEFAULT_COAST_PITCH_BOOST,
        coast_max_total: float = DEFAULT_COAST_MAX_TOTAL,
        # wrist_pitch 俯仰增益（默认 6.0 = 主增益的 2/3；3.0 时转动不明显）
        wrist_pitch_gain: float = DEFAULT_WRIST_PITCH_GAIN_Y,
        # 俯仰增益总缩放（默认 1.5：让垂直单帧推进量对齐水平 yaw）
        pitch_gain_scale: float = DEFAULT_PITCH_GAIN_SCALE,
        detector: Optional[FaceDetector] = None,
        rknn_model_path: Optional[str] = None,
        mnn_model_path: Optional[str] = None,
        # ── 手势识别相关 ──
        hand_enabled: bool = DEFAULT_HAND_ENABLED,
        hand_frame_stride: int = DEFAULT_HAND_FRAME_STRIDE,
        hand_buffer_size: int = DEFAULT_HAND_BUFFER_SIZE,
        hand_vote_threshold: int = DEFAULT_HAND_VOTE_THRESHOLD,
        hand_cooldown_detections: int = DEFAULT_HAND_COOLDOWN_DETECTIONS,
        hand_detector: Optional[HandDetector] = None,
        on_hand_pointing: Optional[Callable[[str], None]] = None,
    ):
        super().__init__("face_tracking")

        if cv2 is None:
            raise ImportError(
                "OpenCV (cv2) 未安装，无法启动人脸追踪服务。请先 `pip install opencv-python`。"
            )

        self.anim_service = anim_service
        self.camera_index = camera_index
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.detect_interval = detect_interval
        self.dead_zone_x = dead_zone_x
        self.dead_zone_y = dead_zone_y
        self.max_step = max_step
        self.lost_frames_threshold = max(1, int(lost_frames_threshold))
        # 距离舵机硬极限 ±100 的安全余量（归一化单位）
        # 例如 5 → 实际允许范围 [-95, +95]，至少留 5 单位防止堵转
        self.hard_margin = max(0.0, float(hard_margin))

        # 预测参数（运行期可改）
        self.prediction_enabled = bool(prediction_enabled)
        self.prediction_lookahead = max(0.0, float(prediction_lookahead))
        self.velocity_alpha = max(0.0, min(1.0, float(velocity_alpha)))
        self.max_velocity = max(0.0, float(max_velocity))
        self.max_pred_offset = max(0.5, float(max_pred_offset))

        # 静止稳定参数（运行期可改）
        self.center_rect_x = max(0.0, float(center_rect_x))
        self.center_rect_y = max(0.0, float(center_rect_y))
        self.static_velocity_thresh = max(0.0, float(static_velocity_thresh))
        self.pose_delta_thresh = max(0.0, float(pose_delta_thresh))

        # ── 人体感应联动状态 ──
        # human_sensor_enabled：总开关。关闭后完全不依赖传感器，保持原有行为。
        # _human_present：True=检测到人体/人体移动(level_state=1)；
        #                 False=无人或人体静止(level_state=0)。
        # 默认 True——未接入/未收到传感器上报时，追踪行为与原来完全一致。
        # 由 RGBService 的串口接收线程通过 set_human_sensor_state() 更新。
        self.human_sensor_enabled = bool(human_sensor_enabled)
        self.human_static_move_thresh = max(0.0, float(human_static_move_thresh))
        self._human_present: bool = True
        self._human_sensor_lock = threading.Lock()
        # 人体静止期间的"冻结参考位置"：进入静止态时的人脸水平像素位置。
        # 之后每帧用人脸中心 x 与它比较——偏离很小视为检测漂移（继续冻结），
        # 偏离超过阈值视为真实移动（恢复跟踪）。仅在追踪线程内读写。
        self._static_ref_cx: Optional[float] = None

        # ── 惯性滑行状态（仅追踪线程读写） ──
        # 人脸快速移出画面时，沿最后记录的运动方向继续滑行几小步。
        # _coast_vx/_coast_vy：丢失瞬间（_lost_frames 从 0 变 1 时）捕获的滤波速度；
        # _coast_frames_left：剩余滑行帧数配额；
        # _coast_accum：滑行期间各关节累计推进量（用于 coast_max_total 封顶）。
        self.coast_enabled = bool(coast_enabled)
        self.coast_frames = max(0, int(coast_frames))
        self.coast_frames_y = max(0, int(coast_frames_y))
        self.coast_min_speed = max(0.0, float(coast_min_speed))
        self.coast_min_speed_y = max(0.0, float(coast_min_speed_y))
        self.coast_speed_ref = max(1.0, float(coast_speed_ref))
        self.coast_pitch_boost = max(1.0, float(coast_pitch_boost))
        self.coast_max_total = max(0.0, float(coast_max_total))
        # wrist_pitch 俯仰增益（运行期可调）：wrist 是灯头前最后一个关节，
        # 转动直接改变灯头朝向；增益太小人脸上下移动时看不出跟随
        self.wrist_pitch_gain = float(wrist_pitch_gain)
        # 俯仰增益总缩放（运行期可调）：整体放大三个俯仰关节的响应速度，
        # 解决"上下跟踪比左右迟钝"（垂直常态推进量只有水平的一半左右）
        self.pitch_gain_scale = max(0.1, float(pitch_gain_scale))
        self._coast_vx: float = 0.0
        self._coast_vy: float = 0.0
        self._coast_frames_left: int = 0
        self._coast_accum: Dict[str, float] = {k: 0.0 for k in self.JOINT_KEYS}

        # 速度估计与上一帧状态（_on_face_detected 维护）
        self._prev_face_cx: Optional[float] = None
        self._prev_face_cy: Optional[float] = None
        self._prev_face_t: Optional[float] = None
        self._vx_smooth: float = 0.0   # 像素/秒，X 方向滤波速度
        self._vy_smooth: float = 0.0   # 像素/秒，Y 方向滤波速度

        # 检测器在追踪线程里初始化（避免主线程阻塞），这里只保存配置
        self._injected_detector = detector
        self._rknn_model_path = rknn_model_path
        self._mnn_model_path = mnn_model_path

        # 摄像头与检测器在 worker 线程里初始化
        self._cap: Optional["cv2.VideoCapture"] = None
        self._detector: Optional[FaceDetector] = None

        # 追踪线程
        self._track_thread: Optional[threading.Thread] = None

        # 基准姿态：从 idle 第一帧加载，作为 _target_pose 的"原点"，
        # 也是限位的中心。开机加载一次，运行期不再变。单位：归一化 -100~+100
        self._base_pose: Dict[str, float] = {k: 0.0 for k in self.JOINT_KEYS}

        # 实际生效的关节限位 [lo, hi]：根据 base_pose + 非对称 RANGE_FROM_BASE
        # 由 _build_joint_limits 计算，并用 hard_margin 做兜底防堵转
        self._joint_limits: Dict[str, tuple] = {}

        # 当前下发到电机的目标位置，初始 = 基准姿态
        self._target_pose: Dict[str, float] = dict(self._base_pose)

        # 上一次成功检测到人脸的时间戳（0 表示从未检测到）
        self._last_seen: float = 0.0

        # 连续未检测到人脸的帧数（按帧计数比按时间计数更抗 Haar 漏检）
        self._lost_frames: int = 0

        # 当前是否处于"追踪激活"状态。只在状态切换时通知 AnimationService。
        self._tracking_active: bool = False

        # 外部 pause/resume 的开关
        self._paused: bool = False

        # ── 手势识别相关状态 ──
        # mediapipe 推理较慢且耗 CPU，放在独立线程；通过 _latest_frame 与抓帧线程共享帧。
        self.hand_enabled = bool(hand_enabled)
        self.hand_frame_stride = max(1, int(hand_frame_stride))
        self.hand_buffer_size = max(1, int(hand_buffer_size))
        self.hand_vote_threshold = max(1, int(hand_vote_threshold))
        self.hand_cooldown_detections = max(0, int(hand_cooldown_detections))
        self._injected_hand_detector = hand_detector
        # 回调：识别到稳定指向时调用 on_hand_pointing("Up"/"Down"/"Left"/"Right")
        # 默认 print 一行，与原 hand_recongnize2.py 行为一致
        self._on_hand_pointing: Callable[[str], None] = (
            on_hand_pointing if on_hand_pointing is not None
            else self._default_on_hand_pointing
        )
        self._hand_detector: Optional[HandDetector] = None
        self._hand_thread: Optional[threading.Thread] = None

    # ====================================================================
    # ServiceBase 接口
    # ====================================================================
    def start(self):
        super().start()
        self._track_thread = threading.Thread(
            target=self._tracking_loop, name="face_tracking_loop", daemon=True
        )
        self._track_thread.start()
        self.logger.info("人脸追踪服务已启动")

    def stop(self, timeout: float = 5.0):
        super().stop(timeout)
        if self._track_thread and self._track_thread.is_alive():
            self._track_thread.join(timeout=timeout)
        # 手势线程也用 _running 标志，super().stop() 已 clear；这里只 join
        if self._hand_thread and self._hand_thread.is_alive():
            self._hand_thread.join(timeout=timeout)
        if self._hand_detector is not None:
            try:
                self._hand_detector.close()
            except Exception:
                pass
            self._hand_detector = None
        # 退出前确保把"追踪激活"标志清掉，让 idle 恢复
        self._set_tracking_active(False)
        self._release_camera()
        self.logger.info("人脸追踪服务已停止")

    def handle_event(self, event_type: str, payload: Any):
        """支持外部 dispatch 的运行期控制事件"""
        if event_type == "pause":
            self._paused = True
            # 暂停时也要把追踪激活清掉，让 idle 接管
            self._set_tracking_active(False)
            self.logger.info("人脸追踪已暂停 (外部 pause)")
        elif event_type == "resume":
            self._paused = False
            self.logger.info("人脸追踪已恢复 (外部 resume)")
        elif event_type == "recenter":
            # 把目标姿态重置为基准姿态，下一帧追踪生效时下发
            self._target_pose = dict(self._base_pose)
            self.logger.info("已请求人脸追踪复位到基准姿态")
        elif event_type == "prediction_on":
            self.prediction_enabled = True
            self._reset_prediction()
            self.logger.info("已开启位置预测")
        elif event_type == "prediction_off":
            self.prediction_enabled = False
            self._reset_prediction()
            self.logger.info("已关闭位置预测（退化为反应式追踪）")
        elif event_type == "set_lookahead":
            # payload = 秒数；用于现场调参
            try:
                v = float(payload)
                self.prediction_lookahead = max(0.0, v)
                self.logger.info(f"lookahead 已设为 {v}s")
            except Exception as e:
                self.logger.warning(f"set_lookahead payload 不合法: {payload} ({e})")
        elif event_type == "set_velocity_alpha":
            try:
                v = float(payload)
                self.velocity_alpha = max(0.0, min(1.0, v))
                self.logger.info(f"velocity_alpha 已设为 {v}")
            except Exception as e:
                self.logger.warning(f"set_velocity_alpha payload 不合法: {payload} ({e})")
        elif event_type == "set_center_rect":
            # payload = (x, y) 或单值（同时设 x 和 y）
            try:
                if isinstance(payload, (tuple, list)) and len(payload) == 2:
                    self.center_rect_x = max(0.0, float(payload[0]))
                    self.center_rect_y = max(0.0, float(payload[1]))
                else:
                    v = max(0.0, float(payload))
                    self.center_rect_x = v
                    self.center_rect_y = v
                self.logger.info(
                    f"中心矩形已设为 ±{self.center_rect_x}×±{self.center_rect_y}"
                )
            except Exception as e:
                self.logger.warning(f"set_center_rect payload 不合法: {payload} ({e})")
        elif event_type == "set_static_velocity_thresh":
            try:
                v = max(0.0, float(payload))
                self.static_velocity_thresh = v
                self.logger.info(f"static_velocity_thresh 已设为 {v} px/s")
            except Exception as e:
                self.logger.warning(f"set_static_velocity_thresh payload 不合法: {payload} ({e})")
        elif event_type == "set_human_sensor":
            # payload = 0/1；用于外部（或测试）模拟人体感应上报
            try:
                self.set_human_sensor_state(int(payload))
            except Exception as e:
                self.logger.warning(f"set_human_sensor payload 不合法: {payload} ({e})")
        elif event_type == "human_sensor_enable":
            # 运行期开关人体感应联动；关闭后恢复不依赖传感器的原有行为
            self.human_sensor_enabled = bool(payload) if payload is not None else True
            if not self.human_sensor_enabled:
                self._static_ref_cx = None
            self.logger.info(
                f"人体感应联动已 {'启用' if self.human_sensor_enabled else '禁用'}"
            )
        elif event_type == "set_human_static_move_thresh":
            try:
                self.human_static_move_thresh = max(0.0, float(payload))
                self.logger.info(f"human_static_move_thresh 已设为 {self.human_static_move_thresh}")
            except Exception as e:
                self.logger.warning(f"set_human_static_move_thresh payload 不合法: {payload} ({e})")
        elif event_type == "coast_enable":
            # 运行期开关"人脸快速移出后的惯性滑行"
            self.coast_enabled = bool(payload) if payload is not None else True
            if not self.coast_enabled:
                self._coast_frames_left = 0
            self.logger.info(f"惯性滑行已 {'启用' if self.coast_enabled else '禁用'}")
        elif event_type == "set_coast_frames":
            # payload = 帧数；水平滑行帧数配额
            try:
                self.coast_frames = max(0, int(payload))
                self.logger.info(f"coast_frames 已设为 {self.coast_frames}")
            except Exception as e:
                self.logger.warning(f"set_coast_frames payload 不合法: {payload} ({e})")
        elif event_type == "set_coast_frames_y":
            # payload = 帧数；垂直（上下）滑行帧数配额，调大可让上下滑行更远
            try:
                self.coast_frames_y = max(0, int(payload))
                self.logger.info(f"coast_frames_y 已设为 {self.coast_frames_y}")
            except Exception as e:
                self.logger.warning(f"set_coast_frames_y payload 不合法: {payload} ({e})")
        elif event_type == "set_coast_max_total":
            # payload = 归一化单位；单关节累计滑行量上限
            try:
                self.coast_max_total = max(0.0, float(payload))
                self.logger.info(f"coast_max_total 已设为 {self.coast_max_total}")
            except Exception as e:
                self.logger.warning(f"set_coast_max_total payload 不合法: {payload} ({e})")
        elif event_type == "set_coast_speed_ref":
            # payload = 像素/秒；达到该速度即满幅滑行，调小可让滑行更明显
            try:
                self.coast_speed_ref = max(1.0, float(payload))
                self.logger.info(f"coast_speed_ref 已设为 {self.coast_speed_ref}")
            except Exception as e:
                self.logger.warning(f"set_coast_speed_ref payload 不合法: {payload} ({e})")
        elif event_type == "set_coast_min_speed_y":
            # payload = 像素/秒；垂直方向滑行触发门槛
            try:
                self.coast_min_speed_y = max(0.0, float(payload))
                self.logger.info(f"coast_min_speed_y 已设为 {self.coast_min_speed_y}")
            except Exception as e:
                self.logger.warning(f"set_coast_min_speed_y payload 不合法: {payload} ({e})")
        elif event_type == "set_coast_pitch_boost":
            # payload = 倍数；垂直滑行强度放大系数（>=1.0）
            try:
                self.coast_pitch_boost = max(1.0, float(payload))
                self.logger.info(f"coast_pitch_boost 已设为 {self.coast_pitch_boost}")
            except Exception as e:
                self.logger.warning(f"set_coast_pitch_boost payload 不合法: {payload} ({e})")
        elif event_type == "set_wrist_pitch_gain":
            # payload = 增益；wrist_pitch 俯仰跟随强度（默认 6.0）
            # 实测若腕部转动仍不明显可加大（如 9.0）；过冲则减小
            try:
                self.wrist_pitch_gain = float(payload)
                self.logger.info(f"wrist_pitch_gain 已设为 {self.wrist_pitch_gain}")
            except Exception as e:
                self.logger.warning(f"set_wrist_pitch_gain payload 不合法: {payload} ({e})")
        elif event_type == "set_pitch_gain_scale":
            # payload = 倍数；俯仰三关节增益总缩放（默认 1.5）。
            # 上下跟踪迟钝→调大（如 1.8~2.0）；上下过冲/抖动→调小（如 1.2）
            try:
                self.pitch_gain_scale = max(0.1, float(payload))
                self.logger.info(f"pitch_gain_scale 已设为 {self.pitch_gain_scale}")
            except Exception as e:
                self.logger.warning(f"set_pitch_gain_scale payload 不合法: {payload} ({e})")
        elif event_type == "hand_enable":
            # 运行期暂时启用 / 禁用手势触发回调
            # 注意：这里只切换"是否触发回调"的标志，不重启 mediapipe 线程，
            # 因为重新加载模型耗时较大。如要彻底关掉线程请重启服务。
            self.hand_enabled = bool(payload) if payload is not None else True
            self.logger.info(f"手势识别已 {'启用' if self.hand_enabled else '禁用'} (运行期开关)")
        elif event_type == "set_hand_stride":
            try:
                v = max(1, int(payload))
                self.hand_frame_stride = v
                self.logger.info(f"hand_frame_stride 已设为 {v}")
            except Exception as e:
                self.logger.warning(f"set_hand_stride payload 不合法: {payload} ({e})")
        elif event_type == "set_hand_callback":
            # 运行期更换回调（例如把"打印"换成"播放动作"）
            if callable(payload):
                self._on_hand_pointing = payload
                self.logger.info("手势回调已更新")
            else:
                self.logger.warning("set_hand_callback payload 必须可调用")
        else:
            self.logger.warning(f"未知事件类型: {event_type}")

    # ====================================================================
    # 摄像头管理
    # ====================================================================
    def _open_camera(self) -> bool:
        try:
            # 优先用 V4L2 后端（树莓派上更稳）
            cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
            if not cap.isOpened():
                # 回退到默认后端
                cap = cv2.VideoCapture(self.camera_index)

            # 关键：尽量在分辨率/帧率之前设 FOURCC=MJPG。USB 摄像头默认 YUYV 时
            # 320x240@30 ≈ 4.6 MB/s，而 USB 2.0 上限 ~30 MB/s 看起来够，但
            # 实测树莓派 3B 上经常协商到 5-10 FPS（驱动保守）。MJPG 编码后
            # 带宽降到 ~0.3 MB/s，几乎一定能跑到 30 FPS。
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            except Exception:
                pass
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            cap.set(cv2.CAP_PROP_FPS, 30)            # 显式请求 30 FPS
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                self.logger.error(f"无法打开摄像头 index={self.camera_index}")
                return False
            self._cap = cap

            # 打印实际协商到的格式（便于诊断）
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            try:
                cc = int(cap.get(cv2.CAP_PROP_FOURCC))
                fourcc = "".join([chr((cc >> 8 * i) & 0xFF) for i in range(4)])
            except Exception:
                fourcc = "?"
            print(f"[face_tracking] 摄像头协商: FOURCC={fourcc} FPS={actual_fps:.1f}")

            # 关键：用 cap.get() 读出来的尺寸不一定准，必须真抓一帧用 frame.shape 校准。
            # 否则当摄像头不接受我们 set 的目标尺寸时，frame_width/height 就是错的，
            # 导致画面中心计算偏移，人脸"靠右"时被算成"靠左"，追踪方向反了。
            ok = False
            for _ in range(5):  # 摄像头第一帧常常 ret=False，多试几次
                ret, frame = cap.read()
                if ret and frame is not None:
                    h, w = frame.shape[:2]
                    self.frame_width = int(w)
                    self.frame_height = int(h)
                    ok = True
                    break
                time.sleep(0.1)

            if not ok:
                # 读不到第一帧也不强制失败，回退到 cap.get() 的值
                self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.frame_width
                self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.frame_height
                self.logger.warning(
                    "首帧读取失败，使用 cap.get() 报告的分辨率: "
                    f"{self.frame_width}x{self.frame_height}（可能不准）"
                )
            else:
                self.logger.info(
                    f"摄像头已打开，实际分辨率: {self.frame_width}x{self.frame_height}, "
                    f"画面中心: ({self.frame_width/2:.1f}, {self.frame_height/2:.1f})"
                )
            return True
        except Exception as e:
            self.logger.error(f"打开摄像头异常: {e}")
            return False

    def _release_camera(self):
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    # ====================================================================
    # 基准姿态与限位
    # ====================================================================
    def _load_base_pose_from_idle(self) -> bool:
        """
        从 anim_service 加载 idle 第一帧作为基准姿态。
        加载失败时使用全 0（虽然不太合理，但至少不会越界）。
        返回 True 表示加载成功。
        """
        try:
            if hasattr(self.anim_service, "get_idle_first_frame"):
                first = self.anim_service.get_idle_first_frame()
            else:
                first = None

            if first is None:
                self.logger.warning("无法读取 idle 第一帧，基准姿态使用全 0")
                self._base_pose = {k: 0.0 for k in self.JOINT_KEYS}
                return False

            for k in self.JOINT_KEYS:
                key = f"{k}.pos"
                if key in first:
                    self._base_pose[k] = float(first[key])
                else:
                    self._base_pose[k] = 0.0

            self.logger.info(f"基准姿态(来自 idle 首帧): {self._base_pose}")
            return True
        except Exception as e:
            self.logger.error(f"加载基准姿态失败，使用全 0: {e}")
            self._base_pose = {k: 0.0 for k in self.JOINT_KEYS}
            return False

    def _build_joint_limits(self):
        """
        根据 idle 基准 + 非对称 RANGE_FROM_BASE 构造每个关节的实际限位 [lo, hi]：
          lo = base - down
          hi = base + up
        然后用 [-100+hard_margin, +100-hard_margin] 兜底，但 **不会强制把 base 本身
        裁出兜底范围**——即使 idle 基准（如 elbow_pitch=+100）已经越过兜底，
        我们也保持 base 不变，只限制追踪指令向"危险方向"再走。

        例如：base=+100, hard_margin=5 → 实际 cap = +95
          - hi 会被裁到 max(base, +95) = +100（因为 base 比 +95 还高，不能再裁低）
          - lo = base - down 不会被影响

        这样既防止追踪超出机械极限，又不会粗暴拉走原本就在极限上的 idle 基准。
        """
        cap_lo = -self.HARD_LIMIT_ABS + self.hard_margin   # = -95
        cap_hi =  self.HARD_LIMIT_ABS - self.hard_margin   # = +95

        warnings = []
        for k in self.JOINT_KEYS:
            base = self._base_pose[k]
            down, up = self.RANGE_FROM_BASE.get(k, (10.0, 10.0))

            lo = base - down
            hi = base + up

            # 兜底裁断（关键：不会把已经越过 cap 的 base 拉走）
            if lo < cap_lo: lo = cap_lo
            if hi > cap_hi: hi = cap_hi

            # 但如果 base 本身就在 cap 之外（如 elbow=+100, cap_hi=+95），
            # 上面的裁剪会让 hi=+95 < base=+100。这时把 hi 拉回 base，
            # 否则限位 [lo, +95] 会强制把 base 从 +100 拉到 +95，启动瞬间一个跳变。
            if hi < base: hi = base
            if lo > base: lo = base

            self._joint_limits[k] = (lo, hi)

            # 检查 base 是否已贴近物理极限
            margin_to_low  = base - (-self.HARD_LIMIT_ABS)
            margin_to_high = self.HARD_LIMIT_ABS - base
            if margin_to_low < self.hard_margin:
                warnings.append(
                    f"  ⚠ {k} 基准 {base:+.2f} 距离 -{self.HARD_LIMIT_ABS} 仅余 {margin_to_low:.2f} (< hard_margin={self.hard_margin})"
                )
            if margin_to_high < self.hard_margin:
                warnings.append(
                    f"  ⚠ {k} 基准 {base:+.2f} 距离 +{self.HARD_LIMIT_ABS} 仅余 {margin_to_high:.2f} (< hard_margin={self.hard_margin})"
                )

            actual_down = base - lo
            actual_up   = hi - base
            print(
                f"[face_tracking] 限位 {k:<13}: base={base:+7.2f} "
                f"配置(down={down:.0f},up={up:.0f}) → 实际 [{lo:+7.2f}, {hi:+7.2f}] "
                f"(向下{actual_down:.1f} / 向上{actual_up:.1f})"
            )

        if warnings:
            print("[face_tracking] 基准姿态接近机械极限：")
            for w in warnings:
                print(w)
            print("  这些方向上追踪行程会受限，但不会损伤舵机。")

    # ====================================================================
    # 主追踪循环（独立线程）
    # ====================================================================
    def _tracking_loop(self):
        # 0. 降低本线程优先级，让音频采集等实时性更强的任务优先（防 input overflow）
        # nice 值范围 -20 ~ +19，值越大优先级越低；普通线程默认 0
        try:
            import os
            os.nice(5)
        except (OSError, AttributeError):
            pass  # 部分平台不支持，忽略

        # 1. 初始化检测器
        # 优先用注入的 detector；否则按 (mnn_model_path, 默认搜寻路径) 逐项尝试
        try:
            if self._injected_detector is not None:
                self._detector = self._injected_detector
                print(f"[face_tracking] 使用注入的检测器: {self._detector.name}")
            else:
                self._detector = create_default_detector(
                    rknn_model_path=self._rknn_model_path,
                    mnn_model_path=self._mnn_model_path,
                )
                print(f"[face_tracking] 使用检测器: {self._detector.name}")
        except Exception as e:
            self.logger.error(f"初始化人脸检测器失败: {e}")
            return

        # 2. 等 anim_service 就绪并加载 idle 基准姿态
        for _ in range(20):  # 最多等 2 秒
            if not self._running.is_set():
                return
            if hasattr(self.anim_service, "get_idle_first_frame") and \
               self.anim_service.get_idle_first_frame() is not None:
                break
            time.sleep(0.1)

        self._load_base_pose_from_idle()
        self._build_joint_limits()
        self._target_pose = dict(self._base_pose)

        # 3. 打开摄像头（带重试）
        retry = 0
        while self._running.is_set() and not self._open_camera():
            retry += 1
            if retry > 10:
                self.logger.error("摄像头打开失败次数过多，人脸追踪线程退出")
                return
            time.sleep(2)

        # 3.5 启动独立的抓帧线程（关键性能优化）
        # 之前主循环每帧 sleep+grab×3+retrieve 是阻塞的，慢驱动下能耗 600-700ms。
        # 改成抓帧线程死循环 grab/retrieve，追踪线程随时取"内存中最新一帧"，
        # 互不阻塞。即使摄像头实际 FPS 很低，追踪也不会被它拖慢。
        self._latest_frame = None
        self._latest_frame_lock = threading.Lock()
        self._latest_frame_seq = 0          # 单调递增，便于检测"是否拿到了新帧"
        self._last_consumed_seq = 0         # 追踪线程已处理到的最后一帧
        self._capture_thread = threading.Thread(
            target=self._capture_loop, name="face_capture_loop", daemon=True
        )
        self._capture_thread.start()

        # 3.6 启动手势识别线程（可选；mediapipe 没装则静默跳过）
        # 复用 _latest_frame，不再开第二个摄像头；推理在独立线程，不阻塞追踪线程。
        if self.hand_enabled:
            try:
                if self._injected_hand_detector is not None:
                    self._hand_detector = self._injected_hand_detector
                    print(f"[face_tracking] 使用注入的手势检测器: {self._hand_detector.name}")
                else:
                    self._hand_detector = create_default_hand_detector()
                if self._hand_detector is not None:
                    self._hand_thread = threading.Thread(
                        target=self._hand_loop, name="hand_gesture_loop", daemon=True
                    )
                    self._hand_thread.start()
                    self.logger.info(
                        f"手势识别线程已启动 (stride={self.hand_frame_stride})"
                    )
                else:
                    self.logger.info("未启用手势识别（mediapipe 不可用）")
            except Exception as e:
                self.logger.warning(f"启动手势识别失败，继续运行人脸追踪: {e}")
                self._hand_detector = None
                self._hand_thread = None
        else:
            self.logger.info("hand_enabled=False，跳过手势识别")

        # 4. 主循环（不再自己 grab/retrieve，改为读 _latest_frame）
        while self._running.is_set():
            try:
                # 4.1 暂停态
                if self._paused:
                    time.sleep(self.detect_interval)
                    continue

                # 4.2 非 idle 动作执行中：完全让位
                if self._is_anim_busy():
                    self._set_tracking_active(False)
                    time.sleep(self.detect_interval)
                    continue

                # 4.3 取最新帧（由抓帧线程独立维护，主循环不再阻塞在 grab）
                # 节流：等到有新帧或 detect_interval 到期
                t_wait_start = time.time()
                frame = None
                while self._running.is_set():
                    with self._latest_frame_lock:
                        if (self._latest_frame is not None
                                and self._latest_frame_seq != self._last_consumed_seq):
                            frame = self._latest_frame  # numpy view，不复制
                            self._last_consumed_seq = self._latest_frame_seq
                            break
                    # 还没新帧——根据情况短睡
                    elapsed = time.time() - t_wait_start
                    if elapsed > 2.0:
                        # 2 秒还没拿到新帧 → 摄像头可能挂了
                        break
                    time.sleep(0.01)

                if frame is None:
                    # 摄像头没在出帧
                    self.logger.warning("超过 2 秒未收到新帧，等待抓帧线程恢复...")
                    self._set_tracking_active(False)
                    time.sleep(0.5)
                    continue

                # 适度节流：避免 detect_interval 设得太小导致刷屏
                # （抓帧线程已经按摄像头实际帧率提供帧，这里再加一点 sleep 让出 CPU）
                # 注意：这个 sleep 不影响"画面延迟"——下一次取帧仍是当时最新的
                time.sleep(self.detect_interval)

                # 4.4 用真实帧尺寸校准画面中心（防止 set 分辨率不生效）
                fh, fw = frame.shape[:2]
                if fw != self.frame_width or fh != self.frame_height:
                    self.logger.info(
                        f"画面尺寸更新: {self.frame_width}x{self.frame_height} -> {fw}x{fh}"
                    )
                    self.frame_width = fw
                    self.frame_height = fh

                # 4.5 检测人脸（通过统一检测器接口，MNN/Haar 透明切换）
                t_detect_start = time.time()
                box = self._detector.detect(frame)
                detect_ms = (time.time() - t_detect_start) * 1000

                # 滑动 FPS 统计（用于诊断推理性能瓶颈）
                self._fps_frame_count = getattr(self, "_fps_frame_count", 0) + 1
                self._fps_window_start = getattr(self, "_fps_window_start", time.time())
                # 每秒更新一次 FPS 显示
                if time.time() - self._fps_window_start >= 1.0:
                    elapsed = time.time() - self._fps_window_start
                    self._track_fps = self._fps_frame_count / elapsed
                    # 平均检测用时
                    self._detect_ms_avg = getattr(self, "_detect_ms_sum", 0) / max(1, self._fps_frame_count)
                    print(f"[track] [perf] FPS={self._track_fps:.1f} 平均 detect={self._detect_ms_avg:.0f}ms")
                    self._fps_frame_count = 0
                    self._fps_window_start = time.time()
                    self._detect_ms_sum = 0.0
                self._detect_ms_sum = getattr(self, "_detect_ms_sum", 0.0) + detect_ms

                if box is not None:
                    x, y, w, h = box
                    self._on_face_detected(x, y, w, h)
                else:
                    self._on_no_face()

                # 不再每帧 gc.collect()——它在 Pi 上会拖慢循环
                del frame, box

            except Exception as e:
                self.logger.error(f"追踪循环异常: {e}")
                time.sleep(0.5)

        # 退出前清理
        self._set_tracking_active(False)
        # 等抓帧线程退出（_running.clear 已经在外面调过）
        if hasattr(self, "_capture_thread") and self._capture_thread is not None:
            self._capture_thread.join(timeout=2.0)
        self._release_camera()
        self.logger.info("追踪循环已退出")

    # ====================================================================
    # 抓帧线程（独立运行，把最新帧放到内存供追踪线程读取）
    # ====================================================================
    def _capture_loop(self):
        """
        死循环 grab + retrieve，最新帧覆盖到 _latest_frame。
        独立线程的好处：
          - 摄像头驱动慢/缓冲区满 → 这个线程被阻塞，不影响追踪线程
          - 追踪线程取帧只锁一下 dict，<<1ms
          - 自动维持"内存里始终是最新一帧"的不变量
        """
        # 也降低本线程优先级
        try:
            import os
            os.nice(5)
        except (OSError, AttributeError):
            pass

        consecutive_fail = 0
        # FPS 诊断
        cap_frame_count = 0
        cap_window_start = time.time()
        cap_read_ms_sum = 0.0

        while self._running.is_set():
            if self._cap is None:
                # 主线程会重开摄像头；这里短睡等待
                time.sleep(0.2)
                continue
            try:
                # 单次 grab + retrieve；不必再 grab×3 丢旧帧——
                # 因为我们已经按摄像头自然帧率持续消费，缓冲区不会积压。
                t0 = time.time()
                ret, frame = self._cap.read()
                read_ms = (time.time() - t0) * 1000
                cap_read_ms_sum += read_ms

                if not ret or frame is None:
                    consecutive_fail += 1
                    if consecutive_fail > 30:
                        # 持续失败 → 触发摄像头重建
                        self.logger.warning("抓帧持续失败，请求摄像头重连")
                        self._release_camera()
                        time.sleep(0.5)
                        if not self._open_camera():
                            time.sleep(2.0)
                        consecutive_fail = 0
                    else:
                        time.sleep(0.05)
                    continue
                consecutive_fail = 0
                # 用锁更新 _latest_frame
                with self._latest_frame_lock:
                    self._latest_frame = frame
                    self._latest_frame_seq += 1

                # 抓帧线程的 FPS 诊断（每 2 秒一次）
                # 这是了解"摄像头能出多少帧"的关键数据：
                # - 如果 cap_fps ≈ 追踪 fps，那瓶颈就是摄像头本身
                # - 如果 cap_fps >> 追踪 fps，那瓶颈在追踪线程
                cap_frame_count += 1
                elapsed = time.time() - cap_window_start
                if elapsed >= 2.0:
                    cap_fps = cap_frame_count / elapsed
                    avg_read_ms = cap_read_ms_sum / cap_frame_count
                    print(f"[capture] FPS={cap_fps:.1f} 平均 read={avg_read_ms:.0f}ms")
                    cap_frame_count = 0
                    cap_read_ms_sum = 0.0
                    cap_window_start = time.time()

            except Exception as e:
                self.logger.error(f"抓帧线程异常: {e}")
                time.sleep(0.2)

        self.logger.info("抓帧线程已退出")

    # ====================================================================
    # 手势识别线程（独立运行，与人脸追踪并行；复用 _latest_frame，不再开摄像头）
    # ====================================================================
    def _default_on_hand_pointing(self, direction: str):
        """默认回调：仅打印日志，与 hand_recongnize2.py 行为一致"""
        print(f"[{time.strftime('%H:%M:%S')}] 手指指向: {direction}")

    def _hand_loop(self):
        """
        手势识别后台循环。
        关键设计：
          - 与人脸追踪线程**完全解耦**：不写 _target_pose、不动电机；
            只通过 _on_hand_pointing 回调上报事件。
          - **复用 _latest_frame**：不打开第二个摄像头，避免与抓帧线程争抢；
            通过 frame_seq 跳过未变化的帧，避免对同一帧重复推理。
          - **跳帧节流**：hand_frame_stride 帧才检测一次；mediapipe 推理 30-80ms，
            stride=3 意味着 ~5-10 FPS 的手势检测，足够指向手势识别，
            而追踪线程那边能保持自己原本 FPS。
          - **降低线程优先级**：让人脸追踪 / 音频采集优先拿 CPU。
          - 追踪暂停 / 动作播放期间也跳过推理，避免在那种场景下平白耗 CPU。
        """
        # 降低线程优先级（比 _tracking_loop 的 nice=5 再低一点，因为这是辅助功能）
        try:
            import os
            os.nice(8)
        except (OSError, AttributeError):
            pass

        direction_buffer = deque(maxlen=self.hand_buffer_size)
        last_direction = ""
        cooldown = 0  # 单位：检测次数（不是相机帧），所以 stride 不会影响冷却时长
        stride_counter = 0
        last_processed_seq = 0

        # FPS 统计窗口
        hand_frame_count = 0
        hand_window_start = time.time()
        hand_infer_ms_sum = 0.0

        self.logger.info(
            f"手势识别循环已启动 (stride={self.hand_frame_stride}, "
            f"buffer={self.hand_buffer_size}, vote={self.hand_vote_threshold})"
        )

        while self._running.is_set():
            try:
                # 暂停 / 动作播放期间 / 运行期手势禁用时，不浪费 CPU
                if self._paused or self._is_anim_busy() or not self.hand_enabled:
                    time.sleep(0.1)
                    continue

                # 1. 取最新帧（跳过已经处理过的帧）
                frame = None
                with self._latest_frame_lock:
                    if (self._latest_frame is not None
                            and self._latest_frame_seq != last_processed_seq):
                        # 复制一份再处理，避免持锁期间 mediapipe 占用 _latest_frame
                        # 抓帧线程随时会覆盖；不复制可能读到半新半旧数据
                        frame = self._latest_frame.copy()
                        last_processed_seq = self._latest_frame_seq

                if frame is None:
                    # 还没新帧，短睡让出 CPU
                    time.sleep(0.02)
                    continue

                # 2. 跳帧策略：每 stride 帧才真正过一次推理
                stride_counter += 1
                if stride_counter < self.hand_frame_stride:
                    continue
                stride_counter = 0

                # 3. 推理
                t0 = time.time()
                result = self._hand_detector.detect(frame)
                infer_ms = (time.time() - t0) * 1000
                hand_infer_ms_sum += infer_ms
                hand_frame_count += 1

                # 4. 投票 + 冷却（与 hand_recongnize2.py 一致）
                if result is not None:
                    direction, _lm = result
                    direction_buffer.append(direction)

                    if len(direction_buffer) >= self.hand_vote_threshold:
                        # 找出出现次数最多的方向
                        most_common = max(set(direction_buffer),
                                          key=direction_buffer.count)
                        if (direction_buffer.count(most_common) >= self.hand_vote_threshold
                                and most_common != last_direction
                                and cooldown == 0):
                            # 触发回调（保护性 try：用户回调出错不能搞挂线程）
                            try:
                                self._on_hand_pointing(most_common)
                            except Exception as e:
                                self.logger.warning(f"手势回调出错: {e}")
                            last_direction = most_common
                            cooldown = self.hand_cooldown_detections
                else:
                    # 当前帧没有指向手势，清空缓冲并允许下一次直接触发
                    direction_buffer.clear()
                    last_direction = ""

                if cooldown > 0:
                    cooldown -= 1

                # 5. FPS 诊断：每 5 秒打一次（不影响主循环）
                elapsed = time.time() - hand_window_start
                if elapsed >= 5.0:
                    hand_fps = hand_frame_count / elapsed
                    avg_infer = hand_infer_ms_sum / max(1, hand_frame_count)
                    print(f"[hand] FPS={hand_fps:.1f} 平均推理={avg_infer:.0f}ms")
                    hand_frame_count = 0
                    hand_infer_ms_sum = 0.0
                    hand_window_start = time.time()

                del frame

            except Exception as e:
                self.logger.error(f"手势识别循环异常: {e}")
                time.sleep(0.3)

        self.logger.info("手势识别循环已退出")

    # ====================================================================
    # 人体感应联动
    # ====================================================================
    def set_human_sensor_state(self, state: int):
        """
        供外部（通常是 RGBService 的人体感应回调，运行在串口接收线程）更新
        人体检测传感器状态：
            state = 0 → 未检测到人体 或 人体静止不动
            state = 1 → 检测到人体 或 人体移动
        线程安全；状态切换时打印日志并重置冻结参考位置。
        """
        present = bool(int(state))
        with self._human_sensor_lock:
            changed = present != self._human_present
            self._human_present = present
        if changed:
            # 传感器恢复"有人/移动"时清掉冻结参考，
            # 由追踪线程在下一帧用最新人脸位置重建
            if present:
                self._static_ref_cx = None
            print(
                f"[track] 人体感应状态: "
                f"{'有人/移动' if present else '无人/静止'}"
                f" → {'正常人脸跟踪' if present else '人脸左右移动很小时将冻结跟踪'}"
            )

    def _get_human_present(self) -> bool:
        """线程安全读取人体感应状态（默认 True：无传感器数据时保持原有行为）"""
        with self._human_sensor_lock:
            return self._human_present

    # ====================================================================
    # 追踪激活状态切换
    # ====================================================================
    def _set_tracking_active(self, active: bool):
        """
        切换追踪激活状态，并通知 AnimationService。
        仅在状态变化时通知，避免日志刷屏。
        """
        if active == self._tracking_active:
            return
        self._tracking_active = active
        # 追踪停止时重置预测状态，避免下次再激活时用过期速度做外推（会跳变）；
        # 同时清掉人体静止的冻结参考——idle 会移动机械臂，旧参考已失效
        if not active:
            self._reset_prediction()
            self._static_ref_cx = None
        try:
            if hasattr(self.anim_service, "set_tracking_active"):
                self.anim_service.set_tracking_active(active)
        except Exception as e:
            self.logger.error(f"通知 AnimationService 切换追踪状态失败: {e}")

    def _reset_prediction(self):
        """清空人脸位置历史与速度估计，下次检测到人脸时从零开始"""
        self._prev_face_cx = None
        self._prev_face_cy = None
        self._prev_face_t = None
        self._vx_smooth = 0.0
        self._vy_smooth = 0.0

    def _is_anim_busy(self) -> bool:
        """非 idle 动作播放中则返回 True"""
        try:
            return bool(getattr(self.anim_service, "is_busy", False))
        except Exception:
            return False

    # ====================================================================
    # 检测结果 → 电机指令
    # ====================================================================
    def _on_face_detected(self, x: int, y: int, w: int, h: int):
        now = time.time()
        self._last_seen = now
        self._lost_frames = 0   # 清零丢失计数
        # 看到人脸：激活追踪
        self._set_tracking_active(True)

        center_x = x + w / 2.0
        center_y = y + h / 2.0

        # ────────────────────────────────────────────────────────
        # Layer 1：中心矩形死区
        # ────────────────────────────────────────────────────────
        # 当测量到的人脸中心落在画面正中的小矩形内，认为"人在画面中央且基本不动"——
        # 直接跳过所有计算与下发，让灯保持当前姿态。
        # 这是解决"人静止时检测框抖动→灯持续微调"的最直接手段。
        # 注意用"测量位置"（center_x/y）判断，不是用"预测位置"——
        # 预测位置受速度滤波影响，刚静止时还可能有残余速度让 pred 跑出矩形。
        rect_x_px = self.center_rect_x * self.frame_width / 2.0
        rect_y_px = self.center_rect_y * self.frame_height / 2.0
        cx_offset = abs(center_x - self.frame_width / 2.0)
        cy_offset = abs(center_y - self.frame_height / 2.0)
        in_center_rect = (cx_offset < rect_x_px and cy_offset < rect_y_px)

        if in_center_rect:
            # 进入中心矩形：保持当前位置，仅维持电机力矩，不下发新指令
            # 重要：在矩形内仍要更新 _prev_face_* 历史（避免下次离开矩形时
            # 用很久之前的位置算速度，得到错误的大速度）
            self._prev_face_cx = center_x
            self._prev_face_cy = center_y
            self._prev_face_t = now
            # 同时让速度估计自然衰减到 0（不重置，避免下次进入抖动）
            self._vx_smooth *= 0.5
            self._vy_smooth *= 0.5

            # 节流打印
            self._print_counter = getattr(self, "_print_counter", 0) + 1
            if self._print_counter >= 6:  # 静止时打印更稀疏
                self._print_counter = 0
                print(
                    f"[track] face=({center_x:.0f},{center_y:.0f}) "
                    f"[中心矩形内 ±{rect_x_px:.0f}×±{rect_y_px:.0f}px，灯保持不动]"
                )
            return  # ← 关键：直接返回，不做下面任何计算和下发

        # ────────────────────────────────────────────────────────
        # Layer 1.5：人体感应静止冻结
        # ────────────────────────────────────────────────────────
        # 人体检测传感器（GPIO9/level_state，经 RGBService 上报）：
        #   无人或人体静止 (level_state=0) 且人脸左右移动距离很小
        #     → 判定画面偏移是检测器漂移，冻结跟踪（不下发任何电机指令），
        #       消除"人静止时机械臂左右抖动"的问题；
        #   有人或人体移动 (level_state=1)，或左右移动距离较大
        #     → 继续正常人脸跟踪。
        if self.human_sensor_enabled and not self._get_human_present():
            if self._static_ref_cx is None:
                # 刚进入"人体静止"状态：记录当前人脸水平位置作为冻结参考
                self._static_ref_cx = center_x
            move_dist = abs(center_x - self._static_ref_cx) / self.frame_width
            if move_dist < self.human_static_move_thresh:
                # 左右移动距离很小 → 检测漂移，冻结跟踪
                # 与中心矩形一样：仍要更新 _prev_face_* 历史并衰减速度估计，
                # 避免恢复跟踪时用过期位置算出虚假大速度
                self._prev_face_cx = center_x
                self._prev_face_cy = center_y
                self._prev_face_t = now
                self._vx_smooth *= 0.5
                self._vy_smooth *= 0.5

                self._print_counter = getattr(self, "_print_counter", 0) + 1
                if self._print_counter >= 6:
                    self._print_counter = 0
                    print(
                        f"[track] face=({center_x:.0f},{center_y:.0f}) "
                        f"水平偏移Δ={move_dist * self.frame_width:.0f}px "
                        f"< {self.human_static_move_thresh * self.frame_width:.0f}px "
                        f"[人体静止，冻结跟踪]"
                    )
                return  # ← 不下发任何电机指令，灯保持当前姿态
            else:
                # 左右移动距离较大 → 真实移动，恢复跟踪；
                # 参考位置跟随当前人脸，移动结束后偏移变小会再次冻结
                self._static_ref_cx = center_x
        else:
            # 人体移动中：参考位置持续跟随当前人脸位置，
            # 传感器切到"静止"时参考即为最新位置
            self._static_ref_cx = center_x

        # ────────────────────────────────────────────────────────
        # 速度估计 + 位置外推（预测）
        # ────────────────────────────────────────────────────────
        # 思路：基于上一帧到本帧人脸像素中心的位移，估计速度（像素/秒），
        # 用一阶低通滤波平滑，再以 lookahead 秒外推得到"未来"位置。
        # 这样灯一次就转向人脸即将到达的位置，减少"差几度→反复修正→抖动"。
        #
        # 注意：速度估计 **始终运行**，不受 prediction_enabled 控制——
        # 除了外推预测，"人脸快速移出画面后的惯性滑行"（_on_no_face/coast）
        # 同样依赖最后记录的运动速度与方向。prediction_enabled 只决定
        # 是否用速度外推预测位置。
        if (
            self._prev_face_cx is not None
            and self._prev_face_t is not None
        ):
            dt = now - self._prev_face_t
            # dt 必须在合理范围，否则估计无意义（比如 5 秒前丢了人脸又出现）
            if 0.005 < dt < 0.5:
                # 1) 原始速度
                vx_raw = (center_x - self._prev_face_cx) / dt
                vy_raw = (center_y - self._prev_face_cy) / dt
                # 2) 速度上限（防误检导致疯狂外推）
                if   vx_raw >  self.max_velocity: vx_raw =  self.max_velocity
                elif vx_raw < -self.max_velocity: vx_raw = -self.max_velocity
                if   vy_raw >  self.max_velocity: vy_raw =  self.max_velocity
                elif vy_raw < -self.max_velocity: vy_raw = -self.max_velocity
                # 3) 一阶 IIR 低通滤波：v_smooth = α·v_prev + (1-α)·v_raw
                self._vx_smooth = self.velocity_alpha * self._vx_smooth \
                                   + (1.0 - self.velocity_alpha) * vx_raw
                self._vy_smooth = self.velocity_alpha * self._vy_smooth \
                                   + (1.0 - self.velocity_alpha) * vy_raw
                # 4) Layer 2：速度死区——|v_smooth| 太小直接归零，
                # 杜绝静止时检测抖动→伪速度→外推到远处→灯跟着动 的链条；
                # 同时保证滑行只在"真的大力移动后丢失"时触发
                if abs(self._vx_smooth) < self.static_velocity_thresh:
                    self._vx_smooth = 0.0
                if abs(self._vy_smooth) < self.static_velocity_thresh:
                    self._vy_smooth = 0.0
            else:
                # dt 异常 → 重置滤波（避免拿过期速度外推）
                self._vx_smooth = 0.0
                self._vy_smooth = 0.0
        else:
            # 第一次看到人脸（或刚激活），无历史 → 速度为 0
            self._vx_smooth = 0.0
            self._vy_smooth = 0.0

        # 外推：仅预测开启时用滤波后的速度推算 lookahead 秒后的位置；
        # 预测关闭时"预测位置"退化为当前测量位置（反应式追踪，与原行为一致）
        if self.prediction_enabled:
            pred_x = center_x + self._vx_smooth * self.prediction_lookahead
            pred_y = center_y + self._vy_smooth * self.prediction_lookahead
        else:
            pred_x, pred_y = center_x, center_y

        # 更新历史（注意：保存"测量到的位置"而不是"预测位置"，否则反馈会失真）
        self._prev_face_cx = center_x
        self._prev_face_cy = center_y
        self._prev_face_t  = now

        # 归一化偏移（用预测位置而不是当前位置——这就是预测追踪的核心）
        offset_x = (pred_x - self.frame_width / 2.0) / (self.frame_width / 2.0)
        offset_y = (pred_y - self.frame_height / 2.0) / (self.frame_height / 2.0)

        # 预测可能把偏移外推到画面外较远位置，做最后保险裁剪
        if   offset_x >  self.max_pred_offset: offset_x =  self.max_pred_offset
        elif offset_x < -self.max_pred_offset: offset_x = -self.max_pred_offset
        if   offset_y >  self.max_pred_offset: offset_y =  self.max_pred_offset
        elif offset_y < -self.max_pred_offset: offset_y = -self.max_pred_offset

        raw_offset_x, raw_offset_y = offset_x, offset_y  # 留作打印

        # 死区（X、Y 独立，因为俯仰对 off_y 更敏感）
        if abs(offset_x) < self.dead_zone_x:
            offset_x = 0.0
        if abs(offset_y) < self.dead_zone_y:
            offset_y = 0.0

        if offset_x == 0.0 and offset_y == 0.0:
            # 已对准：维持位置，下发一帧防漂移
            print(
                f"[track] face=({center_x:.0f},{center_y:.0f}) "
                f"frame={self.frame_width}x{self.frame_height} "
                f"off=({raw_offset_x:+.2f},{raw_offset_y:+.2f}) [死区内，仅维持]"
            )
            self._send_pose_to_motors()
            return

        # 应用方向反转开关（机械装配方向决定，每个俯仰轴独立配置）
        sign_x         = -1.0 if self.INVERT_X            else 1.0
        sign_base_p    = -1.0 if self.INVERT_BASE_PITCH   else 1.0
        sign_elbow     = -1.0 if self.INVERT_ELBOW        else 1.0
        sign_wrist_p   = -1.0 if self.INVERT_WRIST_PITCH  else 1.0

        # 计算各电机位置增量（归一化单位）
        delta_yaw         =  sign_x       * self.YAW_GAIN_X         * offset_x
        # 俯仰三关节统一乘 pitch_gain_scale：对齐水平 yaw 的单帧推进速度
        delta_base_pitch  =  sign_base_p  * self.BASE_PITCH_GAIN_Y  * self.pitch_gain_scale * offset_y
        delta_elbow       =  sign_elbow   * self.ELBOW_GAIN_Y       * self.pitch_gain_scale * offset_y
        delta_wrist_pitch =  sign_wrist_p * self.wrist_pitch_gain   * self.pitch_gain_scale * offset_y
        delta_wrist_roll  =  sign_x       * self.WRIST_ROLL_GAIN    * offset_x

        # 单帧步长限幅
        delta_yaw         = self._clip_step(delta_yaw)
        delta_base_pitch  = self._clip_step(delta_base_pitch)
        delta_elbow       = self._clip_step(delta_elbow)
        delta_wrist_pitch = self._clip_step(delta_wrist_pitch)
        delta_wrist_roll  = self._clip_step(delta_wrist_roll)

        # 累加到目标姿态，并施加软限位（保存旧值以便 Layer 3 比较）
        prev_pose = dict(self._target_pose)
        self._target_pose["base_yaw"]    = self._clip_limit("base_yaw",
                                            self._target_pose["base_yaw"]    + delta_yaw)
        self._target_pose["base_pitch"]  = self._clip_limit("base_pitch",
                                            self._target_pose["base_pitch"]  + delta_base_pitch)
        self._target_pose["elbow_pitch"] = self._clip_limit("elbow_pitch",
                                            self._target_pose["elbow_pitch"] + delta_elbow)
        self._target_pose["wrist_pitch"] = self._clip_limit("wrist_pitch",
                                            self._target_pose["wrist_pitch"] + delta_wrist_pitch)
        self._target_pose["wrist_roll"]  = self._clip_limit("wrist_roll",
                                            self._target_pose["wrist_roll"]  + delta_wrist_roll)

        # Layer 3：所有关节变化都很小 → 跳过下发
        # 即使 Layer 1/2 都通过（说明有微弱运动意图），如果最终 5 个舵机的目标
        # 变化都 < pose_delta_thresh（默认 0.5 单位），这种"亚阈值微动"对舵机来说
        # 几乎产生不了机械位移，只会让舵机内部不停做微小张力调整，反而抖动。
        max_change = max(
            abs(self._target_pose["base_yaw"]    - prev_pose["base_yaw"]),
            abs(self._target_pose["base_pitch"]  - prev_pose["base_pitch"]),
            abs(self._target_pose["elbow_pitch"] - prev_pose["elbow_pitch"]),
            abs(self._target_pose["wrist_pitch"] - prev_pose["wrist_pitch"]),
            abs(self._target_pose["wrist_roll"]  - prev_pose["wrist_roll"]),
        )
        if max_change < self.pose_delta_thresh:
            # 不下发，但保留新 target（下次 delta 累加还从新值起算，避免漂移积累）
            self._print_counter = getattr(self, "_print_counter", 0) + 1
            if self._print_counter >= 6:
                self._print_counter = 0
                print(
                    f"[track] face=({center_x:.0f},{center_y:.0f}) "
                    f"max_Δ={max_change:.2f} < {self.pose_delta_thresh} [跳过下发，避免微抖]"
                )
            return

        # 节流诊断输出：每 N 帧打 1 行，避免 print I/O 阻塞追踪线程
        # （之前每帧都打长 print，CPU 紧张时这是真正的拖累）
        self._print_counter = getattr(self, "_print_counter", 0) + 1
        if self._print_counter >= 3:
            self._print_counter = 0
            print(
                f"[track] face=({center_x:.0f},{center_y:.0f}) "
                f"pred=({pred_x:.0f},{pred_y:.0f}) "
                f"v=({self._vx_smooth:+.0f},{self._vy_smooth:+.0f})px/s "
                f"off=({raw_offset_x:+.2f},{raw_offset_y:+.2f}) "
                f"→ yaw={self._target_pose['base_yaw']:+.1f} "
                f"pitch={self._target_pose['base_pitch']:+.1f} "
                f"elbow={self._target_pose['elbow_pitch']:+.1f} "
                f"wpitch={self._target_pose['wrist_pitch']:+.1f}"
                + (f" | FPS={self._track_fps:.1f}" if hasattr(self, "_track_fps") else "")
            )

        self._send_pose_to_motors()

    def _on_no_face(self):
        """
        本帧未检测到人脸。
        - 刚丢失的瞬间（_lost_frames 从 0 → 1）：捕获最后记录的滤波速度，
          判定是"人脸快速移出"还是普通漏检：
          · 水平/垂直任一轴速度过阈（coast_min_speed / coast_min_speed_y）
            且滑行开启 → 进入惯性滑行，摄像头沿最后运动方向继续转
            coast_frames / coast_frames_y 小步（按主导轴选择配额，"追出去"）
          · 否则（静止时丢失/检测器偶尔漏检）→ 不滑行，原地保持
        - 累计帧数 < lost_frames_threshold：保持当前激活状态等下一帧
          （滑行配额用完后下发当前 pose 维持力矩防漂移）
        - 累计帧数 >= 阈值：判定丢失，停止追踪，把控制权交还给 idle

        按帧计数比按时间稳健：Haar Cascade 偶尔漏检几帧很正常，
        而帧率不稳时按时间会误判。
        """
        if self._last_seen == 0.0:
            # 启动后从未见过人脸 —— 维持初始状态，让 idle 接管
            self._set_tracking_active(False)
            return

        # 刚从"有人脸"切到"无人脸"：捕获最后运动速度，决定是否滑行
        if self._lost_frames == 0:
            self._coast_vx = self._vx_smooth
            self._coast_vy = self._vy_smooth
            # 分轴触发：水平/垂直各自与门槛比较（任一轴过阈即滑行）。
            # 不用合速度：画面高(240px)比宽(320px)短，人脸上下移出时滤波速度
            # 爬升不足，|vy| 常在 180~230 —— 合速度门槛 250 会漏掉纯垂直移出。
            fast_x = abs(self._coast_vx) >= self.coast_min_speed
            fast_y = abs(self._coast_vy) >= self.coast_min_speed_y
            # 滑行帧数配额按"主导轴"选择：|v|/门槛 比值更大的轴决定。
            # 垂直每帧推进量小，默认配额(coast_frames_y)比水平(coast_frames)大。
            rel_x = abs(self._coast_vx) / self.coast_min_speed if self.coast_min_speed > 0 else 0.0
            rel_y = abs(self._coast_vy) / self.coast_min_speed_y if self.coast_min_speed_y > 0 else 0.0
            frames_quota = self.coast_frames if rel_x >= rel_y else self.coast_frames_y
            if (
                self.coast_enabled
                and frames_quota > 0
                and (fast_x or fast_y)
            ):
                self._coast_frames_left = frames_quota
                self._coast_accum = {k: 0.0 for k in self.JOINT_KEYS}
                print(
                    f"[track] 人脸快速移出 "
                    f"(v=({self._coast_vx:+.0f},{self._coast_vy:+.0f})px/s, "
                    f"fast_x={fast_x}, fast_y={fast_y}) "
                    f"→ 沿最后运动方向惯性滑行 {frames_quota} 帧"
                    f"({'水平主导' if rel_x >= rel_y else '垂直主导'})"
                )
            else:
                self._coast_frames_left = 0

        self._lost_frames += 1

        if self._lost_frames >= self.lost_frames_threshold:
            if self._tracking_active:
                print(f"[track] 连续 {self._lost_frames} 帧未检测到人脸，停止追踪，恢复 idle")
            self._set_tracking_active(False)
            # 不复位 _target_pose；下次再看到人脸时仍从最后位置接管。
        else:
            # 短暂丢失：优先消耗滑行配额沿运动方向继续转，
            # 配额用尽后维持当前 pose（下发一帧防漂移）
            if self._tracking_active:
                if self._coast_frames_left > 0:
                    self._coast_frames_left -= 1
                    self._apply_coast_step()
                else:
                    self._send_pose_to_motors()

    def _apply_coast_step(self):
        """
        惯性滑行一步：人脸快速移出画面后，沿最后记录的运动方向再转一小段。
        - 方向/强度：由丢失瞬间捕获的滤波速度 _coast_vx/_coast_vy 决定，
          按 coast_speed_ref 归一化到 [-1,1]（达到该速度即为满幅），
          正负号即"人脸往哪边跑"，复用正常追踪的方向反转开关；
        - 幅度约束：复用单帧步长限幅（max_step）与关节软限位；
          各关节累计推进量按关节独立封顶（达 coast_max_total 即冻结该关节），
          保证"只多走一小段"；
        - 中途重新检测到人脸时 _on_face_detected 会清零 _lost_frames，
          滑行自然终止，交回正常追踪。
        """
        offset_x = self._coast_vx / self.coast_speed_ref
        offset_x = max(-1.0, min(1.0, offset_x))
        # 垂直偏移先乘 coast_pitch_boost 再裁剪：
        # 垂直追踪由 3 个俯仰关节分摊（增益 9/3/3 << yaw 的 18），
        # 且上下移出的滤波速度普遍偏低，不放大则滑行量肉眼不可见
        offset_y = self._coast_vy * self.coast_pitch_boost / self.coast_speed_ref
        offset_y = max(-1.0, min(1.0, offset_y))

        # 应用方向反转开关（与正常追踪一致）
        sign_x       = -1.0 if self.INVERT_X            else 1.0
        sign_base_p  = -1.0 if self.INVERT_BASE_PITCH   else 1.0
        sign_elbow   = -1.0 if self.INVERT_ELBOW        else 1.0
        sign_wrist_p = -1.0 if self.INVERT_WRIST_PITCH  else 1.0

        deltas = {
            "base_yaw":    self._clip_step(sign_x       * self.YAW_GAIN_X         * offset_x),
            # 俯仰三关节同样乘 pitch_gain_scale（offset_y 已含 coast_pitch_boost）
            "base_pitch":  self._clip_step(sign_base_p  * self.BASE_PITCH_GAIN_Y  * self.pitch_gain_scale * offset_y),
            "elbow_pitch": self._clip_step(sign_elbow   * self.ELBOW_GAIN_Y       * self.pitch_gain_scale * offset_y),
            "wrist_pitch": self._clip_step(sign_wrist_p * self.wrist_pitch_gain   * self.pitch_gain_scale * offset_y),
            "wrist_roll":  self._clip_step(sign_x       * self.WRIST_ROLL_GAIN    * offset_x),
        }

        # 按关节独立封顶：某关节累计推进量达到 coast_max_total 后该关节冻结，
        # 其他关节继续消耗自己的配额；全部到顶或帧数用尽才结束滑行。
        # （原先任一关节到顶就停掉整体滑行，累进最快的 base_pitch 会把
        #   wrist_pitch 等慢关节的剩余配额连带砍掉，导致腕部跟随不明显。）
        all_capped = True
        for joint, delta in deltas.items():
            if self._coast_accum.get(joint, 0.0) >= self.coast_max_total:
                continue  # 该关节滑行配额已用完，冻结不动
            all_capped = False
            new_val = self._clip_limit(joint, self._target_pose[joint] + delta)
            # 累计滑行量只计"实际推进量"（被限位截断的部分不计）
            self._coast_accum[joint] = self._coast_accum.get(joint, 0.0) \
                                       + abs(new_val - self._target_pose[joint])
            self._target_pose[joint] = new_val

        accum_max = max(self._coast_accum.values())
        if all_capped:
            # 所有关节都已滑行"一小段"的距离上限，停止后续滑行
            self._coast_frames_left = 0

        print(
            f"[track] [coast] off=({offset_x:+.2f},{offset_y:+.2f}) "
            f"→ yaw={self._target_pose['base_yaw']:+.1f} "
            f"pitch={self._target_pose['base_pitch']:+.1f} "
            f"(累计max={accum_max:.1f}/{self.coast_max_total:.0f}, "
            f"剩余{self._coast_frames_left}帧)"
        )
        self._send_pose_to_motors()

    # ====================================================================
    # 工具
    # ====================================================================
    def _clip_step(self, v: float) -> float:
        if v >  self.max_step: return  self.max_step
        if v < -self.max_step: return -self.max_step
        return v

    def _clip_limit(self, joint: str, v: float) -> float:
        cap_lo = -self.HARD_LIMIT_ABS + self.hard_margin
        cap_hi =  self.HARD_LIMIT_ABS - self.hard_margin
        lo, hi = self._joint_limits.get(joint, (cap_lo, cap_hi))
        if v < lo: return lo
        if v > hi: return hi
        return v

    def _send_pose_to_motors(self):
        """通过 AnimationService 把当前目标姿态下发到电机"""
        action = {f"{k}.pos": v for k, v in self._target_pose.items()}
        try:
            if hasattr(self.anim_service, "send_tracking_action"):
                ok = self.anim_service.send_tracking_action(action)
                if not ok:
                    # 限频提示：连续被拒时只在状态切换时打一次，避免日志洪水
                    if not getattr(self, "_last_send_rejected", False):
                        self.logger.warning(
                            "send_tracking_action 被拒绝（多半是 is_busy=True 或 robot 未连接），"
                            "电机不会动；若动作已结束这条会很快消失"
                        )
                        self._last_send_rejected = True
                else:
                    self._last_send_rejected = False
            else:
                # 兼容回退
                robot = getattr(self.anim_service, "robot", None)
                if robot is not None:
                    robot.send_action(action)
                else:
                    self.logger.error("AnimationService 未暴露 send_tracking_action，且无 robot")
        except Exception as e:
            self.logger.error(f"下发追踪动作失败: {e}")
