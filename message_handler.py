"""
统一的消息处理模块
用于处理实时语音对话 API（Qwen-Omni-Realtime / GLM Realtime）的
WebSocket 通信消息。通过 rt_provider 抽象层与具体 SDK 解耦，
支持音频、函数调用和服务端VAD等多种场景。
消息统一为 dict 格式（含 "type" 字段），事件命名遵循 OpenAI Realtime 风格，
两家后端（Qwen / GLM）均兼容该风格。
"""

import os
import time
import base64
import asyncio
import json
import random
import sounddevice as sd
import functools
from collections.abc import Callable
from typing import Any, Optional

from rt_provider.base import RealtimeProvider

from lelamp.service.rgb.rgb_service import RGBService
from lelamp.service.motors.animation_service import AnimationService
from lelamp.service.vision.vision_service import look_and_describe
from lelamp.service.vision.face_tracking_service import FaceTrackingService
from lelamp.service.audio_playback import play_pcm_data


# ════════════════════════════════════════════════════════════════════
# 手势指向 → 动作录制名 的映射
# 食指方向由 hand_detector 返回 "Up"/"Down"/"Left"/"Right"（首字母大写），
# 这里统一转小写后查表，调用方传大小写都能命中。
# ════════════════════════════════════════════════════════════════════
HAND_DIRECTION_TO_ACTION = {
    "up":    "look_up_action",
    "down":  "look_down_action",
    "left":  "look_left_action",
    "right": "look_right_action",
}

# 注意：按键(Button)与其回调已移到 MessageHandler.__init__ 中创建，
# 因为回调需要访问 self.rgb_service。在模块顶层定义会导致 self 未定义。
# 见 __init__ 末尾的 _setup_button()。

def play_action_pausing_tracking(
    anim_service,
    face_tracking_service,
    action_name: str,
    *,
    start_timeout: float = 1.0,
    finish_timeout: float = 15.0,
    poll_interval: float = 0.05,
):
    """
    播放一个动作，期间暂停人脸追踪 + 手势识别，动作播放结束后再恢复。

    设计要点：
    1. 暂停方式：face_tracking_service.dispatch("pause", None)。
       FaceTrackingService 收到 "pause" 后会把内部 _paused 置 True，
       它的「人脸追踪线程」和「手势识别线程」都会因此跳过工作
       （见 _tracking_loop / _hand_loop 里对 self._paused 的判断），
       因此一次 pause 就同时停掉了「人脸追踪」和「手势识别」。
    2. 播放是异步的：anim_service.dispatch("play", name) 只是把事件入队，
       真正逐帧播放在 AnimationService 自己的事件循环里完成。所以这里通过
       轮询 anim_service.is_busy 来判断动作进度：
         - 先等 is_busy 由 False→True（动作真正开始）
         - 再等 is_busy 由 True→False（动作播完，已开始回插 idle）
       两步都带超时兜底：若对应录制文件不存在导致动作压根没开始，
       start_timeout 到点后直接进入恢复流程，不会卡死。
    3. try/finally 保证：无论中途是否抛异常，最终一定会 resume，
       绝不让人脸追踪被永久卡在暂停态。
    4. 本函数会阻塞调用线程。它被设计为在 FaceTrackingService 的「手势线程」
       里（on_hand_pointing 回调中）调用——阻塞该线程恰好等价于
       「动作期间不再做手势识别」，且不会影响 asyncio 主循环或人脸追踪线程。
    """
    paused = False
    try:
        # 1) 暂停人脸追踪 + 手势识别
        if face_tracking_service is not None:
            try:
                face_tracking_service.dispatch("pause", None)
                paused = True
                print("[手势] 已暂停人脸追踪与手势识别")
            except Exception as e:
                print(f"[手势] 暂停人脸追踪失败（继续播放动作）: {e}")

        # 2) 异步下发动作播放
        anim_service.dispatch("play", action_name)

        # 3) 等动作真正开始（is_busy: False → True）
        deadline = time.time() + start_timeout
        started = False
        while time.time() < deadline:
            if anim_service.is_busy:
                started = True
                break
            time.sleep(poll_interval)
        if not started:
            print(f"[手势] 动作 {action_name} 未在 {start_timeout}s 内开始"
                  f"（可能录制文件缺失），直接恢复追踪")

        # 4) 等动作播放结束（is_busy: True → False）
        if started:
            deadline = time.time() + finish_timeout
            while time.time() < deadline:
                if not anim_service.is_busy:
                    break
                time.sleep(poll_interval)
    finally:
        # 5) 恢复人脸追踪 + 手势识别（无论前面是否异常）
        if paused and face_tracking_service is not None:
            try:
                face_tracking_service.dispatch("resume", None)
                print("[手势] 动作结束，已恢复人脸追踪与手势识别")
            except Exception as e:
                print(f"[手势] 恢复人脸追踪失败: {e}")


class AudioFileSaver:
    def __init__(self, filename, rate):
        self.filename = filename
        self.rate = rate
        self.frames = []
        #self.is_recording = False

    def start(self):
        self.frames = []
        self.audio_data = b''     # ← 加这一行，防止残留旧数据
        #self.is_recording = True
        print(f"开始录制 AI 语音到文件: {self.filename}")

    def add_frame(self, frame_bytes):
        #if self.is_recording:
        #    self.frames.append(frame_bytes)
        self.frames.append(frame_bytes)

    def stop(self):
        #print(f"========> [stop] is_recording={self.is_recording}, frames={len(self.frames)}")  # 加这行
        #if not self.is_recording:
        #    return

        #self.is_recording = False
        if not self.frames:
            print("没有收到音频数据")
            return

        #print(f"正在保存文件，共 {len(self.frames)} 个数据块...")
        self.audio_data = b''.join(self.frames)  # 保存为实例属性，方便后续播放
        print(f"音频数据总大小: {len(self.audio_data)} 字节")

        """
        # 将所有数据块拼接成完整的二进制数据
        audio_data = b''.join(self.frames)
        print(f"保存的音频文件总大小: {len(audio_data)}")

        # 保存为 WAV 文件
        with wave.open(self.filename, 'wb') as wf:
            wf.setnchannels(1)       # 单声道
            wf.setsampwidth(2)       # 16bit (2 bytes)
            wf.setframerate(self.rate)
            wf.writeframes(audio_data)
        print(f"文件已保存: {os.path.abspath(self.filename)}")
        """

    def play_with_aplay(self, is_playing_event=None):
        """用aplay直接播放裸PCM数据（无需保存文件）

        Args:
            is_playing_event: threading.Event 或 asyncio.Event。
                传入后播放前置位、播放结束后清除，通知采集端暂停入队防止回声。
                play_with_aplay 在线程池中执行，直接调用 .set()/.clear() 是线程安全的。

        播放经 audio_playback 的全局锁串行化：GPIO9 无人提示音（no_human.wav）
        与 TTS 同时触发时排队等待，而不是报 "Device or resource busy" 丢掉音频。
        """
        if not self.audio_data:
            print("无音频数据可播放")
            return

        # ── 回声抑制：播放前置位，通知 audio_callback 丢弃采集帧 ──
        if is_playing_event is not None:
            is_playing_event.set()
            print("[回声抑制] 开始播放 TTS，麦克风采集已暂停")

        try:
            play_pcm_data(self.audio_data, self.rate)
        finally:
            # ── 回声抑制：无论成功或异常，播放结束后都要恢复采集 ──
            # 额外等待 200ms：让房间混响尾音消散，避免残余回声被立即采集到
            if is_playing_event is not None:
                time.sleep(0.2)
                is_playing_event.clear()
                print("[回声抑制] TTS 播放结束，麦克风采集已恢复")

    def play_with_sounddevice(self):
        """用sounddevice播放裸PCM字节流（跨平台）"""
        if not hasattr(self, 'audio_data') or len(self.audio_data) == 0:
            print("无音频数据可播放")
            return

        # 将字节流转换为numpy数组（sounddevice要求的格式）
        # 16bit PCM → int16数组
        audio_np = np.frombuffer(self.audio_data, dtype=np.int16)

        print(f"开始播放音频（时长：{len(audio_np)/self.rate:.2f}秒）...")
        # 播放音频（阻塞直到播放完成）
        sd.play(audio_np, samplerate=self.rate, channels=1, device=1)
        sd.wait()  # 等待播放结束
        print("音频播放完成（sounddevice）")

OUTPUT_FILENAME = "/home/pi/Documents/source/python/glm-realtime-sdk/python/ai_response.wav"
SAMPLE_RATE = 24000  # 输出采样率：GLM(pcm) 与 Qwen(PCM_24000HZ_MONO_16BIT) 均为 24000Hz
saver = AudioFileSaver(OUTPUT_FILENAME, SAMPLE_RATE)


class MessageHandler:
    def __init__(self, provider: RealtimeProvider, shutdown_event: Optional[asyncio.Event] = None,
                 is_playing_event: Optional[asyncio.Event] = None):
        """
        初始化消息处理器
        Args:
            provider: 实时对话后端实例（rt_provider.RealtimeProvider）
            shutdown_event: 关闭事件，用于控制消息接收的终止
            is_playing_event: 播放状态标志。置位时 audio_callback 停止入队，
                              防止 TTS 回声被误识别为用户语音发回服务端。
        """
        self.provider = provider
        self.shutdown_event = shutdown_event
        # ── 回声抑制：保存标志引用，播放时传给 play_with_aplay ──
        self.is_playing_event = is_playing_event
        self._custom_handlers: dict[str, Callable] = {}

        # ── 通话挂断：服务端 call.hangup（如静默超时）后置位，
        #    供外层会话循环（run_realtime_app）判断是否需要重连新会话 ──
        self.hangup_event = asyncio.Event()
        self.hangup_reason: Optional[str] = None

        self.rgb_service = RGBService()
        
        # ── 电机开关：根据环境变量/常量决定是否启动电机和人脸追踪 ──
        import os as _os
        _env = _os.environ.get("LELAMP_MOTOR", "").strip().lower()
        if _env in ("0", "false", "off", "no"):
            self._motor_enabled = False
        elif _env in ("1", "true", "on", "yes"):
            self._motor_enabled = True
        else:
            # 默认启用；如需默认关闭，把 True 改成 False
            self._motor_enabled = True

        if self._motor_enabled:
            print("启动灯光服务 + 电机服务")
            self.anim_service = AnimationService(
                port="/dev/ttyACM0",
                lamp_id="bubble",
                fps=25,
                duration=1
            )
            self.rgb_service.start()
            self.anim_service.start()

            # ── 人脸追踪服务：在动作空闲时让灯头追随人脸 ──
            # 通过 anim_service 共享同一条电机总线，避免抢占串口
            try:
                self.face_tracking_service = FaceTrackingService(
                    anim_service=self.anim_service,
                    # 识别到指向手势时回调，根据方向播放对应动作；
                    # 播放期间暂停追踪与手势识别，播完后恢复（见 on_hand_pointing）
                    on_hand_pointing=self.on_hand_pointing,
                )
                self.face_tracking_service.start()
                print("启动人脸追踪服务")
            except Exception as e:
                print(f"[人脸追踪] 启动失败，已跳过: {e}")
                self.face_tracking_service = None
        else:
            print("启动灯光服务（MOTOR_ENABLED=False，跳过电机/人脸追踪）")
            self.anim_service = None
            self.face_tracking_service = None
            self.rgb_service.start()

        # ── 定时随机动作：安静模式标志 & 后台任务句柄 ──
        self._quiet_mode: bool = False
        self._random_action_task: Optional[asyncio.Task] = None
        
        # ── 灯光状态标志：False=关闭，True=开启 ──
        # 语音"开灯/设色"与物理按键短按都会置位；
        # 拾音绿色追逐灯效会暂时打断当前灯光，说话结束后根据该标志
        # 决定恢复原颜色还是熄灭（见 input_audio_buffer.speech_stopped）。
        self._light_on: bool = False
        # ── 用户开灯/设色时记录的灯光颜色，用于灯效打断后恢复 ──
        self._light_color: tuple = (200, 200, 200)

        time.sleep(0.5)
        self.rgb_service.dispatch("solid", (255, 0, 0))
        #rainbow = [(255,0,0), (255,127,0), (255,255,0), (0,255,0), (0,0,255), (75,0,130), (148,0,211)]*9
        #self.rgb_service.dispatch("paint", rainbow[:64]) 
        time.sleep(0.7)
        self.rgb_service.dispatch("solid", (0, 255, 0))
        time.sleep(0.7)
        self.rgb_service.dispatch("solid", (0, 0, 255))
        time.sleep(0.7)
        self.rgb_service.clear()
        
        # ========================================================
        # 新增：注册 RGB 模块短按回调，同步灯光状态
        # ========================================================
        def handle_rgb_short_press(cmd_value: int):
            # 将 0/1 转换为布尔值同步给 _light_on
            self._light_on = bool(cmd_value)
            print(f"[按键同步] 收到短按，_light_on 已更新为: {self._light_on}")

        self.rgb_service.on_short_press = handle_rgb_short_press

        # ========================================================
        # 新增：人体感应 → 人脸追踪联动
        # RGBService 收到 GPIO9/level_state 上报时回调：
        #   0 = 未检测到人体或人体静止不动
        #   1 = 检测到人体或人体移动
        # FaceTrackingService 据此在人静止且人脸左右移动很小时冻结跟踪，
        # 消除检测漂移导致的机械臂左右抖动。
        # ========================================================
        def handle_human_sensor(level_state: int):
            tracker = getattr(self, "face_tracking_service", None)
            if tracker is not None:
                try:
                    tracker.set_human_sensor_state(level_state)
                except Exception as e:
                    print(f"[人体感应] 通知人脸追踪服务失败: {e}")

        self.rgb_service.on_human_sensor = handle_human_sensor

        # ── 按键设置：在这里创建，回调用闭包捕获 self（这样 self.rgb_service 可用）──
        self._setup_button()

    def _setup_button(self):
        """创建 GPIO6 按键并绑定回调。

        回调在 gpiozero 的后台线程里被调用，这里用闭包捕获 self。
        把 Button 存到 self.button，保证其生命周期跟随 MessageHandler，
        不会被垃圾回收掉（否则事件会静默失效）。

        接线提醒：pull_up=True（默认），按键必须接在 GPIO6(物理31脚) 与 GND 之间，
        按下时把引脚拉低才算"按下"。若接到了 3.3V，请改成 Button(6, pull_up=False)。
        """
        """
        # hold_time=5：按住 5 秒触发 when_held；bounce_time=0.2 防抖
        self.button = Button(6, hold_time=5, bounce_time=0.2)

        def on_trigger():
            if not self._light_on:
                # 当前关闭，执行开灯
                print("gpio6 trigger: 开灯")
                self.rgb_service.dispatch("solid", (180, 180, 180))
                self._light_on = True
            else:
                # 当前开启，执行关灯
                print("gpio6 trigger: 关灯")
                self.rgb_service.clear()
                self._light_on = False

        def on_long_press():
            print("long press trigger (5 seconds)")
            subprocess.run(["aplay", "-D", "plughw:seeed2micvoicec,0", "poweroff.wav"])

        # 普通按下
        self.button.when_pressed = on_trigger
        # 长按 5 秒
        self.button.when_held = on_long_press
        print(f"[按键] 已绑定 GPIO6，pin_factory={Device.pin_factory.__class__.__name__}")
        """
        print("还没实现Rock5C的按键功能")

    def register_handler(self, message_type: str, handler: Callable):
        """
        注册自定义消息处理器
        Args:
            message_type: 消息类型
            handler: 处理函数，接收message参数
        """
        self._custom_handlers[message_type] = handler

    def on_hand_pointing(self, direction: str):
        """
        手势指向回调（由 FaceTrackingService 的手势识别线程调用）。

        根据食指指向方向播放对应动作：
            Up    -> self.anim_service.dispatch("play", "look_up_action")
            Down  -> self.anim_service.dispatch("play", "look_down_action")
            Left  -> self.anim_service.dispatch("play", "look_left_action")
            Right -> self.anim_service.dispatch("play", "look_right_action")

        播放期间会暂停人脸追踪与手势识别，动作播完后再恢复
        （具体逻辑见 play_action_pausing_tracking）。
        """
        action_name = HAND_DIRECTION_TO_ACTION.get((direction or "").strip().lower())
        if action_name is None:
            print(f"[手势] 未知指向，忽略: {direction!r}")
            return

        print(f"[手势] 指向 {direction} → 播放动作 {action_name}")
        play_action_pausing_tracking(
            self.anim_service,
            getattr(self, "face_tracking_service", None),
            action_name,
        )

    async def _send_function_output(self, call_id, response: dict):
        """回传函数调用结果并触发模型继续生成响应。"""
        await self.provider.send_function_call_output(
            call_id, json.dumps(response, ensure_ascii=False)
        )
        await self.provider.trigger_response()

    async def _handle_function_call(self, message: dict):
        """处理函数调用消息（统一 dict 消息，Qwen / GLM 后端通用）"""
        name = message.get("name")
        arguments = message.get("arguments")
        # Qwen 后端回传函数结果时必须带 call_id；GLM 后端忽略
        call_id = message.get("call_id")
        print("函数调用参数完成消息")
        if name:
            print(f"  Function Name: {name}")
        print(f"  Arguments: {arguments if arguments else 'None'}")

        # 开灯
        if name == "lightOn":
            try:
                args = json.loads(arguments)
                self.rgb_service.dispatch("solid", (200, 200, 200))
                # ── 灯光标志：语音开灯后置位并记录颜色，拾音灯效结束后恢复 ──
                self._light_on = True
                self._light_color = (200, 200, 200)
                await self._send_function_output(call_id, {"status": "success", "message": f"成功打开灯光"})
            except json.JSONDecodeError as e:
                print(f"解析函数调用参数失败: {e}")

        # 关灯
        if name == "lightOff":
            try:
                self.rgb_service.clear()
                # ── 灯光标志：关灯后清除，拾音灯效结束后不再恢复 ──
                self._light_on = False
                await self._send_function_output(call_id, {"status": "success", "message": f"成功关闭灯光"})
            except json.JSONDecodeError as e:
                print(f"解析函数调用参数失败: {e}")

        # 红色
        if name == "lightRed":
            try:
                self.rgb_service.dispatch("solid", (255, 0, 0))
                # ── 灯光标志：设色视为开灯，记录颜色用于恢复 ──
                self._light_on = True
                self._light_color = (255, 0, 0)
                await self._send_function_output(call_id, {"status": "success", "message": f"已将灯光设置成红色"})
            except json.JSONDecodeError as e:
                print(f"解析函数调用参数失败: {e}")

        # 绿色
        if name == "lightGreen":
            try:
                self.rgb_service.dispatch("solid", (0, 255, 0))
                self._light_on = True
                self._light_color = (0, 255, 0)
                await self._send_function_output(call_id, {"status": "success", "message": f"已将灯光设置成绿色"})
            except json.JSONDecodeError as e:
                print(f"解析函数调用参数失败: {e}")

        # 蓝色
        if name == "lightBlue":
            try:
                self.rgb_service.dispatch("solid", (0, 0, 255))
                self._light_on = True
                self._light_color = (0, 0, 255)
                await self._send_function_output(call_id, {"status": "success", "message": f"已将灯光设置成蓝色"})
            except json.JSONDecodeError as e:
                print(f"解析函数调用参数失败: {e}")

        '''
        # 扫描查看四周
        if name == "actionLookAround":
            try:
                self.rgb_service.dispatch("solid", (200,200,200))
                self.anim_service.dispatch("play", "look_around")
                await asyncio.sleep(1)
                await self._send_function_output(call_id, {"status": "success", "message": f"好的，我看看四周有什么好东西"})
            except json.JSONDecodeError as e:
                print(f"解析函数调用参数失败: {e}")

        # Happy, Yes
        if name == "actionYes":
            try:
                self.anim_service.dispatch("play", "agree")
                await asyncio.sleep(1)
                await self._send_function_output(call_id, {"status": "success", "message": f"是的，确实是这样"})
            except json.JSONDecodeError as e:
                print(f"解析函数调用参数失败: {e}")

        # 查看周围环境并描述
        if name == "lookAndDescribe":
            print("触发 lookAndDescribe：开始拍照识别...")
            # 先做扫描动作，给用户视觉反馈
            self.rgb_service.dispatch("solid", (200,200,200))
            self.anim_service.dispatch("play", "read_with_camera")
            asyncio.create_task(self._look_and_describe_task(call_id))

        # 查看窗外
        if name == "actionCheckWeather":
            print("触发 lookAndDescribe：开始拍照窗外")
            # 先做扫描动作，给用户视觉反馈
            self.rgb_service.dispatch("solid", (200,200,200))
            self.anim_service.dispatch("play", "read_weather")
            asyncio.create_task(self._look_at_window_task(call_id))

        # 进入安静模式
        if name == "enterQuietMode":
            try:
                self._quiet_mode = True
                print("[安静模式] 已进入安静模式，定时随机动作已暂停")
                await self._send_function_output(call_id, {"status": "success", "message": "已进入安静模式，我会保持安静"})
            except Exception as e:
                print(f"enterQuietMode 处理出错: {e}")

        # 进入正常模式
        if name == "enterNormalMode":
            try:
                self._quiet_mode = False
                print("[正常模式] 已恢复正常模式，定时随机动作已恢复")
                await self._send_function_output(call_id, {"status": "success", "message": "已恢复正常模式，每两分钟我会做一个小动作"})
            except Exception as e:
                print(f"enterNormalMode 处理出错: {e}")
        '''

    async def _look_and_describe_task(self, call_id=None):
        try:
            await asyncio.sleep(15)
            print("开始拍照识别...")
            # 在线程池中执行（含拍照/网络请求/播放，全为阻塞操作）
            loop = asyncio.get_event_loop()
            description = await loop.run_in_executor(None, look_and_describe)
            print(f"lookAndDescribe 完成，描述：{description}")
            response = {"status": "success", "message": description}
            #response = {"status": "success", "message": f"好的"}
        except Exception as e:
            print(f"lookAndDescribe 执行出错: {e}")
            description = "抱歉，查看周围时出现了错误。"
            #response = {"status": "error", "message": str(e)}
            response = {"status": "error", "message": description}

        await self._send_function_output(call_id, response)
        
    # ── 定时随机动作 ──────────────────────────────────────────────────────────

    RANDOM_ACTION_INTERVAL = 90  # 2 分钟（秒）
    RANDOM_ACTIONS = [
        ("actionYes",          "agree"),
        ("actionCheckWeather", "look_around"),
        ("actionLookUp", "look_up"),
    ]

    async def _random_action_loop(self):
        """每 2 分钟在非安静模式下随机执行一个动作（不触发 function call 回路）"""
        print("[定时随机动作] 后台任务已启动，间隔 120 秒")
        try:
            while True:
                await asyncio.sleep(self.RANDOM_ACTION_INTERVAL)

                if self._quiet_mode:
                    print("[定时随机动作] 当前为安静模式，跳过本次动作")
                    continue

                if self.anim_service.is_busy:
                    print("[定时随机动作] 当前正在执行其他动作，跳过本次随机动作")
                    continue

                action_name, anim_name = random.choice(self.RANDOM_ACTIONS)
                print(f"[定时随机动作] 执行随机动作: {action_name} -> {anim_name}")

                try:
                    if action_name == "actionCheckWeather":
                        self.rgb_service.dispatch("solid", (200, 200, 200))
                        self.anim_service.dispatch("play", anim_name)
                    else:
                        self.anim_service.dispatch("play", anim_name)
                except Exception as e:
                    print(f"[定时随机动作] 执行出错: {e}")
        except asyncio.CancelledError:
            print("[定时随机动作] 后台任务已取消")

    async def _look_at_window_task(self, call_id=None):
        try:
            await asyncio.sleep(12)
            print("开始拍照窗外...")
            # 在线程池中执行（含拍照/网络请求/播放，全为阻塞操作）
            loop = asyncio.get_event_loop()
            fn = functools.partial(look_and_describe, prompt="用简洁的语音描述图中窗外的天气，回答内容中不要包含：这张图片……类似内容")
            description = await loop.run_in_executor(None, fn)
            print(f"lookAndDescribe 完成，描述：{description}")
            response = {"status": "success", "message": description}
            #response = {"status": "success", "message": f"好的"}
        except Exception as e:
            print(f"lookAndDescribe 执行出错: {e}")
            description = "抱歉，查看周围时出现了错误。"
            #response = {"status": "error", "message": str(e)}
            response = {"status": "error", "message": description}

        await self._send_function_output(call_id, response)

    async def _handle_session_messages(self, message: Any, msg_type: str):
        """处理会话相关消息"""
        match msg_type:
            case "session.created":
                print("会话创建消息")
                print(f"  Session Id: {message.get('session', {}).get('id')}")
            case "session.updated":
                print("会话更新消息")
                print(f"updated session: {message.get('session')}")
            case "session.finished":
                # Qwen 后端 session.finish 的回执（本程序未主动结束会话时也会打印）
                print("会话已结束消息")
            case "error":
                print("错误消息")
                print(f"  Error: {message.get('error')}")

    async def _handle_call_hangup(self, message: dict):
        """处理服务端主动挂断事件（call.hangup，仅 GLM 后端下发）

        GLM Realtime 按"通话"模型管理会话：用户长时间不说话（静默超时）等
        场景下，服务端会下发 call.hangup 并随后关闭连接。

        这里只记录原因并置位 hangup_event，真正退出接收循环；是否重连、
        何时重连由外层会话循环（lelamp_main.run_realtime_app）决定。
        """
        reason = message.get("reason", "unknown")
        end_reason_code = message.get("end_reason_code", "unknown")

        self.hangup_reason = f"{reason}/{end_reason_code}"
        self.hangup_event.set()
        print(f"[通话挂断] 服务端已结束会话: reason={reason}, "
              f"end_reason_code={end_reason_code}")

    async def _handle_audio_input_messages(self, message: dict, msg_type: str):
        """处理音频输入相关消息"""
        match msg_type:
            case "input_audio_buffer.committed":
                print("音频缓冲区提交消息")
                if message.get("item_id"):
                    print(f"  Item Id: {message.get('item_id')}")
            case "input_audio_buffer.speech_started":
                print("语音开始消息")
                # ── 录音指示：用户开始说话时启动绿色追逐灯效 ──
                # 无论之前是否开着灯都播放（会暂时打断当前灯光），
                # 说话结束后按 _light_on 恢复原颜色还是熄灭（见 speech_stopped）。
                self.rgb_service.dispatch("running", (0, 250, 0, 128, 1))
            case "input_audio_buffer.speech_stopped":
                print("语音结束消息")
                # ── 录音指示结束：之前开过灯则恢复之前的灯光颜色，否则熄灭 ──
                if self._light_on:
                    self.rgb_service.dispatch("solid", self._light_color)
                else:
                    self.rgb_service.clear()

    async def _handle_conversation_messages(self, message: dict, msg_type: str):
        """处理会话项目相关消息"""
        match msg_type:
            case "conversation.item.created":
                print("会话项目创建消息")
            case "conversation.item.input_audio_transcription.completed":
                print("输入音频转写完成消息")
                print(f"  Transcript: {message.get('transcript')}")

    async def _handle_response_messages(self, message: dict, msg_type: str):
        """处理响应相关消息"""
        match msg_type:
            case "response.created":
                print("响应创建消息")
                print(f"  Response Id: {message.get('response', {}).get('id')}")
                saver.start() # 开始准备录制
            case "response.done":
                print("响应完成消息")
                response = message.get("response")
                if response:
                    print(f"  Response Id: {response.get('id')}")
                    print(f"  Status: {response.get('status')}")
            case "response.audio.delta":
                print("模型音频增量消息")
                print(f"  Response Id: {message.get('response_id')}")
                delta = message.get("delta")
                if delta:
                    print(f"  Delta Length: {len(delta)}")
                    try:
                        # 1. 解码 Base64 数据（Qwen / GLM 均为 base64 PCM 24kHz）
                        audio_bytes = base64.b64decode(delta)
                        # 2. 将数据交给保存器
                        saver.add_frame(audio_bytes)
                        # 这里也可以选择同时播放音频，如果不需要播放可以注释掉
                        #os.system()
                    except Exception as e:
                        print(f"处理音频数据出错: {e}")
                else:
                    print("  Delta: None")
            case "response.audio.done":
                print("模型音频完成消息")
                try:
                    saver.stop()
                    print("Audio file saved")
                    # ── 回声抑制：通过 run_in_executor 在线程池播放 ──
                    # 原代码 saver.play_with_aplay() 是阻塞调用，直接 await 会卡住事件循环。
                    # 改用 run_in_executor：播放在后台线程执行，asyncio 继续正常调度；
                    # 同时将 is_playing_event 传入，播放期间置位屏蔽麦克风采集。
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None,
                        saver.play_with_aplay,
                        self.is_playing_event
                    )
                except Exception as e:
                    print(f"处理音频数据出错: {e}")
            case "response.audio_transcript.delta":
                print("模型音频文本增量消息")
                print(f"  Response Id: {message.get('response_id')}")
                delta = message.get("delta")
                print(f"  Delta: {delta if delta else 'None'}")
            case "response.audio_transcript.done":
                print("模型音频文本完成消息")

    async def receive_messages(self):
        """
        统一的消息接收处理函数
        处理所有类型的消息，包括音频、视频、函数调用和服务端VAD等场景
        （消息统一为 dict 格式，Qwen / GLM 后端通用）
        """
        # ── 定时随机动作：启动后台任务 ──
        # 按需求 3：注释掉定时随机执行 action 的功能
        # self._random_action_task = asyncio.create_task(self._random_action_loop())

        try:
            while not self.provider.closed:
                if self.shutdown_event and self.shutdown_event.is_set():
                    print("正在停止消息接收...")
                    break

                try:
                    message = await asyncio.wait_for(self.provider.recv(), timeout=5.0)
                    if message is None:
                        continue

                    msg_type = message.get("type")
                    if msg_type is None:
                        print("收到未知类型的消息:", message)
                        continue

                    # 检查是否有自定义处理器
                    if msg_type in self._custom_handlers:
                        await self._custom_handlers[msg_type](message)
                        continue

                    if msg_type == "connection.closed":
                        # 本地合成事件：连接已断开（见 rt_provider.qwen_provider）
                        print(f"实时对话连接已断开: {message.get('message')}")
                        break
                    elif msg_type.startswith("session.") or msg_type == "error":
                        await self._handle_session_messages(message, msg_type)
                    elif msg_type.startswith("input_audio_buffer."):
                        await self._handle_audio_input_messages(message, msg_type)
                    elif msg_type.startswith("conversation.item."):
                        await self._handle_conversation_messages(message, msg_type)
                    elif msg_type.startswith("response."):
                        if msg_type == "response.function_call_arguments.done":
                            await self._handle_function_call(message)
                        else:
                            await self._handle_response_messages(message, msg_type)
                    elif msg_type == "heartbeat":
                        print("心跳消息")
                    elif msg_type == "call.hangup":
                        # 服务端挂断（静默超时等）：记录原因后退出接收循环，
                        # finally 会释放硬件并关闭连接，外层负责重建会话
                        await self._handle_call_hangup(message)
                        break
                    else:
                        print(f"未处理的消息类型: {msg_type}")
                        print(message)

                except TimeoutError:
                    continue  # 超时后继续尝试接收
                except Exception as e:
                    if not self.shutdown_event or not self.shutdown_event.is_set():
                        print(f"接收消息时发生错误: {e}")
                    break
        finally:
            # ── 定时随机动作：停止后台任务 ──
            if self._random_action_task and not self._random_action_task.done():
                self._random_action_task.cancel()
                try:
                    await self._random_action_task
                except asyncio.CancelledError:
                    pass

            # ── 人脸追踪：先停，确保不会再访问 anim_service 的电机总线 ──
            if getattr(self, "face_tracking_service", None) is not None:
                try:
                    print("停止人脸追踪")
                    self.face_tracking_service.stop()
                except Exception as e:
                    print(f"停止人脸追踪出错: {e}")

            print("正在释放硬件资源...")
            try:
                self.rgb_service.clear()
                time.sleep(0.5)
                self.rgb_service.stop()
            except Exception as e:
                print(f"RGB service 停止出错: {e}")

            if self._motor_enabled:
                try:
                    print("停止电机")
                    self.anim_service.stop()
                except Exception as e:
                    print(f"停止电机出错：{e}")
            else:
                print("电机没启动，disenabled")

            if not self.provider.closed:
                try:
                    await self.provider.close()
                    print("实时对话连接已关闭")
                except Exception as e:
                    print(f"关闭实时对话连接失败: {e}")


async def create_message_handler(
    provider: RealtimeProvider,
    shutdown_event: Optional[asyncio.Event] = None,
    is_playing_event: Optional[asyncio.Event] = None,
) -> MessageHandler:
    """
    创建消息处理器实例
    Args:
        provider: 实时对话后端实例（rt_provider.RealtimeProvider）
        shutdown_event: 关闭事件
        is_playing_event: 播放状态标志，用于回声抑制
    Returns:
        MessageHandler实例
    """
    #my_led.breath_effect()
    return MessageHandler(provider, shutdown_event, is_playing_event)
