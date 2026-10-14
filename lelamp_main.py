# Copyright (c) ZhiPu Corporation.
# Licensed under the MIT license.

import asyncio
import os
import signal
import socket
import sys
from typing import Optional
#from gpiozero import Button, LED
import subprocess

import numpy as np
import sounddevice as sd
import wave
import time as _time  # 别名导入，避免与 audio_callback 的 time 参数同名冲突
from dotenv import load_dotenv
from message_handler import (
    create_message_handler,
    play_action_pausing_tracking,
    HAND_DIRECTION_TO_ACTION,
)

# ── 实时对话后端抽象层：通过 LELAMP_RT_PROVIDER 环境变量选择 ──
# qwen（默认，Qwen-Omni-Realtime / dashscope）或 glm（GLM Realtime / rtclient）
from rt_provider import (
    RealtimeConfig,
    create_realtime_provider,
    load_realtime_config,
)


shutdown_event: Optional[asyncio.Event] = None

# ── 回声抑制：全局播放状态标志 ──
# 置位（set）= TTS 正在通过扬声器播放，audio_callback 跳过入队；
# 清除（clear）= 播放结束，恢复正常采集。
is_playing_event: Optional[asyncio.Event] = None

# ════════════════════════════════════════════════════════════════════
# ▼▼▼ 调试开关 ▼▼▼
# 把 AUDIO_ENABLED 设为 False 可暂时关闭音频功能（采集 + 播放 + WebSocket 对话），
# 仅运行人脸追踪服务。用于排查"音频抢 CPU 导致追踪卡顿"的问题。
# 也可以通过环境变量 LELAMP_AUDIO=0 / LELAMP_AUDIO=1 覆盖此默认值，
# 这样不必改代码就能切换，例如：
#     LELAMP_AUDIO=0 python3 lelamp_main.py     # 关音频，仅追踪
#     LELAMP_AUDIO=1 python3 lelamp_main.py     # 完整功能（默认）
# ════════════════════════════════════════════════════════════════════
AUDIO_ENABLED: bool = True   # ← 改这里：True=完整模式，False=纯追踪调试模式
MOTOR_ENABLED: bool = True   # ← 改这里：True=启动电机/人脸追踪，False=仅灯光+音频

# ════════════════════════════════════════════════════════════════════
# ▼▼▼ 回声抑制开关 ▼▼▼
# TTS 播放期间暂停麦克风采集（并在播放结束后多丢弃一段尾音），防止扬声器
# 播出的声音被采回服务端、误触发自动回复。默认开启。
# 可通过环境变量 LELAMP_ECHO_SUPPRESSION=0 / 1 覆盖此默认值（.env 可配）：
#     LELAMP_ECHO_SUPPRESSION=0   # 关闭回声抑制（如使用耳机、外接声卡 AEC 时）
#     LELAMP_ECHO_SUPPRESSION=1   # 开启（默认）
# ════════════════════════════════════════════════════════════════════
ECHO_SUPPRESSION_ENABLED: bool = True   # ← 改这里：True=开启（默认），False=关闭

def _resolve_audio_enabled() -> bool:
    """环境变量 LELAMP_AUDIO 优先，否则用代码常量 AUDIO_ENABLED"""
    env = os.environ.get("LELAMP_AUDIO", "").strip().lower()
    if env in ("0", "false", "off", "no"):
        return False
    if env in ("1", "true", "on", "yes"):
        return True
    return AUDIO_ENABLED

def _resolve_motor_enabled() -> bool:
    """环境变量 LELAMP_MOTOR 优先，否则用代码常量 MOTOR_ENABLED"""
    env = os.environ.get("LELAMP_MOTOR", "").strip().lower()
    if env in ("0", "false", "off", "no"):
        return False
    if env in ("1", "true", "on", "yes"):
        return True
    return MOTOR_ENABLED

def _resolve_echo_suppression_enabled() -> bool:
    """环境变量 LELAMP_ECHO_SUPPRESSION 优先，否则用代码常量 ECHO_SUPPRESSION_ENABLED"""
    env = os.environ.get("LELAMP_ECHO_SUPPRESSION", "").strip().lower()
    if env in ("0", "false", "off", "no"):
        return False
    if env in ("1", "true", "on", "yes"):
        return True
    return ECHO_SUPPRESSION_ENABLED

# ── GPIO5 LED 闪烁控制 ──
#status_led: Optional[LED] = None

def init_status_led():
    """初始化 GPIO5 LED，每秒闪烁一次"""
    """
    global status_led
    try:
        status_led = LED(5)
        status_led.blink(on_time=0.5, off_time=0.5)  # 每秒闪一次（亮0.5s，灭0.5s）
        print("[GPIO] GPIO5 LED 已启动，每秒闪烁一次")
    except Exception as e:
        print(f"[GPIO] 初始化 LED 失败: {e}")
    """
    print("没有实现LED for Rock5C")

def cleanup_status_led():
    """关闭 LED"""
    """
    global status_led
    if status_led is not None:
        try:
            status_led.close()
            print("[GPIO] GPIO5 LED 已关闭")
        except Exception as e:
            print(f"[GPIO] 关闭 LED 失败: {e}")
        status_led = None
    """
    print("没实现LED for Rock5C")
        
# 音频采集配置
SAMPLING_RATE = 16000  # 采样率（需与服务端要求匹配）
CHANNELS = 1  # 单声道
SAMPLE_WIDTH = 2  # 16位深度
PACKET_MS = 100  # 每包音频时长（毫秒）
PACKET_SAMPLES = int(SAMPLING_RATE * PACKET_MS / 1000)  # 每包采样点数

agent_prompt = """
你是一个智能台灯，你可以作为人类的情感陪伴。
"""

# 服务端欢迎词（greeting_config）：只在程序启动后的首个会话播放，重连会话不再播
GREETING_CONTENT = "你好主人，我是你的智能台灯，你可以叫我小智"


def _function_tool(name: str, description: str, parameters: dict = None) -> dict:
    """构造统一的 OpenAI 嵌套格式工具定义。

    Qwen 后端直接使用该格式；GLM 后端由 rt_provider.glm_provider
    自动转换为智谱的扁平格式。
    """
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters or {"type": "object", "properties": {}},
        },
    }


_BRIGHTNESS_PARAM = {
    "type": "string",
    "description": "灯光的亮度",
}

# 会话注册的工具（与原实现保持同一集合）
SESSION_TOOLS = [
    _function_tool("lightOn", "打开台灯，或者说开灯或打开灯光", {"type": "object", "properties": {"brightness": _BRIGHTNESS_PARAM}, "required": ["brightness"]}),
    _function_tool("lightOff", "关闭台灯，或者说关灯或关闭灯光", {"type": "object", "properties": {"brightness": _BRIGHTNESS_PARAM}, "required": ["brightness"]}),
    _function_tool("lightRed", "把灯光设置成红色", {"type": "object", "properties": {"brightness": _BRIGHTNESS_PARAM}, "required": ["brightness"]}),
    _function_tool("lightGreen", "把灯光设置成绿色", {"type": "object", "properties": {"brightness": _BRIGHTNESS_PARAM}, "required": ["brightness"]}),
    _function_tool("lightBlue", "把灯光设置成蓝色", {"type": "object", "properties": {"brightness": _BRIGHTNESS_PARAM}, "required": ["brightness"]}),
]

def init_hardware():
    """初始化硬件权限和状态：串口权限、风扇转速"""
    try:
        # 1. 修改串口权限为 666 (所有用户可读写)
        # 注意：/dev/ttyS4 和 /dev/ttyACM0 是不同的设备，这里分别执行
        subprocess.run(["sudo", "chmod", "666", "/dev/ttyS4"], check=True)
        subprocess.run(["sudo", "chmod", "666", "/dev/ttyACM0"], check=True)
        
        # 2. 设置风扇转速 (PWM)
        # Python 的 subprocess 不支持 shell 的重定向符号 >，
        # 所以使用 'tee' 命令配合 shell=True 来实现写入文件。
        pwm_value = "240"
        subprocess.run(f"echo {pwm_value} | sudo tee /sys/devices/platform/pwm-fan/hwmon/hwmon*/pwm1", shell=True, check=True)
        
        print("硬件初始化完成 (串口权限 & 风扇转速)")
        
    except subprocess.CalledProcessError as e:
        print(f"硬件初始化失败: {e}")
    except Exception as e:
        print(f"发生未知错误: {e}")

def set_system_volume(volume_percent: int):
    """Internal helper to set system volume"""
    try:
        # 按照要求，将命令替换为 amixer -D hw:rockchipes8316 sset 'Headphone' unmute 100%
        # 注意：因为原函数参数 volume_percent 在新命令中没有使用到，这里固定设为 100%
        cmd_line = [
            "amixer", 
            "-D", "hw:rockchipes8316", 
            "sset", "Headphone", 
            "unmute", f"{volume_percent}%"
        ]
        
        subprocess.run(cmd_line, capture_output=True, text=True, timeout=5)
        
    except Exception:
        pass  # Silently fail during initialization


def handle_shutdown(sig=None, frame=None):
    """处理关闭信号"""
    if shutdown_event:
        print("\n正在关闭程序...")
        shutdown_event.set()


async def send_realtime_audio(provider):
    """
    实时采集音频并分帧发送。
    provider 为 rt_provider.RealtimeProvider 实例（GLM / Qwen 等后端透明替换）。
    """
    # ⭐ 用线程安全的 queue.Queue 替代 asyncio.Queue
    import queue as thread_queue
    audio_queue = thread_queue.Queue(maxsize=50)
    loop = asyncio.get_event_loop()

    # ── 回声抑制（采集侧，逐帧粒度）──
    # RESUME_GRACE_S：播放结束后继续丢弃采集帧的时长。
    # 原因：InputStream 用 latency='high'，麦克风从"物理采集"到进入回调有几百 ms 延迟，
    #   所以 TTS 播放最后那段采到的"自己的声音"，会在播放结束、is_playing_event 清除
    #   之后才到达回调。若入队发出去，服务端会把这段尾音回声当成用户说话，触发自动回复
    #   （现象：自动回复总紧跟一次播放结束，转写出"出口/嗯"等碎片）。
    # 做法：回调里维护一个"恢复时刻"，播放期间不断把它后推；播放结束后在 grace 窗口内
    #   继续丢弃，等高延迟缓冲里的尾音回声排空后再恢复入队。
    # 代价：设备说完话后的 RESUME_GRACE_S 秒内用户说话会被忽略，需稍等再开口；
    #   仍偶发自动回复就调大，觉得"抢话"迟钝就调小。
    RESUME_GRACE_S = 1.5
    _resume_state = {"deadline": 0.0}  # 用 dict 以便在闭包回调里可写

    def audio_callback(indata, frames, time, status):
        """音频采集回调函数"""
        if status:
            print(f"音频采集状态异常: {status}", file=sys.stderr)
        if shutdown_event.is_set():
            return

        now = _time.monotonic()
        # 播放中：丢弃，并把"恢复时刻"持续后推到 now + grace
        if is_playing_event is not None and is_playing_event.is_set():
            _resume_state["deadline"] = now + RESUME_GRACE_S
            return
        # 播放刚结束的 grace 窗口内：继续丢弃尾音回声
        if now < _resume_state["deadline"]:
            return

        try:
            # ⭐ seeed HAT 是立体声,取左声道转单声道
            mono = indata[:, 0].copy()
            audio_queue.put_nowait(mono.tobytes())
        except thread_queue.Full:
            pass

    stream = sd.InputStream(
        samplerate=SAMPLING_RATE,
        channels=2,              # ⭐ 硬件原生 2 通道
        dtype='int16',
        blocksize=0,             # ⭐ 系统自适应
        device='USB Audio Device',          # ⭐ 按名字匹配 seeed HAT
        callback=audio_callback,
        latency='high',          # ⭐ 树莓派 3B 用 high 更稳定
    )

    # 用于凑齐 100ms 包的缓冲
    target_bytes = PACKET_SAMPLES * SAMPLE_WIDTH  # 1600 帧 × 2 字节 = 3200
    audio_buffer = bytearray()

    # ── 连接失效保护：连续发送失败达到该次数即判定连接不可用并退出 ──
    # 修复问题：服务端挂断/断网后连接进入 closing 状态，原逻辑只 print
    # 不退出，导致 "Cannot write to closing transport" 每 100ms 刷屏一次
    MAX_SEND_FAILURES = 5
    consecutive_failures = 0

    try:
        stream.start()
        print(f"开始实时采集音频 - 采样率: {SAMPLING_RATE}Hz, 声道数: 1(从立体声左声道提取)")

        while not shutdown_event.is_set():
            if provider.closed:
                print("实时对话连接已关闭，停止音频采集发送")
                break

            try:
                # ⭐ 用 run_in_executor 从线程安全 queue 取数据
                packet_data = await loop.run_in_executor(
                    None, lambda: audio_queue.get(timeout=1.0)
                )
                audio_buffer.extend(packet_data)

                # 凑够 100ms 就发一包（原始 PCM，封装格式由 provider 决定）
                while len(audio_buffer) >= target_bytes:
                    send_data = bytes(audio_buffer[:target_bytes])
                    del audio_buffer[:target_bytes]

                    await provider.send_audio(send_data)
                    consecutive_failures = 0
                    # ⭐ 不再需要 asyncio.sleep,callback 已按采样率节流

            except thread_queue.Empty:
                continue
            except Exception as e:
                consecutive_failures += 1
                print(f"发送音频失败（连续第 {consecutive_failures} 次）: {e}")
                if (
                    shutdown_event.is_set()
                    or provider.closed
                    or consecutive_failures >= MAX_SEND_FAILURES
                ):
                    print("实时对话连接已不可用，停止音频发送")
                    break
                await asyncio.sleep(0.5)
    except Exception as e:
        print(f"实时音频处理失败: {e}")
    finally:
        stream.stop()
        stream.close()
        print("音频采集已停止")


# ════════════════════════════════════════════════════════════════════
# ▼▼▼ 开机网络检测 ▼▼▼
# 程序启动前先确认网络是否可用；不可用则每隔 5 秒播放一次提示音，
# 直到网络恢复才继续往下走，避免联网服务（ASR/TTS/LLM）初始化时直接报错退出。
# ════════════════════════════════════════════════════════════════════

# 用于探测连通性的候选主机（host, port），依次尝试，任意一个成功即视为有网络。
# 默认用国内可达的公共 DNS，避免因为访问境外站点被墙而误判为"无网络"。
_NETWORK_PROBE_HOSTS = [
    ("223.5.5.5", 53),   # 阿里 DNS
    ("119.29.29.29", 53),  # DNSPod
]
_NETWORK_PROBE_TIMEOUT = 3  # 秒

# 提示音播放设备：对应 `rockchip-es8316`（板载耳机/喇叭输出），
# 不用默认设备是因为板子上默认输出设备不一定是这张卡。
# sounddevice 支持按名字子串匹配，不需要写死设备索引（索引可能随插拔 USB 设备变化）。
_OUTPUT_DEVICE = "rockchip-es8316"


def check_network() -> bool:
    """尝试对候选主机建立 TCP 连接，判断当前设备是否联网。"""
    for host, port in _NETWORK_PROBE_HOSTS:
        try:
            with socket.create_connection((host, port), timeout=_NETWORK_PROBE_TIMEOUT):
                return True
        except OSError:
            continue
    return False


def _play_wav_file(wav_path: str) -> None:
    """同步播放一个 wav 文件，播放完成后返回。播放失败只打印日志，不抛异常。"""
    try:
        with wave.open(wav_path, "rb") as wf:
            samplerate = wf.getframerate()
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            raw = wf.readframes(wf.getnframes())

        dtype_map = {1: "int8", 2: "int16", 4: "int32"}
        dtype = dtype_map.get(sampwidth)
        if dtype is None:
            print(f"[网络检测] 不支持的采样位宽: {sampwidth * 8} bit", file=sys.stderr)
            return

        data = np.frombuffer(raw, dtype=dtype)
        if channels > 1:
            data = data.reshape(-1, channels)

        sd.play(data, samplerate, device=_OUTPUT_DEVICE)
        sd.wait()
    except FileNotFoundError:
        print(f"[网络检测] 未找到提示音文件: {wav_path}", file=sys.stderr)
    except Exception as e:
        print(f"[网络检测] 播放提示音失败: {e}", file=sys.stderr)


def wait_for_network(wav_filename: str = "no_network.wav", check_interval: float = 5.0) -> None:
    """
    阻塞直到网络可用。
    无网络期间：每隔 check_interval 秒播放一次 wav_filename 提示用户，同时重新探测网络；
    网络恢复后立即返回，让主流程继续往下走。
    """
    # 用脚本所在目录定位提示音文件，不依赖当前工作目录，更稳妥。
    script_dir = os.path.dirname(os.path.abspath(__file__))
    wav_path = os.path.join(script_dir, wav_filename)

    if check_network():
        print("[网络检测] 网络连接正常")
        return

    print("[网络检测] 当前无网络连接，等待网络恢复...")
    while not check_network():
        _play_wav_file(wav_path)
        _time.sleep(check_interval)

    print("[网络检测] 网络已恢复，继续启动应用")


async def face_tracking_only_main():
    """
    [调试模式] 仅跑硬件服务（RGB / Animation / FaceTracking），不连 WebSocket、不采集/播放音频。
    用于排查"音频抢 CPU 导致追踪卡顿"。

    设计要点：
    - 直接复用 MessageHandler 内部的服务初始化逻辑会把 WebSocket / 音频也带起来，
      所以这里手动拉起最小化的服务集合：RGBService + AnimationService + FaceTrackingService。
    - 不创建 MessageHandler，避免它对 client 的强依赖。
    """
    print("=" * 60)
    print("[调试模式] AUDIO_ENABLED=False")
    print("  - 跳过 WebSocket 连接")
    print("  - 跳过音频采集 / 播放")
    print("  - 仅运行 RGB / 电机 / 人脸追踪")
    print("  - 按 Ctrl+C 退出")
    print("=" * 60)

    # 延迟 import：只在调试模式才需要
    from lelamp.service.rgb.rgb_service import RGBService
    from lelamp.service.motors.animation_service import AnimationService
    from lelamp.service.vision.face_tracking_service import FaceTrackingService

    rgb = RGBService()
    anim = AnimationService(port="/dev/ttyACM0", lamp_id="bubble", fps=25, duration=1)

    # ── 电机开关：MOTOR_ENABLED=False 时不启动 AnimationService / FaceTrackingService ──
    if _resolve_motor_enabled():
        anim = AnimationService(port="/dev/ttyACM0", lamp_id="bubble", fps=25, duration=1)
        rgb.start()
        anim.start()
        print("[调试模式] RGB / 电机服务已启动")
    else:
        anim = None
        rgb.start()
        print("[调试模式] 仅 RGB 服务已启动（MOTOR_ENABLED=False，跳过电机/人脸追踪）")

    tracker = None
    if _resolve_motor_enabled():
        try:
            # 手势识别回调：根据食指指向播放对应动作。
            #   Up    -> anim.dispatch("play", "look_up_action")
            #   Down  -> anim.dispatch("play", "look_down_action")
            #   Left  -> anim.dispatch("play", "look_left_action")
            #   Right -> anim.dispatch("play", "look_right_action")
            # 播放期间暂停人脸追踪与手势识别，动作播完后恢复（见 play_action_pausing_tracking）。
            # 注意：tracker 在下面才赋值，但本回调只会在 tracker.start() 之后才触发，
            # 闭包按名字取值，调用时 tracker 已绑定，可安全使用。
            def on_hand_pointing(direction: str):
                action_name = HAND_DIRECTION_TO_ACTION.get((direction or "").strip().lower())
                if action_name is None:
                    print(f"[手势] 未知指向，忽略: {direction!r}")
                    return
                print(f"[手势] 指向 {direction} → 播放动作 {action_name}")
                play_action_pausing_tracking(anim, tracker, action_name)

            tracker = FaceTrackingService(
                anim_service=anim,
                # ── 手势识别参数（保持温和默认；如想关闭把 hand_enabled 改为 False） ──
                hand_enabled=True,
                hand_frame_stride=3,         # 每 3 个相机帧检测一次手势
                on_hand_pointing=on_hand_pointing,
            )
            tracker.start()

            # ── 人体感应 → 人脸追踪联动（与 MessageHandler 中的接线一致） ──
            # level_state: 0=无人或人体静止，1=有人或人体移动
            def on_human_sensor(level_state: int):
                if tracker is not None:
                    try:
                        tracker.set_human_sensor_state(level_state)
                    except Exception as e:
                        print(f"[人体感应] 通知人脸追踪服务失败: {e}")

            rgb.on_human_sensor = on_human_sensor
            print("[调试模式] 人脸追踪服务已启动（含手势识别）")
        except Exception as e:
            print(f"[调试模式] 人脸追踪启动失败: {e}")

    # 主循环：等 shutdown_event
    try:
        while not shutdown_event.is_set():
            await asyncio.sleep(0.5)
    finally:
        print("[调试模式] 正在停止服务...")
        if tracker is not None:
            try:
                tracker.stop()
            except Exception as e:
                print(f"停止 tracker 异常: {e}")
        try:
            anim.stop()
        except Exception as e:
            print(f"停止 anim 异常: {e}")
        try:
            rgb.stop()
        except Exception as e:
            print(f"停止 rgb 异常: {e}")
        print("[调试模式] 已退出")


# ── 会话重建（自动重连）参数 ──
RECONNECT_INITIAL_DELAY = 10.0   # 首次重连的等待时间
RECONNECT_MAX_DELAY = 30.0      # 指数退避的等待上限
SESSION_HEALTHY_SECONDS = 60.0  # 会话存活超过该时长视为健康，重置退避间隔


async def run_session(config: RealtimeConfig, enable_greeting: bool = True,
                      greeting_state: Optional[dict] = None) -> str:
    """运行一次完整的实时会话（连接 → session.update → 收发 → 清理）。

    Args:
        config: 实时对话后端配置（qwen / glm 由 LELAMP_RT_PROVIDER 决定）
        enable_greeting: 本次会话是否启用欢迎词。
        greeting_state: 可变标记字典。session.update 成功发出后写入
            greeting_state["greeting_armed"]=enable_greeting，供外层循环
            判断欢迎词是否已触发；连接失败（未发出）则不写入，下次重试
            仍会播放。

    返回结束原因：
        'shutdown'       收到退出信号（Ctrl+C / SIGTERM）
        'hangup:<原因>'  服务端主动挂断（如静默超时 silence_timeout）
        'error'          连接异常或其他未预期错误

    结束后硬件资源由 MessageHandler.receive_messages 的 finally 释放，
    重连时重新创建各服务即可。
    """
    provider = create_realtime_provider(config)
    try:
        await provider.connect()
        if shutdown_event.is_set():
            return "shutdown"

        # ── 欢迎词只在首个（成功建立的）会话启用，重连会话不再播 ──
        await provider.update_session(
            instructions=agent_prompt,
            tools=SESSION_TOOLS,
            greeting_text=GREETING_CONTENT if enable_greeting else None,
        )
        if greeting_state is not None and enable_greeting:
            greeting_state["greeting_armed"] = True

        if shutdown_event.is_set():
            return "shutdown"

        # ── 回声抑制：将 is_playing_event 传入 MessageHandler ──
        message_handler = await create_message_handler(
            provider, shutdown_event, is_playing_event
        )

        send_task = asyncio.create_task(send_realtime_audio(provider))
        receive_task = asyncio.create_task(message_handler.receive_messages())

        try:
            await asyncio.gather(send_task, receive_task)
        except Exception as e:
            print(f"任务执行出错: {e}")
            for task in [send_task, receive_task]:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

        if shutdown_event.is_set():
            return "shutdown"
        if message_handler.hangup_event.is_set():
            return f"hangup:{message_handler.hangup_reason}"
        return "error"
    finally:
        try:
            await provider.close()
        except Exception as e:
            print(f"[会话] 关闭实时对话连接出错: {e}")


async def run_realtime_app():
    global shutdown_event, is_playing_event
    shutdown_event = asyncio.Event()

    # ── 回声抑制：创建播放状态标志，传入 MessageHandler 共享 ──
    # 关闭回声抑制时传 None：audio_callback 与 play_with_aplay 对 None
    # 均按"不抑制"处理，麦克风采集不再因播放而暂停。
    if _resolve_echo_suppression_enabled():
        is_playing_event = asyncio.Event()
        print("[回声抑制] 已开启（TTS 播放期间暂停麦克风采集）")
    else:
        is_playing_event = None
        print("[回声抑制] 已关闭（LELAMP_ECHO_SUPPRESSION=0），"
              "播放期间麦克风持续采集，回声可能触发自动回复")

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, handle_shutdown)

    # ── 调试开关：关音频时跳过 WebSocket，进入纯追踪模式 ──
    if not _resolve_audio_enabled():
        await face_tracking_only_main()
        return

    # ── 后端选择：LELAMP_RT_PROVIDER（默认 qwen，可选 glm）──
    config = load_realtime_config()
    backend_desc = config.provider
    if config.provider == "qwen":
        backend_desc += f"（model={config.model}, voice={config.voice}）"
    else:
        backend_desc += f"（voice={config.voice}）"
    print(f"[实时对话] 使用后端: {backend_desc}")

    # ── 会话重建循环：服务端挂断（静默超时）/连接断开后自动重连新会话 ──
    # 不依赖进程退出 + systemd 重启：lamp.service 配了 StartLimitBurst=3 限流，
    # 反复崩溃重启会触发限流把整个服务置为 failed；进程内重连没有该限制。
    reconnect_delay = RECONNECT_INITIAL_DELAY
    session_no = 0
    # ── 欢迎词只播一次：仅在尚未成功播出的会话上启用 ──
    # （首次连接失败时 greeting_armed 不会置位，下次重试仍会播）
    greeting_state = {"greeting_armed": False}
    while not shutdown_event.is_set():
        session_no += 1
        session_start = _time.monotonic()

        try:
            reason = await run_session(
                config,
                enable_greeting=not greeting_state["greeting_armed"],
                greeting_state=greeting_state,
            )
        except Exception as e:
            print(f"[会话] 第 {session_no} 个会话异常退出: {e}")
            reason = "error"

        if reason == "shutdown" or shutdown_event.is_set():
            break

        # 上一个会话存活较久，说明服务端/网络正常，重置退避间隔
        if _time.monotonic() - session_start >= SESSION_HEALTHY_SECONDS:
            reconnect_delay = RECONNECT_INITIAL_DELAY

        print(f"[会话] 第 {session_no} 个会话已结束（原因: {reason}），"
              f"{reconnect_delay:.0f} 秒后自动重建会话...")
        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=reconnect_delay)
        except TimeoutError:
            pass
        reconnect_delay = min(reconnect_delay * 2, RECONNECT_MAX_DELAY)

    print("程序已完成退出")



if __name__ == "__main__":
    # 从脚本所在目录加载 .env（与启动时所在的工作目录无关，systemd/手动行为一致）；
    # 不使用 override：shell 中临时 export 的变量仍可覆盖 .env，方便调试。
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if load_dotenv(_env_path):
        print(f"[配置] 已加载环境参数文件: {_env_path}")
    else:
        print(f"[配置] 未找到环境参数文件（{_env_path}），仅使用当前进程环境变量")

    # ── 联网检测：无网络时循环播报提示音，直到网络恢复才继续 ──
    wait_for_network()

    #init_hardware()
    # ── 初始化 GPIO5 LED ──
    init_status_led()

    try:
        if _resolve_audio_enabled():
            import sounddevice as sd
            sd.check_input_settings()
        else:
            print("[调试模式] AUDIO_ENABLED=False，跳过音频设备检查")
    except ImportError:
        print("请先安装sounddevice库: pip install sounddevice", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"音频设备检查失败: {e}", file=sys.stderr)
        sys.exit(1)

    set_system_volume(90)

    try:
        asyncio.run(run_realtime_app())
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序执行出错: {e}")
    finally:
        cleanup_status_led()
        print("程序已退出")
