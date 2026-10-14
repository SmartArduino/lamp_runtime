"""
统一的 aplay 播放工具。

背景：no_human.wav / poweroff.wav / TTS 裸 PCM 都通过 aplay 输出到板载
rockchip-es8316 声卡。plughw 直开硬件设备不支持多流混音，两个 aplay 并发
打开同一设备会报 "audio open error: Device or resource busy"（GPIO9 无人
提示音与 TTS 同时播放时尤为常见）。

做法：
1. 模块级 threading.Lock 把所有 aplay 播放串行化，后到者等锁排队而不是失败；
2. 提示音 wav 统一按仓库根目录解析绝对路径，不依赖进程工作目录。
"""

import os
import subprocess
import threading

# 输出设备：板载 rockchip-es8316
OUTPUT_DEVICE = "plughw:rockchipes8316,0"

# lelamp/service/audio_playback.py 向上两级即仓库根目录（lamp_runtime/）
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

_PLAYBACK_LOCK = threading.Lock()


def resolve_asset_path(filename: str) -> str:
    """把资源文件名（如 no_human.wav）解析为仓库根目录下的绝对路径"""
    return os.path.join(_PROJECT_ROOT, filename)


def play_wav_file(wav_path: str) -> bool:
    """用 aplay 播放一个 wav 文件（阻塞直到播完）。成功返回 True。"""
    try:
        with _PLAYBACK_LOCK:
            proc = subprocess.run(
                ["aplay", "-D", OUTPUT_DEVICE, wav_path],
                capture_output=True,
            )
    except FileNotFoundError:
        print("未找到aplay命令，请安装alsa-utils：sudo apt install alsa-utils")
        return False
    except Exception as e:
        print(f"aplay 播放异常({wav_path}): {e}")
        return False

    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        print(f"aplay 播放失败({wav_path}): {stderr}")
        return False
    return True


def play_wav_stream(wav_bytes: bytes, device: str = OUTPUT_DEVICE) -> bool:
    """用 aplay 播放含 WAV 头的字节流（阻塞直到播完）。成功返回 True。"""
    if not wav_bytes:
        print("无音频数据可播放")
        return False

    try:
        with _PLAYBACK_LOCK:
            proc = subprocess.Popen(
                ["aplay", "-D", device],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            _, stderr = proc.communicate(input=wav_bytes)
    except FileNotFoundError:
        print("未找到aplay命令，请安装alsa-utils：sudo apt install alsa-utils")
        return False
    except Exception as e:
        print(f"播放出错: {e}")
        return False

    if proc.returncode != 0:
        print(f"aplay播放失败: {stderr.decode('utf-8', errors='replace').strip()}")
        return False
    return True


def play_pcm_data(pcm_data: bytes, rate: int) -> bool:
    """用 aplay 播放裸 PCM 数据（S16_LE，阻塞直到播完）。成功返回 True。"""
    if not pcm_data:
        print("无音频数据可播放")
        return False

    try:
        with _PLAYBACK_LOCK:
            proc = subprocess.Popen(
                [
                    "aplay",
                    "-r", str(rate),
                    "-D", OUTPUT_DEVICE,
                    "-f", "S16_LE",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            _, stderr = proc.communicate(input=pcm_data)
    except FileNotFoundError:
        print("未找到aplay命令，请安装alsa-utils：sudo apt install alsa-utils")
        return False
    except Exception as e:
        print(f"播放出错: {e}")
        return False

    if proc.returncode != 0:
        print(f"aplay播放失败: {stderr.decode('utf-8', errors='replace').strip()}")
        return False
    print("音频播放完成（aplay）")
    return True
