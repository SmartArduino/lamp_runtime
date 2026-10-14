"""
vision_service.py
功能：拍摄一帧图像 → 视觉大模型识别 → （可选）TTS 播放
集成到 lelamp 工程，供 message_handler 在处理 lookAndDescribe tool 时调用。

视觉识别后端跟随实时对话后端（LELAMP_RT_PROVIDER）自动选择：
- qwen（默认）：dashscope MultiModalConversation + Qwen-VL 系列模型
- glm：zai SDK + GLM-4V-Flash

也可以用环境变量 LELAMP_VISION_PROVIDER=qwen|glm 单独指定视觉后端。
各 SDK 在对应分支内延迟导入：只用 qwen 时无需安装 zai，反之亦然。
"""

import base64
import subprocess
import tempfile
import os
import logging
from pathlib import Path
from typing import Any

from ..audio_playback import play_wav_stream

logger = logging.getLogger("vision_service")

# ── 可按需修改的配置 ─────────────────────────────────────────────────────────
CAMERA_DEVICE   = "/dev/video0"          # 摄像头设备（树莓派默认 video0）
CAPTURE_CMD     = "fswebcam"            # 拍照后端："libcamera" 或 "fswebcam"
APLAY_DEVICE    = "plughw:2,0"           # 声卡设备，与 message_handler 保持一致
TTS_VOICE       = "female"               # TTS 音色
TTS_SPEED       = 1.0                    # 语速
TTS_VOLUME      = 1.0                    # 音量
VISION_PROMPT   = "用简洁语言描述你从这张图片看到什么内容，回答时不要包含类似：这张图片……这样的内容"

# 视觉模型名（调用时读取，支持 .env 热更新）
# Qwen：QWEN_VISION_MODEL，默认 qwen-vl-max-latest（长期别名，指向最新 qwen-vl-max）
# GLM：GLM_VISION_MODEL，默认 glm-4v-flash
VISION_MODEL_DEFAULTS = {
    "qwen": "qwen-vl-max-latest",
    "glm": "glm-4v-flash",
}
VISION_MODEL_ENV_VARS = {
    "qwen": "QWEN_VISION_MODEL",
    "glm": "GLM_VISION_MODEL",
}
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_vision_provider() -> str:
    """决定视觉识别使用的后端：LELAMP_VISION_PROVIDER 优先，否则跟随实时对话后端。"""
    env = os.environ.get("LELAMP_VISION_PROVIDER", "").strip().lower()
    if env in ("qwen", "dashscope"):
        return "qwen"
    if env in ("glm", "zhipu"):
        return "glm"
    # 跟随实时对话后端（LELAMP_RT_PROVIDER，默认 qwen）
    from rt_provider import resolve_provider_name
    return resolve_provider_name()


def _capture_image(output_path: str) -> bool:
    """
    调用系统命令拍摄一帧图片，保存到 output_path。
    支持 libcamera-still（树莓派官方摄像头）和 fswebcam（USB摄像头）。
    返回 True 表示成功。
    """
    try:
        if CAPTURE_CMD == "libcamera":
            cmd = [
                "libcamera-still",
                "--nopreview",
                "--immediate",
                "-o", output_path,
                "--width", "640",
                "--height", "480",
                "--timeout", "500",   # ms，快速抓帧
            ]
        else:  # fswebcam（USB 摄像头）
            cmd = [
                "fswebcam",
                "-d", CAMERA_DEVICE,
                "-r", "640x480",
                "--no-banner",
                output_path,
            ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            logger.error(f"拍照失败: {result.stderr.strip()}")
            return False

        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            logger.error("拍照命令执行成功，但未生成图片文件")
            return False

        logger.info(f"拍照成功: {output_path}")
        print(f"已经拍照")
        return True

    except FileNotFoundError as e:
        logger.error(f"拍照命令未找到，请检查 CAPTURE_CMD 配置: {e}")
        return False
    except subprocess.TimeoutExpired:
        logger.error("拍照超时")
        return False
    except Exception as e:
        logger.error(f"拍照异常: {e}")
        return False


def _image_to_base64(image_path: str) -> tuple[str, str]:
    """
    将图片文件转换为 base64 字符串，同时返回 MIME 类型。
    返回 (base64_str, mime_type)
    """
    ext = Path(image_path).suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    mime_type = mime_map.get(ext, "image/jpeg")

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    return b64, mime_type


def _vision_model(provider: str) -> str:
    from rt_provider import clean_env_value

    env_var = VISION_MODEL_ENV_VARS[provider]
    return (clean_env_value(env_var, os.environ.get(env_var, ""))
            or VISION_MODEL_DEFAULTS[provider])


def _recognize_image_qwen(image_path: str, prompt: str) -> str:
    """调用 Qwen-VL 系列模型（dashscope）识别图片内容，返回描述文本。"""
    import dashscope  # 延迟导入：仅 qwen 后端需要
    from rt_provider import clean_env_value

    api_key = clean_env_value(
        "DASHSCOPE_API_KEY", os.environ.get("DASHSCOPE_API_KEY", "")
    )
    if not api_key:
        raise OSError("环境变量 'DASHSCOPE_API_KEY' 未设置或为空（Qwen 视觉识别需要）")

    model = _vision_model("qwen")
    logger.info(f"调用 {model} 识别图片...")
    b64, mime_type = _image_to_base64(image_path)

    response = dashscope.MultiModalConversation.call(
        model=model,
        api_key=api_key,
        messages=[
            {
                "role": "user",
                "content": [
                    # data URI 形式的 base64 图片
                    {"image": f"data:{mime_type};base64,{b64}"},
                    {"text": prompt},
                ],
            }
        ],
        max_tokens=256,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"dashscope 视觉识别失败: code={getattr(response, 'code', None)}, "
            f"message={getattr(response, 'message', None)}"
        )

    # content 通常是 [{"text": "..."}] 列表，兼容纯字符串
    content = response.output.choices[0].message.content
    if isinstance(content, str):
        description = content
    else:
        description = "".join(
            item.get("text", "") for item in content if isinstance(item, dict)
        ).strip()

    logger.info(f"图片识别结果: {description}")
    return description


def _recognize_image_glm(image_path: str, prompt: str) -> str:
    """调用 GLM-4V-Flash（zai SDK）识别图片内容，返回描述文本。"""
    from zai import ZhipuAiClient  # 延迟导入：仅 GLM 后端需要
    from rt_provider import clean_env_value

    api_key = clean_env_value(
        "ZHIPU_API_KEY", os.environ.get("ZHIPU_API_KEY", "")
    )
    if not api_key:
        raise OSError("环境变量 'ZHIPU_API_KEY' 未设置或为空（GLM 视觉识别需要）")

    client = ZhipuAiClient(api_key=api_key)
    model = _vision_model("glm")
    logger.info(f"调用 {model} 识别图片...")
    b64, mime_type = _image_to_base64(image_path)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{b64}"
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
        max_tokens=256,
    )

    description = response.choices[0].message.content
    logger.info(f"图片识别结果: {description}")
    return description


def _recognize_image(image_path: str, prompt: str) -> str:
    """按当前后端分发图片识别。"""
    provider = _resolve_vision_provider()
    if provider == "qwen":
        return _recognize_image_qwen(image_path, prompt)
    return _recognize_image_glm(image_path, prompt)


def _tts_to_bytes(client: Any, text: str) -> bytes:
    """
    调用 GLM-TTS（zai SDK）将文字转为语音，返回 WAV 字节流。
    注意：仅 GLM 后端可用；当前播放链路已停用（见 look_and_describe），
    如需启用请先按后端补齐 Qwen TTS 分支。
    """
    logger.info("调用 GLM-TTS 生成语音...")
    response = client.audio.speech(
        model="glm-tts",
        input=text,
        voice=TTS_VOICE,
        response_format="wav",
        speed=TTS_SPEED,
        volume=TTS_VOLUME,
    )

    # 将响应流读取为 bytes
    audio_bytes = b""
    for chunk in response.iter_bytes():
        audio_bytes += chunk

    logger.info(f"TTS 音频大小: {len(audio_bytes)} 字节")
    return audio_bytes


def _play_wav_bytes(wav_bytes: bytes, device: str = APLAY_DEVICE) -> None:
    """
    通过 aplay 播放 WAV 字节流（含 WAV 头，无需指定采样率格式）。
    播放经 audio_playback 全局锁串行化，避免与其他 aplay 并发抢占声卡。
    """
    if not wav_bytes:
        logger.error("无音频数据，跳过播放")
        return

    logger.info("开始播放 TTS 音频...")
    if play_wav_stream(wav_bytes, device):
        logger.info("TTS 音频播放完成")


# ── 对外暴露的主函数 ───────────────────────────────────────────────────────────

def look_and_describe(prompt: str = VISION_PROMPT) -> str:
    """
    完整流程：拍照 → 图片识别（按后端选择 Qwen-VL / GLM-4V-Flash）→ TTS 生成 → 播放。
    参数:
        prompt: 传给视觉模型的提示词，作为 text 字段的值。
                默认使用模块级常量 VISION_PROMPT。
    返回识别到的描述文本（供 function call 回传给模型）。
    出错时返回错误提示文字。
    """
    # 1. 拍照（使用临时文件，用完自动删除）
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        if not _capture_image(tmp_path):
            return "抱歉，摄像头拍照失败，无法查看周围环境。"

        # 2. 图片识别（传入 prompt，内部按后端分发）
        try:
            description = _recognize_image(tmp_path, prompt)
        except Exception as e:
            logger.error(f"图片识别失败: {e}")
            return f"抱歉，图像识别出错"

        # 3. TTS 生成音频
        """
        try:
            wav_bytes = _tts_to_bytes(client, description)
        except Exception as e:
            logger.error(f"TTS 生成失败: {e}")
            # TTS 失败不阻断，返回文字让模型继续说
            return description

        # 4. 播放音频
        _play_wav_bytes(wav_bytes)
        """
        return description

    finally:
        # 清理临时图片
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
