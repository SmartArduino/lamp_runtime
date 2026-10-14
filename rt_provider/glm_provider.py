"""
GLM Realtime 后端：基于智谱 rtclient SDK（原 lelamp_main.py / message_handler.py
中的 rtclient 调用逻辑完整迁移至此，行为保持不变）。

注意：本模块仅在选择 GLM 后端时才会被导入，rtclient 需按 README 手动安装。
"""

import asyncio
import base64
import wave
from io import BytesIO
from typing import Any, Dict, List, Optional

from rtclient import RTLowLevelClient
from rtclient.models import (
    FunctionCallOutputItem,
    InputAudioBufferAppendMessage,
    ItemCreateMessage,
    ServerVAD,
    SessionUpdateMessage,
    SessionUpdateParams,
)

from .base import RealtimeProvider

GLM_REALTIME_URL = "wss://open.bigmodel.cn/api/paas/v4/realtime"

# 输入音频约定：16kHz / 单声道 / 16bit，GLM 要求 WAV 封装后 base64
INPUT_SAMPLE_RATE = 16000
INPUT_CHANNELS = 1
INPUT_SAMPLE_WIDTH = 2


def _pcm_to_wav_base64(pcm: bytes) -> str:
    """把一段裸 PCM 封装成 WAV 再 base64 编码（GLM input_audio_format="wav"）。"""
    wav_io = BytesIO()
    with wave.open(wav_io, "wb") as wav_out:
        wav_out.setnchannels(INPUT_CHANNELS)
        wav_out.setsampwidth(INPUT_SAMPLE_WIDTH)
        wav_out.setframerate(INPUT_SAMPLE_RATE)
        wav_out.writeframes(pcm)
    wav_io.seek(0)
    return base64.b64encode(wav_io.getvalue()).decode("utf-8")


def _tools_to_glm_format(tools: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """把统一的 OpenAI 嵌套 tool 定义转换为 GLM 的扁平格式。

    OpenAI:  {"type": "function", "function": {"name", "description", "parameters"}}
    GLM:     {"type": "function", "name", "description", "parameters"}
    """
    result: List[Dict[str, Any]] = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function")
        if tool.get("type") == "function" and isinstance(fn, dict):
            result.append(
                {
                    "type": "function",
                    "name": fn.get("name"),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters")
                    or {"type": "object", "properties": {}},
                }
            )
        else:
            # 已是扁平格式或其他格式，原样透传
            result.append(tool)
    return result


def _message_to_dict(message: Any) -> Optional[Dict[str, Any]]:
    """把 rtclient 返回的消息（pydantic 对象或原始 dict）归一化为 dict。"""
    if message is None:
        return None
    if isinstance(message, dict):
        return message
    data: Any = None
    if hasattr(message, "model_dump"):  # pydantic v2
        data = message.model_dump()
    elif hasattr(message, "dict"):  # pydantic v1
        data = message.dict()
    if not isinstance(data, dict):
        data = {}
    if "type" not in data and hasattr(message, "type"):
        data["type"] = message.type
    return data


class GlmRealtimeProvider(RealtimeProvider):
    provider_name = "glm"

    def __init__(
        self,
        api_key: str,
        voice: str = "male-qn-daxuesheng",
        url: str = GLM_REALTIME_URL,
    ):
        self._api_key = api_key
        self._voice = voice
        self._url = url
        self._client: Optional[RTLowLevelClient] = None

    # ─────────────────── 生命周期 ───────────────────

    async def connect(self) -> None:
        if self._client is not None and not self._client.closed:
            return
        client = RTLowLevelClient(
            url=self._url,
            headers={"Authorization": f"Bearer {self._api_key}"},
        )
        # 原代码使用 `async with RTLowLevelClient(...) as client`，
        # 这里手动进入上下文以纳入 provider 生命周期管理
        await client.__aenter__()
        self._client = client

    async def close(self) -> None:
        client, self._client = self._client, None
        if client is None or client.closed:
            return
        try:
            await client.close()
        except Exception as e:
            print(f"[GLM] 关闭连接异常: {e}")

    @property
    def closed(self) -> bool:
        return self._client is None or self._client.closed

    # ─────────────────── 会话配置 ───────────────────

    async def update_session(
        self,
        instructions: str,
        tools: List[Dict[str, Any]],
        greeting_text: Optional[str] = None,
    ) -> None:
        if self._client is None or self._client.closed:
            raise ConnectionError("GLM realtime connection is not open")

        # beta_fields 与原实现一致；欢迎词通过 greeting_config 由服务端播放
        beta_fields = {"chat_mode": "audio", "tts_source": "e2e", "auto_search": False}
        if greeting_text:
            beta_fields["greeting_config"] = {"enable": True, "content": greeting_text}

        session_message = SessionUpdateMessage(
            session=SessionUpdateParams(
                instructions=instructions,
                input_audio_format="wav",
                output_audio_format="pcm",
                voice=self._voice,
                modalities={"audio", "text"},
                turn_detection=ServerVAD(),
                beta_fields=beta_fields,
                tools=_tools_to_glm_format(tools),
            )
        )
        await self._client.send(session_message)

    # ─────────────────── 上行 ───────────────────

    async def send_audio(self, pcm_chunk: bytes) -> None:
        if self._client is None or self._client.closed:
            raise ConnectionError("GLM realtime connection is not open")
        loop = asyncio.get_running_loop()
        message = InputAudioBufferAppendMessage(
            audio=_pcm_to_wav_base64(pcm_chunk),
            client_timestamp=int(loop.time() * 1000),
        )
        await self._client.send(message)

    async def send_function_call_output(
        self, call_id: Optional[str], output: str
    ) -> None:
        if self._client is None or self._client.closed:
            raise ConnectionError("GLM realtime connection is not open")
        # GLM 的函数调用结果回传不需要 call_id（保持原实现行为）
        output_item = FunctionCallOutputItem(output=output)
        await self._client.send(ItemCreateMessage(item=output_item))

    async def trigger_response(self) -> None:
        if self._client is None or self._client.closed:
            raise ConnectionError("GLM realtime connection is not open")
        await self._client.send_json({"type": "response.create"})

    # ─────────────────── 下行 ───────────────────

    async def recv(self) -> Optional[Dict[str, Any]]:
        if self._client is None:
            return None
        message = await self._client.recv()
        return _message_to_dict(message)
