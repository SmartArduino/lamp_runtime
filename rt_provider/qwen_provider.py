"""
Qwen-Omni-Realtime 后端：基于阿里云百炼 dashscope SDK
（dashscope.audio.qwen_omni.OmniRealtimeConversation）。

dashscope 的 OmniRealtimeConversation 是同步实现（内部通过 websocket-client
在自己的线程里收发消息、回调事件），本模块负责把它桥接到 asyncio 世界：

- 上行：所有阻塞调用（connect/update_session/append_audio/...）通过
  run_in_executor 串行执行，避免卡住事件循环；
- 下行：on_event 回调（SDK 线程）经 loop.call_soon_threadsafe 投递进
  asyncio.Queue，recv() 再异步取出，消息统一为 dict；
- on_close 回调置位 closed 标志并投递 "connection.closed" 哨兵事件，
  供上层接收循环退出。

注意：本模块仅在选择 Qwen 后端时才会被导入，需要安装 dashscope>=1.27.7。
"""

import asyncio
import base64
import uuid
from typing import Any, Callable, Dict, List, Optional

from dashscope.audio.qwen_omni import (
    AudioFormat,
    MultiModality,
    OmniRealtimeCallback,
    OmniRealtimeConversation,
)

from .base import RealtimeProvider

QWEN_REALTIME_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"

# 连接关闭哨兵事件类型（本地合成，非服务端下发）
EVENT_CONNECTION_CLOSED = "connection.closed"


class _QwenCallbackBridge(OmniRealtimeCallback):
    """把 dashscope 的线程回调桥接到 asyncio 事件循环。"""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        on_event: Callable[[Dict[str, Any]], None],
        on_close: Callable[[int, str], None],
    ):
        self._loop = loop
        self._on_event = on_event
        self._on_close = on_close

    def on_open(self) -> None:
        pass

    def on_close(self, close_status_code, close_msg) -> None:
        self._loop.call_soon_threadsafe(self._on_close, close_status_code, close_msg)

    def on_event(self, message) -> None:
        if isinstance(message, dict):
            self._loop.call_soon_threadsafe(self._on_event, message)


class QwenOmniRealtimeProvider(RealtimeProvider):
    provider_name = "qwen"

    def __init__(
        self,
        api_key: str,
        model: str = "qwen3.5-omni-plus-realtime",
        voice: str = "Tina",
        url: str = QWEN_REALTIME_URL,
    ):
        self._api_key = api_key
        self._model = model
        self._voice = voice
        self._url = url
        self._conversation: Optional[OmniRealtimeConversation] = None
        self._queue: Optional[asyncio.Queue] = None
        self._closed = True

    # ─────────────────── 生命周期 ───────────────────

    async def connect(self) -> None:
        if self._conversation is not None and not self._closed:
            return
        loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()
        bridge = _QwenCallbackBridge(
            loop,
            on_event=self._queue.put_nowait,
            on_close=self._handle_ws_closed,
        )
        conversation = OmniRealtimeConversation(
            model=self._model,
            callback=bridge,
            url=self._url,
            api_key=self._api_key,
        )
        self._conversation = conversation
        self._closed = False
        # connect() 同步阻塞（内部起线程并等待握手完成），放到线程池执行
        await loop.run_in_executor(None, conversation.connect)

    async def close(self) -> None:
        conversation, self._conversation = self._conversation, None
        self._closed = True
        if conversation is None:
            return
        try:
            await asyncio.get_running_loop().run_in_executor(None, conversation.close)
        except Exception as e:
            print(f"[Qwen] 关闭连接异常: {e}")

    @property
    def closed(self) -> bool:
        return self._closed

    def _handle_ws_closed(self, close_status_code, close_msg) -> None:
        """SDK 线程 on_close 回调（经 call_soon_threadsafe 在 loop 线程执行）。"""
        if self._closed:
            return
        self._closed = True
        print(f"[Qwen] 连接已关闭: code={close_status_code}, msg={close_msg}")
        if self._queue is not None:
            self._queue.put_nowait(
                {
                    "type": EVENT_CONNECTION_CLOSED,
                    "code": close_status_code,
                    "message": close_msg,
                }
            )

    def _ensure_open(self) -> OmniRealtimeConversation:
        if self._conversation is None or self._closed:
            raise ConnectionError("Qwen realtime connection is not open")
        return self._conversation

    # ─────────────────── 会话配置 ───────────────────

    async def update_session(
        self,
        instructions: str,
        tools: List[Dict[str, Any]],
        greeting_text: Optional[str] = None,
    ) -> None:
        conversation = self._ensure_open()
        loop = asyncio.get_running_loop()
        # instructions / tools 通过 SDK 的 **kwargs 平铺进 session 配置，
        # tools 直接使用 OpenAI 嵌套格式（官方示例用法）
        await loop.run_in_executor(
            None,
            lambda: conversation.update_session(
                output_modalities=[MultiModality.AUDIO, MultiModality.TEXT],
                voice=self._voice,
                input_audio_format=AudioFormat.PCM_16000HZ_MONO_16BIT,
                output_audio_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
                enable_input_audio_transcription=True,
                enable_turn_detection=True,
                turn_detection_type="server_vad",
                instructions=instructions,
                tools=list(tools or []),
            ),
        )
        # Qwen 没有 GLM 的服务端 greeting_config，改为主动触发一次
        # response 让模型朗读欢迎词
        if greeting_text:
            await self._create_response(
                instructions=(
                    "请直接朗读下面这句话向用户打招呼，"
                    "不要添加、不要修改、不要解释："
                    + greeting_text
                )
            )

    # ─────────────────── 上行 ───────────────────

    async def send_audio(self, pcm_chunk: bytes) -> None:
        conversation = self._ensure_open()
        # Qwen 输入为裸 PCM（16k/16bit/mono）直接 base64，无需 WAV 封装
        audio_b64 = base64.b64encode(pcm_chunk).decode("ascii")
        await asyncio.get_running_loop().run_in_executor(
            None, conversation.append_audio, audio_b64
        )

    async def send_function_call_output(
        self, call_id: Optional[str], output: str
    ) -> None:
        conversation = self._ensure_open()
        item = {
            "id": "item_" + uuid.uuid4().hex,
            "type": "function_call_output",
            "call_id": call_id,
            "output": output,
        }
        await asyncio.get_running_loop().run_in_executor(
            None, conversation.create_item, item
        )

    async def trigger_response(self) -> None:
        await self._create_response()

    async def _create_response(self, instructions: Optional[str] = None) -> None:
        conversation = self._ensure_open()
        modalities = [MultiModality.AUDIO, MultiModality.TEXT]
        await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: conversation.create_response(
                instructions=instructions,
                output_modalities=modalities,
            ),
        )

    # ─────────────────── 下行 ───────────────────

    async def recv(self) -> Optional[Dict[str, Any]]:
        if self._queue is None:
            return None
        return await self._queue.get()
