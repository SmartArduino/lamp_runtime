"""
实时语音对话 Provider 抽象层（接口定义）。

将上层业务（音频采集/播放、函数调用、灯光/电机控制）与具体的实时对话
SDK 解耦，当前有两种实现：

- glm_provider.GlmRealtimeProvider        智谱 GLM Realtime（rtclient SDK）
- qwen_provider.QwenOmniRealtimeProvider  阿里 Qwen-Omni-Realtime（dashscope SDK）

两家服务端事件均遵循 OpenAI Realtime 风格，抽象层统一约定：

- 输入音频：send_audio() 接收 16kHz / 16bit / mono 的原始 PCM 字节流，
  各实现自行完成 WAV 封装（GLM 需要）或直接 base64（Qwen 需要）；
- 输出音频：统一通过 "response.audio.delta" 事件下发，delta 为 base64
  编码的 PCM，采样率由各实现保证为 24000Hz（message_handler 按此播放）；
- 事件消息：recv() 一律返回 dict（含 "type" 字段），字段命名与
  OpenAI Realtime 事件对齐；
- tools 定义：统一使用 OpenAI 嵌套格式，需要扁平格式的后端（GLM）自行转换：
    {"type": "function",
     "function": {"name": ..., "description": ..., "parameters": {...}}}
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class RealtimeProvider(ABC):
    """实时对话后端的统一接口。每个会话创建一个实例。"""

    provider_name: str = "unknown"

    # ─────────────────── 生命周期 ───────────────────

    @abstractmethod
    async def connect(self) -> None:
        """建立连接。已连接时重复调用应直接返回。"""

    @abstractmethod
    async def close(self) -> None:
        """关闭连接。要求幂等：未连接/已关闭时调用不抛异常。"""

    @property
    @abstractmethod
    def closed(self) -> bool:
        """连接是否已关闭或不可用。"""

    # ─────────────────── 会话配置 ───────────────────

    @abstractmethod
    async def update_session(
        self,
        instructions: str,
        tools: List[Dict[str, Any]],
        greeting_text: Optional[str] = None,
    ) -> None:
        """下发 session.update：人设提示词、工具定义、欢迎词。

        Args:
            instructions: 系统提示词（人设/行为规则）。
            tools: OpenAI 嵌套格式的工具定义列表。
            greeting_text: 开场欢迎词。None 表示不启用；
                各后端用自己的机制实现（GLM: greeting_config；
                Qwen: 连接后主动触发一次 response）。
        """

    # ─────────────────── 上行 ───────────────────

    @abstractmethod
    async def send_audio(self, pcm_chunk: bytes) -> None:
        """发送一段 16kHz / 16bit / mono 的原始 PCM 音频。"""

    @abstractmethod
    async def send_function_call_output(
        self, call_id: Optional[str], output: str
    ) -> None:
        """回传函数调用结果。

        Args:
            call_id: 函数调用 ID。部分后端（Qwen）必须回传；
                不需要的后端（GLM）可忽略。
            output: 结果文本（通常是 JSON 字符串）。
        """

    @abstractmethod
    async def trigger_response(self) -> None:
        """主动触发一次 response.create（函数调用结果回传后使用）。"""

    # ─────────────────── 下行 ───────────────────

    @abstractmethod
    async def recv(self) -> Optional[Dict[str, Any]]:
        """取出一条服务端事件（dict）。

        由调用方通过 asyncio.wait_for 控制超时；连接已关闭且无缓存
        消息时返回 None。
        """
