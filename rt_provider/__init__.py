"""
实时语音对话 Provider 工厂与配置。

通过环境变量（.env 支持 load_dotenv）选择后端，与项目既有的
LELAMP_AUDIO / LELAMP_MOTOR 开关风格一致：

- LELAMP_RT_PROVIDER: qwen（默认，Qwen-Omni-Realtime / dashscope SDK）
                      glm（GLM Realtime / rtclient SDK）
  别名：qwen-omni-realtime / dashscope 等价 qwen；zhipu / rtclient 等价 glm。

Qwen 后端（默认）配置：
- DASHSCOPE_API_KEY: 阿里云百炼 API Key（必填）
- QWEN_OMNI_MODEL:   模型名，默认 qwen3.5-omni-plus-realtime
- QWEN_OMNI_VOICE:   音色，默认 Tina

GLM 后端配置：
- ZHIPU_API_KEY: 智谱 API Key（必填）
- ZHIPU_VOICE:   音色，默认 male-qn-daxuesheng
"""

import os
from dataclasses import dataclass
from typing import Optional

from .base import RealtimeProvider

PROVIDER_QWEN = "qwen"
PROVIDER_GLM = "glm"

# ── 默认后端：Qwen-Omni-Realtime ──
DEFAULT_PROVIDER = PROVIDER_QWEN

QWEN_DEFAULT_MODEL = "qwen3.5-omni-plus-realtime"
QWEN_DEFAULT_VOICE = "Tina"
GLM_DEFAULT_VOICE = "male-qn-daxuesheng"

__all__ = [
    "RealtimeProvider",
    "RealtimeConfig",
    "PROVIDER_QWEN",
    "PROVIDER_GLM",
    "DEFAULT_PROVIDER",
    "resolve_provider_name",
    "clean_env_value",
    "load_realtime_config",
    "create_realtime_provider",
]


@dataclass
class RealtimeConfig:
    provider: str
    api_key: str
    voice: str
    model: str = ""  # 仅 Qwen 使用；GLM 的模型由服务端 URL 决定


def clean_env_value(var_name: str, value: str) -> str:
    """清理环境变量值：截掉行内注释（'#' 及之后内容）并去除首尾空白。

    .env 值后跟行内注释时（如 KEY=value  # 说明），bash 的 source 与
    systemd 的 EnvironmentFile 都不识别行内注释，会把注释文字并入变量值，
    导致 API Key 无效（401 Unauthorized）或模型名错误。API Key / 模型名 /
    音色均不含 '#'，统一截取是安全的。
    """
    cleaned = value.split("#", 1)[0].strip()
    if cleaned != value and value.strip():
        print(
            f"[实时对话] 警告: {var_name} 的值带有行内注释或多余空白"
            f"（{len(value)} 字符 -> {len(cleaned)} 字符），已自动清理；"
            f"建议修改 .env，将注释单独成行"
        )
    return cleaned


def resolve_provider_name() -> str:
    """解析要使用的后端：环境变量 LELAMP_RT_PROVIDER 优先，否则用默认值。"""
    env = clean_env_value(
        "LELAMP_RT_PROVIDER", os.environ.get("LELAMP_RT_PROVIDER", "")
    ).lower()
    if env in ("qwen", "qwen-omni-realtime", "qwen3-omni-realtime", "dashscope"):
        return PROVIDER_QWEN
    if env in ("glm", "zhipu", "bigmodel", "rtclient"):
        return PROVIDER_GLM
    if env:
        print(
            f"[实时对话] 未知的 LELAMP_RT_PROVIDER={env!r}，"
            f"回退到默认后端 {DEFAULT_PROVIDER}"
        )
    return DEFAULT_PROVIDER


def _required_env(var_name: str, provider_desc: str) -> str:
    value = clean_env_value(var_name, os.environ.get(var_name, ""))
    if not value:
        raise OSError(
            f"环境变量 '{var_name}' 未设置或为空（当前实时对话后端：{provider_desc}）"
        )
    return value


def _mask_key(key: str) -> str:
    """脱敏显示 API Key：仅保留前 3 位、后 4 位与总长度，用于排查 401 类问题。"""
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:3]}****{key[-4:]}（共 {len(key)} 字符）"


def load_realtime_config() -> RealtimeConfig:
    """读取环境变量并构造后端配置（缺少 API Key 时抛 OSError）。"""
    provider = resolve_provider_name()
    if provider == PROVIDER_QWEN:
        config = RealtimeConfig(
            provider=provider,
            api_key=_required_env("DASHSCOPE_API_KEY", "Qwen-Omni-Realtime"),
            voice=(clean_env_value("QWEN_OMNI_VOICE", os.environ.get("QWEN_OMNI_VOICE", ""))
                   or QWEN_DEFAULT_VOICE),
            model=(clean_env_value("QWEN_OMNI_MODEL", os.environ.get("QWEN_OMNI_MODEL", ""))
                   or QWEN_DEFAULT_MODEL),
        )
    else:
        config = RealtimeConfig(
            provider=provider,
            api_key=_required_env("ZHIPU_API_KEY", "GLM Realtime"),
            voice=(clean_env_value("ZHIPU_VOICE", os.environ.get("ZHIPU_VOICE", ""))
                   or GLM_DEFAULT_VOICE),
        )
    # 打印 Key 指纹：systemctl 与手动运行时指纹应一致，否则说明环境变量来源有问题
    print(f"[实时对话] {config.provider} API Key: {_mask_key(config.api_key)}")
    return config


def create_realtime_provider(config: RealtimeConfig) -> RealtimeProvider:
    """按配置创建 provider 实例（每个会话应创建一个新实例）。

    各 SDK 在此按需延迟导入：只用 GLM 时不要求安装 dashscope，
    只用 Qwen 时不要求安装 rtclient。
    """
    if config.provider == PROVIDER_QWEN:
        from .qwen_provider import QwenOmniRealtimeProvider

        return QwenOmniRealtimeProvider(
            api_key=config.api_key,
            model=config.model,
            voice=config.voice,
        )
    from .glm_provider import GlmRealtimeProvider

    return GlmRealtimeProvider(api_key=config.api_key, voice=config.voice)
