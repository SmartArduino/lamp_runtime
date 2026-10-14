#!/usr/bin/env python3
"""
修复 uv sync 报错：

  Requirements contain conflicting indexes for package `torch` in split
  `platform_machine == 'aarch64' and sys_platform == 'linux'`:
    - https://download.pytorch.org/whl/cpu
    - https://download.pytorch.org/whl/cu128

原因：
  本地修补版 lerobot 的 pyproject.toml 在 [tool.uv.sources] 中把
  torch / torchvision 指向 CUDA 源（pytorch-cuXXX），且 marker 覆盖了
  aarch64 Linux；而 lamp_runtime 自身声明 aarch64 Linux 使用 CPU 源
  （pytorch-cpu）。uv 解析 path 依赖时会应用其 [tool.uv.sources]，
  于是同一平台分叉上 torch 被要求来自两个索引，产生冲突。
  （此前 uv.lock 缓存有效从未重新解析，修改 lamp_runtime 的
  pyproject.toml 会触发重新解析，潜伏的冲突随之暴露）

做法：
  把 lerobot pyproject.toml 中 torch / torchvision 的 CUDA 源条目
  marker 追加排除 aarch64（and platform_machine != 'aarch64'），
  让出 aarch64 Linux 给 lamp_runtime 声明的 CPU 源；
  其他平台（x86_64 Linux 的 cu128 等）行为不变。

用法（在板子上 lamp_runtime 目录下执行）：
  python3 tools/fix_lerobot_torch_sources.py
  python3 tools/fix_lerobot_torch_sources.py --lerobot /path/to/lerobot

说明：
  - 幂等：已修复/已有平台限定的条目不会重复修改
  - 安全：修改前自动备份为 pyproject.toml.bak
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

DEFAULT_LEROBOT_PATHS = [
    "/home/radxa/source/python/lerobot",
    str(Path.home() / "source/python/lerobot"),
]

TARGET_KEYS = ("torch", "torchvision")

# 单条 source 条目，如 { index = "pytorch-cu128", marker = "..." }
ENTRY_RE = re.compile(r"\{[^{}]*\}")
MARKER_RE = re.compile(r'marker\s*=\s*"([^"]*)"')


def patch_entry(entry: str):
    """修补单条 { index = "pytorch-cuXXX", ... } 条目。

    返回 (新条目, 是否修改, 说明)。
    """
    if "pytorch-cu" not in entry:
        return entry, False, None

    m = MARKER_RE.search(entry)
    if m is None:
        # 无 marker：该 CUDA 源覆盖所有平台（含 aarch64 Linux），追加排除
        body = entry[:-1].rstrip().rstrip(",")  # 去掉收尾 "}" 与多余逗号
        new = body + ', marker = "platform_machine != \'aarch64\'"}'
        return new, True, "无 marker，追加排除 aarch64"

    marker = m.group(1)
    if "platform_machine" in marker:
        # 已有平台限定（本就不覆盖 aarch64，或语义需人工确认），不动
        return entry, False, f"已有平台限定，跳过（marker={marker}）"
    new_marker = marker + " and platform_machine != 'aarch64'"
    new_entry = entry[: m.start(1)] + new_marker + entry[m.end(1):]
    return new_entry, True, f"marker: {marker} -> {new_marker}"


def patch_line(line: str):
    """修补一行（可能含多条 entry）。返回 (新行, 修改说明列表)。"""
    notes = []

    def _sub(m):
        new, changed, note = patch_entry(m.group(0))
        if changed:
            notes.append(note)
        return new

    new_line = ENTRY_RE.sub(_sub, line)
    return new_line, notes


def patch_pyproject(path: Path) -> bool:
    """修补 pyproject.toml，返回是否发生修改。"""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    out = []
    in_target = False
    all_notes = []
    opened_re = [re.compile(rf"^{key}\s*=\s*\[") for key in TARGET_KEYS]

    for line in lines:
        stripped = line.strip()

        if not in_target:
            if any(r.match(line) for r in opened_re):
                in_target = True
                if stripped.endswith("]") and not stripped.endswith("["):
                    # 单行数组：处理完本行即闭合
                    new_line, notes = patch_line(line)
                    out.append(new_line)
                    all_notes.extend(notes)
                    in_target = False
                    continue
        else:
            if stripped.endswith("]"):
                in_target = False

        new_line, notes = patch_line(line)
        out.append(new_line)
        all_notes.extend(notes)

    if not all_notes:
        if "pytorch-cu" in text:
            print("[提示] torch/torchvision 的 CUDA 源条目已有平台限定或已修复，无需修改")
        else:
            print("[提示] 未发现 pytorch-cu 源配置，无需修改")
        return False

    backup = path.with_suffix(".toml.bak")
    shutil.copy2(path, backup)
    path.write_text("".join(out), encoding="utf-8")

    print(f"[修补] {path}（备份: {backup}）")
    for note in all_notes:
        print(f"  - {note}")
    return True


def find_lerobot_pyproject(cli_path: str = None) -> Path:
    candidates = []
    if cli_path:
        candidates.append(cli_path)
    for p in DEFAULT_LEROBOT_PATHS:
        candidates.append(p)
    for p in candidates:
        pyproject = Path(p) / "pyproject.toml"
        if pyproject.is_file():
            return pyproject
    print("未找到 lerobot 的 pyproject.toml，尝试过:")
    for p in candidates:
        print(f"  {p}")
    print("请用 --lerobot 指定 lerobot 目录后重试")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="修补 lerobot 的 torch/torchvision CUDA 源 marker，排除 aarch64")
    parser.add_argument("--lerobot", default=None, help="lerobot 源码目录（缺省尝试常见路径）")
    args = parser.parse_args()

    pyproject = find_lerobot_pyproject(args.lerobot)
    print(f"[目标] {pyproject}")

    try:
        changed = patch_pyproject(pyproject)
    except Exception as e:
        print(f"[错误] 修补失败: {e}")
        print("请按 README（4.4.1 节）手动修改 [tool.uv.sources] 中 torch/torchvision 的 marker")
        sys.exit(1)

    if changed:
        print("\n修补完成，请回到 lamp_runtime 目录执行:")
        print("  uv sync")
    else:
        print("\n若 uv sync 仍报冲突，请按 README（4.4.1 节）手动检查该文件")


if __name__ == "__main__":
    main()
