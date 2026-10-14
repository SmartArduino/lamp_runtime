"""bubble.json 校准文件的统一读取与换算(仿真推理 / 真机部署共用)。

bubble.json 每个关节字段:
    id            舵机 ID(1..5), 对应 robot.xml 的 joint1..joint5,
                  也是策略观测/动作向量里的关节顺序
    drive_mode    方向位: bit0=1 表示该关节角度增大时舵机原始值减小(反向)
    homing_offset 中位偏移, 单位 0.01 度(home_raw = 2048 + offset*0.01*4096/360)
    range_min     舵机原始值运动下限(步)
    range_max     舵机原始值运动上限(步)
    (舵机 0..4095 步对应 360 度, 即 4096 步/圈)

换算约定(theta 为弧度, 与 MuJoCo qpos / 策略动作同坐标系):
    raw(theta) = home_raw + direction * theta * steps_per_rad
    theta(raw) = direction * (raw - home_raw) / steps_per_rad
    关节弧度限位 = [range_min, range_max] 按上式换算

用法:
    python calib.py            # 打印各关节换算表并自检
    cal = calib.load()         # 代码中加载

注意: load() 会做自洽性检查(零位必须落在限位范围内)。若换用其它校准工具
重新生成 bubble.json, 且 homing_offset 单位是"步"(lerobot 风格)而非 0.01 度,
用 calib.load(path, homing_unit="step") 切换; auto 模式会自动尝试两种单位。
"""

import json
import math
import sys

import numpy as np

STEPS_PER_REV = 4096.0   # 舵机一圈对应的原始步数(0..4095 = 360 度)
RAW_CENTER = 2048.0      # 舵机电气中位


class ArmCalib:
    def __init__(self, names, ids, home_raw, direction, range_min, range_max, homing_unit):
        self.names = list(names)
        self.ids = list(ids)
        self.home_raw = np.asarray(home_raw, dtype=float)
        self.direction = np.asarray(direction, dtype=float)
        self.range_min = np.asarray(range_min, dtype=float)
        self.range_max = np.asarray(range_max, dtype=float)
        self.homing_unit = homing_unit
        self.steps_per_rad = STEPS_PER_REV / (2.0 * math.pi)

        lo = self.direction * (self.range_min - self.home_raw) / self.steps_per_rad
        hi = self.direction * (self.range_max - self.home_raw) / self.steps_per_rad
        self.rad_limits = np.sort(np.stack([lo, hi], axis=1), axis=1)

    def raw_to_rad(self, raw):
        """舵机原始值 -> 弧度(策略坐标系, 真机读反馈用)"""
        raw = np.asarray(raw, dtype=float)
        return self.direction * (raw - self.home_raw) / self.steps_per_rad

    def rad_to_raw(self, rad, clip=True):
        """弧度(策略坐标系) -> 舵机原始值; clip=True 时硬限幅到 [range_min, range_max]"""
        rad = np.asarray(rad, dtype=float)
        raw = self.home_raw + self.direction * rad * self.steps_per_rad
        if clip:
            raw = np.clip(raw, self.range_min, self.range_max)
        return raw

    def clip_radians(self, rad):
        """把弧度动作截断到各关节软限位内"""
        rad = np.asarray(rad, dtype=float)
        return np.clip(rad, self.rad_limits[:, 0], self.rad_limits[:, 1])

    def beyond_radians(self, rad):
        """各关节超出软限位的弧度数(>=0), 用于告警/急停判断"""
        rad = np.asarray(rad, dtype=float)
        return np.maximum.reduce([rad - self.rad_limits[:, 1],
                                  self.rad_limits[:, 0] - rad,
                                  np.zeros_like(rad)])

    def problems(self):
        """自检: 校准数据必须自洽, 否则换算结果不可信, 拒绝用于真机"""
        probs = []
        for i, name in enumerate(self.names):
            if not 0 <= self.range_min[i] < self.range_max[i] <= STEPS_PER_REV - 1:
                probs.append(f"{name}: 原始值范围 [{self.range_min[i]:.0f}, {self.range_max[i]:.0f}] 非法(应为 0..{STEPS_PER_REV - 1:.0f} 内的递增区间)")
            if not self.range_min[i] <= self.home_raw[i] <= self.range_max[i]:
                probs.append(f"{name}: 零位原始值 {self.home_raw[i]:.0f} 不在 [{self.range_min[i]:.0f}, {self.range_max[i]:.0f}] 内, homing 单位假设({self.homing_unit})可能不对")
        return probs

    def table(self):
        head = (f"{'joint':<14}{'id':>4}{'dir':>5}{'home_raw':>10}"
                f"{'raw_min':>9}{'raw_max':>9}{'rad_min':>9}{'rad_max':>9}")
        rows = []
        for i, name in enumerate(self.names):
            rows.append(f"{name:<14}{self.ids[i]:>4}{self.direction[i]:>+5.0f}{self.home_raw[i]:>10.0f}"
                        f"{self.range_min[i]:>9.0f}{self.range_max[i]:>9.0f}"
                        f"{self.rad_limits[i, 0]:>9.4f}{self.rad_limits[i, 1]:>9.4f}")
        return "\n".join([head] + rows)


def _build(data, homing_unit):
    names = sorted(data, key=lambda k: data[k]["id"])
    ids = [data[k]["id"] for k in names]
    direction = [-1.0 if data[k].get("drive_mode", 0) & 1 else 1.0 for k in names]
    range_min = [float(data[k]["range_min"]) for k in names]
    range_max = [float(data[k]["range_max"]) for k in names]
    offsets = [float(data[k]["homing_offset"]) for k in names]

    if homing_unit == "centideg":
        home_raw = [RAW_CENTER + o * 0.01 * (STEPS_PER_REV / 360.0) for o in offsets]
    elif homing_unit == "step":
        home_raw = [RAW_CENTER + o for o in offsets]
    else:
        raise ValueError(f"未知 homing 单位: {homing_unit}")
    return ArmCalib(names, ids, home_raw, direction, range_min, range_max, homing_unit)


def load(path="bubble.json", homing_unit="auto"):
    """加载校准文件。homing_unit: auto(默认, 逐个尝试并选自洽的) / centideg / step"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    units = ("centideg", "step") if homing_unit == "auto" else (homing_unit,)
    fallback = None
    for unit in units:
        cal = _build(data, unit)
        if not cal.problems():
            return cal
        fallback = cal
    raise ValueError(
        f"{path} 与 homing 单位假设({'/'.join(units)})不自洽:\n  "
        + "\n  ".join(fallback.problems())
        + "\n请核对校准工具的导出格式, 修正 bubble.json 或指定 calib.load(..., homing_unit=...)"
    )


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "bubble.json"
    cal = load(path)
    n = len(cal.names)
    print(f"校准文件: {path}  (homing 单位: {cal.homing_unit}, {STEPS_PER_REV:.0f} 步 = 360 度)")
    print(cal.table())
    print("自检通过; 零位(弧度0)对应舵机原始值:",
          np.round(cal.rad_to_raw(np.zeros(n))).astype(int).tolist())
    print("换算示例: 各关节 +0.5rad -> 舵机原始值:",
          np.round(cal.rad_to_raw(np.full(n, 0.5))).astype(int).tolist())
