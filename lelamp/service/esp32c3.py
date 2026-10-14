#!/usr/bin/env python3
""" 
ESPC3 LED 控制协议 - Python 实现 
================================ 
帧格式: 
+--------+--------+--------+--------+--------+--------+--------+--------+ 
| 0xAA   | 0x55   | CMD    | LEN    | DATA   | ...    | CRC8   | 0x0D   | 
+--------+--------+--------+--------+--------+--------+--------+--------+ 
| 帧头1  | 帧头2  | 命令   | 长度   | 数据   |        | 校验   | 帧尾   | 
CRC8: 多项式 0x07，初始值 0x00，从 CMD 到 DATA 最后一个字节 
""" 
from typing import Optional, List, Tuple, Callable 

# ===================================================================== 
# 常量定义 
# ===================================================================== 
FRAME_HEAD1 = 0xAA 
FRAME_HEAD2 = 0x55 
FRAME_TAIL = 0x0D 

# 命令码 
CMD_SET_SINGLE = 0x01 
CMD_SET_ALL = 0x02 
CMD_SET_MULTI = 0x03 
CMD_EFFECT_BREATH = 0x04 
CMD_EFFECT_RAINBOW = 0x05 
CMD_EFFECT_RUNNING = 0x06 
CMD_EFFECT_CHASE = 0x07 
CMD_SET_BRIGHTNESS = 0x08 
CMD_GET_STATUS = 0x09 
CMD_STOP_EFFECT = 0x0A 
CMD_CLEAR = 0x2B 

# 新增：外接 LED 控制命令 (避开被用作帧尾的 0x0D)
CMD_EXT_LED_ON = 0x0C    #示例： AA 55 0C 00 FC 0D
CMD_EXT_LED_OFF = 0x0E   #示例： AA 55 0E 00 D6 0D

# 频闪特效命令码
CMD_EFFECT_BLINK = 0x10  #示例： aa 55 10 01 c8 c1 0d

CMD_ACK = 0x0F 

# ACK 状态码 
ACK_OK = 0x00 
ACK_CRC_ERROR = 0x01 
ACK_PARAM_ERROR = 0x02 
ACK_CMD_UNKNOWN = 0x03 

# ===================================================================== 
# CRC8 算法 
# ===================================================================== 
def crc8(data: bytes) -> int: 
    """ 
    CRC8 校验算法。 
    多项式: 0x07, 初始值: 0x00, 无输入/输出反转, 无异或输出。 
    """ 
    crc = 0x00 
    for byte in data: 
        crc ^= byte 
        for _ in range(8): 
            if crc & 0x80: 
                crc = ((crc << 1) ^ 0x07) & 0xFF 
            else: 
                crc = (crc << 1) & 0xFF 
    return crc 

# ===================================================================== 
# 帧构建 
# ===================================================================== 
def build_frame(cmd: int, data: bytes = b'') -> bytes: 
    """ 
    构建完整帧: AA 55 CMD LEN DATA[...] CRC8 0D 
    CRC8 计算范围: CMD + LEN + DATA 
    """ 
    length = len(data) 
    crc_payload = bytes([cmd, length]) + data 
    crc_val = crc8(crc_payload) 
    frame = bytes([FRAME_HEAD1, FRAME_HEAD2]) + crc_payload + bytes([crc_val, FRAME_TAIL]) 
    return frame 

# ===================================================================== 
# 帧解析 
# ===================================================================== 
def parse_frame(raw: bytes) -> Optional[dict]: 
    """ 
    解析一帧数据。 
    返回: {cmd, length, data, crc_recv, crc_calc, crc_ok, raw} 
    帧不完整或帧头/帧尾错误返回 None。 
    """ 
    if len(raw) < 6: # 最小帧: AA 55 CMD LEN CRC 0D 
        return None 
    if raw[0] != FRAME_HEAD1 or raw[1] != FRAME_HEAD2: 
        return None 
    
    cmd = raw[2] 
    length = raw[3] 
    data_end = 4 + length 
    if len(raw) < data_end + 2: # 还需 CRC + TAIL 
        return None 
        
    data = raw[4:data_end] 
    crc_recv = raw[data_end] 
    tail = raw[data_end + 1] 
    
    if tail != FRAME_TAIL: 
        return None 
        
    crc_calc = crc8(raw[2:data_end]) # CMD ~ DATA 
    return { 
        'cmd': cmd, 
        'length': length, 
        'data': data, 
        'crc_recv': crc_recv, 
        'crc_calc': crc_calc, 
        'crc_ok': crc_recv == crc_calc, 
        'raw': raw[:data_end + 2], 
    } 

# ===================================================================== 
# 流式接收缓冲区 
# ===================================================================== 
class FrameReceiver: 
    """ 
    流式接收缓冲区，从连续字节流中提取完整帧。 
    支持数据分多次到达、前导垃圾数据跳过、连续多帧解析。 
    典型用法:: 
        receiver = FrameReceiver() 
        receiver.feed(serial.read(100)) 
        while True: 
            frame = receiver.get_frame() 
            if frame is None: 
                break 
            print(frame) 
    """ 
    def __init__(self): 
        self._buf = bytearray() 

    def feed(self, data: bytes): 
        """将接收到的原始数据喂入缓冲区。""" 
        self._buf.extend(data) 

    def get_frame(self) -> Optional[dict]: 
        """尝试从缓冲区提取一帧，成功返回解析结果并移除该帧。""" 
        while len(self._buf) >= 6: 
            # 寻找帧头 AA 55 
            idx = -1 
            for i in range(len(self._buf) - 1): 
                if self._buf[i] == FRAME_HEAD1 and self._buf[i + 1] == FRAME_HEAD2: 
                    idx = i 
                    break 
            if idx == -1: 
                if len(self._buf) > 1: 
                    self._buf = self._buf[-1:] # 保留最后1字节防止帧头跨包 
                return None 
            if idx > 0: 
                del self._buf[:idx] # 丢弃帧头前无效数据 
            if len(self._buf) < 6: 
                return None 
            length = self._buf[3] 
            frame_len = 4 + length + 2 # HEAD(2)+CMD(1)+LEN(1)+DATA(n)+CRC(1)+TAIL(1) 
            if len(self._buf) < frame_len: 
                return None # 等待更多数据 
            frame_bytes = bytes(self._buf[:frame_len]) 
            result = parse_frame(frame_bytes) 
            if result is None: 
                del self._buf[:2] # 帧尾不匹配，跳过帧头继续搜索 
                continue 
            del self._buf[:frame_len] 
            return result 
        return None 

    def clear(self): 
        """清空接收缓冲区。""" 
        self._buf.clear() 

# ===================================================================== 
# 响应解析辅助函数 
# ===================================================================== 
def parse_ack(data: bytes) -> Optional[dict]: 
    """ 
    解析 ACK 应答帧 DATA 字段。 
    DATA: [status, cmd] 
    """ 
    if len(data) < 2: 
        return None 
    return {'status': data[0], 'cmd': data[1]} 

def parse_status(data: bytes) -> Optional[dict]: 
    """ 
    解析 GET_STATUS 响应帧 DATA 字段。 
    DATA: [led_num, current_effect, brightness, effect_running, reserved, reserved] 
    """ 
    if len(data) < 6: 
        return None 
    return { 
        'led_num': data[0], 
        'current_effect': data[1], 
        'brightness': data[2], 
        'effect_running': bool(data[3]), 
        'reserved': (data[4], data[5]), 
    } 

# ===================================================================== 
# 上层协议封装 
# ===================================================================== 
class ESPC3Protocol: 
    """ 
    ESPC3 LED 协议收发器，封装所有命令的构建和发送。 
    用法示例:: 
        import serial 
        ser = serial.Serial('/dev/ttyUSB0', 115200) 
        proto = ESPC3Protocol(send_func=ser.write) 
        proto.set_all(255, 0, 0) # 全部红色 
        proto.set_single(3, 0, 255, 0) # 第3个LED绿色 
        proto.effect_rainbow(150, 200) # 彩虹效果 
        proto.clear() # 关闭所有LED 
    """ 
    def __init__(self, send_func: Callable[[bytes], int] = None): 
        self._send = send_func 

    def _send_frame(self, cmd: int, data: bytes = b'') -> bytes: 
        """构建并发送一帧，返回帧字节。""" 
        frame = build_frame(cmd, data) 
        if self._send: 
            self._send(frame) 
        return frame 

    # ---------- LED 设置命令 ---------- 
    def set_single(self, index: int, r: int, g: int, b: int) -> bytes: 
        """设置单个 LED 颜色 (CMD=0x01)。DATA: [index, r, g, b]""" 
        return self._send_frame(CMD_SET_SINGLE, bytes([index & 0xFF, r & 0xFF, g & 0xFF, b & 0xFF])) 

    def set_all(self, r: int, g: int, b: int) -> bytes: 
        """设置全部 LED 为同一颜色 (CMD=0x02)。DATA: [r, g, b]""" 
        return self._send_frame(CMD_SET_ALL, bytes([r & 0xFF, g & 0xFF, b & 0xFF])) 

    def set_multi(self, leds: List[Tuple[int, int, int, int]]) -> bytes: 
        """ 
        设置多个 LED 颜色 (CMD=0x03)。 
        DATA: [n, idx1, r1, g1, b1, idx2, r2, g2, b2, ...] 
        :param leds: [(index, r, g, b), ...] 
        """ 
        data = bytearray([len(leds) & 0xFF]) 
        for idx, r, g, b in leds: 
            data.extend([idx & 0xFF, r & 0xFF, g & 0xFF, b & 0xFF]) 
        return self._send_frame(CMD_SET_MULTI, bytes(data)) 

    # ---------- 效果命令 ---------- 
    def effect_breath(self, r: int, g: int, b: int, speed: int) -> bytes: 
        """呼吸灯效果 (CMD=0x04)。DATA: [r, g, b, speed]""" 
        return self._send_frame(CMD_EFFECT_BREATH, bytes([r & 0xFF, g & 0xFF, b & 0xFF, speed & 0xFF])) 

    def effect_rainbow(self, speed: int, brightness: int) -> bytes: 
        """彩虹效果 (CMD=0x05)。DATA: [speed, brightness]""" 
        return self._send_frame(CMD_EFFECT_RAINBOW, bytes([speed & 0xFF, brightness & 0xFF])) 

    def effect_running(self, r: int, g: int, b: int, speed: int, direction: int) -> bytes: 
        """流水灯效果 (CMD=0x06)。DATA: [r, g, b, speed, direction]""" 
        return self._send_frame(CMD_EFFECT_RUNNING, bytes([r & 0xFF, g & 0xFF, b & 0xFF, speed & 0xFF, direction & 0xFF])) 

    def effect_chase(self, r: int, g: int, b: int, speed: int, length: int) -> bytes: 
        """追逐效果 (CMD=0x07)。DATA: [r, g, b, speed, length]""" 
        return self._send_frame(CMD_EFFECT_CHASE, bytes([r & 0xFF, g & 0xFF, b & 0xFF, speed & 0xFF, length & 0xFF])) 

    def effect_blink(self, speed: int) -> bytes: 
        """ 
        频闪效果 (CMD=0x10)。DATA: [speed] 
        :param speed: 闪烁速度 (0-255，值越大闪烁越快)
        """ 
        return self._send_frame(CMD_EFFECT_BLINK, bytes([speed & 0xFF]))

    def stop_effect(self) -> bytes: 
        """停止当前效果 (CMD=0x0A)。""" 
        return self._send_frame(CMD_STOP_EFFECT) 

    def clear(self) -> bytes: 
        """关闭所有 LED (CMD=0x0B)。""" 
        return self._send_frame(CMD_CLEAR) 

    # ---------- 状态命令 ---------- 
    def set_brightness(self, brightness: int) -> bytes: 
        """设置全局亮度 (CMD=0x08)。DATA: [brightness]""" 
        return self._send_frame(CMD_SET_BRIGHTNESS, bytes([brightness & 0xFF])) 

    def get_status(self) -> bytes: 
        """获取设备状态 (CMD=0x09)。返回帧需用 parse_status() 解析。""" 
        return self._send_frame(CMD_GET_STATUS) 

    # ---------- 外接 LED 控制 ---------- 
    def ext_led_on(self) -> bytes: 
        """打开外接 LED (CMD=0x0C)。无 DATA。""" 
        return self._send_frame(CMD_EXT_LED_ON) 

    def ext_led_off(self) -> bytes: 
        """关闭外接 LED (CMD=0x0E)。无 DATA。""" 
        return self._send_frame(CMD_EXT_LED_OFF) 

# ===================================================================== 
# 测试 
# ===================================================================== 
def run_tests(): 
    """运行协议自测。""" 
    print("=" * 70) 
    print("ESPC3 协议实现测试") 
    print("=" * 70) 
    passed = 0 
    failed = 0 

    def check(name, ok, expected="", actual=""): 
        nonlocal passed, failed 
        if ok: 
            passed += 1 
            print(f"  ✅ PASS | {name}") 
        else: 
            failed += 1 
            print(f"  ❌ FAIL | {name}") 
            print(f"     期望: {expected}") 
            print(f"     实际: {actual}") 

    def hexstr(b): 
        return ' '.join(f'{x:02X}' for x in b) 

    print("\n--- 1. 帧构建测试 ---") 
    # 测试1: SET_ALL 红色 
    frame = build_frame(CMD_SET_ALL, bytes([0xFF, 0x00, 0x00])) 
    check("SET_ALL 帧头", frame[:2] == b'\xAA\x55', "AA 55", hexstr(frame[:2])) 
    check("SET_ALL CMD", frame[2] == 0x02, "02", f"{frame[2]:02X}") 
    check("SET_ALL LEN", frame[3] == 0x03, "03", f"{frame[3]:02X}") 
    check("SET_ALL DATA", frame[4:7] == bytes([0xFF, 0x00, 0x00])) 
    check("SET_ALL 帧尾", frame[-1:] == b'\x0D', "0D", f"{frame[-1]:02X}") 
    print(f"    完整帧: {hexstr(frame)}") 

    # 测试2: SET_SINGLE 第3个LED绿色 
    frame = build_frame(CMD_SET_SINGLE, bytes([0x03, 0x00, 0xFF, 0x00])) 
    check("SET_SINGLE CMD", frame[2] == 0x01) 
    check("SET_SINGLE LEN", frame[3] == 0x04) 
    check("SET_SINGLE DATA", frame[4:8] == bytes([0x03, 0x00, 0xFF, 0x00])) 
    print(f"    完整帧: {hexstr(frame)}") 

    # 测试3: CLEAR 
    frame = build_frame(CMD_CLEAR, b'') 
    check("CLEAR CMD", frame[2] == 0x0B) 
    check("CLEAR LEN", frame[3] == 0x00) 
    check("CLEAR 帧长度", len(frame) == 6, "6", str(len(frame))) 
    print(f"    完整帧: {hexstr(frame)}") 

    # 测试4: GET_STATUS 
    frame = build_frame(CMD_GET_STATUS, b'') 
    check("GET_STATUS CMD", frame[2] == 0x09) 
    check("GET_STATUS LEN", frame[3] == 0x00) 
    print(f"    完整帧: {hexstr(frame)}") 

    # 测试5: SET_BRIGHTNESS 
    frame = build_frame(CMD_SET_BRIGHTNESS, bytes([0x80])) 
    check("SET_BRIGHTNESS CMD", frame[2] == 0x08) 
    check("SET_BRIGHTNESS DATA", frame[4:5] == bytes([0x80])) 
    print(f"    完整帧: {hexstr(frame)}") 

    # 测试6: EFFECT_RAINBOW 
    frame = build_frame(CMD_EFFECT_RAINBOW, bytes([0x96, 0xC8])) 
    check("EFFECT_RAINBOW CMD", frame[2] == 0x05) 
    check("EFFECT_RAINBOW DATA", frame[4:6] == bytes([0x96, 0xC8])) 
    print(f"    完整帧: {hexstr(frame)}") 

    print("\n--- 2. 帧解析测试 ---") 
    # 解析 SET_ALL 
    frame = build_frame(CMD_SET_ALL, bytes([0xFF, 0x00, 0x00])) 
    result = parse_frame(frame) 
    check("解析 SET_ALL cmd", result['cmd'] == 0x02) 
    check("解析 SET_ALL data", result['data'] == bytes([0xFF, 0x00, 0x00])) 
    check("解析 SET_ALL crc_ok", result['crc_ok'] == True) 

    # 解析 GET_STATUS 响应 
    resp = build_frame(CMD_GET_STATUS, bytes([0x08, 0x00, 0xFF, 0x01, 0x00, 0x00])) 
    result = parse_frame(resp) 
    check("解析 STATUS cmd", result['cmd'] == 0x09) 
    status = parse_status(result['data']) 
    check("解析 STATUS led_num", status['led_num'] == 8) 
    check("解析 STATUS brightness", status['brightness'] == 255) 
    check("解析 STATUS effect_running", status['effect_running'] == True) 

    print("\n--- 3. CRC 错误检测测试 ---") 
    bad_frame = bytearray(build_frame(CMD_SET_ALL, bytes([0xFF, 0x00, 0x00]))) 
    bad_frame[-2] = 0x00 # 篡改 CRC 
    result = parse_frame(bytes(bad_frame)) 
    check("CRC错误检测", result['crc_ok'] == False) 

    print("\n--- 4. 流式接收测试 ---") 
    receiver = FrameReceiver() 
    # 不完整帧 
    receiver.feed(bytes([0xAA, 0x55, 0x02, 0x03])) 
    check("不完整帧返回None", receiver.get_frame() is None) 
    # 补全后解析 
    frame1 = build_frame(CMD_SET_ALL, bytes([0xFF, 0x00, 0x00])) 
    receiver.feed(frame1[4:]) 
    f = receiver.get_frame() 
    check("补全后解析成功", f is not None and f['cmd'] == 0x02) 

    # 前导垃圾数据 
    receiver.clear() 
    frame2 = build_frame(CMD_CLEAR, b'') 
    receiver.feed(bytes([0x00, 0x11, 0x22]) + frame2) 
    f = receiver.get_frame() 
    check("跳过垃圾数据", f is not None and f['cmd'] == 0x0B) 

    # 连续多帧 
    receiver.clear() 
    fa = build_frame(CMD_STOP_EFFECT, b'') 
    fb = build_frame(CMD_CLEAR, b'') 
    fc = build_frame(CMD_SET_ALL, bytes([0x00, 0xFF, 0x00])) 
    receiver.feed(fa + fb + fc) 
    f1 = receiver.get_frame() 
    f2 = receiver.get_frame() 
    f3 = receiver.get_frame() 
    f4 = receiver.get_frame() 
    check("连续帧1", f1 is not None and f1['cmd'] == 0x0A) 
    check("连续帧2", f2 is not None and f2['cmd'] == 0x0B) 
    check("连续帧3", f3 is not None and f3['cmd'] == 0x02) 
    check("连续帧4为None", f4 is None) 

    # 逐字节接收 
    receiver.clear() 
    fd = build_frame(CMD_SET_SINGLE, bytes([0x03, 0x00, 0xFF, 0x00])) 
    got = None 
    for i in range(len(fd)): 
        receiver.feed(bytes([fd[i]])) 
        got = receiver.get_frame() 
    check("逐字节接收", got is not None and got['cmd'] == 0x01) 

    print("\n--- 5. ACK 响应解析测试 ---") 
    ack_frame = build_frame(CMD_ACK, bytes([0x00, 0x02])) 
    result = parse_frame(ack_frame) 
    ack = parse_ack(result['data']) 
    check("ACK status", ack['status'] == 0x00) 
    check("ACK cmd", ack['cmd'] == 0x02) 

    ack_err = build_frame(CMD_ACK, bytes([0x01, 0x02])) 
    result = parse_frame(ack_err) 
    ack = parse_ack(result['data']) 
    check("ACK_ERR status", ack['status'] == 0x01) 

    print("\n--- 6. 上层 API 测试 ---") 
    sent_frames = [] 
    proto = ESPC3Protocol(send_func=lambda d: sent_frames.append(d)) 
    proto.set_all(255, 0, 0) 
    proto.set_single(3, 0, 255, 0) 
    proto.set_multi([(0, 255, 0, 0), (1, 0, 255, 0), (2, 0, 0, 255)]) 
    proto.effect_breath(255, 0, 0, 100) 
    proto.effect_rainbow(150, 200) 
    proto.effect_running(0, 255, 0, 80, 1) 
    proto.effect_chase(255, 255, 0, 60, 3) 
    proto.effect_blink(200) 
    proto.ext_led_on()     # 新增：测试外接 LED 开
    proto.ext_led_off()    # 新增：测试外接 LED 关
    proto.set_brightness(128) 
    proto.get_status() 
    proto.stop_effect() 
    proto.clear() 

    print(f"    共发送 {len(sent_frames)} 帧:") 
    all_ok = True 
    for i, f in enumerate(sent_frames): 
        result = parse_frame(f) 
        ok = result is not None and result['crc_ok'] 
        if not ok: 
            all_ok = False 
        print(f"    帧{i+1} [{'✅' if ok else '❌'}]: {hexstr(f)}") 
    check("API 所有帧 CRC 校验", all_ok) 

    # 验证 SET_MULTI 数据 
    multi_result = parse_frame(sent_frames[2]) 
    check("SET_MULTI CMD", multi_result['cmd'] == 0x03) 
    check("SET_MULTI LED数", multi_result['data'][0] == 3) 

    # 验证 EXT_LED_ON 数据 
    on_result = parse_frame(sent_frames[8])   # 第9帧 (索引8) 
    check("EXT_LED_ON CMD", on_result['cmd'] == CMD_EXT_LED_ON) 
    check("EXT_LED_ON LEN", on_result['length'] == 0) 

    # 验证 EXT_LED_OFF 数据 
    off_result = parse_frame(sent_frames[9])  # 第10帧 (索引9) 
    check("EXT_LED_OFF CMD", off_result['cmd'] == CMD_EXT_LED_OFF) 
    check("EXT_LED_OFF LEN", off_result['length'] == 0) 

    # 结果汇总 
    print(f"\n{'=' * 70}") 
    print(f"测试结果: ✅ {passed} 通过 | ❌ {failed} 失败") 
    if failed == 0: 
        print("🎉 全部测试通过!") 
    else: 
        print("⚠️ 存在失败项!") 
    print(f"{'=' * 70}") 

if __name__ == '__main__': 
    run_tests()
