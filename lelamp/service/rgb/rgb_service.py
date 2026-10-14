# service/rgb/rgb_service.py
"""
基于 ESPC3 二进制协议的 RGB 控制服务。
适配 lelamp_runtime 的 ServiceBase 接口，支持通过串口发送灯光指令。

用法:
    rgb = RGBService()
    rgb.start()
    
    # 颜色设置
    rgb.dispatch("solid", (r, g, b), Priority.HIGH)
    rgb.dispatch("paint", [(r,g,b), ...])
    rgb.dispatch("single", (index, r, g, b))
    
    # 特效事件 (来自 rgb_c3.py)
    rgb.dispatch("breath", (r, g, b, speed))
    rgb.dispatch("rainbow", (speed, brightness))
    rgb.dispatch("running", (r, g, b, speed, direction))
    rgb.dispatch("chase", (r, g, b, speed, length))
    
    rgb.dispatch("clear", None)
    rgb.stop()
"""

import time
import threading
import serial
from typing import Any, Optional, List, Tuple, Callable

from ..base import ServiceBase, Priority
from ..audio_playback import play_wav_file, resolve_asset_path
from ..esp32c3 import (
    ESPC3Protocol, FrameReceiver, parse_ack, parse_status,
    CMD_ACK, CMD_GET_STATUS,CMD_EXT_LED_ON,CMD_EXT_LED_OFF
)


class RGBService(ServiceBase):
    def __init__(self, name: str = "rgb", port: str = "/dev/ttyS4", baudrate: int = 115200):
        super().__init__(name)
        self._port = port
        self._baudrate = baudrate
        self._serial: Optional[serial.Serial] = None
        self._proto: Optional[ESPC3Protocol] = None
        self._receiver = FrameReceiver()
        self._rx_thread: Optional[threading.Thread] = None
        # 新增：LED 闪烁线程
        self._blink_thread: Optional[threading.Thread] = None
        self._blink_enabled = False
        
        # 新增：短按事件回调 
        self.on_short_press: Optional[Callable[[int], None]] = None

        # 新增：人体感应事件回调（GPIO9/level_state 上报）
        # state: 0=未检测到人体或人体静止不动；1=检测到人体或人体移动
        self.on_human_sensor: Optional[Callable[[int], None]] = None
        # 最新一次收到的人体感应状态（默认 1：未收到上报前保持"有人/移动"，
        # 让人脸追踪维持原有行为，不会被传感器缺数据误冻结）
        self._human_sensor_state: int = 1

    def start(self):
        if self.is_running:
            self.logger.warning(f"Service {self.name} is already running")
            return
        
        try:
            self._serial = serial.Serial(self._port, self._baudrate, timeout=0.01)
            self._proto = ESPC3Protocol(send_func=self._serial.write)
        except Exception as e:
            self.logger.error(f"Failed to open serial port {self._port}: {e}")
            return

        super().start()  # 启动事件循环线程
        
        # 启动串口接收线程
        self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._rx_thread.start()
        
        # 启动 LED 闪烁线程
        self.start_ext_led_blink()
        
        self.logger.info(f"RGBService started on {self._port}")

    def stop(self, timeout: float = 5.0):
        if not self.is_running:
            return
            
        # 先停止闪烁线程
        self.stop_ext_led_blink()
        
        # 发送关闭灯光命令
        if self._proto:
            self._proto.clear()
            time.sleep(0.05)
            
        super().stop(timeout)  # 停止事件循环线程
        
        if self._rx_thread and self._rx_thread.is_alive():
            self._rx_thread.join(timeout=1.0)
            
        if self._serial:
            self._serial.close()
            self._serial = None
        self.logger.info(f"RGBService stopped")
    
    # ==================== 新增：外接 LED 闪烁控制 ====================

    def start_ext_led_blink(self, interval: float = 1.0):
        """
        启动外接 LED 闪烁任务。
        每隔 interval 秒切换一次 LED 的亮灭状态（亮0.5秒，灭0.5秒，周期1秒）。
        """
        if self._blink_thread and self._blink_thread.is_alive():
            self.logger.warning("Ext LED blink is already running")
            return

        self._blink_enabled = True
        self._blink_thread = threading.Thread(
            target=self._blink_loop, args=(interval,), daemon=True
        )
        self._blink_thread.start()
        self.logger.info(f"Ext LED blink started (interval={interval}s)")

    def stop_ext_led_blink(self):
        """停止外接 LED 闪烁任务，并关闭 LED。"""
        self._blink_enabled = False
        if self._blink_thread and self._blink_thread.is_alive():
            self._blink_thread.join(timeout=1.0)
        # 停止后确保 LED 关闭
        if self._proto:
            self._proto.ext_led_off()
        self.logger.info("Ext LED blink stopped")

    def _blink_loop(self, interval: float):
        """
        LED 闪烁线程的主循环。
        每个周期内亮 interval/2 秒，灭 interval/2 秒。
        """
        half = interval / 2.0
        while self.is_running and self._blink_enabled:
            try:
                if self._proto:
                    self._proto.ext_led_on()
                    time.sleep(half)
                    self._proto.ext_led_off()
                    time.sleep(half)
                    #print("LED blinking...")
            except Exception as e:
                self.logger.error(f"Error in blink loop: {e}")
                time.sleep(0.5)

    # ================================================================

    # ---- 兼容性方法 ----
    # 旧代码可能直接调用这些方法，而不是走 dispatch
    # 这里做一层桥接，将同步调用转为异步事件投递

    @property
    def human_sensor_state(self) -> int:
        """最新的人体感应状态（0=无人或静止，1=有人或移动）"""
        return self._human_sensor_state

    def clear(self):
        """关闭所有 LED（通过事件队列异步执行）"""
        self.dispatch("clear", None, Priority.HIGH)

    def solid(self, r: int, g: int, b: int, priority: Priority = Priority.NORMAL):
        """设置全部 LED 为同一颜色（兼容旧接口）"""
        self.dispatch("solid", (r, g, b), priority)

    def paint(self, colors: list, priority: Priority = Priority.NORMAL):
        """设置多个 LED 颜色（兼容旧接口）"""
        self.dispatch("paint", colors, priority)

    # ---- 内部实现 ----

    def _rx_loop(self):
        """串口接收线程，负责读取ACK和状态响应"""
        while self.is_running and self._serial:
            try:
                if self._serial.in_waiting > 0:
                    data = self._serial.read(self._serial.in_waiting)
                    #print(f"[RX Raw data]: {data.hex()}", flush=True)
                    self._receiver.feed(data)
                    
                    while True:
                        frame = self._receiver.get_frame()
                        if frame is None:
                            break
                        if not frame['crc_ok']:
                            self.logger.warning(f"CRC error in received frame: {frame['raw'].hex()}")
                            continue
                            
                        if frame['cmd'] == CMD_ACK:
                            ack = parse_ack(frame['data'])
                            if ack:
                                if ack['status'] == 0x00:
                                    self.logger.debug(f"ACK OK for cmd {ack['cmd']:#04X}")
                                elif ack['status'] == 0xff:
                                    print("短按按钮")
                                    # 新增：触发回调，将 ack['cmd'] 的值传出去 
                                    if self.on_short_press:
                                        try:
                                            self.on_short_press(ack['cmd'])
                                        except Exception as e:
                                            self.logger.error(f"短按回调执行出错: {e}")
                                elif ack['status'] == 0xfe:
                                    print("长按按钮，关机")
                                    play_wav_file(resolve_asset_path("poweroff.wav"))
                                elif ack['status'] == 0xfd:
                                    # 新增：GPIO9电平变化上报，ack['cmd'] 对应 current_level_state
                                    # level_state: 0=未检测到人体或人体静止不动，1=检测到人体或人体移动
                                    level_state = ack['cmd']
                                    print(f"GPIO9电平状态: {level_state}")
                                    self._human_sensor_state = level_state
                                    if level_state == 0:
                                        #play_wav_file(resolve_asset_path("no_human.wav"))
                                        print('Human Sensor detect nothing')
                                    # 触发回调，把人体感应状态传给人脸追踪等服务
                                    if self.on_human_sensor:
                                        try:
                                            self.on_human_sensor(level_state)
                                        except Exception as e:
                                            self.logger.error(f"人体感应回调执行出错: {e}")
                                else:
                                    self.logger.warning(f"ACK Error {ack['status']} for cmd {ack['cmd']:#04X}")
                        elif frame['cmd'] == CMD_GET_STATUS:
                            status = parse_status(frame['data'])
                            self.logger.info(f"Device Status: {status}")
                else:
                    time.sleep(0.01)
            except Exception as e:
                self.logger.error(f"Error in RX loop: {e}")

    def handle_event(self, event_type: str, payload: Any):
        """处理事件队列中的灯光事件"""
        if not self._proto:
            self.logger.warning("Protocol not initialized")
            return

        try:
            if event_type == "solid":
                r, g, b = payload
                self._proto.set_all(r, g, b)
                
            elif event_type == "single":
                index, r, g, b = payload
                self._proto.set_single(index, r, g, b)
                
            elif event_type == "paint":
                # payload: [(r, g, b), ...] 或 [(index, r, g, b), ...]
                if payload and len(payload[0]) == 4:
                    self._proto.set_multi(payload)
                else:
                    # 如果只是颜色列表，扩展为带索引的多LED设置
                    leds = [(i, r, g, b) for i, (r, g, b) in enumerate(payload)]
                    self._proto.set_multi(leds)
                    
            elif event_type == "breath":
                r, g, b, speed = payload
                self._proto.effect_breath(r, g, b, speed)
                
            elif event_type == "rainbow":
                speed, brightness = payload
                self._proto.effect_rainbow(speed, brightness)
                
            elif event_type == "running":
                r, g, b, speed, direction = payload
                self._proto.effect_running(r, g, b, speed, direction)
                
            elif event_type == "chase":
                r, g, b, speed, length = payload
                self._proto.effect_chase(r, g, b, speed, length)
            
            elif event_type == "blink":
                speed = payload
                self._proto.effect_blink(speed)
            
            elif event_type == "ext_led_on":
                self._proto.ext_led_on()
            elif event_type == "ext_led_off":
                self._proto.ext_led_off()
                
            elif event_type == "brightness":
                brightness = payload
                self._proto.set_brightness(brightness)
                
            elif event_type == "stop_effect":
                self._proto.stop_effect()
                
            elif event_type == "clear":
                self._proto.clear()
                
            elif event_type == "get_status":
                self._proto.get_status()
                
            else:
                self.logger.warning(f"Unknown event type: {event_type}")
                
        except Exception as e:
            self.logger.error(f"Error handling event {event_type}: {e}")
