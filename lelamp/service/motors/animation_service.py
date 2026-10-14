import os
import csv
import time
import threading
from typing import Any, List, Dict, Optional, Tuple
from lelamp.follower import LeLampFollowerConfig, LeLampFollower


class AnimationService:
    def __init__(self, port: str, lamp_id: str, fps: int = 30, duration: float = 5.0, idle_recording: str = "idle"):
        self.port = port
        self.lamp_id = lamp_id
        self.fps = fps
        self.duration = duration
        self.idle_recording = idle_recording
        self.robot_config = LeLampFollowerConfig(port=port, id=lamp_id)
        self.robot: LeLampFollower = None
        self.recordings_dir = os.path.join(os.path.dirname(__file__), "..", "..", "recordings")
        
        # State management
        self._recording_cache: Dict[str, List[Dict[str, float]]] = {}
        self._current_state: Optional[Dict[str, float]] = None
        self._current_recording: Optional[str] = None
        self._current_frame_index: int = 0
        self._current_actions: List[Dict[str, float]] = []
        self._interpolation_frames: int = 0
        self._interpolation_target: Optional[Dict[str, float]] = None
        
        # Custom event handling
        self._running = threading.Event()
        self._event_queue = []
        self._event_lock = threading.Lock()
        self._event_thread: Optional[threading.Thread] = None

        # 追踪指令与动作回放共用同一条电机总线，需要互斥访问
        # （send_tracking_action 与 _continue_playback 不能同时调用 robot.send_action）
        self._robot_lock = threading.Lock()

        # 人脸追踪是否处于激活状态（由 FaceTrackingService 通过
        # set_tracking_active() 设置）。激活时 idle 循环让位，由追踪线程接管
        # 电机指令；非 idle 动作（curious/happy 等）不受影响。
        self._tracking_active: bool = False
    
    def start(self):
        self.robot = LeLampFollower(self.robot_config)
        self.robot.connect(calibrate=False)
        print(f"Animation service connected to {self.port}")
        
        # Start event processing thread
        self._running.set()
        self._event_thread = threading.Thread(target=self._event_loop, daemon=True)
        self._event_thread.start()
        
        # Initialize with idle recording via self dispatch
        self.dispatch("play", self.idle_recording)

    def stop(self, timeout: float = 5.0):
        # Stop event processing
        self._running.clear()
        if self._event_thread and self._event_thread.is_alive():
            self._event_thread.join(timeout=timeout)
        
        if self.robot:
            self.robot.disconnect()
            self.robot = None
    
    def dispatch(self, event_type: str, payload: Any):
        """Dispatch an event - same interface as ServiceBase"""
        if not self._running.is_set():
            print(f"Animation service is not running, ignoring event {event_type}")
            return
        
        with self._event_lock:
            self._event_queue.append((event_type, payload))
    
    def _event_loop(self):
        """Custom event loop that supports interruption"""
        while self._running.is_set():
            # Check for events
            with self._event_lock:
                if self._event_queue:
                    event_type, payload = self._event_queue.pop(0)
                else:
                    event_type, payload = None, None
            
            if event_type:
                try:
                    self.handle_event(event_type, payload)
                except Exception as e:
                    print(f"Error handling event {event_type}: {e}")
            
            # Continue current playback
            self._continue_playback()
            
            time.sleep(1.0 / self.fps)  # Frame rate timing
    
    def handle_event(self, event_type: str, payload: Any):
        if event_type == "play":
            self._handle_play(payload)
        else:
            print(f"Unknown event type: {event_type}")
    
    def _handle_play(self, recording_name: str):
        """Start playing a recording with interpolation from current state"""
        if not self.robot:
            print("Robot not connected")
            return
        
        # Load the recording
        actions = self._load_recording(recording_name)
        if actions is None:
            return
        
        print(f"Starting {recording_name} with interpolation")
        
        # Set up new playback
        self._current_recording = recording_name
        self._current_actions = actions
        self._current_frame_index = 0
        
        # If we have a current state, set up interpolation to the first frame
        if self._current_state is not None:
            self._interpolation_frames = int(self.duration * self.fps)
            self._interpolation_target = actions[0]
        else:
            self._interpolation_frames = 0
            self._interpolation_target = None
    
    def _continue_playback(self):
        """Continue current playback - called every frame"""
        if not self._current_recording or not self._current_actions:
            return

        # 人脸追踪激活时，idle 循环让位：
        # - 仅当当前是 idle 录制、且不在插值过程中时让位（避免插值半截被打断造成卡顿）
        # - 非 idle 动作（curious/happy 等）始终正常播放，不受追踪影响
        if (
            self._tracking_active
            and self._current_recording == self.idle_recording
            and self._interpolation_frames == 0
        ):
            return

        try:
            # Handle interpolation to first frame
            if self._interpolation_frames > 0 and self._interpolation_target is not None:
                # Calculate interpolation progress
                progress = 1.0 - (self._interpolation_frames / (self.duration * self.fps))
                progress = max(0.0, min(1.0, progress))
                
                # Interpolate between current state and target
                interpolated_action = {}
                for joint in self._interpolation_target.keys():
                    current_val = self._current_state.get(joint, 0)
                    target_val = self._interpolation_target[joint]
                    interpolated_action[joint] = current_val + (target_val - current_val) * progress
                
                with self._robot_lock:
                    self.robot.send_action(interpolated_action)
                self._current_state = interpolated_action.copy()
                self._interpolation_frames -= 1
                return
            
            # Play current frame
            if self._current_frame_index < len(self._current_actions):
                action = self._current_actions[self._current_frame_index]
                with self._robot_lock:
                    self.robot.send_action(action)
                self._current_state = action.copy()
                self._current_frame_index += 1
            else:
                # Recording finished
                if self._current_recording != self.idle_recording:
                    # Interpolate back to idle
                    idle_actions = self._load_recording(self.idle_recording)
                    if idle_actions is not None and len(idle_actions) > 0:
                        self._current_recording = self.idle_recording
                        self._current_actions = idle_actions
                        self._current_frame_index = 0
                        # Set up interpolation back to idle
                        if self._current_state is not None:
                            self._interpolation_frames = int(self.duration * self.fps)
                            self._interpolation_target = idle_actions[0]
                else:
                    # Loop idle recording
                    self._current_frame_index = 0
                    
        except Exception as e:
            print(f"Error in playback: {e}")
            # Reset to safe state
            self._current_recording = None
            self._current_actions = []
            self._current_frame_index = 0
    

        """Get list of recording names available for this lamp ID"""
        if not os.path.exists(self.recordings_dir):
            return []
        
        recordings = []
        suffix = f".csv"
        
        for filename in os.listdir(self.recordings_dir):
            if filename.endswith(suffix):
                # Remove the lamp_id suffix to get the recording name
                recording_name = filename[:-len(suffix)]
                recordings.append(recording_name)
        
        return sorted(recordings)
    
    def _load_recording(self, recording_name: str) -> Optional[List[Dict[str, float]]]:
        """Load a recording from cache or file"""
        # Check cache first
        if recording_name in self._recording_cache:
            return self._recording_cache[recording_name]
        
        csv_filename = f"{recording_name}.csv"
        csv_path = os.path.join(self.recordings_dir, csv_filename)
        
        if not os.path.exists(csv_path):
            print(f"Recording not found: {csv_path}")
            return None
        
        try:
            with open(csv_path, 'r') as csvfile:
                csv_reader = csv.DictReader(csvfile)
                actions = []
                for row in csv_reader:
                    # Extract action data (exclude timestamp column)
                    action = {key: float(value) for key, value in row.items() if key != 'timestamp'}
                    actions.append(action)
            
            # Cache the recording
            self._recording_cache[recording_name] = actions
            return actions
            
        except Exception as e:
            print(f"Error loading recording {recording_name}: {e}")
            return None

    @property
    def is_busy(self) -> bool:
        """当前是否正在播放非 idle 动作（线程安全）"""
        return (
            self._current_recording is not None
            and self._current_recording != self.idle_recording
        )

    # ------------------------------------------------------------------
    # 人脸追踪通道
    # ------------------------------------------------------------------
    def send_tracking_action(self, action: Dict[str, float]) -> bool:
        """
        供 FaceTrackingService 等"非动作回放"的调用方下发瞬时电机指令。

        与 _continue_playback 共用 _robot_lock，确保不会在 robot.send_action
        正写一帧时被另一线程并发写入。

        当 is_busy=True（正在播放非 idle 动作）时直接拒绝，
        避免追踪指令把动作打断。返回 True 表示已下发。
        """
        if self.robot is None:
            return False
        if self.is_busy:
            return False
        try:
            with self._robot_lock:
                self.robot.send_action(action)
            # 同步当前状态，避免下次插值从旧状态开始
            # action 的 key 形如 "base_yaw.pos"
            self._current_state = action.copy()
            return True
        except Exception as e:
            print(f"send_tracking_action 失败: {e}")
            return False

    def set_tracking_active(self, active: bool) -> None:
        """
        由 FaceTrackingService 调用，告诉 AnimationService 追踪是否处于激活状态。
        - active=True：检测到人脸正在追踪 → idle 循环让位
        - active=False：未检测到人脸 → idle 恢复正常循环
        """
        if active and not self._tracking_active:
            print("[AnimationService] 人脸追踪激活，idle 循环让位")
        elif not active and self._tracking_active:
            print("[AnimationService] 人脸追踪停止，idle 循环恢复")
        self._tracking_active = active

    def get_idle_first_frame(self) -> Optional[Dict[str, float]]:
        """
        返回 idle 录制的第一帧（dict，key 形如 'base_yaw.pos'），
        供 FaceTrackingService 初始化基准姿态使用。加载失败返回 None。
        """
        actions = self._load_recording(self.idle_recording)
        if actions is None or len(actions) == 0:
            return None
        return actions[0].copy()
