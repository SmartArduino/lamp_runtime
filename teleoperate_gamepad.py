"""Teleoperate LeLamp with a Logitech F710 gamepad, with optional video capture.

Usage (from the root of lelamp_runtime):
    uv run teleoperate_gamepad.py --id lelamp --port /dev/ttyACM0

Capture video from the device's cameras while teleoperating (lerobot
API-example style, cameras are passed to LeLampFollowerConfig):
    uv run teleoperate_gamepad.py --id lelamp --port /dev/ttyACM0 \
        --cameras "{lamp: {type: opencv, index_or_path: 0, fps: 30, width: 1280, height: 720}}"

If the camera rejects the requested fps/resolution ("failed to set fps=..."
error), use its defaults instead:
    uv run teleoperate_gamepad.py --id lelamp --port /dev/ttyACM0 --cameras 0
Check the modes your camera actually supports with:
    v4l2-ctl -d /dev/video0 --list-formats-ext

Requires pygame: run `uv add pygame` inside lelamp_runtime.
Set the mode switch on the back of the F710 to "X" (XInput).

While teleoperating, camera frames are captured and saved as one mp4 file
per camera under --save-dir. Run `lerobot-find-cameras opencv` to list the
camera indices/paths available on your device. For smooth capture, use a
camera fps at least equal to --fps.

Live viewing: use --web-port 8080 and open http://<device-ip>:8080 in any
browser on the same network (recommended for headless devices), or --display
for a local preview window (requires a desktop and non-headless opencv).

Control mapping (5 joints):
    Left stick X      -> base_yaw
    Left stick Y      -> base_pitch
    Right stick Y     -> elbow_pitch
    Right stick X     -> wrist_roll
    Triggers (LT/RT)  -> wrist_pitch
    LB (hold)         -> fine control (slower / reduced range)
    START             -> move all joints back to home (0)
    B                 -> quit
"""

import argparse
import os
import re
import socket
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:
    import pygame
except ImportError:
    raise SystemExit(
        "pygame is required for gamepad support. "
        "Install it inside lelamp_runtime with: uv add pygame"
    )

import numpy as np
import yaml

if sys.platform == "win32" and "OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS" not in os.environ:
    os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lelamp.follower import LeLampFollower, LeLampFollowerConfig

JOINTS = ["base_yaw", "base_pitch", "elbow_pitch", "wrist_roll", "wrist_pitch"]

AXIS_LEFT_X = 0
AXIS_LEFT_Y = 1
AXIS_LEFT_TRIGGER = 2
AXIS_RIGHT_X = 3
AXIS_RIGHT_Y = 4
AXIS_RIGHT_TRIGGER = 5

BUTTON_LB = 4
BUTTON_START = 7
BUTTON_B = 1

AMPLITUDE = 100.0
TRIGGER_DEADZONE = 0.05


def apply_deadzone(value: float, deadzone: float) -> float:
    magnitude = abs(value)
    if magnitude <= deadzone:
        return 0.0
    sign = 1.0 if value >= 0.0 else -1.0
    return sign * (magnitude - deadzone) / (1.0 - deadzone)


class F710Gamepad:
    def __init__(self, deadzone: float = 0.15, fine_factor: float = 0.25):
        self.deadzone = deadzone
        self.fine_factor = fine_factor
        self.joystick = None

    @property
    def is_connected(self) -> bool:
        return self.joystick is not None

    def connect(self) -> None:
        if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            self._diagnose_linux()
            raise ConnectionError(
                "No gamepad detected. Plug in the F710 USB receiver and try again."
            )

        index = 0
        for i in range(pygame.joystick.get_count()):
            if "f710" in pygame.joystick.Joystick(i).get_name().lower():
                index = i
                break
        self.joystick = pygame.joystick.Joystick(index)
        self.joystick.init()

        name = self.joystick.get_name()
        print(f"Connected gamepad: {name}")
        if "f710" not in name.lower() and "xbox" not in name.lower():
            print(
                f"Warning: expected a Logitech F710 in XInput mode, got '{name}'. "
                "Make sure the switch on the back of the gamepad is set to 'X'."
            )
        if self.joystick.get_numaxes() < 6:
            print(
                f"Warning: gamepad reports {self.joystick.get_numaxes()} axes, expected at least 6. "
                "Analog triggers need XInput mode ('X' switch on the back of the F710)."
            )
        self._print_controls()

    def _diagnose_linux(self) -> None:
        if not sys.platform.startswith("linux"):
            return
        print("Diagnostics: SDL/pygame cannot see any joystick.")
        nodes = []
        try:
            with open("/proc/bus/input/devices", encoding="utf-8") as f:
                for block in f.read().split("\n\n"):
                    if "f710" not in block.lower():
                        continue
                    name = "unknown"
                    handlers = []
                    for line in block.splitlines():
                        if line.startswith("N: Name="):
                            name = line.split("=", 1)[1].strip()
                        if line.startswith("H: Handlers="):
                            handlers = line.split("=", 1)[1].split()
                    nodes = [h for h in handlers if h.startswith(("event", "js"))]
                    print(f"Kernel device: {name}")
                    for n in nodes:
                        print(f"  /dev/input/{n}")
        except OSError:
            print("Could not read /proc/bus/input/devices.")

        if not nodes:
            print("The kernel does not report an F710 joystick device. Replug the USB receiver.")
            return

        unreadable = [
            n for n in nodes if not os.access(os.path.join("/dev/input", n), os.R_OK)
        ]
        if unreadable:
            for n in unreadable:
                print(f"  /dev/input/{n}: NOT readable by current user")
            print("Permission problem detected. Fix it with:")
            print("    sudo usermod -aG input $USER")
            print("then log out and back in (or reboot) and run this script again.")
            print("Quick check that permissions are the cause:")
            print(
                '    sudo uv run python -c "import pygame; pygame.init(); '
                'pygame.joystick.init(); print(pygame.joystick.get_count())"'
            )
        else:
            print("All device nodes are readable. Extra hints:")
            print(
                "  - Force SDL to use the device explicitly, e.g.: "
                "SDL_JOYSTICK_DEVICE=/dev/input/event4 uv run teleoperate_gamepad.py ..."
            )
            print(
                "  - Flip the F710 mode switch on the back to 'X' (XInput), "
                "replug the receiver, then retry."
            )

    def _print_controls(self) -> None:
        print("Gamepad controls:")
        print("  Left stick X       -> base_yaw")
        print("  Left stick Y       -> base_pitch")
        print("  Right stick Y      -> elbow_pitch")
        print("  Right stick X      -> wrist_roll")
        print("  Triggers LT / RT   -> wrist_pitch")
        print("  LB (hold)          -> fine control")
        print("  START              -> back to home position")
        print("  B                  -> quit")

    def read(self) -> dict:
        pygame.event.pump()

        left_x = apply_deadzone(self.joystick.get_axis(AXIS_LEFT_X), self.deadzone)
        left_y = apply_deadzone(self.joystick.get_axis(AXIS_LEFT_Y), self.deadzone)
        right_x = apply_deadzone(self.joystick.get_axis(AXIS_RIGHT_X), self.deadzone)
        right_y = apply_deadzone(self.joystick.get_axis(AXIS_RIGHT_Y), self.deadzone)

        left_trig = (self.joystick.get_axis(AXIS_LEFT_TRIGGER) + 1.0) / 2.0
        right_trig = (self.joystick.get_axis(AXIS_RIGHT_TRIGGER) + 1.0) / 2.0
        left_trig = apply_deadzone(left_trig, TRIGGER_DEADZONE)
        right_trig = apply_deadzone(right_trig, TRIGGER_DEADZONE)

        return {
            "base_yaw": left_x,
            "base_pitch": -left_y,
            "elbow_pitch": -right_y,
            "wrist_roll": right_x,
            "wrist_pitch": right_trig - left_trig,
            "fine": bool(self.joystick.get_button(BUTTON_LB)),
            "home": bool(self.joystick.get_button(BUTTON_START)),
            "exit": bool(self.joystick.get_button(BUTTON_B)),
        }

    def disconnect(self) -> None:
        if self.joystick is not None:
            self.joystick.quit()
            self.joystick = None
        if pygame.get_init():
            pygame.joystick.quit()
            pygame.quit()


def clamp(value: float, lo: float = -AMPLITUDE, hi: float = AMPLITUDE) -> float:
    return max(lo, min(hi, value))


CAMERA_OPTIONS = ("index_or_path", "fps", "width", "height", "rotation", "warmup_s")


def parse_camera_configs(spec: str | None) -> dict[str, OpenCVCameraConfig]:
    if not spec:
        return {}

    spec = spec.strip()
    if re.fullmatch(r"\d+", spec) or ("/" in spec and " " not in spec):
        parsed = {"lamp": {"index_or_path": spec}}
    else:
        parsed = yaml.safe_load(spec)

    if not isinstance(parsed, dict) or not parsed:
        raise ValueError(
            f"Invalid --cameras value: {spec!r}. Expected a YAML dict like "
            '"{lamp: {index_or_path: 0, fps: 30, width: 1280, height: 720}}".'
        )

    configs = {}
    for name, cfg in parsed.items():
        if not isinstance(cfg, dict):
            cfg = {"index_or_path": cfg}
        cfg = dict(cfg)
        cam_type = cfg.pop("type", "opencv")
        if cam_type != "opencv":
            raise ValueError(
                f"Camera {name!r}: unsupported type {cam_type!r}, only 'opencv' is supported."
            )
        unknown = sorted(set(cfg) - set(CAMERA_OPTIONS))
        if unknown:
            raise ValueError(
                f"Camera {name!r}: unknown option(s) {unknown}, valid options are {list(CAMERA_OPTIONS)}."
            )
        index_or_path = cfg.pop("index_or_path", 0)
        if isinstance(index_or_path, str) and index_or_path.isdigit():
            index_or_path = int(index_or_path)
        configs[name] = OpenCVCameraConfig(index_or_path=index_or_path, **cfg)
    return configs


def fill_camera_defaults(configs: dict[str, OpenCVCameraConfig]) -> None:
    for name, cfg in configs.items():
        if cfg.width and cfg.height and cfg.fps:
            continue
        cap = cv2.VideoCapture(cfg.index_or_path)
        if cap.isOpened():
            width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
            height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
        else:
            cap.release()
            width, height, fps = 640, 480, 0.0
            print(
                f"Warning: could not probe camera {name!r} ({cfg.index_or_path}); "
                "falling back to 640x480 @ 30 fps."
            )
        if width <= 0 or height <= 0:
            print(f"Warning: camera {name!r} reported {width}x{height}; using 640x480.")
            width, height = 640, 480
        if fps <= 0:
            print(f"Warning: camera {name!r} reported fps={fps}; using 30 fps.")
            fps = 30.0
        if cfg.width is None:
            cfg.width = width
        if cfg.height is None:
            cfg.height = height
        if cfg.fps is None:
            cfg.fps = int(round(fps))
        print(f"Camera {name!r}: {cfg.width}x{cfg.height} @ {cfg.fps} fps")


class VideoRecorder:
    def __init__(self, path: Path, fps: float):
        self.path = path
        self.fps = max(1.0, float(fps))
        self.writer = None
        self.num_frames = 0

    def write(self, frame_rgb: np.ndarray) -> np.ndarray:
        if self.writer is None:
            height, width = frame_rgb.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, (width, height))
            if not self.writer.isOpened():
                raise RuntimeError(f"Could not open video file for writing: {self.path}")
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        self.writer.write(frame_bgr)
        self.num_frames += 1
        return frame_bgr

    def release(self) -> None:
        if self.writer is not None:
            self.writer.release()
            print(f"Saved {self.num_frames} frames to {self.path}")
            self.writer = None


class MjpegStream:
    def __init__(self, names):
        self.names = list(names)
        self.lock = threading.Lock()
        self.frames = {}

    def update(self, name: str, frame_bgr: np.ndarray) -> None:
        ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            with self.lock:
                self.frames[name] = buf.tobytes()


def _make_stream_handler(stream: MjpegStream):
    class StreamHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send_index()
            elif self.path.startswith("/stream/"):
                self._send_stream(self.path[len("/stream/"):])
            else:
                self.send_error(404)

        def _send_index(self):
            parts = [
                f"<div><h2>{name}</h2><img src='/stream/{name}' "
                "style='max-width:100%;height:auto'></div>"
                for name in stream.names
            ]
            body = "\n".join(parts) or "<p>No cameras configured.</p>"
            html = (
                "<!doctype html><html><head><meta charset='utf-8'>"
                "<title>LeLamp live camera</title></head>"
                f"<body><h1>LeLamp live camera</h1>{body}</body></html>"
            )
            data = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_stream(self, name):
            if name not in stream.names:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    with stream.lock:
                        data = stream.frames.get(name)
                    if data is not None:
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                            + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n"
                        )
                        self.wfile.flush()
                    time.sleep(1.0 / 30.0)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError):
                pass

    return StreamHandler


def get_lan_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def start_web_server(stream: MjpegStream, port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("0.0.0.0", port), _make_stream_handler(stream))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def precise_sleep(seconds: float) -> None:
    if seconds > 0.0:
        time.sleep(seconds)


def main():
    parser = argparse.ArgumentParser(
        description="Teleoperate LeLamp with a Logitech F710 gamepad",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Example: uv run teleoperate_gamepad.py --id lelamp --port /dev/ttyACM0",
    )
    parser.add_argument("--id", type=str, required=True, help="ID of the lamp")
    parser.add_argument("--port", type=str, required=True, help="Serial port for the lamp")
    parser.add_argument(
        "--mode",
        choices=["velocity", "absolute"],
        default="velocity",
        help="velocity: stick deflection drives joint speed; "
        "absolute: stick position sets joint position",
    )
    parser.add_argument("--fps", type=int, default=30, help="Control loop frequency")
    parser.add_argument(
        "--max-speed",
        type=float,
        default=60.0,
        help="Max joint speed in normalized units per second (velocity mode)",
    )
    parser.add_argument("--deadzone", type=float, default=0.15, help="Stick deadzone")
    parser.add_argument(
        "--max-relative-target",
        type=int,
        default=None,
        help="Safety clamp on goal position change per sent action (recommended in absolute mode)",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        default=None,
        help='Cameras to capture while teleoperating, as a YAML dict like '
        '"{lamp: {type: opencv, index_or_path: 0, fps: 30, width: 1280, height: 720}}", '
        "or a shorthand camera index/path (e.g. 0 or /dev/video0). "
        "Omit to teleoperate without cameras. "
        "Run `lerobot-find-cameras opencv` to list available cameras. "
        "Omitted fps/width/height are probed from the camera's current defaults; "
        "if connecting fails with 'failed to set fps/width/height', the camera does "
        "not support that mode.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="videos",
        help="Directory where captured camera videos are saved as mp4 files",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show a live camera preview window (requires a display and non-headless opencv)",
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=None,
        help="Serve a live camera stream over HTTP on this port (e.g. 8080); "
        "open http://<device-ip>:<port> in a browser on the same network",
    )
    args = parser.parse_args()

    try:
        camera_config = parse_camera_configs(args.cameras)
    except ValueError as e:
        raise SystemExit(str(e))
    fill_camera_defaults(camera_config)

    robot_config = LeLampFollowerConfig(
        port=args.port,
        id=args.id,
        max_relative_target=args.max_relative_target,
        cameras=camera_config,
    )
    robot = LeLampFollower(robot_config)
    robot.connect()
    if camera_config:
        print(f"Cameras connected: {', '.join(robot.cameras)}")

    gamepad = F710Gamepad(deadzone=args.deadzone)
    gamepad.connect()

    dt = 1.0 / args.fps
    obs = robot.get_observation()
    current = {f"{joint}.pos": float(obs[f"{joint}.pos"]) for joint in JOINTS}

    if args.mode == "velocity":
        target = dict(current)
    else:
        base = dict(current)
        if args.max_relative_target is None:
            print(
                "Tip: in absolute mode, consider --max-relative-target (e.g. 5) "
                "to limit how fast the lamp can move per action."
            )

    print(f"Teleoperating {robot} in {args.mode} mode at {args.fps} fps. Press B on the gamepad to quit.")

    recorders = {}
    if camera_config:
        if args.display and not hasattr(cv2, "imshow"):
            print("Warning: cv2.imshow is unavailable (opencv-python-headless); disabling --display.")
            args.display = False

        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for name, cam_cfg in camera_config.items():
            fps = robot.cameras[name].fps or cam_cfg.fps or args.fps
            recorders[name] = VideoRecorder(save_dir / f"{args.id}_{name}_{stamp}.mp4", fps)
        print(f"Recording camera video to {save_dir} while teleoperating.")

    stream = None
    if args.web_port:
        if recorders:
            stream = MjpegStream(recorders.keys())
            start_web_server(stream, args.web_port)
            print(
                f"Live camera view: http://{get_lan_ip()}:{args.web_port}/ "
                "(open in a browser on the same network)"
            )
        else:
            print("Warning: --web-port ignored because no cameras are configured (--cameras).")

    try:
        while True:
            t0 = time.perf_counter()

            state = gamepad.read()
            if state["exit"]:
                print("Quit button pressed. Shutting down teleop...")
                break

            if state["home"]:
                if args.mode == "absolute":
                    base = {f"{joint}.pos": 0.0 for joint in JOINTS}
                target = {f"{joint}.pos": 0.0 for joint in JOINTS}
            else:
                scale = gamepad.fine_factor if state["fine"] else 1.0
                for joint in JOINTS:
                    key = f"{joint}.pos"
                    if args.mode == "velocity":
                        target[key] = clamp(target[key] + state[joint] * args.max_speed * scale * dt)
                    else:
                        target[key] = clamp(base[key] + state[joint] * AMPLITUDE * scale)

            robot.send_action(target)

            if recorders:
                try:
                    observation = robot.get_observation()
                    for name in recorders:
                        frame_bgr = recorders[name].write(observation[name])
                        if stream is not None:
                            stream.update(name, frame_bgr)
                        if args.display:
                            cv2.imshow(f"lelamp_{name}", frame_bgr)
                except TimeoutError:
                    pass
                if args.display:
                    cv2.waitKey(1)

            precise_sleep(dt - (time.perf_counter() - t0))
    except KeyboardInterrupt:
        print("Shutting down teleop...")
    finally:
        for recorder in recorders.values():
            recorder.release()
        if args.display and recorders:
            cv2.destroyAllWindows()
        gamepad.disconnect()
        robot.disconnect()


if __name__ == "__main__":
    main()
