#uv run teleoperate_gamepad.py --id bubble --port /dev/ttyACM0  --cameras "{lamp: {type: opencv, index_or_path: 0, fps: 10, width: 640, height: 480}}" --web-port 8080
uv run teleoperate_gamepad.py --id bubble --port /dev/ttyACM0 --cameras 0 --web-port 8080
