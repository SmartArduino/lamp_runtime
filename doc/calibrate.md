# Calibrate servo

Motor Setup and Calibration

## 1. Find the servo driver port:
This command finds the port your motor driver is connected to.

`uv run lerobot-find-port`

## 2. Setup motors with unique IDs:
This command set up each motor of LeLamp with an unique ID.

`uv run -m lelamp.setup_motors --id your_lamp_name --port the_port_found_in_previous_step`

## 3. Calibrate motors:

This command calibrate your motors.

`sudo uv run -m lelamp.calibrate --id your_lamp_name --port the_port_found_in_previous_step`

The calibration process will:

* Calibrate both follower and leader modes
* Ensure proper servo positioning and response
* Set baseline positions for accurate movement