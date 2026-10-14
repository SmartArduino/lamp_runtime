# Calibrate servo

This command calibrate your motors.

`sudo uv run -m lelamp.calibrate --id your_lamp_name --port the_port_found_in_previous_step`

The calibration process will:

* Calibrate both follower and leader modes
* Ensure proper servo positioning and response
* Set baseline positions for accurate movement