# Semaphore Games

### A full-body game controller

[![demo](demo.gif)](https://youtu.be/Vi3Li3TkUVY)

View a fuller demo and more background on the project at https://youtu.be/Vi3Li3TkUVY

Semaphore Games uses [OpenCV](https://github.com/opencv/opencv-python) and MediaPipe's [Pose detection](https://google.github.io/mediapipe/solutions/pose.html#python-solution-api) to perform real-time detection of body landmarks from video input. From there, relative differences are calculated to determine specific positions and translate those into keys and commands sent via [keyboard](https://github.com/boppreh/keyboard).

For the best experience, you will need a big, well-lit space with a plain background and some combination of an external monitor, external webcam, or extremely good eyesight; standing far enough away from a built-in camera to fit multiple bodies with extended limbs takes significant distance.

The primary way to "type" custom defined keys or key combinations is by extending arms and raising legs at various angles, with some additional recognized motions such as covering the mouth, crossing the arms, jumping, and squatting.

Movements are mapped to keys based on a customizable CSV input file. See `data/default.csv` for the default example file. Using the `--map` option you can create and load in your own key mappings for different games.

Example common usage:

`python play.py -f 1 -l 1 -s 2 -r 1 -m pong.csv` - Flip the camera, draw body 
landmarks, split the screen for two players, record the results to a timestamped AVI, and use the `pong.csv` map for input controls.

By default, extended arm angles snap to the closest 45 degrees, and leg angles are set to 18 degrees off of vertical (i.e. 18 for 'low kick' and 36 for 'high kick'). Modify `LEG_EXTEND_ANGLE` and the resulting angles in mapped leg extensions to update this sensitivity.

An input file for key mapping should follow the format in `default.csv`, with the following columns:

- 0: player number, starting at 1
- 1: body part or action, such as `left arm` or `jump`; full list below
- 2: value for part or action: degrees such as 45 for arms, or 0/1/2 for legs
- 3: key or keys to press, separated by spaces
- 4: user-defined action description, such as `accelerate` or `low punch`

The first _row_ of each mapping file is skipped; it can be used for notes such as identifying or linking to the relevant game.

The full set of parts/actions available are as follows:
- `left arm`: takes values rounded to 45 degrees for fully extended arm; see below for 
degrees
- `right arm`: same as above
- `left leg`: takes values of 0 (low straight leg lift), 1 (high leg lift), or 2 (leg lift, 
bent at knee)
- `right leg`: same as above
- `crossed arms`: only takes the value `1`; both palms touching opposite elbows
- `jump`: only takes the value `1`; hips quickly rising then falling
- `squat`: only takes the value `1`; both thighs parallel to ground, overrides leg lifts
- `mouth`: only takes the value `1`; cover mouth with both palms

Right arm degrees: 90 straight up to 0 right to -90 down by side
Left arm degrees: 90 straight up to 180 left to 269... -> -90 down by side

Running on latest MacOS from Terminal, toggle the following for keyboard access:
`System Settings -> Privacy & Security -> Accessibility -> Terminal -> slide to allow`

For Mac, this uses a [custom keyboard library](https://github.com/everythingishacked/keyboard) imported locally to correct key mappings.
