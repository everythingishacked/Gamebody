import argparse
from contextlib import ExitStack
import csv

from keyboard import keyboard # local fork

import mediapipe as mp
import cv2

from datetime import datetime
from math import atan, atan2, pi, degrees
from numpy import concatenate
from scipy.spatial import distance as dist

DEFAULT_LANDMARKS_STYLE = mp.solutions.drawing_styles.get_default_pose_landmarks_style()

# Optionally record the video feed to a timestamped AVI in the current directory
RECORDING_FILENAME = str(datetime.now()).replace('.','').replace(':','') + '.avi'
FPS = 10

VISIBILITY_THRESHOLD = .8 # amount of certainty that a body landmark is visible
STRAIGHT_LIMB_MARGIN = 20 # degrees from 180
EXTENDED_LIMB_MARGIN = .8 # lower limb length as fraction of upper limb

ARM_CROSSED_RATIO = 2 # max distance from wrist to opposite elbow, relative to mouth width

MOUTH_COVER_THRESHOLD = .03 # hands over mouth max distance error out of 1

LEG_LIFT_MIN = -10 # degrees from horizontal

SQUAT_THRESHOLD = .1

JUMP_THRESHOLD = .0001

# R side: 90 top to 0 right to -90 bottom
# L side: 90 top to 180 left to 269... -> -90 bottom
semaphores = {}

LEG_EXTEND_ANGLE = 18 # degrees from vertical standing; should be divisor of 90
leg_extension_angles = {
  (-90, -90 + LEG_EXTEND_ANGLE): (True, 0), # right leg, low
  (-90, -90 + 2*LEG_EXTEND_ANGLE): (True, 1), # right leg, high
  (270 - LEG_EXTEND_ANGLE, -90): (False, 0), # left leg, low
  (270 - 2*LEG_EXTEND_ANGLE, -90): (False, 1), #left leg high
}

FRAME_HISTORY = 8 # pose history is compared against FRAME_HISTORY recent frames
HALF_HISTORY = int(FRAME_HISTORY/2)

empty_frame = {
  'hipL_y': 0,
  'hipR_y': 0,
  'hips_dy': 0,
}
last_frames = FRAME_HISTORY*[empty_frame.copy()]

frame_midpoint = (0,0)

last_keys = [[]]

def map_keys(file_name, player_count):
  global semaphores

  with open('maps/' + (file_name or 'default.csv')) as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader) # skip first row
    for player_cardinal, part, position, keys, repeat, action_name in csv_reader:
      player_index = player_count - int(player_cardinal) # reverse and 0-index
      semaphores[(player_index, part, int(position))] = {
        'keys': keys.split(' '),
        'name': action_name,
        'repeat': bool(int(repeat)),
      }
    print("Successfully read in:", semaphores)

def get_angle(a, b, c):
  ang = degrees(atan2(c['y']-b['y'], c['x']-b['x']) - atan2(a['y']-b['y'], a['x']-b['x']))
  return ang + 360 if ang < 0 else ang

def is_missing(part):
  return any(joint['visibility'] < VISIBILITY_THRESHOLD for joint in part)

def is_limb_pointing(upper, mid, lower):
  if is_missing([upper, mid, lower]):
    return False
  limb_angle = get_angle(upper, mid, lower)
  is_in_line = abs(180 - limb_angle) < STRAIGHT_LIMB_MARGIN
  if is_in_line:
    upper_length = dist.euclidean([upper['x'], upper['y']], [mid['x'], mid['y']])
    lower_length = dist.euclidean([lower['x'], lower['y']], [mid['x'], mid['y']])
    is_extended = lower_length > EXTENDED_LIMB_MARGIN * upper_length
    return is_extended
  return False

def get_limb_direction(arm, closest_degrees=45):
  # should also use atan2 but I don't want to do more math
  dy = arm[2]['y'] - arm[0]['y'] # wrist -> shoulder
  dx = arm[2]['x'] - arm[0]['x']
  angle = degrees(atan(dy/dx))
  if (dx < 0):
    angle += 180

  # collapse to nearest closest_degrees; 45 for semaphore
  mod_close = angle % closest_degrees
  angle -= mod_close
  if mod_close > closest_degrees/2:
    angle += closest_degrees

  angle = int(angle)
  if angle == 270:
    angle = -90

  return angle

def is_arm_crossed(elbow, wrist, max_dist):
  return dist.euclidean([elbow['x'], elbow['y']], [wrist['x'], wrist['y']]) < max_dist

def is_arms_crossed(elbowL, wristL, elbowR, wristR, mouth_width):
  max_dist = mouth_width * ARM_CROSSED_RATIO
  return is_arm_crossed(elbowL, wristR, max_dist) and is_arm_crossed(elbowR, wristL, max_dist)

def is_leg_lifted(leg):
  if is_missing(leg):
    return False
  dy = leg[1]['y'] - leg[0]['y'] # knee -> hip
  dx = leg[1]['x'] - leg[0]['x']
  angle = degrees(atan2(dy, dx))
  return angle > LEG_LIFT_MIN

def is_jumping(i, hipL, hipR):
  global last_frames

  if is_missing([hipL, hipR]):
    return False

  last_frames[i][-1]['hipL_y'] = hipL['y']
  last_frames[i][-1]['hipR_y'] = hipR['y']

  if (hipL['y'] > last_frames[i][-2]['hipL_y'] + JUMP_THRESHOLD) and (
      hipR['y'] > last_frames[i][-2]['hipR_y'] + JUMP_THRESHOLD):
    last_frames[i][-1]['hips_dy'] = 1 # rising
  elif (hipL['y'] < last_frames[i][-2]['hipL_y'] - JUMP_THRESHOLD) and (
        hipR['y'] < last_frames[i][-2]['hipR_y'] - JUMP_THRESHOLD):
    last_frames[i][-1]['hips_dy'] = -1 # falling
  else:
    last_frames[i][-1]['hips_dy'] = 0 # not significant dy

  # consistently rising first half, lowering second half
  jump_up = all(frame['hips_dy'] == 1 for frame in last_frames[i][:HALF_HISTORY])
  get_down = all(frame['hips_dy'] == -1 for frame in last_frames[i][HALF_HISTORY:])
  return jump_up and get_down

def is_mouth_covered(mouth, palms):
  if is_missing(palms):
    return False
  dxL = (mouth[0]['x'] - palms[0]['x'])
  dyL = (mouth[0]['y'] - palms[0]['y'])
  dxR = (mouth[1]['x'] - palms[1]['x'])
  dyR = (mouth[1]['y'] - palms[1]['y'])
  return all(abs(d) < MOUTH_COVER_THRESHOLD for d in [dxL, dyL, dxR, dyR])

def is_squatting(hipL, kneeL, hipR, kneeR):
  if is_missing([hipL, kneeL, hipR, kneeR]):
    return False
  dyL = abs(hipL['y'] - kneeL['y'])
  dyR = abs(hipR['y'] - kneeR['y'])
  return (dyL < SQUAT_THRESHOLD) and (dyR < SQUAT_THRESHOLD)

def match_and_type(player_num, parts_and_actions, image, display_only):
  global semaphores, last_keys

  new_keys = []
  new_keys_to_repeat = []

  for (part_or_action, position) in parts_and_actions:
    match = semaphores.get((player_num, part_or_action, position), '')
    if match:
      if match.get('repeat'):
        new_keys_to_repeat += [match.get('keys', '')]
      else:
        new_keys += [match.get('keys', '')]

  all_new_keys = new_keys + new_keys_to_repeat

  for hotkey in last_keys[player_num]:
    if (hotkey not in all_new_keys):
      print("releasing:", hotkey)
      keyboard.release(hotkey)

  output(new_keys, last_keys[player_num], False, image, display_only)
  output(new_keys_to_repeat, last_keys[player_num], True, image, display_only)
  last_keys[player_num] = all_new_keys

def output(keys, previous_keys, repeat, image, display_only):
  for hotkey in keys:
    keystring = '+'.join(key for key in hotkey if key not in previous_keys)
    if len(keystring):
      if display_only:
        cv2.putText(image, keystring, frame_midpoint,
          cv2.FONT_HERSHEY_SIMPLEX, 20, (0,0,255), 20)
      else:
        if repeat:
          print("REPEAT: press & release", keystring)
          keyboard.press_and_release(keystring)
        else:
          print("pressing:", keystring)
          keyboard.press(keystring)

def render_and_maybe_exit(image, recording):
  cv2.imshow('Semaphore Games', image)
  if recording:
    recording.write(image)
  return cv2.waitKey(5) & 0xFF == 27

def process_poses(image, pose_models, draw_landmarks, flip, display_only):
  global last_frames, frame_midpoint, last_keys

  image.flags.writeable = False
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  width = image.shape[1]
  splits = len(pose_models)
  split_len = width // splits
  images = [image[:, i:i+split_len] for i in range(0, width, split_len)]

  for mark in range(0, width, split_len):
    cv2.line(image, (mark,0), (mark,width), (255,255,255), 1)

  pose_results = [pose_models[i].process(images[i]) for i in range(0, splits)]

  if draw_landmarks:
    for i, image in enumerate(images):
      mp.solutions.drawing_utils.draw_landmarks(
        image,
        pose_results[i].pose_landmarks,
        mp.solutions.pose.POSE_CONNECTIONS,
        DEFAULT_LANDMARKS_STYLE)

  image = concatenate(images, axis=1)
  image.flags.writeable = True
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

  if flip: # selfie view
    image = cv2.flip(image, 1)

  for player_num, pose_result in enumerate(pose_results):
    actions = []

    if pose_result.pose_landmarks:
      # prepare to store most recent frame of movement updates over time
      last_frames[player_num] = last_frames[player_num][1:] + [empty_frame.copy()]

      body = []
      # (0,0) bottom left to (1,1) top right
      for point in pose_result.pose_landmarks.landmark:
        body.append({
          'x': 1 - point.x,
          'y': 1 - point.y,
          'visibility': point.visibility
        })

      kneeL, kneeR = body[25], body[26]
      hipL, hipR = body[23], body[24]
      legL = (hipL, kneeL, body[27]) # + ankle
      legR = (hipR, kneeR, body[28]) # + ankle

      if is_squatting(hipL, kneeL, hipR, kneeR):
        # squat (hips <> knees ~horizontal)
        actions += [('squat', 1)]
      elif is_leg_lifted(legL): # one hip <> knee ~horizontal
        actions += [('left leg', 2)]
      elif is_leg_lifted(legR):
        actions += [('right leg', 2)]
      else:
        # leg extension angles
        if is_limb_pointing(*legL) and is_limb_pointing(*legR):
          legL_angle = get_limb_direction(legL, LEG_EXTEND_ANGLE)
          legR_angle = get_limb_direction(legR, LEG_EXTEND_ANGLE)
          is_right, is_high = leg_extension_angles.get((legL_angle, legR_angle), (None, None))
          if is_high is not None:
            which_leg = ('right' if is_right else 'left') + ' leg'
            actions += [(which_leg, is_high)]

      # jump (hips rise + fall)
      if is_jumping(player_num, hipL, hipR):
        actions += [('jump', 1)]

      # mouth covered by both palms
      mouth = (body[9], body[10])
      palms = (body[19], body[20])
      if is_mouth_covered(mouth, palms):
        actions += [('mouth', 1)]

      # arms crossed: wrists near opposite elbows
      shoulderL, elbowL, wristL = body[11], body[13], body[15]
      armL = (shoulderL, elbowL, wristL)
      shoulderR, elbowR, wristR = body[12], body[14], body[16]
      armR = (shoulderR, elbowR, wristR)
      mouth_width = abs(mouth[1]['x']-mouth[0]['x'])
      if is_arms_crossed(elbowL, wristL, elbowR, wristR, mouth_width):
        actions += [('crossed arms', 1)]

      # single arm extension angles
      for (arm, is_right) in [(armL, False), (armR, True)]:
        if is_limb_pointing(*arm):
          arm_angle = get_limb_direction(arm)
          which_arm = ('right' if is_right else 'left') + ' arm'
          actions += [(which_arm, arm_angle)]

    if actions or last_keys[player_num]:
      match_and_type(player_num, actions, image, display_only)

  return image


def main():
  global last_frames, last_keys, frame_midpoint

  parser = argparse.ArgumentParser()
  parser.add_argument('--map', '-m', help='File to import for mapped keys')
  parser.add_argument('--input', '-i', help='Input video device or file (number or path), defaults to 0', default='0')
  parser.add_argument('--flip', '-f', help='Set to any value to flip resulting output (selfie view)')
  parser.add_argument('--landmarks', '-l', help='Set to any value to draw body landmarks')
  parser.add_argument('--record', '-r', help='Set to any value to save a timestamped AVI in current directory')
  parser.add_argument('--display', '-d', help='Set to any value to only visually display output rather than type')
  parser.add_argument('--split', '-s', help='Split the screen into a positive integer of separate regions, defaults to 1', default='1')
  args = parser.parse_args()

  INPUT = int(args.input) if args.input.isdigit() else args.input
  FLIP = args.flip is not None
  DRAW_LANDMARKS = args.landmarks is not None
  RECORDING = args.record is not None
  DISPLAY_ONLY = args.display is not None
  SPLIT = int(args.split)

  last_frames = SPLIT * [last_frames.copy()]
  last_keys = SPLIT * [[]]

  cap = cv2.VideoCapture(INPUT)

  frame_size = (int(cap.get(3)), int(cap.get(4)))
  frame_midpoint = (int(frame_size[0]/2), int(frame_size[1]/2))

  recording = cv2.VideoWriter(RECORDING_FILENAME,
    cv2.VideoWriter_fourcc(*'MJPG'), FPS, frame_size) if RECORDING else None

  MAP_FILE = args.map
  map_keys(MAP_FILE, SPLIT)

  with ExitStack() as stack:
    pose_models = SPLIT*[stack.enter_context(mp.solutions.pose.Pose())]

    while cap.isOpened():
      success, image = cap.read()
      if not success: break

      image = process_poses(image, pose_models, DRAW_LANDMARKS, FLIP, DISPLAY_ONLY)

      if render_and_maybe_exit(image, recording):
        break

  if RECORDING:
    recording.release()

  cap.release()
  cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
