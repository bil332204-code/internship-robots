"""
object_finder.py
-------------------
"Object Finder" -- the main assistant script.

Flow:
  1. Take a text command (e.g. "find my bottle")
  2. Gemini figures out WHAT object to look for, and checks if YOLO
     can even recognize that kind of object
  3. Head scans through positions: CENTER, LEFT, RIGHT, UP, DOWN
  4. At each position, take a photo and check with YOLO if the target
     object is visible
  5. If found: point the matching arm toward it, announce it
     If not found after all positions: apologize
  6. Ask "anything else?" and loop

Run with ARC_SCRIPT.py already running (updated version with
pan+tilt head control and arm control).

NOTE: Head tilt (up/down) and arm "pointing" angles below are
PLACEHOLDER values -- test these with your real robot and update
the TODO-marked constants once you have it back.
"""

import os
import time
import msvcrt
from yolo_detector import detect_objects, detect_with_positions
from vision import extract_target_object, describe_object_location
from pid_controller import PIDController
from face_recognizer import recognize_people, get_known_names, recognize_people_with_positions

# Relative paths, same convention as the rest of the project (gesture.txt,
# speech.txt live in the project folder itself, not a hardcoded absolute
# path) -- this way it works on any machine/laptop without editing paths.
TRIGGER_FILE = "trigger.txt"
READY_FILE = "ready.txt"
SPEAK_FILE = "speech.txt"  # merged with the rest of the project's speech.txt
SERVO_FILE = "servo.txt"
ARM_FILE = "arm.txt"
LED_FILE = "led.txt"
SNAPSHOT_FILE = "snapshot.jpg"
COMMAND_FILE = "command.txt"

READY_TIMEOUT_SECONDS = 15

# ---------------------------------------------------------------
# Confirmed on real robot:
#   pan:  90 = center, <90 = left, >90 = right
#   tilt: 90 = center, 60 = UP, 120 = DOWN
# ---------------------------------------------------------------
SCAN_POSITIONS = [
    ("CENTER", 90, 90),
    ("LEFT",   60, 90),
    ("RIGHT", 120, 90),
    ("UP",     90, 60),
    ("DOWN",   90, 120),
]

# Confirmed on real robot: d4=180 makes the right arm point straight.
# TODO: left arm (d8) needs separate testing -- may not be the same angle
# due to mirrored mechanics. Test Servo.setPosition(d8, ...) similarly.
ARM_POINT_ANGLE = 180
ARM_REST_DELAY_SECONDS = 3  # how long to hold the point before resting

# Fine-centering: once the object is found via the coarse scan, nudge
# the head to actually center it in the camera view, using the same
# tuned PID values from pid_simulation.py.
CENTER_TOLERANCE_PX = 20     # close enough -- stop nudging within this many pixels
MAX_CENTERING_STEPS = 5       # safety cap on how many nudges to attempt
PAN_TILT_MIN = 60              # safety range, matches earlier tested bounds
PAN_TILT_MAX = 120

# If the object consistently ends up off-center vertically after
# centering (e.g. always stays too high), flip this to -1 and test again.
# Pan direction was already confirmed correct; tilt direction for the
# fine-centering correction was not separately tested.
TILT_CORRECTION_SIGN = -1


def ensure_files_exist():
    for path in [TRIGGER_FILE, READY_FILE, SPEAK_FILE, SERVO_FILE, ARM_FILE, LED_FILE, COMMAND_FILE]:
        if not os.path.exists(path):
            with open(path, "w") as f:
                f.write("")


def flash_success_light():
    with open(LED_FILE, "w") as f:
        f.write("SUCCESS")


def move_head(pan: int, tilt: int):
    with open(SERVO_FILE, "w") as f:
        f.write(f"{pan},{tilt}")


def point_arm(side: str, angle: int):
    with open(ARM_FILE, "w") as f:
        f.write(f"{side}:{angle}")


def rest_arm(side: str):
    # TODO: replace 90 with whatever your robot's real "resting" angle is
    point_arm(side, 90)


ANNOUNCEMENTS = []  # web dashboard reads this after each command to show JD's response


def announce(text: str):
    print(f"JD: {text}")
    ANNOUNCEMENTS.append(text)
    with open(SPEAK_FILE, "w") as f:
        f.write(text)

    # Wait for ARC to pick up the message (it clears the file once
    # it starts speaking) -- prevents the next announce() call from
    # overwriting this one before ARC even sees it.
    waited = 0
    while waited < 5:
        with open(SPEAK_FILE, "r") as f:
            if f.read().strip() == "":
                break
        time.sleep(0.3)
        waited += 0.3

    # Give it extra time to actually finish speaking out loud
    # (rough estimate: ~0.55s per word, plus a flat buffer -- commas/
    # pauses make real speech slower than a raw word-count guess)
    estimated_speech_time = max(2.0, len(text.split()) * 0.55 + 0.5)
    time.sleep(estimated_speech_time)


def request_photo_and_wait() -> bool:
    with open(TRIGGER_FILE, "w") as f:
        f.write("1")
    waited = 0
    while waited < READY_TIMEOUT_SECONDS:
        if os.path.exists(READY_FILE):
            with open(READY_FILE, "r") as f:
                if f.read().strip() == "1":
                    with open(READY_FILE, "w") as f:
                        f.write("")
                    return True
        time.sleep(0.5)
        waited += 0.5
    return False


def wait_for_voice_command() -> str:
    """
    Waits for EITHER:
      - ARC's Speech Recognition writing to command.txt (voice), OR
      - You typing a command in this terminal and pressing Enter (backup)

    Whichever comes first wins. This exists because free-form speech
    recognition (Windows SAPI) can be unreliable -- this way a demo
    never gets stuck waiting on a mic that didn't catch a word.
    """
    print("\nListening... say something like 'find my bottle'")
    print("(or just type it here and press Enter)")

    typed = ""
    while True:
        # Check if ARC wrote a voice command
        with open(COMMAND_FILE, "r") as f:
            content = f.read().strip()
        if content:
            with open(COMMAND_FILE, "w") as f:
                f.write("")
            print(f"Heard: \"{content}\"")
            return content

        # Check if the user is typing (non-blocking, Windows only)
        if msvcrt.kbhit():
            ch = msvcrt.getwche()
            if ch in ("\r", "\n"):
                if typed.strip():
                    result = typed.strip()
                    print(f"\nTyped: \"{result}\"")
                    return result
                typed = ""
            elif ch == "\b":
                typed = typed[:-1]
            else:
                typed += ch

        time.sleep(0.2)


def side_for_position(position_label: str) -> str:
    """One arm (right) is used for pointing regardless of where the
    object was found -- confirmed sufficient for this robot."""
    return "RIGHT"


def search_for_object(target_label: str):
    """
    Scans through SCAN_POSITIONS looking for target_label.
    Returns the position label where it was found, or None if not found.
    """
    for label, pan, tilt in SCAN_POSITIONS:
        print(f"Scanning {label} (pan={pan}, tilt={tilt})...")
        move_head(pan, tilt)
        time.sleep(1.5)  # give the servo time to physically get there

        if not request_photo_and_wait():
            print("Timed out waiting for ARC during scan. Skipping this position.")
            continue

        detections = detect_objects(SNAPSHOT_FILE)
        print(f"  Detected: {detections if detections else 'nothing'}")

        if target_label in detections:
            return label

    return None


def fine_center_on_object(target_label: str, pan: int, tilt: int) -> tuple:
    """
    Once the object has been roughly found (via the coarse 5-position
    scan), this nudges the head with small PID-driven corrections so
    the object ends up centered in the camera view, not just "somewhere
    in frame". Returns the final (pan, tilt) used.
    """
    pid_pan = PIDController(Kp=0.4, Ki=0.002, Kd=0.3, max_correction=15)
    pid_tilt = PIDController(Kp=0.4, Ki=0.002, Kd=0.3, max_correction=15)

    for step in range(MAX_CENTERING_STEPS):
        if not request_photo_and_wait():
            print("Timed out during centering. Stopping here.")
            break

        detections, image_width, image_height = detect_with_positions(SNAPSHOT_FILE)
        matches = [d for d in detections if d["label"] == target_label]
        if not matches:
            print("Lost sight of the object while centering. Stopping here.")
            break

        target = max(matches, key=lambda d: d["confidence"])
        error_x = (image_width / 2) - target["center_x"]
        error_y = (image_height / 2) - target["center_y"]

        print(f"Centering step {step + 1}: error_x={error_x:.1f}, error_y={error_y:.1f}")

        if abs(error_x) < CENTER_TOLERANCE_PX and abs(error_y) < CENTER_TOLERANCE_PX:
            print("Object is centered.")
            break

        pan += pid_pan.step(error_x)
        tilt += TILT_CORRECTION_SIGN * pid_tilt.step(error_y)
        pan = max(PAN_TILT_MIN, min(PAN_TILT_MAX, pan))
        tilt = max(PAN_TILT_MIN, min(PAN_TILT_MAX, tilt))

        move_head(pan, tilt)
        time.sleep(1.2)  # give the servo time to physically get there

    return pan, tilt


def handle_who_is_this():
    """Takes a photo and announces the names of anyone recognized in it."""
    announce("Let me take a look.")

    if not request_photo_and_wait():
        announce("Sorry, I couldn't get a clear look right now.")
        return

    try:
        names = recognize_people(SNAPSHOT_FILE)
    except FileNotFoundError:
        announce("I haven't been trained to recognize faces yet.")
        return
    except Exception as e:
        print(f"Face recognition error: {e}")
        announce("Sorry, something went wrong trying to recognize anyone.")
        return

    known = [n for n in names if n != "Unknown"]
    unknown_count = names.count("Unknown")

    if not names:
        announce("I don't see anyone in front of me.")
    elif known and not unknown_count:
        if len(known) == 1:
            announce(f"That's {known[0]}!")
        else:
            announce("I see " + ", ".join(known[:-1]) + f" and {known[-1]}.")
    elif known and unknown_count:
        announce("I recognize " + ", ".join(known) + f", and {unknown_count} other person I don't know.")
    else:
        announce("I see someone, but I don't recognize them.")


def extract_person_name(command: str):
    """
    Checks if the command mentions a known trained person's name
    (e.g. "scan to find Ahmed" -> matches "Ahmed Ali"). Returns the
    matching name, or None if no known person is mentioned.
    """
    command_lower = command.lower()
    for name in get_known_names():
        first_word = name.split()[0].lower()
        if first_word in command_lower or name.lower() in command_lower:
            return name
    return None


def search_for_person(target_name: str):
    """
    Scans through SCAN_POSITIONS looking for target_name's face,
    same pattern as search_for_object(). Returns the position label
    where they were found, or None if not found anywhere.
    """
    for label, pan, tilt in SCAN_POSITIONS:
        print(f"Scanning {label} for {target_name} (pan={pan}, tilt={tilt})...")
        move_head(pan, tilt)
        time.sleep(1.5)

        if not request_photo_and_wait():
            print("Timed out waiting for ARC during scan. Skipping this position.")
            continue

        try:
            names = recognize_people(SNAPSHOT_FILE)
        except Exception as e:
            print(f"Face recognition error: {e}")
            continue

        print(f"  Recognized: {names if names else 'nobody'}")
        if target_name in names:
            return label

    return None


def fine_center_on_person(target_name: str, pan: int, tilt: int) -> tuple:
    """
    Same idea as fine_center_on_object, but for a recognized person's
    face -- nudges the head with PID corrections until their face is
    centered in the camera view.
    """
    pid_pan = PIDController(Kp=0.4, Ki=0.002, Kd=0.3, max_correction=15)
    pid_tilt = PIDController(Kp=0.4, Ki=0.002, Kd=0.3, max_correction=15)

    for step in range(MAX_CENTERING_STEPS):
        if not request_photo_and_wait():
            print("Timed out during centering. Stopping here.")
            break

        try:
            results, image_width, image_height = recognize_people_with_positions(SNAPSHOT_FILE)
        except Exception as e:
            print(f"Face recognition error during centering: {e}")
            break

        matches = [r for r in results if r["name"] == target_name]
        if not matches:
            print("Lost sight of the person while centering. Stopping here.")
            break

        target = matches[0]
        error_x = (image_width / 2) - target["center_x"]
        error_y = (image_height / 2) - target["center_y"]

        print(f"Centering step {step + 1}: error_x={error_x:.1f}, error_y={error_y:.1f}")

        if abs(error_x) < CENTER_TOLERANCE_PX and abs(error_y) < CENTER_TOLERANCE_PX:
            print("Person is centered.")
            break

        pan += pid_pan.step(error_x)
        tilt += TILT_CORRECTION_SIGN * pid_tilt.step(error_y)
        pan = max(PAN_TILT_MIN, min(PAN_TILT_MAX, pan))
        tilt = max(PAN_TILT_MIN, min(PAN_TILT_MAX, tilt))

        move_head(pan, tilt)
        time.sleep(1.2)

    return pan, tilt


def handle_find_person(target_name: str):
    announce(f"Please wait, I'm scanning around to find {target_name}.")

    found_at = search_for_person(target_name)

    if found_at:
        flash_success_light()

        coarse_pan, coarse_tilt = next(
            (pan, tilt) for label, pan, tilt in SCAN_POSITIONS if label == found_at
        )
        print("Fine-centering on the person...")
        fine_center_on_person(target_name, coarse_pan, coarse_tilt)

        announce(f"I found {target_name}!")
        side = side_for_position(found_at)
        point_arm(side, ARM_POINT_ANGLE)
        time.sleep(ARM_REST_DELAY_SECONDS)
        rest_arm(side)
        move_head(90, 90)
    else:
        announce(f"Sorry, I couldn't find {target_name}.")
        move_head(90, 90)


def is_who_is_this_command(command: str) -> bool:
    command = command.lower()
    return "who" in command  # e.g. "who is this", "who am I looking at"


def process_command(command: str):
    print("Thinking about what you asked...")
    result = extract_target_object(command)

    if not result["supported"]:
        announce("Sorry, I can't recognize that object yet.")
        return

    target = result["object"]
    announce(f"Please wait, I'm scanning around to find your {target}.")

    found_at = search_for_object(target)

    if found_at:
        flash_success_light()  # celebratory RGB flash

        # Get the pan/tilt that worked for the coarse position, then
        # fine-tune to actually center the object in view.
        coarse_pan, coarse_tilt = next(
            (pan, tilt) for label, pan, tilt in SCAN_POSITIONS if label == found_at
        )
        print("Fine-centering on the object...")
        fine_center_on_object(target, coarse_pan, coarse_tilt)

        # SNAPSHOT_FILE now holds the freshly centered photo
        print("Asking Gemini to describe where it is...")
        with open(SNAPSHOT_FILE, "rb") as f:
            image_bytes = f.read()
        try:
            location = describe_object_location(image_bytes, target)
            announce(f"I found your {target}! It's {location}.")
        except Exception as e:
            # No internet / Gemini quota used up -- don't say anything
            # generic, just confirm it was found without location detail.
            print(f"Location description failed ({e}), skipping location detail.")
            announce(f"I found your {target}!")

        side = side_for_position(found_at)
        point_arm(side, ARM_POINT_ANGLE)
        time.sleep(ARM_REST_DELAY_SECONDS)
        rest_arm(side)
        move_head(90, 90)  # return head to center
    else:
        announce(f"Sorry, I couldn't find your {target}.")
        move_head(90, 90)


def run_object_finder_command(command: str) -> str:
    """
    Web-friendly version of one turn of the console loop: takes ONE typed
    or spoken command, runs the same object/person/who-is-this logic as
    the console version, and returns everything JD announced during it
    (joined into one string) so the web page can show it as a chat reply.
    """
    ensure_files_exist()
    ANNOUNCEMENTS.clear()

    person_name = extract_person_name(command)
    if is_who_is_this_command(command):
        handle_who_is_this()
    elif person_name:
        handle_find_person(person_name)
    else:
        process_command(command)

    return " ".join(ANNOUNCEMENTS) if ANNOUNCEMENTS else "(JD didn't say anything for that one.)"


def run_object_finder():
    ensure_files_exist()
    print("Object Finder is ready.")

    announce("Hello everyone! I'm JD, your object-finding assistant. "
              "How can I help you find things today?")
    time.sleep(3)  # give it time to actually finish speaking before listening

    try:
        command = wait_for_voice_command()
        while True:
            person_name = extract_person_name(command)
            if is_who_is_this_command(command):
                handle_who_is_this()
            elif person_name:
                handle_find_person(person_name)
            else:
                process_command(command)
            announce("Anything else you'd like me to find?")
            command = wait_for_voice_command().lower()
            if "no" in command or "nothing" in command or "stop" in command:
                announce("You're welcome. Have a great day!")
                break
            # otherwise, treat their response as the next command directly
    except KeyboardInterrupt:
        print("\nStopped by user.")


if __name__ == "__main__":
    run_object_finder()