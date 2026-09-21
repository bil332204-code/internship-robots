"""
main_controller.py — JD Unified Control Center

Single entry point for every mode in the project. Picks up:
  - Gesture / Emotion / Pose / Speech control (hand signals, facial
    expression, body pose, voice commands -> JD performs an action)
  - Ask JD (voice, console)      -- talk to JD, get spoken AI answers
  - Ask JD (voice, web browser)  -- same, but from a phone/laptop browser
  - Find Object / Who Is This    -- JD scans the room, finds objects or
                                     people using its camera + AI vision
  - Train New Faces              -- one-time setup so JD can recognize
                                     new people

Each option's heavy dependencies (Whisper, YOLO, DeepFace, Gemini, Flask)
are imported ONLY when that option is actually chosen -- picking Gesture
Mode, for example, never loads Whisper or YOLO in the background, so
startup stays fast regardless of which mode you pick.
"""

MENU = """
==================================================
   🤖  JD ROBOT — UNIFIED CONTROL CENTER
==================================================
  Movement & Reaction Modes
  [1] Gesture Control       🖐️   hand signals -> action
  [2] Emotion Control       😊   facial expression -> action
  [3] Pose Control          🕺   body pose -> action
  [4] Speech Control        🎤   voice command -> action

  Ask JD (conversational AI)
  [5] Ask JD — Voice        🎙️   speak into your mic (console)
  [6] Ask JD — Web Chat     🌐   open a browser, mic or type

  Vision & People
  [7] Find Object           🔍   "find my bottle", "who is this"
  [8] Train New Faces       👤   teach JD to recognize someone new

  [0] Exit
==================================================
"""


def main():
    while True:
        print(MENU)
        choice = input("Choose an option (0-8): ").strip()

        if choice == "1":
            from gesture_control import run_gesture
            run_gesture()

        elif choice == "2":
            from emotion_control import run_emotion
            run_emotion()

        elif choice == "3":
            from pose_control import run_pose
            run_pose()

        elif choice == "4":
            from speech_control import run_speech_control
            run_speech_control()

        elif choice == "5":
            from ask_jd import run_ask_jd
            run_ask_jd()

        elif choice == "6":
            from voice_jd import run_ask_jd_web
            run_ask_jd_web()

        elif choice == "7":
            from object_finder import run_object_finder
            run_object_finder()

        elif choice == "8":
            from face_trainer import run_face_trainer
            run_face_trainer()

        elif choice == "0" or choice.lower() == "q":
            print("👋 Goodbye!")
            break

        else:
            print("⚠️ Invalid choice, please enter a number from 0-8.")

        print("\n" + "=" * 50)
        input("Press Enter to return to the menu...")


if __name__ == "__main__":
    main()
