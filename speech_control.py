import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import time

GESTURE_FILE = "gesture.txt"

# Loaded lazily on first use (not at import time), so the web dashboard
# can share ONE Whisper instance across Ask JD, Speech Control, etc.
# instead of loading it multiple times.
whisper_model = None


def load_whisper_model():
    global whisper_model
    if whisper_model is None:
        print("🎙️ Loading Whisper model (first run may take a moment)...")
        whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
        print("✅ Whisper model ready!")

# Every confirmed valid AutoPositionAction name, mapped to spoken keywords
# a person would naturally say to trigger it.
commands = {
    "bow": "Bow",
    "disco": "Disco Dance",
    "dance": "Disco Dance",
    "fly": "Fly",
    "forward": "Forward",
    "get up": "Getup",
    "getup": "Getup",
    "gorilla": "Gorilla",
    "grab": "Grab",
    "hands dance": "Hands Dance",
    "happy hands": "Happy Hands",
    "happy": "Happy Hands",
    "head bob feet": "Head Bob Feet",
    "head bob": "Head Bob",
    "headstand": "Headstand",
    "hand stand": "Headstand",
    "jump jack": "Jump Jack",
    "jumping jack": "Jump Jack",
    "jump": "Jump Jack",
    "kick": "Kick",
    "go left": "Left",
    "turn left": "Left",
    "lunge": "Lunge Singing",
    "pass the mic": "Pass the Mic",
    "point": "Point",
    "predance": "Predance",
    "pushup": "Pushups",
    "push up": "Pushups",
    "pushups": "Pushups",
    "reverse": "Reverse",
    "go back": "Reverse",
    "go right": "Right",
    "turn right": "Right",
    "roll hands": "Roll Hands",
    "shimmy": "Shimmy",
    "sing": "Singing",
    "singing hands in": "Singing Hands In",
    "singing with hands": "Singing with Hands",
    "sit down": "Sit Down",
    "sit wave": "Sit Wave",
    "situp": "Situps",
    "sit up": "Situps",
    "situps": "Situps",
    "split": "Splits",
    "splits": "Splits",
    "stand up": "Stand From Sit",
    "stand from sit": "Stand From Sit",
    "stop": "Stop",
    "somersault": "Summersault",
    "summersault": "Summersault",
    "flip": "Summersault",
    "think": "Thinking",
    "thinking": "Thinking",
    "throw the mic": "Throw Mic",
    "throw mic": "Throw Mic",
    "wave": "Wave",
    "ymca dance": "YMCA Dance",
    "ymca march": "YMCA March",
    "ymca": "YMCA Dance",
}

# Sort keywords longest-first, so multi-word phrases like "jump jack"
# get matched before the shorter, more generic "jump" would steal the match
sorted_commands = sorted(commands.items(), key=lambda x: len(x[0]), reverse=True)


def send_to_arc(action):
    with open(GESTURE_FILE, "w", encoding="utf-8") as f:
        f.write(action)
    print(f"✅ Sent to JD: {action}")


def calibrate_silence_threshold(samplerate=16000, calibration_duration=1.5):
    print("🔧 Calibrating microphone... stay quiet for a moment.")
    chunk_duration = 0.1
    chunk_samples = int(chunk_duration * samplerate)
    num_chunks = int(calibration_duration / chunk_duration)

    stream = sd.InputStream(samplerate=samplerate, channels=1, dtype='int16')
    stream.start()

    volumes = []
    for _ in range(num_chunks):
        chunk, _ = stream.read(chunk_samples)
        volumes.append(np.abs(chunk).mean())

    stream.stop()
    stream.close()

    ambient_level = np.mean(volumes)
    threshold = ambient_level * 3 + 100
    print(f"✅ Calibration done. Threshold set to: {threshold:.0f}")
    return threshold


def record_until_silence(samplerate=16000, silence_threshold=None,
                          silence_duration=1.2, max_duration=8):
    if silence_threshold is None:
        silence_threshold = SILENCE_THRESHOLD

    print("🎤 Listening for a command...")
    chunk_duration = 0.1
    chunk_samples = int(chunk_duration * samplerate)
    recorded_chunks = []
    silent_chunks_needed = int(silence_duration / chunk_duration)
    silent_chunk_count = 0
    has_spoken = False

    stream = sd.InputStream(samplerate=samplerate, channels=1, dtype='int16')
    stream.start()
    start_time = time.time()

    while True:
        chunk, _ = stream.read(chunk_samples)
        recorded_chunks.append(chunk.copy())
        volume = np.abs(chunk).mean()

        if volume > silence_threshold:
            has_spoken = True
            silent_chunk_count = 0
        else:
            silent_chunk_count += 1

        if has_spoken and silent_chunk_count >= silent_chunks_needed:
            break
        if time.time() - start_time > max_duration:
            break

    stream.stop()
    stream.close()
    full_audio = np.concatenate(recorded_chunks, axis=0)
    return full_audio, samplerate


def transcribe_audio(audio_data, samplerate):
    audio_float = audio_data.astype(np.float32) / 32768.0
    audio_flat = audio_float.flatten()

    segments, info = whisper_model.transcribe(audio_flat, beam_size=5, vad_filter=True)
    text = " ".join([segment.text for segment in segments]).strip()

    if len(text) == 0:
        return None
    return text


def match_command(spoken_text):
    lowered = spoken_text.lower()
    for keyword, action in sorted_commands:
        if keyword in lowered:
            return action
    return None


def run_speech_control():
    global SILENCE_THRESHOLD
    load_whisper_model()
    SILENCE_THRESHOLD = calibrate_silence_threshold()

    print(f"\n🤖 Speech Control Active! {len(set(commands.values()))} actions available.")
    print("Say a command, or 'exit' to quit.\n")

    while True:
        try:
            audio_data, samplerate = record_until_silence()
            spoken_text = transcribe_audio(audio_data, samplerate)

            if spoken_text is None:
                print("❌ Didn't catch that, try again...")
                continue

            print(f"Heard: \"{spoken_text}\"")

            if "exit" in spoken_text.lower():
                print("Stopping speech control...")
                break

            action = match_command(spoken_text)
            if action:
                send_to_arc(action)
            else:
                print("⚠️ No matching command recognized. Try again.")

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    run_speech_control()