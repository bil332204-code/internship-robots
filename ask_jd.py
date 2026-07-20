import speech_recognition as sr
import sounddevice as sd
import numpy as np
import ollama
import time
import json
import os

GESTURE_FILE = "gesture.txt"
SPEECH_FILE = "speech.txt"
HISTORY_FILE = "conversation_history.json"

MAX_HISTORY_MESSAGES = 50  # keep memory from growing forever (across restarts too)

DEBUG_MEMORY = True  # set to False once you've confirmed memory works

SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are JD, a friendly helpful robot assistant. Keep answers short, "
        "1-3 sentences, simple and clear. You DO have access to the full "
        "conversation history above — always check it before answering "
        "questions about what was said earlier, and use it naturally when "
        "relevant."
    )
}


def load_history():
    """
    Loads conversation_history from disk if it exists, so JD remembers
    past sessions even after the script is closed and reopened.
    Falls back to a fresh history (just the system prompt) if the file
    doesn't exist yet or is corrupted.
    """
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
            if isinstance(history, list) and len(history) > 0:
                print(f"📂 Loaded {len(history)} messages from previous sessions.")
                return history
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️ Couldn't load history file ({e}), starting fresh.")

    print("📂 No previous history found, starting a new conversation.")
    return [SYSTEM_PROMPT]


def save_history(history):
    """Writes conversation_history to disk after every exchange."""
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except IOError as e:
        print(f"⚠️ Couldn't save history: {e}")


# Conversation memory — loaded from disk on startup, persists across sessions
conversation_history = load_history()


def send_to_arc(action):
    with open(GESTURE_FILE, "w", encoding="utf-8") as f:
        f.write(action)
    print(f"✅ Sent gesture to JD: {action}")


def send_speech_to_arc(text):
    with open(SPEECH_FILE, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"🗣️ Sent speech to JD: {text}")


def pick_reaction(answer_text):
    text = answer_text.lower()
    if any(word in text for word in ["sorry", "don't know", "not sure", "cannot"]):
        return "Thinking"
    elif any(word in text for word in ["great", "yes", "awesome", "cool", "happy"]):
        return "Happy Hands"
    elif "?" in text:
        return "Point"
    else:
        return "Wave"


def record_until_silence(samplerate=16000, silence_threshold=500,
                          silence_duration=1.5, max_duration=15):
    """
    Records audio and automatically stops when silence is detected.
    - silence_threshold: how quiet counts as 'silence' (lower = more sensitive)
    - silence_duration: how many seconds of silence before stopping
    - max_duration: absolute maximum recording length as a safety limit
    """
    print("🎤 Listening... (speak now, I'll stop when you pause)")

    chunk_duration = 0.1  # process in 100ms chunks
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

        # Stop if we've heard speech AND then enough silence
        if has_spoken and silent_chunk_count >= silent_chunks_needed:
            break

        # Safety cutoff so it never listens forever
        if time.time() - start_time > max_duration:
            print("⏱️ Max recording time reached.")
            break

    stream.stop()
    stream.close()

    full_audio = np.concatenate(recorded_chunks, axis=0)
    return full_audio, samplerate


def transcribe_audio(audio_data, samplerate):
    recognizer = sr.Recognizer()
    audio = sr.AudioData(audio_data.tobytes(), samplerate, 2)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return None


def get_confirmed_question():
    """
    Records a question, transcribes it, and asks for confirmation.
    Returns the confirmed question text, or None if user wants to cancel.
    """
    while True:
        audio_data, samplerate = record_until_silence()
        question = transcribe_audio(audio_data, samplerate)

        if question is None:
            print("❌ Sorry, I couldn't understand that. Let's try again.")
            continue

        print(f"\n📝 I heard: \"{question}\"")
        print("Say 'yes' to confirm, or 'no' to try again...")

        # Shorter silence_duration here since "yes"/"no" is a quick word,
        # not a full sentence.
        confirm_audio, confirm_rate = record_until_silence(
            silence_duration=1.0, max_duration=5
        )
        confirmation = transcribe_audio(confirm_audio, confirm_rate)

        if confirmation is None:
            print("Didn't catch that, let's try the question again.")
            continue

        confirmation = confirmation.lower()
        if "yes" in confirmation or "correct" in confirmation or "right" in confirmation:
            return question
        elif "no" in confirmation or "wrong" in confirmation or "repeat" in confirmation:
            print("🔄 Okay, let's try again...")
            continue
        else:
            print("I didn't understand your confirmation, let's try the question again.")
            continue


def ask_ollama_with_memory(question):
    global conversation_history

    conversation_history.append({"role": "user", "content": question})

    if DEBUG_MEMORY:
        print(f"\n📚 DEBUG — sending {len(conversation_history)} messages to Ollama:")
        for i, msg in enumerate(conversation_history):
            preview = msg["content"][:70].replace("\n", " ")
            print(f"   [{i}] {msg['role']}: {preview}")

    response = ollama.chat(
        model="llama3.2",
        messages=conversation_history
    )
    answer = response['message']['content']

    conversation_history.append({"role": "assistant", "content": answer})

    # Trim history so it doesn't grow forever (keep system prompt + most recent)
    if len(conversation_history) > MAX_HISTORY_MESSAGES:
        conversation_history = [conversation_history[0]] + conversation_history[-(MAX_HISTORY_MESSAGES - 1):]

    # Persist to disk after every exchange so a crash/close doesn't lose it
    save_history(conversation_history)

    return answer


def run_ask_jd():
    print("🤖 Ask JD Mode Active! JD will remember this conversation — even after restart.")
    print("Ask JD any question out loud. Say 'exit' during confirmation to quit anytime.")

    while True:
        try:
            question = get_confirmed_question()

            if question is None:
                continue

            if "exit" in question.lower():
                print("Stopping Ask JD mode... (history saved for next time)")
                break

            print(f"\n✅ Confirmed question: {question}")
            print("🧠 JD is thinking...")

            answer = ask_ollama_with_memory(question)
            print(f"JD says: {answer}")

            reaction = pick_reaction(answer)
            send_to_arc(reaction)
            send_speech_to_arc(answer)

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    run_ask_jd()