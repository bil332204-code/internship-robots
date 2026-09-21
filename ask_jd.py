import sounddevice as sd
import numpy as np
import ollama
import time
import json
import os
import random
from faster_whisper import WhisperModel

GESTURE_FILE = "gesture.txt"
SPEECH_FILE = "speech.txt"
HISTORY_FILE = "conversation_history.json"

MAX_HISTORY_MESSAGES = 50
DEBUG_MEMORY = True

FAST_MODEL = "llama3.2"
DEEP_MODEL = "llama3.1:8b"

# ─────────────────────────────────────────────
# Whisper setup — loaded lazily on first use (not at import time), so
# importing this module from main_controller.py doesn't pay Whisper's
# load cost unless Ask JD (Voice) is actually the mode chosen.
# ─────────────────────────────────────────────
whisper_model = None


def load_whisper_model():
    global whisper_model
    if whisper_model is None:
        print("🎙️ Loading Whisper model (first run may take a moment)...")
        whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
        print("✅ Whisper model ready!")

# ─────────────────────────────────────────────
# System prompt — subject restriction + followup awareness
# ─────────────────────────────────────────────
SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are JD, a friendly robot assistant built for a robotics internship demo. "
        "You ONLY answer questions about robotics, artificial intelligence, computer science, "
        "and technology topics (e.g. how robots work, sensors, AI concepts, programming, "
        "electronics, machine learning basics). "
        "Keep answers short, 1-3 sentences, simple and clear, suitable for a student audience. "
        "If a question is NOT about robotics, AI, or technology, politely say you can only "
        "discuss robotics and tech topics, and ask if they have a question about that instead. "
        "Do not answer questions about unrelated topics like politics, medical advice, or "
        "personal opinions on controversial subjects. "
        "IMPORTANT: Many user messages are short follow-ups like 'explain further', 'why', "
        "'go on', 'tell me more', 'give an example', or 'what do you mean'. These are NOT "
        "off-topic just because they don't repeat robotics/AI/tech words — always check what "
        "the CURRENT topic of conversation already is (from the recent messages above) before "
        "deciding whether to refuse. Only refuse if the conversation topic itself, not just the "
        "current message's wording, is unrelated to robotics/AI/tech. "
        "You DO have access to the full conversation history above — always check it before "
        "answering questions about what was said earlier."
    )
}

DEEP_DIVE_INSTRUCTION = (
    "The student is asking for a DEEP, THOROUGH explanation this time. "
    "Give a detailed, multi-paragraph answer covering the underlying concepts, "
    "how it works step by step, and key terminology — still clear and educational, "
    "but do not artificially shorten it. Aim for genuine depth, like explaining "
    "to a computer science student who wants to really understand the topic."
)

STRONG_FOLLOWUP_PHRASES = [
    "explain further", "explain more", "tell me more", "go on", "continue",
    "elaborate", "give an example", "example", "what do you mean",
    "more detail", "more details", "can you explain", "in more detail",
    "expand on that", "say more",
    "explain that", "explain this", "explain it", "explain that thing",
    "explain this thing", "what about that", "what about it",
    "tell me about that", "tell me about it",
]
STANDALONE_ONLY_WORDS = ["why", "how", "why is that", "that", "this", "it"]

DEEP_DIVE_TRIGGERS = [
    "in depth", "in detail", "deeply", "thoroughly", "full explanation",
    "explain fully", "everything about", "how does it actually work",
    "train it", "how to train", "step by step", "math behind", "the math",
    "algorithm behind", "under the hood", "technical details",
]

SAFE_FALLBACK_RESPONSES = [
    "Hmm, I'm not confident about that one. Could you ask me something else about robotics or AI?",
    "I don't have a good answer for that right now. Try asking me about how robots work!",
    "That's a bit outside what I know well. Ask me about sensors, AI, or programming instead!",
]

# ─────────────────────────────────────────────
# Persistent memory (survives restarts)
# ─────────────────────────────────────────────
def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
            if isinstance(history, list) and len(history) > 0:
                print(f"📂 Loaded {len(history)} messages from previous sessions.")
                history[0] = SYSTEM_PROMPT
                return history
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️ Couldn't load history file ({e}), starting fresh.")
    print("📂 No previous history found, starting a new conversation.")
    return [SYSTEM_PROMPT]

def save_history(history):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except IOError as e:
        print(f"⚠️ Couldn't save history: {e}")

conversation_history = load_history()

# ─────────────────────────────────────────────
# Robot control — gesture + speech file bridge
# ─────────────────────────────────────────────
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
    if any(word in text for word in ["sorry", "don't know", "not sure", "cannot", "outside what i know"]):
        return "Thinking"
    elif any(word in text for word in ["great", "yes", "awesome", "cool", "happy"]):
        return "Happy Hands"
    elif "?" in text:
        return "Point"
    else:
        return "Wave"

# ─────────────────────────────────────────────
# Follow-up question detection & expansion
# ─────────────────────────────────────────────
def get_last_topic(history):
    for msg in reversed(history):
        if msg["role"] == "user":
            return msg["content"]
    return None

def expand_followup_question(question, history):
    cleaned = question.strip().lower().rstrip(".!?")
    word_count = len(cleaned.split())

    filler_prefixes = ["i said ", "please ", "can you ", "could you ", "so "]
    stripped = cleaned
    for filler in filler_prefixes:
        if stripped.startswith(filler):
            stripped = stripped[len(filler):]

    trailing_fillers = [" thing", " to me", " for me"]
    for trailing in trailing_fillers:
        if stripped.endswith(trailing):
            stripped = stripped[: -len(trailing)]

    matches_strong_phrase = (
        word_count <= 8
        and any(stripped == phrase or stripped.startswith(phrase) for phrase in STRONG_FOLLOWUP_PHRASES)
    )
    matches_standalone_word = (
        word_count <= 2 and stripped in STANDALONE_ONLY_WORDS
    )

    is_short_followup = matches_strong_phrase or matches_standalone_word

    if not is_short_followup:
        return question

    last_topic = get_last_topic(history)
    if not last_topic:
        return question

    expanded = f"Regarding my previous question '{last_topic}', please {question.lower()}."

    if DEBUG_MEMORY:
        print(f"🔗 DEBUG — expanded follow-up: \"{question}\" -> \"{expanded}\"")

    return expanded

# ─────────────────────────────────────────────
# Deep-dive detection
# ─────────────────────────────────────────────
def wants_deep_dive(question):
    lowered = question.lower()
    return any(trigger in lowered for trigger in DEEP_DIVE_TRIGGERS)

# ─────────────────────────────────────────────
# Safety fallback check — with debug visibility + deep-dive aware length limit
# ─────────────────────────────────────────────
def is_answer_safe(answer_text, deep_dive=False):
    if not answer_text or len(answer_text.strip()) == 0:
        return False, "Answer was empty"
    if len(answer_text.strip()) < 3:
        return False, "Answer was too short (< 3 characters)"
    max_length = 6000 if deep_dive else 2000
    if len(answer_text) > max_length:
        return False, f"Answer was too long ({len(answer_text)} characters, limit is {max_length})"
    confusion_markers = ["i am an ai", "as a language model", "i cannot browse",
                          "i don't have access to real-time"]
    lowered = answer_text.lower()
    for marker in confusion_markers:
        if marker in lowered:
            return False, f"Answer contained confusion marker phrase: \"{marker}\""
    return True, None

def get_safe_answer(raw_answer, deep_dive=False):
    safe, reason = is_answer_safe(raw_answer, deep_dive=deep_dive)
    if safe:
        return raw_answer
    else:
        print(f"⚠️ Safety check failed. Reason: {reason}")
        print(f"   Raw answer was: \"{raw_answer}\"")
        print(f"   Length: {len(raw_answer)} characters")
        return random.choice(SAFE_FALLBACK_RESPONSES)

# ─────────────────────────────────────────────
# Microphone calibration
# ─────────────────────────────────────────────
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
    print(f"✅ Calibration done. Ambient level: {ambient_level:.0f}, threshold set to: {threshold:.0f}")
    return threshold

# ─────────────────────────────────────────────
# Audio recording with silence detection
# ─────────────────────────────────────────────
def record_until_silence(samplerate=16000, silence_threshold=None,
                          silence_duration=1.5, max_duration=15):
    if silence_threshold is None:
        silence_threshold = SILENCE_THRESHOLD

    print("🎤 Listening... (speak now, I'll stop when you pause)")
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
            print("⏱️ Max recording time reached.")
            break

    stream.stop()
    stream.close()
    full_audio = np.concatenate(recorded_chunks, axis=0)
    return full_audio, samplerate

def transcribe_audio(audio_data, samplerate):
    audio_float = audio_data.astype(np.float32) / 32768.0
    audio_flat = audio_float.flatten()

    segments, info = whisper_model.transcribe(
        audio_flat,
        beam_size=5,
        vad_filter=True,
    )

    text = " ".join([segment.text for segment in segments]).strip()

    if len(text) == 0:
        return None
    return text

# ─────────────────────────────────────────────
# Question confirmation flow
# ─────────────────────────────────────────────
def get_confirmed_question():
    while True:
        audio_data, samplerate = record_until_silence()
        question = transcribe_audio(audio_data, samplerate)

        if question is None:
            print("❌ Sorry, I couldn't understand that. Let's try again.")
            continue

        print(f"\n📝 I heard: \"{question}\"")
        print("Say 'yes' to confirm, or 'no' to try again...")

        confirm_audio, confirm_rate = record_until_silence(silence_duration=1.0, max_duration=5)
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

# ─────────────────────────────────────────────
# Ollama call with memory + followup expansion + deep-dive + safety check
# ─────────────────────────────────────────────
def ask_ollama_with_memory(question):
    global conversation_history

    question_to_send = expand_followup_question(question, conversation_history)
    deep_dive = wants_deep_dive(question_to_send)
    model_to_use = DEEP_MODEL if deep_dive else FAST_MODEL

    conversation_history.append({"role": "user", "content": question_to_send})

    # Build the message list, using a temporarily adjusted system prompt
    # for deep-dive questions, without permanently altering saved history
    messages_to_send = list(conversation_history)

    if deep_dive:
        deep_system = dict(SYSTEM_PROMPT)
        deep_system["content"] = SYSTEM_PROMPT["content"].replace(
            "Keep answers short, 1-3 sentences, simple and clear, suitable for a student audience.",
            DEEP_DIVE_INSTRUCTION
        )
        messages_to_send[0] = deep_system
        print(f"🔬 DEBUG — Deep-dive mode activated. Using model: {model_to_use}")

    if DEBUG_MEMORY:
        print(f"\n📚 DEBUG — sending {len(messages_to_send)} messages to Ollama ({model_to_use}):")
        for i, msg in enumerate(messages_to_send):
            preview = msg["content"][:70].replace("\n", " ")
            print(f"   [{i}] {msg['role']}: {preview}")

    response = ollama.chat(model=model_to_use, messages=messages_to_send)
    raw_answer = response['message']['content']
    answer = get_safe_answer(raw_answer, deep_dive=deep_dive)

    conversation_history.append({"role": "assistant", "content": answer})

    if len(conversation_history) > MAX_HISTORY_MESSAGES:
        conversation_history = [conversation_history[0]] + conversation_history[-(MAX_HISTORY_MESSAGES - 1):]

    save_history(conversation_history)
    return answer

# ─────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────
def run_ask_jd():
    global SILENCE_THRESHOLD
    load_whisper_model()
    SILENCE_THRESHOLD = calibrate_silence_threshold()

    print("🤖 Ask JD Mode Active! JD only discusses robotics/AI/tech topics, and remembers this conversation.")
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