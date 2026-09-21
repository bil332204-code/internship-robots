from flask import Flask, request, jsonify, render_template_string
import ollama
import json
import os
import random
import socket
import numpy as np
from faster_whisper import WhisperModel
import tempfile

app = Flask(__name__)

GESTURE_FILE = "gesture.txt"
SPEECH_FILE = "speech.txt"
HISTORY_FILE = "conversation_history.json"

MAX_HISTORY_MESSAGES = 50
DEBUG_MEMORY = True

FAST_MODEL = "llama3.2"
DEEP_MODEL = "llama3.1:8b"

# Loaded lazily on first use (not at import time), so importing this module
# from main_controller.py doesn't pay Whisper's load cost unless Ask JD
# (Web) is actually the mode chosen.
whisper_model = None


def load_whisper_model():
    global whisper_model
    if whisper_model is None:
        print("🎙️ Loading Whisper model (first run may take a moment)...")
        whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
        print("✅ Whisper model ready!")

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
    "explain fully", "how does it actually work",
    "how to train a model", "step by step", "math behind", "the math",
    "algorithm behind", "under the hood", "technical details",
]

SAFE_FALLBACK_RESPONSES = [
    "Hmm, I'm not confident about that one. Could you ask me something else about robotics or AI?",
    "I don't have a good answer for that right now. Try asking me about how robots work!",
    "That's a bit outside what I know well. Ask me about sensors, AI, or programming instead!",
]


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


def send_to_arc(action):
    try:
        with open(GESTURE_FILE, "w", encoding="utf-8") as f:
            f.write(action)
        print(f"✅ Sent gesture to JD: {action}")
    except IOError as e:
        print(f"⚠️ Couldn't write gesture: {e}")


def send_speech_to_arc(text):
    try:
        with open(SPEECH_FILE, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"🗣️ Sent speech to JD: {text}")
    except IOError as e:
        print(f"⚠️ Couldn't write speech: {e}")


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


def wants_deep_dive(question):
    lowered = question.lower()
    return any(trigger in lowered for trigger in DEEP_DIVE_TRIGGERS)


def is_answer_safe(answer_text, deep_dive=False):
    if not answer_text or len(answer_text.strip()) == 0:
        return False, "Answer was empty"
    if len(answer_text.strip()) < 3:
        return False, "Answer was too short (< 3 characters)"
    # Raised from 900 -> 2000 for normal answers. 900 characters was
    # rejecting entirely reasonable 3-4 sentence answers from llama3.2,
    # which were then silently replaced by a generic fallback message —
    # this is what looked like "long answer, then says it can't answer."
    max_length = 3000 if deep_dive else 2000
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
    print(f"⚠️ Safety check failed. Reason: {reason}")
    print(f"   Raw answer was: \"{raw_answer}\"")
    return random.choice(SAFE_FALLBACK_RESPONSES)


def ask_ollama_with_memory(question):
    global conversation_history

    question_to_send = expand_followup_question(question, conversation_history)
    deep_dive = wants_deep_dive(question_to_send)
    model_to_use = DEEP_MODEL if deep_dive else FAST_MODEL

    conversation_history.append({"role": "user", "content": question_to_send})

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


CHAT_PAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Voice-to-JD</title>
<style>
  :root {
    --bg: #0f172a; --panel: #1e293b; --accent: #38bdf8;
    --user-bubble: #38bdf8; --jd-bubble: #334155;
    --text: #e2e8f0; --muted: #94a3b8; --rec: #ef4444;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg); color: var(--text); display: flex; justify-content: center; height: 100vh;
  }
  .chat-container { width: 100%; max-width: 480px; display: flex; flex-direction: column; height: 100vh; }
  header { padding: 16px; background: var(--panel); border-bottom: 1px solid #334155; text-align: center; }
  header h1 { margin: 0; font-size: 18px; }
  header p { margin: 4px 0 0; font-size: 12px; color: var(--muted); }
  #messages { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 10px; }
  .bubble { max-width: 78%; padding: 10px 14px; border-radius: 16px; line-height: 1.4; font-size: 14px; white-space: pre-wrap; }
  .user { align-self: flex-end; background: var(--user-bubble); color: #04121f; border-bottom-right-radius: 4px; }
  .jd { align-self: flex-start; background: var(--jd-bubble); border-bottom-left-radius: 4px; }
  .jd.thinking { color: var(--muted); font-style: italic; }
  .controls { display: flex; padding: 16px; gap: 12px; background: var(--panel); border-top: 1px solid #334155; align-items: center; justify-content: center; }
  #mic-btn {
    width: 72px; height: 72px; border-radius: 50%; border: none; background: var(--accent);
    color: #04121f; font-size: 28px; cursor: pointer; display: flex; align-items: center;
    justify-content: center; transition: all 0.15s ease;
  }
  #mic-btn.recording { background: var(--rec); animation: pulse 1s infinite; }
  #mic-btn:disabled { opacity: 0.4; cursor: not-allowed; }
  @keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(239,68,68,0.6); }
    70% { box-shadow: 0 0 0 16px rgba(239,68,68,0); }
    100% { box-shadow: 0 0 0 0 rgba(239,68,68,0); }
  }
  #status-text { font-size: 13px; color: var(--muted); text-align: center; margin-top: 8px; min-height: 16px; }
</style>
</head>
<body>
  <div class="chat-container">
    <header>
      <h1>🤖 Voice-to-JD</h1>
      <p>Tap the mic, ask your question out loud</p>
    </header>
    <div id="messages"></div>
    <div>
      <div class="controls">
        <button id="mic-btn">🎤</button>
      </div>
      <div id="status-text">Tap the mic to start</div>
    </div>
  </div>
<script>
  const messagesEl = document.getElementById("messages");
  const micBtn = document.getElementById("mic-btn");
  const statusText = document.getElementById("status-text");

  let mediaRecorder;
  let audioChunks = [];
  let isRecording = false;

  function addBubble(text, sender) {
    const bubble = document.createElement("div");
    bubble.className = "bubble " + sender;
    bubble.textContent = text;
    messagesEl.appendChild(bubble);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return bubble;
  }

  async function startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(stream);
      audioChunks = [];

      mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data);
      mediaRecorder.onstop = handleRecordingStop;

      mediaRecorder.start();
      isRecording = true;
      micBtn.classList.add("recording");
      micBtn.textContent = "⏹️";
      statusText.textContent = "Listening... tap again to stop";
    } catch (err) {
      statusText.textContent = "Couldn't access microphone: " + err.message;
    }
  }

  function stopRecording() {
    if (mediaRecorder && isRecording) {
      mediaRecorder.stop();
      mediaRecorder.stream.getTracks().forEach(track => track.stop());
      isRecording = false;
      micBtn.classList.remove("recording");
      micBtn.textContent = "🎤";
    }
  }

  async function handleRecordingStop() {
    micBtn.disabled = true;
    statusText.textContent = "Transcribing...";

    const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
    const formData = new FormData();
    formData.append("audio", audioBlob, "question.webm");

    try {
      const transcribeRes = await fetch("/transcribe", {
        method: "POST",
        body: formData,
      });
      const transcribeData = await transcribeRes.json();

      if (transcribeData.error || !transcribeData.text) {
        statusText.textContent = "Couldn't understand that, try again.";
        micBtn.disabled = false;
        return;
      }

      const question = transcribeData.text;
      addBubble(question, "user");
      statusText.textContent = "JD is thinking...";
      const thinkingBubble = addBubble("JD is thinking...", "jd thinking");

      const askRes = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      const askData = await askRes.json();
      thinkingBubble.remove();

      if (askData.error) {
        addBubble("Something went wrong: " + askData.error, "jd");
      } else {
        addBubble(askData.answer, "jd");
      }
      statusText.textContent = "Tap the mic to ask again";

    } catch (err) {
      statusText.textContent = "Error: " + err.message;
    } finally {
      micBtn.disabled = false;
    }
  }

  micBtn.addEventListener("click", () => {
    if (!isRecording) {
      startRecording();
    } else {
      stopRecording();
    }
  });

  addBubble("Hi! I'm JD. Tap the mic and ask me anything about robotics, AI, or tech.", "jd");
</script>
</body>
</html>
"""


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "unknown — run 'ipconfig' (Windows) to find it manually"


@app.route("/")
def index():
    return render_template_string(CHAT_PAGE_HTML)


@app.route("/transcribe", methods=["POST"])
def transcribe():
    """
    Receives an audio file recorded by the browser's microphone,
    saves it temporarily, runs Whisper on it, and returns the text.
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio file received"}), 400

    audio_file = request.files["audio"]

    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        segments, info = whisper_model.transcribe(tmp_path, beam_size=5, vad_filter=True)
        text = " ".join([segment.text for segment in segments]).strip()
        print(f"🎙️ Transcribed: \"{text}\"")
        return jsonify({"text": text})
    except Exception as e:
        print(f"⚠️ Transcription error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "Empty question"}), 400

    print(f"💬 Voice question received: {question}")
    answer = ask_ollama_with_memory(question)
    reaction = pick_reaction(answer)

    send_to_arc(reaction)
    send_speech_to_arc(answer)

    print(f"JD says: {answer}")
    return jsonify({"answer": answer, "reaction": reaction})


def run_ask_jd_web():
    load_whisper_model()
    local_ip = get_local_ip()
    print("🤖 Voice-to-JD is starting...")
    print("💻 On THIS laptop, open:      http://localhost:5000")
    print(f"💻 On a PHONE (same WiFi):    http://{local_ip}:5000")
    print("⚠️ Note: microphone access in browsers usually requires HTTPS or 'localhost' —")
    print("   if using the phone URL and the mic button doesn't work, that's likely why.")
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    run_ask_jd_web()