from fastapi import FastAPI
import socket

app = FastAPI()

JD_IP = "192.168.1.1"
JD_PORT = 23

def send_to_jd(command: str):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(3)
        s.connect((JD_IP, JD_PORT))
        s.send((command + "\r\n").encode())
        s.close()
        return {"status": "success", "command": command}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/")
def home():
    return {"message": "JD Robot Controller Running!"}

@app.get("/move/{action}")
def move(action: str):
    return send_to_jd(action)