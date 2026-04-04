import cv2
import numpy as np
import tempfile
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

import supervision as sv
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from mediapipe.python.solutions.pose import Pose

pose = Pose(min_detection_confidence=0.5)

# MODELS

from inference import get_model
model2 = get_model("rfdetr-medium")

# BI-LSTM MODEL

class LSTMModel(nn.Module):
    def __init__(self, input_size=12, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        h = torch.cat((h[-2], h[-1]), dim=1)
        return self.fc(h).squeeze()

lstm_model = LSTMModel()
lstm_model.load_state_dict(torch.load("Path_to_Model.pth", map_location="cpu"))
lstm_model.eval()

# GLOBALS

cap = cv2.VideoCapture(0)
MAX_SEQ_LEN = 150
CONFIDENCE_THRESHOLD = 0.3
NMS_THRESHOLD = 0.3
PAD = 30

left_knee_angles, right_knee_angles = [], []
left_ankle_angles, right_ankle_angles = [], []
risk_history = []
MAX_HISTORY = 30

# HELPERS

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def build_sequence():
    if len(left_knee_angles) < 2:
        return None

    lk = np.array(left_knee_angles)
    rk = np.array(right_knee_angles)
    la = np.array(left_ankle_angles)
    ra = np.array(right_ankle_angles)

    kc = np.gradient(np.stack([lk, rk], axis=1), axis=0)
    ac = np.gradient(np.stack([la, ra], axis=1), axis=0)

    seq_base = np.stack([lk, rk, la, ra, kc[:,0], ac[:,0]], axis=1)
    vel = np.gradient(seq_base, axis=0)
    seq = np.concatenate([seq_base, vel], axis=1)

    seq = (seq - seq.mean(axis=0)) / (seq.std(axis=0) + 1e-6)

    length = len(seq)
    if length >= MAX_SEQ_LEN:
        seq = seq[-MAX_SEQ_LEN:]
        length = MAX_SEQ_LEN
    else:
        pad_arr = np.zeros((MAX_SEQ_LEN - length, seq.shape[1]))
        seq = np.vstack([pad_arr, seq])

    return seq, length

def aggregate_results(clip_risks, clip_duration=20):
    if len(clip_risks) == 0:
        return {"risk":0.0, "overall_risk":0.0}
    clip_risks = np.array(clip_risks)
    overall_risk = float(np.mean(clip_risks))
    return {
        "risk": round(overall_risk,3),
        "overall_risk": round(overall_risk,3),
    }

# REAL-TIME CALLBACK

def callback(frame, i):
    result2 = model2.infer(frame, confidence=CONFIDENCE_THRESHOLD)[0]
    detections = sv.Detections.from_inference(result2).with_nms(threshold=NMS_THRESHOLD)
    person_mask = detections.data["class_name"] == "person"
    detections = detections[person_mask]

    annotated_image = frame.copy()
    if len(detections) == 0:
        return annotated_image

    areas = [(x2-x1)*(y2-y1) for x1,y1,x2,y2 in detections.xyxy]
    idx = np.argmax(areas)
    x1, y1, x2, y2 = map(int, detections.xyxy[idx])
    h, w, _ = frame.shape
    x1, y1 = max(0,x1-PAD), max(0,y1-PAD)
    x2, y2 = min(w,x2+PAD), min(h,y2+PAD)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return annotated_image

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    results_pose = pose.process(rgb)

    if results_pose.pose_landmarks:
        h_crop, w_crop, _ = crop.shape
        def get(idx): return np.array([results_pose.pose_landmarks.landmark[idx].x*w_crop,
                                        results_pose.pose_landmarks.landmark[idx].y*h_crop])
        try:
            hip_l,knee_l,ankle_l,foot_l = get(23), get(25), get(27), get(31)
            hip_r,knee_r,ankle_r,foot_r = get(24), get(26), get(28), get(32)
            left_knee_angles.append(calculate_angle(hip_l,knee_l,ankle_l))
            right_knee_angles.append(calculate_angle(hip_r,knee_r,ankle_r))
            left_ankle_angles.append(calculate_angle(knee_l,ankle_l,foot_l))
            right_ankle_angles.append(calculate_angle(knee_r,ankle_r,foot_r))
            if len(left_knee_angles)>MAX_SEQ_LEN:
                left_knee_angles.pop(0); right_knee_angles.pop(0)
                left_ankle_angles.pop(0); right_ankle_angles.pop(0)
        except: pass

    seq_data = build_sequence()
    if seq_data:
        seq, length = seq_data
        seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
        lengths_tensor = torch.tensor([length], dtype=torch.int64)
        with torch.no_grad():
            risk = float(torch.sigmoid(lstm_model(seq_tensor, lengths_tensor)).item())
        risk_history.append(risk)
        if len(risk_history) > MAX_HISTORY:
            risk_history.pop(0)
        
        # Overlay risk on the frame
        cv2.putText(annotated_image, f"Risk: {risk:.2f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    return annotated_image

# VIDEO STREAM GENERATOR

def generate_frames():
    i=0
    while True:
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.resize(frame, (640,480))
        try: frame = callback(frame,i)
        except Exception as e: print("Error:", e)
        i+=1
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+buffer.tobytes()+b'\r\n')

# FASTAPI SETUP

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

# Live feed 
@app.get("/live_feed")
def live_feed():
    return StreamingResponse(generate_frames(),
                             media_type="multipart/x-mixed-replace; boundary=frame")

# Live risk JSON 
@app.get("/live_risk")
def live_risk():
    if len(risk_history)==0: return {"status":"warming_up"}
    return aggregate_results(risk_history)

# Recorded video 
@app.post("/predict_video")
async def predict_video(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False,suffix=".mp4") as temp:
        temp.write(await file.read()); path=temp.name
    return predict_video_global(path)

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head>
        <title>Upload Video for Risk Prediction</title>
    </head>
    <body>
        <h2>Upload a video to get DVT risk score</h2>
        <form action="/predict_video" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="video/*" required>
            <input type="submit" value="Upload & Predict">
        </form>
        <hr>
        <h2>Live Monitoring</h2>
        <img src="/live_feed" width="640" height="480" id="live_video">
        <p>Live Risk: <span id="risk_value">warming up...</span></p>
        <script>
        setInterval(async () => {
            const res = await fetch('/live_risk');
            const data = await res.json();
            document.getElementById('risk_value').innerText = data.risk?.toFixed(2) ?? 'warming up...';
        }, 1000);
        </script>
    </body>
    </html>
    """

# VIDEO PROCESSING FUNCTIONS

def split_video_to_clips(video_path,duration=20):
    cap=cv2.VideoCapture(video_path)
    fps=cap.get(cv2.CAP_PROP_FPS)
    clips,frames=[],[]
    while True:
        ret, frame=cap.read()
        if not ret: break
        frames.append(frame)
        if len(frames)>=int(fps*duration):
            clips.append(frames); frames=[]
    if frames: clips.append(frames)
    cap.release(); return clips

def process_clip(frames):
    lk,rk,la,ra=[],[],[],[]
    for frame in frames:
        result2=model2.infer(frame, confidence=CONFIDENCE_THRESHOLD)[0]
        detections=sv.Detections.from_inference(result2)
        person_mask=detections.data["class_name"]=="person"
        detections=detections[person_mask]
        if len(detections)==0: continue
        areas=[(x2-x1)*(y2-y1) for x1,y1,x2,y2 in detections.xyxy]
        idx=np.argmax(areas)
        x1,y1,x2,y2=map(int,detections.xyxy[idx])
        crop=frame[max(0,y1-PAD):y2+PAD, max(0,x1-PAD):x2+PAD]
        if crop.size==0: continue
        rgb=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        results_pose=pose.process(rgb)
        if not results_pose.pose_landmarks: continue
        h,w,_=crop.shape
        def get(i): return np.array([results_pose.pose_landmarks.landmark[i].x*w,
                                     results_pose.pose_landmarks.landmark[i].y*h])
        try:
            hip_l,knee_l,ankle_l,foot_l=get(23),get(25),get(27),get(31)
            hip_r,knee_r,ankle_r,foot_r=get(24),get(26),get(28),get(32)
            lk.append(calculate_angle(hip_l,knee_l,ankle_l))
            rk.append(calculate_angle(hip_r,knee_r,ankle_r))
            la.append(calculate_angle(knee_l,ankle_l,foot_l))
            ra.append(calculate_angle(knee_r,ankle_r,foot_r))
        except: continue
    if len(lk)<2: return None

    lk=np.array(lk); rk=np.array(rk); la=np.array(la); ra=np.array(ra)
    kc=np.gradient(np.stack([lk,rk],axis=1), axis=0)
    ac=np.gradient(np.stack([la,ra],axis=1), axis=0)
    seq_base=np.stack([lk,rk,la,ra,kc[:,0],ac[:,0]],axis=1)
    vel=np.gradient(seq_base, axis=0)
    seq=np.concatenate([seq_base,vel],axis=1)
    seq=(seq-seq.mean(axis=0))/(seq.std(axis=0)+1e-6)
    length=len(seq)
    if length>=MAX_SEQ_LEN: seq=seq[-MAX_SEQ_LEN:]; length=MAX_SEQ_LEN
    else: pad_arr=np.zeros((MAX_SEQ_LEN-length,seq.shape[1])); seq=np.vstack([pad_arr,seq])
    return seq,length

def predict_video_global(path):
    clips=split_video_to_clips(path)
    preds=[]
    for clip in clips:
        seq_data=process_clip(clip)
        if seq_data is None: continue
        seq,length=seq_data
        seq_tensor=torch.tensor(seq,dtype=torch.float32).unsqueeze(0)
        lengths_tensor=torch.tensor([length],dtype=torch.int64)
        with torch.no_grad(): preds.append(float(torch.sigmoid(lstm_model(seq_tensor,lengths_tensor)).item()))
    if not preds: return None
    return aggregate_results(preds)