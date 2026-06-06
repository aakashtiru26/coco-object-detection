import os
import shutil
import tempfile
import cv2
import base64
import numpy as np
import time
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

app = FastAPI()

# Allow CORS for local development from the frontend (which is likely served on a different port or file://)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLOv8 model
model = YOLO('yolov8n.pt')
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"YOLOv8 initialized on device: {device}")

@app.post("/api/detect")
async def detect_objects(file: UploadFile = File(...)):
    temp_dir = tempfile.mkdtemp()
    input_path = os.path.join(temp_dir, file.filename)
    
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    output_filename = f"out_{file.filename}.mp4"
    output_path = os.path.join(temp_dir, output_filename)
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Could not open uploaded video file.")
        
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps:
        fps = 25.0
        
    # Use mp4v fourcc for general mp4 compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Inference
        results = model(frame, verbose=False)
        
        # Plot results on frame
        annotated_frame = results[0].plot()
        
        out.write(annotated_frame)
        frame_count += 1
        
    cap.release()
    out.release()
    
    if frame_count == 0:
        raise HTTPException(status_code=400, detail="Video was opened but no frames could be read.")
    
    return FileResponse(output_path, media_type="video/mp4", filename=output_filename)

@app.websocket("/api/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    last_time = time.time()
    try:
        while True:
            # Receive base64 string from frontend
            data = await websocket.receive_text()
            
            # Remove "data:image/jpeg;base64," header if present
            if "," in data:
                data = data.split(",")[1]
                
            # Decode base64 to image
            img_bytes = base64.b64decode(data)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue
                
            # FPS Calculation
            current_time = time.time()
            fps = 1.0 / (current_time - last_time + 1e-6)
            last_time = current_time
                
            # Run YOLOv8 inference with reduced resolution and hardware acceleration
            results = model(frame, imgsz=320, verbose=False, device=device)
            boxes_data = []
            
            if len(results) > 0:
                result = results[0]
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = model.names[cls_id]
                    
                    boxes_data.append({
                        "x": x1,
                        "y": y1,
                        "w": x2 - x1,
                        "h": y2 - y1,
                        "conf": conf,
                        "class": cls_name
                    })
            
            # Send lightweight JSON back to frontend
            await websocket.send_json({
                "fps": round(fps, 1),
                "boxes": boxes_data
            })
            
    except WebSocketDisconnect:
        print("Client disconnected from WebSocket.")
