# VisionTrack: Real-Time Object Detection

Welcome to **VisionTrack**! This is a full-stack, real-time object detection application capable of identifying over 80 distinct categories from the COCO dataset directly from your webcam.

You can check out the live site here: [VisionTrack Live](https://aakashtiru26.github.io/coco-object-detection/)

## 🚀 The Tech Stack
* **Frontend:** Vanilla HTML, CSS, JavaScript, and a custom **Three.js** WebGL engine for the interactive 3D particle background. Hosted on **GitHub Pages**.
* **Backend:** Python, **FastAPI**, **WebSockets**, and **OpenCV** to capture, compress, and stream video frames. Hosted on **Hugging Face Spaces** (Docker).
* **AI Model:** **YOLOv8n** (You Only Look Once), originally in PyTorch, but fully exported to **ONNX Runtime** for massive CPU acceleration.

## 🛠️ The Journey & Challenges
Building this application was a massive learning experience, especially when it came to crossing the gap from a local machine to a production deployment. Here were the biggest hurdles and how they were solved:

### 1. The Deployment Architecture Gap
Initially, everything ran on `localhost`. The frontend grabbed the webcam, the backend ran PyTorch, and everything felt instantaneous. When it was time to deploy, I realized standard free hosting (like Vercel or Netlify) couldn't support long-running WebSockets or the heavy RAM footprint of PyTorch. 

**The Fix:** I split the monolithic architecture. The static frontend went to GitHub Pages, and I built a custom Docker container to run the PyTorch API on a free Hugging Face Space.

### 2. The PyTorch 2.6 Security Wall (`UnpicklingError`)
During deployment to Hugging Face, the build crashed entirely. PyTorch 2.6 recently introduced a strict security patch (`weights_only=True`) that blocked the serialized Python objects inside the YOLOv8 `.pt` file from loading.

**The Fix:** Rather than rewriting the Ultralytics loading logic, I explicitly pinned `torch==2.5.1` in the `requirements.txt` to safely roll back to a stable checkpoint.

### 3. The CPU & Network Bottleneck
Once deployed, the FPS plummeted. Hugging Face's free tier uses a 2-core CPU, and transmitting high-res JPEGs over WebSockets to a distant server was too slow.

**The Fix:** I implemented a triple-optimization strategy:
1. **Model Export:** I dumped the raw PyTorch model and exported YOLOv8 to the **ONNX Runtime**, which executes matrix math significantly faster on standard CPUs.
2. **Inference Sizing:** I reduced the backend inference resolution to `256px`. The CPU does less math, but the bounding boxes are accurately scaled back up on the frontend.
3. **Payload Compression:** I dialed down the JPEG compression quality of the frames before they enter the WebSocket, massively reducing network transmission times over standard internet connections.

## 💻 How to Run Locally

1. Clone the repository.
2. Open `index.html` and change `PRODUCTION_BACKEND` back to `localhost:8000`.
3. Start the frontend:
   ```bash
   python3 -m http.server 8080
   ```
4. Start the backend:
   ```bash
   cd backend
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   uvicorn main:app --reload --port 8000
   ```
5. Visit `http://localhost:8080` in your browser.

---
*Built by Aakash Tiruveedula*
