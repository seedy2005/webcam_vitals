# 💓 Webcam Vitals (rPPG)
> **Contactless heart rate & breathing rate in 30 seconds using only your webcam. No hardware. No wearables. 100% private.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-FaceLandmarker_WASM-0072B5?logo=google)](https://developers.google.com/mediapipe)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?logo=openai)](https://openai.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🌟 The Big Idea

Basic vital-sign checks require hardware: a pulse oximeter, a blood-pressure cuff, or an expensive smartwatch. Community screening camps, rural clinics, and tele-consultations often have none of these, but **almost everyone has a camera**.

**Webcam Vitals** transforms any standard laptop webcam or smartphone camera (via DroidCam) into a contact-free pulse sensor. Seated still in normal room light, you receive your heart rate within 5 bpm of a clinical pulse oximeter in under 35 seconds.

---

## 🔬 How Does It Work? (The Science of rPPG)

### "Can a camera really see my heartbeat?"
**Yes!** The technology is called **Remote Photoplethysmography (rPPG)**:

1. **The Pulse Cycle:** Every time your heart beats, blood is pushed into the microscopic capillary vessels just beneath your facial skin.
2. **Light Absorption:** Blood contains hemoglobin, which absorbs certain wavelengths of light (especially green light).
3. **Subtle Color Shifts:** As blood volume peaks and dips with each beat, the amount of light reflected off your skin changes by **less than 1%**.
4. **Signal Extraction:** While invisible to human eyes, digital camera sensors capture these minute color fluctuations. By isolating pure skin patches on your forehead and cheeks, our algorithm filters out the noise and pinpoints your exact pulse rhythm.

```
┌─────────────┐       ┌──────────────────────┐       ┌──────────────────────┐       ┌─────────────┐
│  Heartbeat  │ ───>  │ Skin Color Micro-    │ ───>  │ POS Algorithm +      │ ───>  │  Live BPM   │
│   Pumps     │       │ Fluctuations (Green) │       │ Butterworth Filtering│       │   Display   │
└─────────────┘       └──────────────────────┘       └──────────────────────┘       └─────────────┘
```

---

## ✨ Features

- 🫀 **Instant Heart Rate (HR):** Accurate BPM estimation in ~30 seconds.
- 🌬️ **Breathing Rate (RR):** Tracks low-frequency respiratory sinus arrhythmia (6–30 breaths/min).
- 📈 **Real-Time Pulse Waveform:** Watch your pulse signal trace live on screen while the scan runs.
- 🔒 **Zero-Video-Upload Privacy:** Video frames **never leave your browser**. Only lightweight color averages (~15 KB of numbers) are analyzed locally.
- 📱 **DroidCam & Phone Camera Ready:** Built-in auto-detection for DroidCam so you can use high-resolution phone cameras with zero configuration.
- 🛡️ **Signal Quality Gating:** The app checks ambient brightness, head motion, and spectral signal-to-noise ratio (SNR). If you cover your face or move around, it asks you to retry rather than guessing a false number.
- 🩺 **Ground-Truth Comparator:** A dedicated field allows judges and doctors to type a real pulse oximeter reading and see the absolute error computed live on screen.
- 🤖 **AI Wellness Summary:** Summarizes readings into 3 clear, empathetic sentences powered by OpenAI `gpt-4o-mini` (with an instant, offline clinical template fallback).
- 🎬 **Offline Video Fallback:** Allows uploading a pre-recorded face video file for stage demos when room lighting is unpredictable.

---

## 🚀 Quick Start (60 Seconds)

### Prerequisites
- Windows 10/11 (or macOS / Linux)
- Python 3.10 or higher
- A modern browser (Google Chrome or Microsoft Edge recommended)

---

### Method 1: One-Click Launch (Windows)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/seedy2005/webcam-vitals.git
   cd webcam-vitals
   ```

2. **Start the Backend API:**
   Double-click **`start-backend.bat`** (or run `start-backend.bat` in Command Prompt).  
   *Runs FastAPI on `http://localhost:8000`.*

3. **Start the Frontend Web App:**
   Double-click **`start-frontend.bat`**.  
   *Opens the web app on `http://localhost:5173`.*

4. **Scan:** Open `http://localhost:5173`, choose your camera, and click **▶ Start 30-second Scan**!

---

### Method 2: Manual Terminal Setup

#### 1. Backend Server
```bash
cd api
python -m pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

#### 2. Frontend Server
```bash
cd web
python -m http.server 5173
```
Now navigate to **`http://localhost:5173`** in your browser.

---

### 📱 Using Your Phone as a Camera (DroidCam)

If your laptop camera is low resolution or grainy:
1. Install **DroidCam** on your phone (Android/iOS) and PC ([dev47apps.com](https://www.dev47apps.com/)).
2. Connect both to the same Wi-Fi network (or connect via USB cable).
3. Open `http://localhost:5173` — the app will automatically detect **`DroidCam Source [Phone Camera]`** as the primary video input!

---

## 🔑 Optional: AI Wellness Summary Setup

The application works 100% out of the box with the built-in clinical rule template. If you want dynamic OpenAI summaries:
1. Copy `.env.example` to `api/.env`:
   ```bash
   cp .env.example api/.env
   ```
2. Insert your OpenAI API key:
   ```env
   OPENAI_API_KEY=sk-proj-your-api-key-here
   ```
*(If the key is missing or out of credits, the app seamlessly falls back to the local clinical template without interrupting your scan).*

---

## 🛠️ System Architecture

```mermaid
graph TD
    subgraph Browser ["Frontend (HTML5 + Vanilla JS + WASM)"]
        Cam[Webcam / DroidCam Stream] --> MP[MediaPipe Face Landmarker WASM]
        MP --> ROI[Forehead & Cheek Skin Patches]
        ROI --> Sampler[Fast Hardware-Accelerated Color Sampler]
        Sampler --> Waveform[Live Canvas Pulse Waveform]
        Sampler --> Traces[30-Second RGB Array]
    end

    subgraph Backend ["Python Backend (FastAPI + SciPy + NumPy)"]
        Traces --> Resample[Uniform 30 Hz Interpolation]
        Resample --> POS[POS Motion-Color Projection]
        POS --> Filter[4th-Order Butterworth Filter (0.7 - 3.0 Hz)]
        Filter --> Welch[Welch PSD + Sub-Bin Parabolic Interpolation]
        Filter --> RR[Respiratory Rate Extraction (0.1 - 0.5 Hz)]
        Welch --> Median[Sliding 15s Median Filter]
        Welch --> Gate[Bandwidth-Normalized SNR & Prominence Check]
        Gate --> LLM[OpenAI GPT-4o-mini / Fallback]
    end

    Backend --> Results[Results Screen: HR, RR, Quality Badge, Error vs Reference]
```

### Digital Signal Processing (DSP) Pipeline:
1. **Face Tracking:** MediaPipe Face Landmarker runs locally via WebAssembly, continuously locking onto 468 facial points.
2. **Skin Region (ROI):** Isolates the central forehead and both zygomatic cheek zones while completely ignoring eyebrows, facial hair, and eyes.
3. **POS Algorithm:** Implements Wang et al. (2017) *Plane-Orthogonal-to-Skin* projection to eliminate ambient illumination variations and minor head movements:
   $$S_1 = G_n - B_n, \quad S_2 = G_n + B_n - 2R_n, \quad h = S_1 + \frac{\sigma(S_1)}{\sigma(S_2)}S_2$$
4. **Filtering:** Applies a zero-phase 4th-order Butterworth bandpass filter (`filtfilt`) isolating **0.7 Hz to 3.0 Hz** (42 to 180 bpm).
5. **Spectral Peak Detection:** Welch Power Spectral Density (PSD) combined with sub-bin parabolic peak interpolation resolves frequency down to decimal fractions.
6. **Median Smoothing:** Evaluates 2-second incremental updates and takes the median over the final 15 seconds to dismiss blinks and minor shifts.

---

## 📂 Repository Structure

```
webcam-vitals/
├── api/
│   ├── main.py              # FastAPI endpoints (POST /analyze, GET /health)
│   ├── rppg.py              # POS algorithm, Butterworth filtering, Welch FFT, SNR gate
│   ├── summary.py           # OpenAI GPT-4o-mini integration & clinical template fallback
│   ├── requirements.txt     # Python backend dependencies
│   └── .env.example         # Template for environment variables
├── web/
│   ├── index.html           # Modern 5-screen interface (Start, Scan, Results, etc.)
│   ├── app.js               # MediaPipe face tracking, Canvas waveforms, state machine
│   └── styles.css           # Glassmorphism dark-mode UI with neon-green pulse glow
├── start-backend.bat        # 1-click Windows starter for FastAPI
├── start-frontend.bat       # 1-click Windows starter for web frontend
└── README.md                # Project documentation
```

---

## 🩺 Ground-Truth Validation & Testing

| Metric | Target | How Measured |
| :--- | :--- | :--- |
| **HR Error** | Mean absolute error $\le 5\text{ bpm}$ | Compared directly against a fingertip pulse oximeter |
| **RR Error** | Mean absolute error $\le 3\text{ breaths/min}$ | Manual breath count over 30 seconds |
| **Scan Duration** | $\le 35\text{ seconds}$ | 5s camera stabilization + 30s capture |
| **Frame Rate** | $\ge 24\text{ fps}$ | In-browser FPS monitor |
| **Bad-Signal Gating** | 100% rejection of shaking/covered face | Trigger retry prompt instead of guessing numbers |

---

## ⚠️ Wellness Disclaimer

> **This application is a screening aid and wellness estimation tool, NOT a diagnostic medical device.**  
> It does not claim clinical certification and should not be used for diagnosis, prevention, or treatment of medical conditions. Environmental factors (such as dim lighting, excessive motion, or low camera sensitivity) may impact accuracy. Always consult a certified healthcare professional for medical concerns.

---

## 📄 License

Distributed under the MIT License. Built with ❤️ for hackathons and health accessibility.
