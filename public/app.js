/**
 * app.js — Webcam Vitals rPPG frontend
 *
 * Responsibilities:
 *  - Manage 3-screen UI state machine (Start → Scan → Results)
 *  - Camera capture via getUserMedia
 *  - Face landmark detection + ROI extraction via MediaPipe
 *  - RGB trace collection at ~30 fps
 *  - Motion estimation from face bounding box
 *  - Live pulse waveform rendering (Canvas)
 *  - Countdown ring animation
 *  - Real-time guidance messages
 *  - POST traces to /analyze and display results
 *  - Recorded-video fallback mode
 *  - Reference HR comparison
 */

// ─── Configuration ────────────────────────────────────────────────────────────

const CONFIG = {
  API_BASE: (window.location.hostname === 'localhost' && window.location.port === '5173')
    ? 'http://localhost:8000'
    : '',
  SCAN_DURATION_S:  30,
  WARMUP_S:         5,       // discard first 5 s while AE settles
  TARGET_FPS:       30,
  ROI_UPDATE_HZ:    30,
  WAVEFORM_HISTORY: 200,     // samples to show in live strip
  WAVEFORM_WIN_S:   6,       // seconds of waveform history shown
  GUIDANCE_DELAY_S: 2,       // seconds before showing guidance
  LIVE_HR_MIN_S:    10,      // start showing live HR estimate after this
  POS_WIN_S:        1.6,     // POS window (must match backend)
  HR_LO_HZ:        0.7,
  HR_HI_HZ:        3.0,
};

// MediaPipe face landmark indices for forehead and both cheeks
// (forehead top, forehead left, forehead right, left cheek, right cheek, nose bridge)
const ROI_INDICES = {
  FOREHEAD: [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
  FOREHEAD_ONLY: [10, 67, 109, 338, 297],
  LEFT_CHEEK:  [187, 123, 147, 213, 192, 214, 207, 205, 36, 142, 126, 217],
  RIGHT_CHEEK: [411, 352, 376, 433, 416, 434, 427, 425, 266, 371, 355, 437],
};

// ─── State ───────────────────────────────────────────────────────────────────

const state = {
  screen: 'start',         // 'start' | 'scan' | 'processing' | 'results' | 'error'
  mode: 'live',            // 'live' | 'upload'
  stream: null,
  animFrame: null,
  scanStartTime: null,
  warmupDone: false,
  rgbTraces: [],
  timestamps: [],
  motionSamples: [],
  lastFaceBox: null,
  faceMesh: null,
  waveformBuffer: [],      // filtered pulse samples for live display
  liveHR: 0,
  guidanceState: 'looking',// 'looking' | 'dark' | 'motion' | 'good' | 'warning'
  noFaceFrames: 0,
  motionFrames: 0,
  countdownPaused: false,
  pausedElapsed: 0,
  pauseStart: null,
  availableCameras: [],
  selectedDeviceId: localStorage.getItem('preferredCameraId') || null,
  userExplicitlySelectedCamera: !!localStorage.getItem('preferredCameraId'),
};

// ─── DOM refs ─────────────────────────────────────────────────────────────────

const dom = {
  screens: {
    start:      document.getElementById('screen-start'),
    scan:       document.getElementById('screen-scan'),
    processing: document.getElementById('screen-processing'),
    results:    document.getElementById('screen-results'),
    error:      document.getElementById('screen-error'),
  },
  startBtn:        document.getElementById('btn-start-scan'),
  startUploadBtn:  document.getElementById('btn-start-upload'),
  cancelBtn:       document.getElementById('btn-cancel'),
  scanAgainBtn:    document.getElementById('btn-scan-again'),
  errorRetryBtn:   document.getElementById('btn-error-retry'),
  cameraSelect:    document.getElementById('camera-select'),
  cameraHint:      document.getElementById('camera-hint'),
  video:           document.getElementById('video-el'),
  overlayCanvas:   document.getElementById('overlay-canvas'),
  waveformCanvas:  document.getElementById('waveform-canvas'),
  resultWaveform:  document.getElementById('result-waveform-canvas'),
  countdownNum:    document.getElementById('countdown-num'),
  ringProg:        document.getElementById('ring-prog'),
  guidanceDot:     document.getElementById('guidance-dot'),
  guidanceText:    document.getElementById('guidance-text'),
  liveHREl:        document.getElementById('live-hr'),
  videoWrapper:    document.getElementById('video-wrapper'),
  // Results
  resultHR:        document.getElementById('result-hr'),
  resultRR:        document.getElementById('result-rr'),
  resultConf:      document.getElementById('result-confidence'),
  resultSNR:       document.getElementById('result-snr-fill'),
  summaryText:     document.getElementById('summary-text'),
  summaryLoading:  document.getElementById('summary-loading'),
  refInput:        document.getElementById('ref-hr-input'),
  refResult:       document.getElementById('ref-result'),
  errorTitle:      document.getElementById('error-title'),
  errorMsg:        document.getElementById('error-msg'),
  // Upload
  uploadZone:      document.getElementById('upload-zone'),
  fileInput:       document.getElementById('file-input'),
  uploadVideoEl:   document.getElementById('upload-video-el'),
  modeLive:        document.getElementById('mode-live'),
  modeUpload:      document.getElementById('mode-upload'),
};

// ─── Utility ─────────────────────────────────────────────────────────────────

const sleep = ms => new Promise(r => setTimeout(r, ms));

function setScreen(name) {
  state.screen = name;
  Object.entries(dom.screens).forEach(([k, el]) => {
    if (!el) return;
    el.classList.toggle('active', k === name);
  });
}

function setGuidance(type, msg) {
  state.guidanceState = type;
  dom.guidanceDot.className = 'guidance-dot';
  if (type === 'good')    dom.guidanceDot.classList.add('green');
  else if (type === 'warning' || type === 'motion') dom.guidanceDot.classList.add('red');
  else                    dom.guidanceDot.classList.add('amber');
  dom.guidanceText.textContent = msg;

  dom.videoWrapper.classList.remove('detecting', 'tracking', 'warning');
  if (type === 'good')    dom.videoWrapper.classList.add('tracking');
  else if (type === 'warning' || type === 'motion') dom.videoWrapper.classList.add('warning');
  else                    dom.videoWrapper.classList.add('detecting');
}

// Simple in-browser band-pass using a cascaded biquad approximation (IIR)
// Used for live waveform display only; accuracy not critical.
class SimpleBandPass {
  constructor(lo_hz, hi_hz, fs) {
    this.buf = new Array(8).fill(0);
    // Use Web Audio API-style coefficients approximation (not used here).
    // Actually, we'll just use a ring buffer + moving-average trick for display.
    this.lo = lo_hz; this.hi = hi_hz; this.fs = fs;
    this.ring = [];
    this.winLen = Math.round(fs / lo_hz);
    this.dcBuf = new Array(this.winLen).fill(0);
    this.dcIdx = 0; this.dcSum = 0;
  }
  push(val) {
    // Remove DC with moving average
    const old = this.dcBuf[this.dcIdx];
    this.dcBuf[this.dcIdx] = val;
    this.dcIdx = (this.dcIdx + 1) % this.dcBuf.length;
    this.dcSum += val - old;
    const dc = this.dcSum / this.dcBuf.length;
    return val - dc;
  }
}

// POS single-sample approximation for live display
// Full POS done in backend on complete trace; here we do a simplified version
class LivePOS {
  constructor(fs) {
    this.fs = fs;
    this.winLen = Math.round(CONFIG.POS_WIN_S * fs);
    this.rBuf = []; this.gBuf = []; this.bBuf = [];
    this.bp = new SimpleBandPass(CONFIG.HR_LO_HZ, CONFIG.HR_HI_HZ, fs);
    this.out = [];
  }
  push(r, g, b) {
    this.rBuf.push(r); this.gBuf.push(g); this.bBuf.push(b);
    if (this.rBuf.length > this.winLen) {
      this.rBuf.shift(); this.gBuf.shift(); this.bBuf.shift();
    }
    if (this.rBuf.length < 8) return 0;
    const W = this.rBuf.length;
    const meanR = this.rBuf.reduce((a,b)=>a+b,0)/W;
    const meanG = this.gBuf.reduce((a,b)=>a+b,0)/W;
    const meanB = this.bBuf.reduce((a,b)=>a+b,0)/W;
    if (meanR < 1 || meanG < 1 || meanB < 1) return 0;
    // POS projection
    const S1 = (g - meanG)/meanG - (b - meanB)/meanB;
    const S2 = (g - meanG)/meanG + (b - meanB)/meanB - 2*(r - meanR)/meanR;
    // adaptive weight (simplified: use fixed weight)
    const h = S1 + S2;
    this.out.push(h);
    if (this.out.length > this.winLen * 2) this.out.shift();
    return this.bp.push(h);
  }
}

// Live HR estimation from waveform buffer via zero-crossing rate
function estimateLiveHR(buf, fs) {
  if (buf.length < fs * 5) return 0;
  const seg = buf.slice(-Math.round(fs * 8));
  const mean = seg.reduce((a,b)=>a+b,0)/seg.length;
  let crossings = 0;
  let prev = seg[0] > mean;
  for (let i = 1; i < seg.length; i++) {
    const cur = seg[i] > mean;
    if (cur !== prev && cur) crossings++;
    prev = cur;
  }
  const durationS = seg.length / fs;
  return Math.round((crossings / durationS) * 60);
}

// ─── MediaPipe Setup ──────────────────────────────────────────────────────────

let faceLandmarker = null;

async function initMediaPipe() {
  const { FaceLandmarker, FilesetResolver } = await import(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs'
  );

  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
  );

  faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
      delegate: 'GPU',
    },
    runningMode: 'VIDEO',
    numFaces: 1,
    outputFaceBlendshapes: false,
    outputFacialTransformationMatrixes: false,
  });
}

// ─── ROI Extraction ───────────────────────────────────────────────────────────

function getLandmarkPoint(lm, idx, w, h) {
  return [lm[idx].x * w, lm[idx].y * h];
}

// Persistent canvas reused across frames (avoids GC pauses & dropped frames)
let sharedOffCanvas = null;
let sharedOffCtx = null;

function getROIRects(landmarks, w, h) {
  // Forehead: center forehead patch between brow and top of forehead
  // Landmarks: 10 (top), 9 (glabella), 109 (left forehead), 338 (right forehead)
  const p10 = getLandmarkPoint(landmarks, 10, w, h);
  const p9 = getLandmarkPoint(landmarks, 9, w, h);
  const p109 = getLandmarkPoint(landmarks, 109, w, h);
  const p338 = getLandmarkPoint(landmarks, 338, w, h);

  const fWidth = Math.abs(p338[0] - p109[0]) * 0.55;
  const fHeight = Math.abs(p9[1] - p10[1]) * 0.55;
  const fCenterX = (p10[0] + p9[0]) / 2;
  const fCenterY = (p10[1] + p9[1]) / 2;

  const forehead = {
    x: Math.max(0, Math.round(fCenterX - fWidth / 2)),
    y: Math.max(0, Math.round(fCenterY - fHeight / 2)),
    w: Math.max(8, Math.round(fWidth)),
    h: Math.max(8, Math.round(fHeight)),
  };

  // Left cheek: centered at landmark 117
  const pCheekL = getLandmarkPoint(landmarks, 117, w, h);
  const cheekWidth = Math.max(8, Math.round(fWidth * 0.5));
  const cheekHeight = Math.max(8, Math.round(fHeight * 0.75));

  const leftCheek = {
    x: Math.max(0, Math.round(pCheekL[0] - cheekWidth / 2)),
    y: Math.max(0, Math.round(pCheekL[1] - cheekHeight / 2)),
    w: cheekWidth,
    h: cheekHeight,
  };

  // Right cheek: centered at landmark 346
  const pCheekR = getLandmarkPoint(landmarks, 346, w, h);
  const rightCheek = {
    x: Math.max(0, Math.round(pCheekR[0] - cheekWidth / 2)),
    y: Math.max(0, Math.round(pCheekR[1] - cheekHeight / 2)),
    w: cheekWidth,
    h: cheekHeight,
  };

  return [forehead, leftCheek, rightCheek];
}

function extractROIColor(ctx, landmarks, w, h) {
  if (!sharedOffCanvas) {
    sharedOffCanvas = document.createElement('canvas');
    sharedOffCtx = sharedOffCanvas.getContext('2d', { willReadFrequently: true });
  }
  if (sharedOffCanvas.width !== w || sharedOffCanvas.height !== h) {
    sharedOffCanvas.width = w;
    sharedOffCanvas.height = h;
  }

  // Draw current video frame onto persistent canvas
  sharedOffCtx.drawImage(dom.video, 0, 0, w, h);

  const rects = getROIRects(landmarks, w, h);
  let rSum = 0, gSum = 0, bSum = 0, count = 0;

  for (const r of rects) {
    if (r.w < 4 || r.h < 4 || r.x + r.w > w || r.y + r.h > h) continue;
    const imgData = sharedOffCtx.getImageData(r.x, r.y, r.w, r.h);
    const data = imgData.data;
    const step = 2; // sample every 2nd pixel for smooth 30 fps
    const rowBytes = r.w * 4;

    for (let py = 0; py < r.h; py += step) {
      const rowOffset = py * rowBytes;
      for (let px = 0; px < r.w; px += step) {
        const i4 = rowOffset + px * 4;
        rSum += data[i4];
        gSum += data[i4 + 1];
        bSum += data[i4 + 2];
        count++;
      }
    }
  }

  if (count === 0) return null;
  return [rSum / count, gSum / count, bSum / count];
}

function drawROIOverlay(ctx, landmarks, w, h, hasMotion) {
  ctx.clearRect(0, 0, w, h);

  const color = hasMotion ? '#ff5c7a' : '#00ffa3';
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.8;
  ctx.fillStyle = hasMotion ? 'rgba(255, 92, 122, 0.15)' : 'rgba(0, 255, 163, 0.15)';

  const rects = getROIRects(landmarks, w, h);
  for (const r of rects) {
    ctx.beginPath();
    if (ctx.roundRect) {
      ctx.roundRect(r.x, r.y, r.w, r.h, 6);
    } else {
      ctx.rect(r.x, r.y, r.w, r.h);
    }
    ctx.fill();
    ctx.stroke();
  }

  // Draw dots at key anchor points
  ctx.fillStyle = color;
  ctx.globalAlpha = 0.85;
  for (const idx of [10, 9, 117, 346]) {
    if (idx >= landmarks.length) continue;
    const [x, y] = getLandmarkPoint(landmarks, idx, w, h);
    ctx.beginPath();
    ctx.arc(x, y, 2.5, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
}

function getFaceBox(landmarks, w, h) {
  const xs = landmarks.map(l => l.x * w);
  const ys = landmarks.map(l => l.y * h);
  return {
    x: Math.min(...xs), y: Math.min(...ys),
    w: Math.max(...xs) - Math.min(...xs),
    h: Math.max(...ys) - Math.min(...ys),
    cx: (Math.min(...xs) + Math.max(...xs)) / 2,
    cy: (Math.min(...ys) + Math.max(...ys)) / 2,
  };
}

// ─── Camera Device Management ──────────────────────────────────────────────────

function scoreCamera(device) {
  const label = ((device.label || '') + ' ' + (device.deviceId || '')).toLowerCase();

  // 1. DroidCam phone camera — Top Priority!
  if (label.includes('droidcam')) {
    return 200;
  }

  // 2. Integrated laptop webcams
  if (
    label.includes('integrated') ||
    label.includes('internal') ||
    label.includes('built-in') ||
    label.includes('builtin') ||
    label.includes('laptop') ||
    label.includes('webcam') ||
    label.includes('front') ||
    label.includes('chicony') ||
    label.includes('sunplus') ||
    label.includes('realtek') ||
    label.includes('facetime')
  ) {
    return 100;
  }

  // 3. Other phone-link / Windows LinkToDevice virtual cameras
  if (
    label.includes('link to windows') ||
    label.includes('phone link') ||
    label.includes('linktodevice') ||
    label.includes('virtual') ||
    label.includes('obs')
  ) {
    return -100;
  }

  return 0;
}

async function updateCameraList() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
    return;
  }

  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter(d => d.kind === 'videoinput');
    state.availableCameras = videoDevices;

    if (!dom.cameraSelect) return;
    dom.cameraSelect.innerHTML = '';

    if (videoDevices.length === 0) {
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = 'No camera found';
      dom.cameraSelect.appendChild(opt);
      return;
    }

    // Sort devices so DroidCam appears first, laptop webcams second, others last
    const sorted = [...videoDevices].sort((a, b) => scoreCamera(b) - scoreCamera(a));

    // Choose preferred camera: if DroidCam is present, select it automatically!
    const droidCamDev = sorted.find(d => (d.label || '').toLowerCase().includes('droidcam'));
    let currentId = state.selectedDeviceId;
    const exists = sorted.some(d => d.deviceId === currentId);

    if (droidCamDev && (!currentId || !exists || !state.userExplicitlySelectedCamera)) {
      currentId = droidCamDev.deviceId;
      state.selectedDeviceId = currentId;
      localStorage.setItem('preferredCameraId', currentId);
    } else if (!currentId || !exists) {
      currentId = sorted[0].deviceId;
      state.selectedDeviceId = currentId;
      if (currentId) localStorage.setItem('preferredCameraId', currentId);
    }

    sorted.forEach((dev, idx) => {
      const opt = document.createElement('option');
      opt.value = dev.deviceId;
      const score = scoreCamera(dev);
      let badge = '';
      if (score >= 200) badge = ' 📱 [DroidCam Phone Camera]';
      else if (score > 0) badge = ' 💻 [Laptop Webcam]';
      else if (score < 0) badge = ' 🔗 [Virtual / Link]';

      opt.textContent = (dev.label || `Camera ${idx + 1}`) + badge;
      if (dev.deviceId === state.selectedDeviceId) {
        opt.selected = true;
      }
      dom.cameraSelect.appendChild(opt);
    });

    if (dom.cameraHint) {
      const chosen = sorted.find(d => d.deviceId === state.selectedDeviceId);
      const chosenLabel = ((chosen?.label || '')).toLowerCase();
      if (chosenLabel.includes('droidcam')) {
        dom.cameraHint.textContent = '📱 DroidCam Phone Camera connected';
        dom.cameraHint.style.color = 'var(--green)';
      } else if (chosen && scoreCamera(chosen) < 0) {
        dom.cameraHint.textContent = 'Virtual camera selected';
        dom.cameraHint.style.color = 'var(--amber)';
      } else {
        dom.cameraHint.textContent = 'Laptop webcam active';
        dom.cameraHint.style.color = 'var(--green)';
      }
    }
  } catch (err) {
    console.warn('Could not enumerate camera devices:', err);
  }
}

async function startCamera(preferredId = null) {
  try {
    let targetId = preferredId || state.selectedDeviceId;

    // If no targetId is set yet, inspect available devices to prioritize DroidCam
    if (!targetId && navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(d => d.kind === 'videoinput');
        if (videoDevices.length > 0) {
          const sorted = [...videoDevices].sort((a, b) => scoreCamera(b) - scoreCamera(a));
          targetId = sorted[0].deviceId;
        }
      } catch (e) {}
    }

    const buildConstraints = (devId) => {
      const c = {
        width: { ideal: 640 },
        height: { ideal: 480 },
        frameRate: { ideal: 30, min: 24 },
      };
      if (devId) {
        c.deviceId = { ideal: devId };
      }
      return c;
    };

    let stream = null;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: buildConstraints(targetId),
        audio: false,
      });
    } catch (err) {
      // Fallback without deviceId restriction
      console.warn('Specific deviceId request failed, falling back:', err);
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30, min: 24 } },
        audio: false,
      });
    }

    // Inspect the active track to see what camera was actually opened
    let activeTrack = stream.getVideoTracks()[0];
    let trackLabel = (activeTrack?.label || '').toLowerCase();

    // If active track is DroidCam, keep it! If it fell back to broken Phone Link, swap to DroidCam or Laptop cam
    if (trackLabel.includes('droidcam')) {
      console.log('Streaming from DroidCam phone camera:', activeTrack.label);
    } else if (!state.userExplicitlySelectedCamera && (trackLabel.includes('link to windows') || trackLabel.includes('phonelink') || trackLabel.includes('linktodevice'))) {
      try {
        const allDevices = await navigator.mediaDevices.enumerateDevices();
        const betterCam = allDevices.find(d => (d.label || '').toLowerCase().includes('droidcam'))
          || allDevices.find(d => d.kind === 'videoinput' && scoreCamera(d) > 0);
        if (betterCam && betterCam.deviceId) {
          console.log('Swapping from Windows link to:', betterCam.label);
          activeTrack.stop();
          stream = await navigator.mediaDevices.getUserMedia({
            video: {
              deviceId: { exact: betterCam.deviceId },
              width: { ideal: 640 },
              height: { ideal: 480 },
              frameRate: { ideal: 30, min: 24 },
            },
            audio: false,
          });
          activeTrack = stream.getVideoTracks()[0];
          targetId = betterCam.deviceId;
        }
      } catch (swapErr) {
        console.warn('Auto-switch failed:', swapErr);
      }
    }

    state.stream = stream;
    dom.video.srcObject = stream;
    await new Promise(resolve => { dom.video.onloadedmetadata = resolve; });
    dom.video.play();

    // Update settings and state
    if (activeTrack) {
      const settings = activeTrack.getSettings();
      const actualId = settings.deviceId || targetId;
      if (actualId) {
        state.selectedDeviceId = actualId;
        localStorage.setItem('preferredCameraId', actualId);
      }
    }

    // Refresh dropdown now that user permission reveals device labels
    await updateCameraList();

    return true;
  } catch (err) {
    console.error('Camera error:', err);
    return false;
  }
}

function stopCamera() {
  if (state.stream) {
    state.stream.getTracks().forEach(t => t.stop());
    state.stream = null;
  }
  dom.video.srcObject = null;
}

// ─── Waveform Canvas ──────────────────────────────────────────────────────────

function drawWaveform(canvas, data, color = '#00ffa3', fill = true) {
  const ctx = canvas.getContext('2d');
  const w = canvas.clientWidth || canvas.width;
  const h = canvas.clientHeight || canvas.height;
  canvas.width = w;
  canvas.height = h;
  ctx.clearRect(0, 0, w, h);

  if (data.length < 2) return;

  const maxVal = Math.max(...data.map(Math.abs), 0.001);
  const mid = h / 2;
  const xStep = w / (data.length - 1);

  // Gradient fill
  if (fill) {
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, color.replace(')', ', 0.25)').replace('rgb', 'rgba').replace('#00ffa3', 'rgba(0,255,163,0.25)'));
    grad.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.beginPath();
    ctx.moveTo(0, mid);
    data.forEach((v, i) => {
      const x = i * xStep;
      const y = mid - (v / maxVal) * (mid * 0.85);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.lineTo((data.length - 1) * xStep, mid);
    ctx.lineTo(0, mid);
    ctx.closePath();
    ctx.fillStyle = 'rgba(0,255,163,0.12)';
    ctx.fill();
  }

  // Line
  ctx.beginPath();
  ctx.lineWidth = 2;
  ctx.strokeStyle = color;
  ctx.shadowColor = color;
  ctx.shadowBlur = 6;
  data.forEach((v, i) => {
    const x = i * xStep;
    const y = mid - (v / maxVal) * (mid * 0.85);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.shadowBlur = 0;

  // Zero line
  ctx.beginPath();
  ctx.strokeStyle = 'rgba(255,255,255,0.05)';
  ctx.lineWidth = 1;
  ctx.moveTo(0, mid); ctx.lineTo(w, mid);
  ctx.stroke();
}

// ─── Countdown ring ───────────────────────────────────────────────────────────

function updateCountdown(elapsed) {
  const total = CONFIG.SCAN_DURATION_S;
  const remaining = Math.max(0, total - elapsed);
  dom.countdownNum.textContent = Math.ceil(remaining);

  const r = 30;
  const circ = 2 * Math.PI * r;
  const prog = elapsed / total;
  dom.ringProg.style.strokeDasharray = circ;
  dom.ringProg.style.strokeDashoffset = circ * (1 - prog);
}

// ─── Motion detection ─────────────────────────────────────────────────────────

function computeMotion(newBox, fps) {
  if (!state.lastFaceBox || !newBox) return 0;
  const dx = newBox.cx - state.lastFaceBox.cx;
  const dy = newBox.cy - state.lastFaceBox.cy;
  const dist = Math.sqrt(dx * dx + dy * dy);
  return dist * fps; // pixels per second
}

// ─── Scan loop ────────────────────────────────────────────────────────────────

let livePos = null;
let lastTimestamp = 0;
let frameCount = 0;
let scanDoneCallback = null;

async function startScan() {
  // Reset state
  state.rgbTraces = [];
  state.timestamps = [];
  state.motionSamples = [];
  state.waveformBuffer = [];
  state.liveHR = 0;
  state.noFaceFrames = 0;
  state.motionFrames = 0;
  state.warmupDone = false;
  state.countdownPaused = false;
  state.pausedElapsed = 0;
  state.pauseStart = null;
  state.lastFaceBox = null;
  livePos = new LivePOS(CONFIG.TARGET_FPS);
  frameCount = 0;
  lastTimestamp = 0;
  state.scanStartTime = performance.now();

  dom.liveHREl.innerHTML = '<span id="live-hr-val">--</span> <span class="unit">bpm</span>';

  // Initialize countdown ring
  const r = 30;
  const circ = 2 * Math.PI * r;
  dom.ringProg.style.strokeDasharray = circ;
  dom.ringProg.style.strokeDashoffset = circ;
  dom.countdownNum.textContent = CONFIG.SCAN_DURATION_S;

  setGuidance('looking', '👁 Looking for your face…');

  return new Promise(resolve => {
    scanDoneCallback = resolve;
    requestAnimationFrame(scanFrame);
  });
}

async function scanFrame(ts) {
  if (state.screen !== 'scan') return;

  const video = dom.video;
  if (video.readyState < 2) {
    state.animFrame = requestAnimationFrame(scanFrame);
    return;
  }

  const vw = video.videoWidth;
  const vh = video.videoHeight;

  // Resize canvases to match video
  const oc = dom.overlayCanvas;
  if (oc.width !== vw) oc.width = vw;
  if (oc.height !== vh) oc.height = vh;

  // Run face landmarker
  let landmarks = null;
  let faceBox = null;
  try {
    if (faceLandmarker) {
      const result = faceLandmarker.detectForVideo(video, ts);
      if (result.faceLandmarks && result.faceLandmarks.length > 0) {
        landmarks = result.faceLandmarks[0];
        faceBox = getFaceBox(landmarks, vw, vh);
      }
    }
  } catch (e) {
    // MediaPipe may throw on first few frames
  }

  const now = performance.now();
  const totalElapsed = (now - state.scanStartTime) / 1000;
  const fps = lastTimestamp > 0 ? 1000 / (ts - lastTimestamp) : CONFIG.TARGET_FPS;
  lastTimestamp = ts;
  frameCount++;

  // Guidance logic
  const brightness = landmarks ? await getFrameBrightness(video, vw, vh) : 0;

  if (!landmarks) {
    state.noFaceFrames++;
    const ctx = dom.overlayCanvas.getContext('2d');
    ctx.clearRect(0, 0, vw, vh);

    if (state.noFaceFrames > 2) {
      setGuidance('looking', '👁 Looking for your face…');
      if (!state.countdownPaused) {
        state.countdownPaused = true;
        state.pauseStart = now;
      }
    }
  } else {
    state.noFaceFrames = 0;

    // Motion
    const motionPxps = computeMotion(faceBox, fps);
    state.motionSamples.push(motionPxps);
    if (state.motionSamples.length > 60) state.motionSamples.shift();

    const highMotion = motionPxps > 18;

    // Draw overlay
    drawROIOverlay(dom.overlayCanvas.getContext('2d'), landmarks, vw, vh, highMotion);
    state.lastFaceBox = faceBox;

    // Brightness check
    const tooDark = brightness < 35;
    const tooBright = brightness > 220;

    if (tooDark) {
      setGuidance('warning', '☀️ Move closer to a steady light');
      state.countdownPaused = true; state.pauseStart = state.pauseStart || now;
    } else if (highMotion) {
      setGuidance('motion', '🛑 Hold still');
      state.motionFrames++;
      // Pause and auto-resume after 10 s of motion
      state.countdownPaused = true; state.pauseStart = state.pauseStart || now;
    } else {
      if (state.countdownPaused) {
        state.pausedElapsed += (now - (state.pauseStart || now)) / 1000;
        state.countdownPaused = false; state.pauseStart = null;
      }
      state.motionFrames = Math.max(0, state.motionFrames - 3);
      setGuidance('good', '✅ Great, keep still');
    }

    // Extract RGB only after warmup
    if (totalElapsed > CONFIG.WARMUP_S) {
      if (!state.warmupDone) { state.warmupDone = true; state.scanStartTime = now; }

      const colors = extractROIColor(dom.overlayCanvas.getContext('2d'), landmarks, vw, vh);
      if (colors) {
        const t = (now - state.scanStartTime) / 1000;
        state.rgbTraces.push(colors);
        state.timestamps.push(t);

        // Live POS + waveform
        const posVal = livePos.push(colors[0], colors[1], colors[2]);
        state.waveformBuffer.push(posVal);
        const maxWin = Math.round(CONFIG.TARGET_FPS * CONFIG.WAVEFORM_WIN_S);
        if (state.waveformBuffer.length > maxWin) state.waveformBuffer.shift();

        drawWaveform(dom.waveformCanvas, state.waveformBuffer);

        // Live HR estimate (after 10 s of data)
        if (state.timestamps.length > CONFIG.TARGET_FPS * CONFIG.LIVE_HR_MIN_S) {
          const hr = estimateLiveHR(state.waveformBuffer, CONFIG.TARGET_FPS);
          if (hr >= 40 && hr <= 180) {
            state.liveHR = hr;
            const el = document.getElementById('live-hr-val');
            if (el) el.textContent = hr;
          }
        }
      }
    } else {
      setGuidance('looking', `⏳ Settling… ${Math.ceil(CONFIG.WARMUP_S - totalElapsed)} s`);
    }
  }

  // Countdown (only counts data-collection time, not warmup)
  const dataElapsed = state.timestamps.length > 0
    ? state.timestamps[state.timestamps.length - 1]
    : 0;
  updateCountdown(Math.min(dataElapsed, CONFIG.SCAN_DURATION_S));

  // Check if scan is done
  if (dataElapsed >= CONFIG.SCAN_DURATION_S) {
    const ctx = dom.overlayCanvas.getContext('2d');
    ctx.clearRect(0, 0, vw, vh);
    if (scanDoneCallback) scanDoneCallback();
    return;
  }

  state.animFrame = requestAnimationFrame(scanFrame);
}

async function getFrameBrightness(video, w, h) {
  try {
    const c = document.createElement('canvas');
    c.width = 16; c.height = 16;
    const ctx = c.getContext('2d');
    ctx.drawImage(video, 0, 0, 16, 16);
    const d = ctx.getImageData(0, 0, 16, 16).data;
    let sum = 0;
    for (let i = 0; i < d.length; i += 4) sum += (d[i] + d[i+1] + d[i+2]) / 3;
    return sum / (d.length / 4);
  } catch { return 128; }
}

// ─── Analysis ─────────────────────────────────────────────────────────────────

async function analyzeTraces() {
  const meanMotion = state.motionSamples.length > 0
    ? state.motionSamples.reduce((a,b)=>a+b,0) / state.motionSamples.length
    : 0;

  const payload = {
    rgb_traces: state.rgbTraces,
    timestamps: state.timestamps,
    motion_pxps: meanMotion,
  };

  const resp = await fetch(`${CONFIG.API_BASE}/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: 'Server error' }));
    throw new Error(err.detail || `HTTP ${resp.status}`);
  }

  return resp.json();
}

// ─── Results rendering ────────────────────────────────────────────────────────

function renderResults(data) {
  state.lastResult = data;

  if (!data.quality_ok) {
    showError(
      data.error_message || 'Signal quality too low',
      "Couldn't get a reliable reading. Check lighting, hold still, and try again."
    );
    return;
  }

  // HR
  dom.resultHR.textContent = Math.round(data.hr_bpm);

  // RR
  dom.resultRR.textContent = data.rr_bpm > 0 ? Math.round(data.rr_bpm) : '--';

  // Confidence badge
  dom.resultConf.textContent = data.confidence;
  dom.resultConf.className = `confidence-badge ${data.confidence}`;

  // SNR quality bar
  const snrPct = Math.min(100, Math.max(0, (data.snr_db / 15) * 100));
  dom.resultSNR.style.width = `${snrPct}%`;

  // Waveform
  if (data.pulse_waveform && data.pulse_waveform.length > 0) {
    drawWaveform(dom.resultWaveform, data.pulse_waveform, '#00ffa3', true);
  }

  // Warn if unusual
  if (data.error_message) {
    const warn = document.getElementById('result-warning');
    if (warn) { warn.textContent = `⚠️ ${data.error_message}`; warn.style.display = 'block'; }
  }

  // Summary
  dom.summaryLoading.style.display = 'flex';
  dom.summaryText.style.display = 'none';
  if (data.summary) {
    dom.summaryLoading.style.display = 'none';
    dom.summaryText.style.display = 'block';
    dom.summaryText.textContent = data.summary;
  }

  setScreen('results');
}

// ─── Reference HR ─────────────────────────────────────────────────────────────

function updateReferenceError() {
  if (!state.lastResult) return;
  const ref = parseFloat(dom.refInput.value);
  if (isNaN(ref) || ref < 30 || ref > 200) {
    dom.refResult.innerHTML = '<span style="color:var(--text-muted)">Enter a reading from a pulse oximeter</span>';
    return;
  }
  const err = Math.abs(state.lastResult.hr_bpm - ref);
  const cls = err <= 5 ? 'good' : err <= 10 ? 'ok' : 'bad';
  const emoji = err <= 5 ? '✅' : err <= 10 ? '⚠️' : '❌';
  dom.refResult.innerHTML = `
    ${emoji} Error: <span class="error-val ${cls}">${err.toFixed(1)} bpm</span>
    <span style="color:var(--text-muted);font-size:0.8rem;margin-left:8px">
      (App: ${Math.round(state.lastResult.hr_bpm)} bpm · Ref: ${ref} bpm)
    </span>`;
}

// ─── Error screen ─────────────────────────────────────────────────────────────

function showError(title, msg) {
  dom.errorTitle.textContent = title;
  dom.errorMsg.textContent = msg;
  setScreen('error');
}

// ─── Upload mode ──────────────────────────────────────────────────────────────

function setupUploadMode() {
  dom.uploadZone.addEventListener('click', () => dom.fileInput.click());
  dom.fileInput.addEventListener('change', e => handleFileUpload(e.target.files[0]));

  dom.uploadZone.addEventListener('dragover', e => {
    e.preventDefault();
    dom.uploadZone.classList.add('drag-over');
  });
  dom.uploadZone.addEventListener('dragleave', () => dom.uploadZone.classList.remove('drag-over'));
  dom.uploadZone.addEventListener('drop', e => {
    e.preventDefault();
    dom.uploadZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('video/')) handleFileUpload(file);
  });
}

async function handleFileUpload(file) {
  if (!file) return;
  const url = URL.createObjectURL(file);
  dom.uploadVideoEl.src = url;
  await new Promise(r => { dom.uploadVideoEl.onloadedmetadata = r; });
  dom.uploadVideoEl.style.display = 'block';

  setScreen('scan');
  await processVideoFile(dom.uploadVideoEl);
}

async function processVideoFile(videoEl) {
  // Similar to live scan but reads from an uploaded video
  state.rgbTraces = [];
  state.timestamps = [];
  state.motionSamples = [];
  livePos = new LivePOS(CONFIG.TARGET_FPS);

  // Use the upload video element as source for MediaPipe
  const canvas = document.createElement('canvas');
  canvas.width = videoEl.videoWidth || 640;
  canvas.height = videoEl.videoHeight || 480;
  const ctx = canvas.getContext('2d');

  videoEl.currentTime = CONFIG.WARMUP_S;
  await new Promise(r => { videoEl.onseeked = r; });

  setScreen('processing');

  // Process frame by frame at ~10 fps for speed
  const duration = Math.min(videoEl.duration - CONFIG.WARMUP_S, 35);
  const frameStep = 1 / 10;

  for (let t = 0; t < duration; t += frameStep) {
    videoEl.currentTime = CONFIG.WARMUP_S + t;
    await new Promise(r => { videoEl.onseeked = r; });
    ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);

    try {
      if (faceLandmarker) {
        const result = faceLandmarker.detectForVideo(videoEl, t * 1000);
        if (result.faceLandmarks && result.faceLandmarks.length > 0) {
          const lm = result.faceLandmarks[0];
          const colors = extractROIColorFromCanvas(ctx, lm, canvas.width, canvas.height);
          if (colors) {
            state.rgbTraces.push(colors);
            state.timestamps.push(t);
          }
        }
      }
    } catch (e) {}

    if (state.timestamps.length >= CONFIG.SCAN_DURATION_S * 10) break;
  }

  await runAnalysis();
}

function extractROIColorFromCanvas(ctx, landmarks, w, h) {
  // Simplified: just sample forehead region
  const pts = ROI_INDICES.FOREHEAD_ONLY.map(i => getLandmarkPoint(landmarks, i, w, h));
  const xs = pts.map(p=>p[0]), ys = pts.map(p=>p[1]);
  const x0 = Math.max(0, Math.floor(Math.min(...xs)));
  const y0 = Math.max(0, Math.floor(Math.min(...ys)));
  const x1 = Math.min(w-1, Math.ceil(Math.max(...xs)));
  const y1 = Math.min(h-1, Math.ceil(Math.max(...ys)));
  if (x1-x0 < 4 || y1-y0 < 4) return null;
  const imgData = ctx.getImageData(x0, y0, x1-x0, y1-y0);
  const d = imgData.data;
  let r=0,g=0,b=0,n=0;
  for (let i=0;i<d.length;i+=4) { r+=d[i];g+=d[i+1];b+=d[i+2];n++; }
  return n > 0 ? [r/n,g/n,b/n] : null;
}

// ─── Main flow ────────────────────────────────────────────────────────────────

async function runAnalysis() {
  setScreen('processing');
  stopCamera();

  try {
    const data = await analyzeTraces();
    renderResults(data);
  } catch (err) {
    console.error('Analysis error:', err);
    showError('Analysis failed', err.message || 'Could not connect to the analysis server. Make sure the backend is running on port 8000.');
  }
}

async function onStartScan() {
  setScreen('scan');

  // Load MediaPipe if not yet done
  if (!faceLandmarker) {
    setGuidance('looking', '⏳ Loading face tracker…');
    try {
      await initMediaPipe();
    } catch (e) {
      showError('Face tracker failed to load', 'Could not initialize MediaPipe. Check your internet connection and try again.');
      return;
    }
  }

  const camOk = await startCamera();
  if (!camOk) {
    showError(
      'Camera access denied',
      'Allow camera access in your browser settings, or use the "Upload video" mode instead.'
    );
    return;
  }

  // Wait for video to be ready
  await sleep(300);

  await startScan();
  await runAnalysis();
}

function onCancel() {
  if (state.animFrame) cancelAnimationFrame(state.animFrame);
  stopCamera();
  setScreen('start');
}

function onScanAgain() {
  state.lastResult = null;
  dom.refInput.value = '';
  dom.refResult.innerHTML = '<span style="color:var(--text-muted)">Enter a reading from a pulse oximeter</span>';
  setScreen('start');
}

// ─── Mode toggle ──────────────────────────────────────────────────────────────

function setMode(mode) {
  state.mode = mode;
  dom.modeLive.classList.toggle('active', mode === 'live');
  dom.modeUpload.classList.toggle('active', mode === 'upload');

  const liveSection = document.getElementById('live-section');
  const uploadSection = document.getElementById('upload-section');
  if (liveSection) liveSection.style.display = mode === 'live' ? 'flex' : 'none';
  if (uploadSection) uploadSection.style.display = mode === 'upload' ? 'flex' : 'none';
}

// ─── Init ─────────────────────────────────────────────────────────────────────

function init() {
  // Bind buttons
  dom.startBtn?.addEventListener('click', onStartScan);
  dom.startUploadBtn?.addEventListener('click', () => dom.fileInput.click());
  dom.cancelBtn?.addEventListener('click', onCancel);
  dom.scanAgainBtn?.addEventListener('click', onScanAgain);
  dom.errorRetryBtn?.addEventListener('click', onScanAgain);
  dom.refInput?.addEventListener('input', updateReferenceError);
  dom.modeLive?.addEventListener('click', () => setMode('live'));
  dom.modeUpload?.addEventListener('click', () => setMode('upload'));

  // Camera selection handling
  dom.cameraSelect?.addEventListener('change', async (e) => {
    const newId = e.target.value;
    state.selectedDeviceId = newId;
    state.userExplicitlySelectedCamera = true;
    localStorage.setItem('preferredCameraId', newId);

    if (dom.cameraHint) {
      const chosen = state.availableCameras.find(d => d.deviceId === newId);
      if (chosen && scoreCamera(chosen) < 0) {
        dom.cameraHint.textContent = 'Phone camera selected';
        dom.cameraHint.style.color = 'var(--amber)';
      } else {
        dom.cameraHint.textContent = 'Laptop webcam active';
        dom.cameraHint.style.color = 'var(--green)';
      }
    }

    // If currently in a scan, hot-swap the video stream
    if (state.screen === 'scan' && state.stream) {
      stopCamera();
      await startCamera(newId);
    }
  });

  // Listen for hardware camera connect / disconnect
  if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
    navigator.mediaDevices.addEventListener('devicechange', updateCameraList);
  }

  // Initial camera list check
  updateCameraList();

  setupUploadMode();
  setScreen('start');

  // Preload MediaPipe in background for faster first scan
  setTimeout(() => {
    if (!faceLandmarker) initMediaPipe().catch(() => {});
  }, 2000);
}

document.addEventListener('DOMContentLoaded', init);
