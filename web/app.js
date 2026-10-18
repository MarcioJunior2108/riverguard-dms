/**
 * DriverGuard DMS Mobile - Aplicação Web com MediaPipe Tasks Vision WebAssembly
 * Sistema Completo de Visão Computacional para Detecção de Fadiga Veicular
 */

import {
  FaceLandmarker,
  FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

// ==========================================
// CONFIGURAÇÕES E LIMIARES BIOMÉTRICOS
// ==========================================
const CONFIG = {
  DEFAULT_EAR_THRESHOLD: 0.22,
  DEFAULT_MAR_THRESHOLD: 0.65,
  DROWSY_TIME_SEC: 1.3,
  CRITICAL_SLEEP_SEC: 2.5,
  YAWN_TIME_SEC: 1.8,
  YAWN_COOLDOWN_SEC: 4.0,
  HEAD_NOD_PITCH_THRESHOLD: -8.0,
  HEAD_NOD_DELTA_THRESHOLD: 12.0,
  HEAD_NOD_RATIO_THRESHOLD: 0.55,
  HEAD_NOD_TIME_SEC: 0.7,
  HEAD_YAW_THRESHOLD: 28.0,
  DISTRACTION_TIME_SEC: 2.0,
  CALIBRATION_DURATION_SEC: 3.0,
};

// Índices de Landmarks Faciais
const INDICES = {
  // Olho Esquerdo (do ponto de vista do observador)
  LEFT_EYE: [33, 160, 158, 133, 153, 144],
  // Olho Direito
  RIGHT_EYE: [362, 385, 387, 263, 373, 380],
  // Lábios Internos para MAR
  MOUTH_INNER: {
    upperMid: 13,
    lowerMid: 14,
    upperLeft: 82,
    lowerLeft: 87,
    upperRight: 312,
    lowerRight: 317,
    cornerLeft: 78,
    cornerRight: 308
  },
  // Pontos Anatômicos da Cabeça
  HEAD: {
    forehead: 10,
    noseTip: 1,
    chin: 152,
    leftCheek: 33,
    rightCheek: 263
  }
};

// ==========================================
// ESTADOS OPERACIONAIS DO CONDUTOR
// ==========================================
const DriverState = {
  ATTENTIVE: "ATTENTIVE",
  DROWSY: "DROWSY",
  CRITICAL_SLEEP: "CRITICAL_SLEEP",
  HEAD_NODDING: "HEAD_NODDING",
  YAWNING: "YAWNING",
  DISTRACTED: "DISTRACTED",
  CALIBRATING: "CALIBRATING",
  NO_FACE: "NO_FACE"
};

// ==========================================
// SINTETIZADOR DE ÁUDIO WEB (Web Audio API)
// ==========================================
class AudioSynthesizer {
  constructor() {
    this.ctx = null;
    this.isMuted = false;
    this.activeOscillators = [];
    this.lastAlarmTime = 0;
  }

  init() {
    if (!this.ctx) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (AudioCtx) this.ctx = new AudioCtx();
    }
    if (this.ctx && this.ctx.state === "suspended") {
      this.ctx.resume();
    }
  }

  toggleMute() {
    this.isMuted = !this.isMuted;
    if (this.isMuted) this.stopCurrent();
    return this.isMuted;
  }

  stopCurrent() {
    this.activeOscillators.forEach(osc => {
      try { osc.stop(); osc.disconnect(); } catch (e) {}
    });
    this.activeOscillators = [];
  }

  playCriticalAlarm() {
    if (this.isMuted) return;
    this.init();
    if (!this.ctx) return;

    const now = this.ctx.currentTime;
    if (now - this.lastAlarmTime < 0.25) return;
    this.lastAlarmTime = now;

    // Sirene de emergência automotiva (dual tone alternado 1200Hz / 850Hz)
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();

    osc.type = "sawtooth";
    osc.frequency.setValueAtTime(1200, now);
    osc.frequency.exponentialRampToValueAtTime(800, now + 0.22);

    gain.gain.setValueAtTime(0.35, now);
    gain.gain.exponentialRampToValueAtTime(0.01, now + 0.22);

    osc.connect(gain);
    gain.connect(this.ctx.destination);

    osc.start(now);
    osc.stop(now + 0.23);
  }

  playWarningChime() {
    if (this.isMuted) return;
    this.init();
    if (!this.ctx) return;

    const now = this.ctx.currentTime;
    if (now - this.lastAlarmTime < 1.2) return;
    this.lastAlarmTime = now;

    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();

    osc.type = "sine";
    osc.frequency.setValueAtTime(587.33, now); // D5
    osc.frequency.exponentialRampToValueAtTime(440.00, now + 0.35); // A4

    gain.gain.setValueAtTime(0.2, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.35);

    osc.connect(gain);
    gain.connect(this.ctx.destination);

    osc.start(now);
    osc.stop(now + 0.36);
  }
}

// ==========================================
// APLICAÇÃO PRINCIPAL DRIVERGUARD
// ==========================================
class DriverGuardApp {
  constructor() {
    // Elementos DOM
    this.video = document.getElementById("webcam");
    this.canvas = document.getElementById("hud-canvas");
    this.ctx = this.canvas.getContext("2d");
    this.startOverlay = document.getElementById("start-overlay");
    this.btnStart = document.getElementById("btn-start");
    this.alarmBorder = document.getElementById("alarm-border");
    this.statusBanner = document.getElementById("status-banner");
    this.statusTitle = document.getElementById("status-title");
    this.statusSub = document.getElementById("status-sub");
    this.fpsCounter = document.getElementById("fps-counter");
    this.telemetryPanel = document.getElementById("telemetry-panel");

    // Botões do Dock
    this.btnCalibrate = document.getElementById("btn-calibrate");
    this.btnCameraFlip = document.getElementById("btn-camera-flip");
    this.btnMute = document.getElementById("btn-mute");
    this.btnHaptic = document.getElementById("btn-haptic");
    this.btnTogglePanel = document.getElementById("btn-toggle-panel");
    this.btnToggleTelemetry = document.getElementById("btn-toggle-telemetry");

    // Indicadores Telemetria
    this.valEar = document.getElementById("val-ear");
    this.valMar = document.getElementById("val-mar");
    this.valHead = document.getElementById("val-head");
    this.valFatigue = document.getElementById("val-fatigue");
    this.meterEar = document.getElementById("meter-ear");
    this.meterMar = document.getElementById("meter-mar");
    this.meterPitch = document.getElementById("meter-pitch");
    this.meterFatigue = document.getElementById("meter-fatigue");
    this.cntBlinks = document.getElementById("cnt-blinks");
    this.cntYawns = document.getElementById("cnt-yawns");
    this.cntDrowsy = document.getElementById("cnt-drowsy");

    // Audio & Haptic
    this.audio = new AudioSynthesizer();
    this.hapticEnabled = true;

    // Câmera & MediaPipe
    this.landmarker = null;
    this.currentFacingMode = "user"; // Câmera frontal padrão
    this.isRunning = false;
    this.lastVideoTime = -1;
    this.prevFpsTime = performance.now();
    this.fps = 30;

    // FSM State Tracking
    this.currentState = DriverState.ATTENTIVE;
    this.earThreshold = CONFIG.DEFAULT_EAR_THRESHOLD;
    this.marThreshold = CONFIG.DEFAULT_MAR_THRESHOLD;
    this.baselinePitch = 0.0;

    // Timers
    this.eyeClosedStartTime = null;
    this.mouthOpenStartTime = null;
    this.headNodStartTime = null;
    this.headYawStartTime = null;
    this.headDownOpenEyesStartTime = null;
    this.lastYawnRecordedTime = 0;
    this.lastFaceSeenTime = performance.now() / 1000;

    // Durações
    this.eyeClosedDuration = 0.0;
    this.mouthOpenDuration = 0.0;
    this.headNodDuration = 0.0;
    this.headYawDuration = 0.0;
    this.headDownOpenEyesDuration = 0.0;

    // Contadores
    this.blinkCount = 0;
    this.yawnCount = 0;
    this.drowsyEventsCount = 0;
    this.isCurrentlyBlinking = false;
    this.fatigueIndex = 0;

    // Calibração
    this.isCalibrating = false;
    this.calibrationStartTime = 0;
    this.calibEarSamples = [];
    this.calibPitchSamples = [];

    // Flash visual
    this.isFlashing = false;
    this.lastFlashToggle = 0;
  }

  async init() {
    this.bindEvents();
    this.resizeCanvas();
    window.addEventListener("resize", () => this.resizeCanvas());
  }

  bindEvents() {
    this.btnStart.addEventListener("click", async () => {
      this.btnStart.textContent = "CARREGANDO MODELO NEURAL...";
      this.btnStart.disabled = true;
      this.audio.init();
      await this.startSystem();
    });

    this.btnCalibrate.addEventListener("click", () => this.startCalibration());

    this.btnCameraFlip.addEventListener("click", () => this.flipCamera());

    this.btnMute.addEventListener("click", () => {
      const isMuted = this.audio.toggleMute();
      document.getElementById("mute-icon").textContent = isMuted ? "🔇" : "🔔";
      document.getElementById("mute-text").textContent = isMuted ? "MUDO" : "Áudio ON";
      this.btnMute.classList.toggle("active", !isMuted);
    });

    this.btnHaptic.addEventListener("click", () => {
      this.hapticEnabled = !this.hapticEnabled;
      this.btnHaptic.classList.toggle("active", this.hapticEnabled);
      if (this.hapticEnabled && navigator.vibrate) navigator.vibrate(50);
    });

    const togglePanel = () => {
      this.telemetryPanel.classList.toggle("collapsed");
    };
    this.btnTogglePanel.addEventListener("click", togglePanel);
    this.btnToggleTelemetry.addEventListener("click", togglePanel);
  }

  resizeCanvas() {
    this.canvas.width = window.innerWidth;
    this.canvas.height = window.innerHeight;
  }

  async startSystem() {
    try {
      // 1. Inicializar MediaPipe Vision Tasks WebAssembly
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
      );

      this.landmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
          delegate: "GPU"
        },
        outputFaceBlendshapes: true,
        outputFacialTransformationMatrixes: true,
        runningMode: "VIDEO",
        numFaces: 1
      });

      // 2. Iniciar Câmera Frontal
      await this.startCamera();

      this.startOverlay.classList.add("hidden");
      this.isRunning = true;
      requestAnimationFrame(() => this.processLoop());
    } catch (err) {
      console.error("Erro ao inicializar:", err);
      alert("Não foi possível acessar a câmera ou carregar o modelo de visão. Verifique permissões: " + err.message);
      this.btnStart.textContent = "TENTAR NOVAMENTE";
      this.btnStart.disabled = false;
    }
  }

  async startCamera() {
    if (this.video.srcObject) {
      this.video.srcObject.getTracks().forEach(track => track.stop());
    }

    const constraints = {
      video: {
        facingMode: this.currentFacingMode,
        width: { ideal: 1280 },
        height: { ideal: 720 }
      },
      audio: false
    };

    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    this.video.srcObject = stream;
    await new Promise(resolve => {
      this.video.onloadedmetadata = () => {
        this.video.play();
        resolve();
      };
    });
  }

  async flipCamera() {
    this.currentFacingMode = this.currentFacingMode === "user" ? "environment" : "user";
    this.video.style.transform = this.currentFacingMode === "user" ? "scaleX(-1)" : "none";
    await this.startCamera();
  }

  startCalibration() {
    this.isCalibrating = true;
    this.calibrationStartTime = performance.now() / 1000;
    this.calibEarSamples = [];
    this.calibPitchSamples = [];
    this.updateStatusBanner("CALIBRANDO PERFIL...", "Mantenha a cabeça ereta e olhos abertos (3s)", "state-calibrating");
  }

  // ==========================================
  // CÁLCULOS BIOMÉTRICOS MATEMÁTICOS
  // ==========================================
  dist2D(p1, p2) {
    const dx = p1.x - p2.x;
    const dy = p1.y - p2.y;
    return Math.sqrt(dx * dx + dy * dy);
  }

  calculateEAR(landmarks) {
    // Olho Esquerdo
    const lP1 = landmarks[INDICES.LEFT_EYE[0]];
    const lP2 = landmarks[INDICES.LEFT_EYE[1]];
    const lP3 = landmarks[INDICES.LEFT_EYE[2]];
    const lP4 = landmarks[INDICES.LEFT_EYE[3]];
    const lP5 = landmarks[INDICES.LEFT_EYE[4]];
    const lP6 = landmarks[INDICES.LEFT_EYE[5]];

    const lVert = this.dist2D(lP2, lP6) + this.dist2D(lP3, lP5);
    const lHoriz = 2.0 * this.dist2D(lP1, lP4);
    const earLeft = lHoriz > 0 ? lVert / lHoriz : 0.0;

    // Olho Direito
    const rP1 = landmarks[INDICES.RIGHT_EYE[0]];
    const rP2 = landmarks[INDICES.RIGHT_EYE[1]];
    const rP3 = landmarks[INDICES.RIGHT_EYE[2]];
    const rP4 = landmarks[INDICES.RIGHT_EYE[3]];
    const rP5 = landmarks[INDICES.RIGHT_EYE[4]];
    const rP6 = landmarks[INDICES.RIGHT_EYE[5]];

    const rVert = this.dist2D(rP2, rP6) + this.dist2D(rP3, rP5);
    const rHoriz = 2.0 * this.dist2D(rP1, rP4);
    const earRight = rHoriz > 0 ? rVert / rHoriz : 0.0;

    return (earLeft + earRight) / 2.0;
  }

  calculateMAR(landmarks) {
    const m = INDICES.MOUTH_INNER;
    const pCornerL = landmarks[m.cornerLeft];
    const pCornerR = landmarks[m.cornerRight];
    const pMidTop = landmarks[m.upperMid];
    const pMidBot = landmarks[m.lowerMid];
    const pLeftTop = landmarks[m.upperLeft];
    const pLeftBot = landmarks[m.lowerLeft];
    const pRightTop = landmarks[m.upperRight];
    const pRightBot = landmarks[m.lowerRight];

    const v1 = this.dist2D(pLeftTop, pLeftBot);
    const v2 = this.dist2D(pMidTop, pMidBot);
    const v3 = this.dist2D(pRightTop, pRightBot);
    const horiz = 2.0 * this.dist2D(pCornerL, pCornerR);

    return horiz > 0 ? (v1 + v2 + v3) / horiz : 0.0;
  }

  calculateHeadPose(landmarks) {
    const pForehead = landmarks[INDICES.HEAD.forehead];
    const pChin = landmarks[INDICES.HEAD.chin];
    const pNose = landmarks[INDICES.HEAD.noseTip];
    const pLeft = landmarks[INDICES.HEAD.leftCheek];
    const pRight = landmarks[INDICES.HEAD.rightCheek];

    // Vetor espinhal do queixo para a testa
    const vy = pChin.y - pForehead.y;
    const vz = (pChin.z || 0.0) - (pForehead.z || 0.0);
    const norm = Math.sqrt(vy * vy + vz * vz);
    const pitch = norm > 0 ? -Math.asin(Math.max(-1, Math.min(1, vz / norm))) * (180.0 / Math.PI) : 0.0;

    // Rotação horizontal (Yaw)
    const yaw = ((pRight.z || 0.0) - (pLeft.z || 0.0)) * 120.0;

    // Razão de perspectiva 2D
    const noseChinDist = this.dist2D(pNose, pChin);
    const foreheadNoseDist = this.dist2D(pForehead, pNose);
    const chinRatio = foreheadNoseDist > 0 ? noseChinDist / foreheadNoseDist : 1.0;

    return { pitch, yaw, chinRatio };
  }

  // ==========================================
  // LOOP DE INFERÊNCIA E MÁQUINA DE ESTADOS
  // ==========================================
  processLoop() {
    if (!this.isRunning) return;

    const nowMs = performance.now();
    const nowSec = nowMs / 1000;

    // FPS
    const diff = nowMs - this.prevFpsTime;
    if (diff > 0) {
      this.fps = 0.9 * this.fps + 0.1 * (1000 / diff);
      this.prevFpsTime = nowMs;
      this.fpsCounter.textContent = `FPS: ${Math.round(this.fps)}`;
    }

    // Toggle flash visual
    if (nowSec - this.lastFlashToggle > 0.22) {
      this.isFlashing = !this.isFlashing;
      this.lastFlashToggle = nowSec;
    }

    if (this.video.currentTime !== this.lastVideoTime && this.landmarker) {
      this.lastVideoTime = this.video.currentTime;
      const results = this.landmarker.detectForVideo(this.video, nowMs);
      this.handleVisionResults(results, nowSec);
    }

    requestAnimationFrame(() => this.processLoop());
  }

  handleVisionResults(results, now) {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    const hasFace = results && results.faceLandmarks && results.faceLandmarks.length > 0;

    // ==========================================
    // TRATAMENTO DE PERDA DE ROSTO / OCLUSÃO
    // ==========================================
    if (!hasFace) {
      if (this.headNodStartTime !== null) {
        this.headNodDuration = now - this.headNodStartTime;
        if (this.headNodDuration >= CONFIG.HEAD_NOD_TIME_SEC) {
          if (this.currentState !== DriverState.HEAD_NODDING) this.drowsyEventsCount++;
          this.currentState = DriverState.HEAD_NODDING;
          this.triggerAlerts();
          this.updateHUD(null, 0, 0, 0, 0);
          return;
        }
      }

      if (now - this.lastFaceSeenTime > 1.5) {
        this.resetTimers();
        this.currentState = DriverState.NO_FACE;
        this.updateStatusBanner("SEM CONDUTOR", "Posicione a câmera frontal em seu rosto", "state-neutral");
      }
      this.alarmBorder.className = "alarm-border";
      return;
    }

    this.lastFaceSeenTime = now;
    const landmarks = results.faceLandmarks[0];
    const blendshapes = results.faceBlendshapes && results.faceBlendshapes[0] ? results.faceBlendshapes[0].categories : [];

    // 1. Extração Biométrica
    const ear = this.calculateEAR(landmarks);
    const mar = this.calculateMAR(landmarks);
    const { pitch, yaw, chinRatio } = this.calculateHeadPose(landmarks);

    // Mapeamento de blendshapes
    const blinkLeft = blendshapes.find(b => b.categoryName === "eyeBlinkLeft")?.score || 0;
    const blinkRight = blendshapes.find(b => b.categoryName === "eyeBlinkRight")?.score || 0;
    const blinkScore = (blinkLeft + blinkRight) / 2.0;

    // Modo Calibração
    if (this.isCalibrating) {
      this.processCalibration(ear, pitch, now);
      this.drawFaceBox(landmarks, "CALIBRANDO", "#00e5ff");
      return;
    }

    // 2. Olhos
    const isEyesClosed = (ear < this.earThreshold) || (blinkScore > 0.65);
    if (isEyesClosed) {
      if (this.eyeClosedStartTime === null) this.eyeClosedStartTime = now;
      this.eyeClosedDuration = now - this.eyeClosedStartTime;
      this.isCurrentlyBlinking = true;
    } else {
      if (this.isCurrentlyBlinking) {
        this.blinkCount++;
        this.isCurrentlyBlinking = false;
      }
      this.eyeClosedStartTime = null;
      this.eyeClosedDuration = 0.0;
    }

    // 3. Boca
    const isMouthOpen = mar > this.marThreshold;
    if (isMouthOpen) {
      if (this.mouthOpenStartTime === null) this.mouthOpenStartTime = now;
      this.mouthOpenDuration = now - this.mouthOpenStartTime;
      if (this.mouthOpenDuration >= CONFIG.YAWN_TIME_SEC) {
        if (now - this.lastYawnRecordedTime > CONFIG.YAWN_COOLDOWN_SEC) {
          this.yawnCount++;
          this.lastYawnRecordedTime = now;
        }
      }
    } else {
      this.mouthOpenStartTime = null;
      this.mouthOpenDuration = 0.0;
    }

    // 4. Cabeça Baixa e FUSÃO MULTI-MODAL DE SEGURANÇA
    const isPitchDown = (pitch < CONFIG.HEAD_NOD_PITCH_THRESHOLD) ||
                        (pitch < (this.baselinePitch - CONFIG.HEAD_NOD_DELTA_THRESHOLD));
    const isRatioDown = chinRatio < CONFIG.HEAD_NOD_RATIO_THRESHOLD;
    const isHeadDropped = isPitchDown || isRatioDown;

    // REGRA DE OURO: Cochilo com cabeça caída EXIGE Cabeça Inclinada + Olhos Fechados!
    const isSleepNodding = isHeadDropped && isEyesClosed;

    if (isSleepNodding) {
      if (this.headNodStartTime === null) this.headNodStartTime = now;
      this.headNodDuration = now - this.headNodStartTime;
      this.headDownOpenEyesStartTime = null;
      this.headDownOpenEyesDuration = 0.0;
    } else {
      this.headNodStartTime = null;
      this.headNodDuration = 0.0;

      // Se a cabeça está voltada para baixo com os OLHOS ABERTOS:
      if (isHeadDropped && !isEyesClosed) {
        if (this.headDownOpenEyesStartTime === null) this.headDownOpenEyesStartTime = now;
        this.headDownOpenEyesDuration = now - this.headDownOpenEyesStartTime;
      } else {
        this.headDownOpenEyesStartTime = null;
        this.headDownOpenEyesDuration = 0.0;
      }
    }

    // 5. Desvio Lateral (Yaw)
    const isLookingAway = Math.abs(yaw) > CONFIG.HEAD_YAW_THRESHOLD;
    if (isLookingAway) {
      if (this.headYawStartTime === null) this.headYawStartTime = now;
      this.headYawDuration = now - this.headYawStartTime;
    } else {
      this.headYawStartTime = null;
      this.headYawDuration = 0.0;
    }

    // ==========================================
    // 6. MÁQUINA DE ESTADOS FINITA (FSM)
    // ==========================================
    let boxColor = "#00e676";
    let boxLabel = "CONDUTOR ATENTO";

    if (this.eyeClosedDuration >= CONFIG.CRITICAL_SLEEP_SEC) {
      if (this.currentState !== DriverState.CRITICAL_SLEEP) this.drowsyEventsCount++;
      this.currentState = DriverState.CRITICAL_SLEEP;
      this.updateStatusBanner("PERIGO: MICRO-SONO!", `Olhos fechados: ${this.eyeClosedDuration.toFixed(1)}s - ACORDE!`, "state-danger");
      boxColor = "#ff1744";
      boxLabel = "PERIGO - SONO";

    } else if (this.headNodDuration >= CONFIG.HEAD_NOD_TIME_SEC) {
      if (this.currentState !== DriverState.HEAD_NODDING) this.drowsyEventsCount++;
      this.currentState = DriverState.HEAD_NODDING;
      this.updateStatusBanner("PERIGO: COCHILO AO VOLANTE!", "Cabeça inclinada + Olhos fechados - ACORDE!", "state-danger");
      boxColor = "#ff1744";
      boxLabel = "COCHILO AO VOLANTE";

    } else if (this.eyeClosedDuration >= CONFIG.DROWSY_TIME_SEC) {
      if (this.currentState !== DriverState.DROWSY) this.drowsyEventsCount++;
      this.currentState = DriverState.DROWSY;
      this.updateStatusBanner("ALERTA: SONOLÊNCIA", `Olhos fechados: ${this.eyeClosedDuration.toFixed(1)}s`, "state-danger");
      boxColor = "#ff1744";
      boxLabel = "SONOLÊNCIA";

    } else if (this.headYawDuration >= CONFIG.DISTRACTION_TIME_SEC || this.headDownOpenEyesDuration >= CONFIG.DISTRACTION_TIME_SEC) {
      this.currentState = DriverState.DISTRACTED;
      const sub = this.headDownOpenEyesDuration >= CONFIG.DISTRACTION_TIME_SEC ? "Cabeça baixa prolongada - Olhe para a via" : "Cabeça virada - Olhos na pista";
      this.updateStatusBanner("ATENÇÃO: OLHOS NA VIA", sub, "state-warning");
      boxColor = "#ffab00";
      boxLabel = "DISTRAÇÃO";

    } else if (this.mouthOpenDuration >= CONFIG.YAWN_TIME_SEC) {
      this.currentState = DriverState.YAWNING;
      this.updateStatusBanner("AVISO: BOCEJO PROLONGADO", `Fadiga detectada (${this.mouthOpenDuration.toFixed(1)}s)`, "state-warning");
      boxColor = "#ffab00";
      boxLabel = "BOCEJO";

    } else {
      this.currentState = DriverState.ATTENTIVE;
      this.updateStatusBanner("MOTORISTA ATENTO", "Monitoramento normal ativo", "state-attentive");
      boxColor = "#00e676";
      boxLabel = "CONDUTOR ATENTO";
    }

    // Disparos de Alarme e Feedback Háptico
    this.triggerAlerts();

    // Desenhar Bounding Box no Canvas
    this.drawFaceBox(landmarks, boxLabel, boxColor);

    // Atualizar Telemetria
    this.updateHUD(ear, mar, pitch, yaw);
  }

  triggerAlerts() {
    const isCritical = [DriverState.DROWSY, DriverState.CRITICAL_SLEEP, DriverState.HEAD_NODDING].includes(this.currentState);

    if (isCritical) {
      this.alarmBorder.className = `alarm-border ${this.isFlashing ? "flashing" : ""}`;
      this.audio.playCriticalAlarm();
      // Vibração no celular
      if (this.hapticEnabled && navigator.vibrate) {
        navigator.vibrate([200, 100, 200]);
      }
    } else if (this.currentState === DriverState.YAWNING || this.currentState === DriverState.DISTRACTED) {
      this.alarmBorder.className = "alarm-border";
      this.audio.playWarningChime();
    } else {
      this.alarmBorder.className = "alarm-border";
    }
  }

  processCalibration(ear, pitch, now) {
    this.calibEarSamples.push(ear);
    this.calibPitchSamples.push(pitch);

    const elapsed = now - this.calibrationStartTime;
    if (elapsed >= CONFIG.CALIBRATION_DURATION_SEC) {
      if (this.calibEarSamples.length > 10) {
        const meanEar = this.calibEarSamples.reduce((a, b) => a + b, 0) / this.calibEarSamples.length;
        const meanPitch = this.calibPitchSamples.reduce((a, b) => a + b, 0) / this.calibPitchSamples.length;
        this.earThreshold = Math.max(0.16, Math.min(0.28, meanEar * 0.72));
        this.baselinePitch = meanPitch;
      }
      this.isCalibrating = false;
      this.updateStatusBanner("CALIBRAÇÃO CONCLUÍDA", `EAR: ${this.earThreshold.toFixed(2)} | Pitch: ${this.baselinePitch.toFixed(0)}°`, "state-attentive");
    }
  }

  resetTimers() {
    this.eyeClosedStartTime = null;
    this.mouthOpenStartTime = null;
    this.headNodStartTime = null;
    this.headYawStartTime = null;
    this.headDownOpenEyesStartTime = null;
    this.eyeClosedDuration = 0.0;
    this.mouthOpenDuration = 0.0;
    this.headNodDuration = 0.0;
    this.headYawDuration = 0.0;
    this.headDownOpenEyesDuration = 0.0;
  }

  // ==========================================
  // RENDERIZAÇÃO GRÁFICA NO CANVAS
  // ==========================================
  drawFaceBox(landmarks, label, color) {
    let minX = 1, minY = 1, maxX = 0, maxY = 0;
    for (const p of landmarks) {
      if (p.x < minX) minX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.x > maxX) maxX = p.x;
      if (p.y > maxY) maxY = p.y;
    }

    const cw = this.canvas.width;
    const ch = this.canvas.height;

    // Espelha coordenadas se estiver na câmera frontal
    let x1 = (1.0 - maxX) * cw;
    let x2 = (1.0 - minX) * cw;
    let y1 = minY * ch;
    let y2 = maxY * ch;

    // Padding de conforto
    const padX = (x2 - x1) * 0.12;
    const padY = (y2 - y1) * 0.12;
    x1 = Math.max(0, x1 - padX);
    x2 = Math.min(cw, x2 + padX);
    y1 = Math.max(0, y1 - padY);
    y2 = Math.min(ch, y2 + padY);

    const w = x2 - x1;
    const h = y2 - y1;

    // Retângulo Base com transparência
    this.ctx.strokeStyle = color;
    this.ctx.lineWidth = 2;
    this.ctx.strokeRect(x1, y1, w, h);

    // Cantos Reforçados (L-brackets estilizados)
    const corner = Math.min(24, w / 4);
    this.ctx.lineWidth = 4;
    this.ctx.beginPath();
    // Sup Esq
    this.ctx.moveTo(x1, y1 + corner); this.ctx.lineTo(x1, y1); this.ctx.lineTo(x1 + corner, y1);
    // Sup Dir
    this.ctx.moveTo(x2 - corner, y1); this.ctx.lineTo(x2, y1); this.ctx.lineTo(x2, y1 + corner);
    // Inf Esq
    this.ctx.moveTo(x1, y2 - corner); this.ctx.lineTo(x1, y2); this.ctx.lineTo(x1 + corner, y2);
    // Inf Dir
    this.ctx.moveTo(x2 - corner, y2); this.ctx.lineTo(x2, y2); this.ctx.lineTo(x2, y2 - corner);
    this.ctx.stroke();
  }

  updateHUD(ear, mar, pitch, yaw) {
    if (ear !== null) {
      this.valEar.textContent = ear.toFixed(2);
      this.meterEar.style.width = `${Math.min(100, (ear / 0.40) * 100)}%`;
      this.meterEar.className = `meter-fill ${ear < this.earThreshold ? "fill-danger" : "fill-ok"}`;

      this.valMar.textContent = mar.toFixed(2);
      this.meterMar.style.width = `${Math.min(100, (mar / 0.90) * 100)}%`;
      this.meterMar.className = `meter-fill ${mar > this.marThreshold ? "fill-warning" : "fill-accent"}`;

      this.valHead.textContent = `P:${pitch.toFixed(0)}° / Y:${yaw.toFixed(0)}°`;
      this.meterPitch.style.width = `${Math.max(10, Math.min(100, 50 + pitch * 2))}%`;

      // Cálculo do Índice de Fadiga
      let score = (this.yawnCount * 15.0) + (this.drowsyEventsCount * 25.0);
      if (this.eyeClosedDuration > 0.5) score += Math.min(30, (this.eyeClosedDuration / CONFIG.DROWSY_TIME_SEC) * 30);
      if (this.headNodDuration > 0.3) score += Math.min(30, (this.headNodDuration / CONFIG.HEAD_NOD_TIME_SEC) * 30);
      this.fatigueIndex = Math.min(100, Math.round(score));

      this.valFatigue.textContent = `${this.fatigueIndex}%`;
      this.meterFatigue.style.width = `${this.fatigueIndex}%`;
      this.meterFatigue.className = `meter-fill ${this.fatigueIndex <= 40 ? "fill-ok" : this.fatigueIndex <= 70 ? "fill-warning" : "fill-danger"}`;

      this.cntBlinks.textContent = this.blinkCount;
      this.cntYawns.textContent = this.yawnCount;
      this.cntDrowsy.textContent = this.drowsyEventsCount;
    }
  }

  updateStatusBanner(title, sub, stateClass) {
    this.statusTitle.textContent = title;
    this.statusSub.textContent = sub;
    this.statusBanner.className = `status-banner ${stateClass}`;
  }
}

// Inicializar aplicação ao carregar
window.addEventListener("DOMContentLoaded", () => {
  const app = new DriverGuardApp();
  app.init();

  // Registrar Service Worker para PWA Offline e instalação na tela inicial
  if ("serviceWorker" in navigator) {
    navigator.serviceWorker.register("sw.js").catch(e => console.log("SW bypass:", e));
  }
});
