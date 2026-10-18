"""
config.py - Configurações Centrais do Sistema de Monitoramento de Sono (DMS)
DriverGuard DMS: Detecção de Fadiga e Sonolência ao Volante
"""

import os
from pathlib import Path

# ==========================================
# DIRETÓRIOS E MODELOS
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
SOUNDS_DIR = ASSETS_DIR / "sounds"
MODEL_PATH = BASE_DIR / "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

# Garantir existência de diretórios essenciais
SOUNDS_DIR.mkdir(parents=True, exist_ok=True)

# Sons de Alerta (WAV gerados automaticamente caso ausentes)
SOUND_DROWSY = SOUNDS_DIR / "alarm_drowsy.wav"
SOUND_YAWN = SOUNDS_DIR / "alert_yawn.wav"
SOUND_DISTRACTED = SOUNDS_DIR / "alert_distracted.wav"

# ==========================================
# PARÂMETROS DE DETECÇÃO (THRESHOLDS PADRÃO)
# ==========================================
# Olhos (EAR - Eye Aspect Ratio)
DEFAULT_EAR_THRESHOLD = 0.21       # Abaixo deste valor, os olhos são considerados fechados
DROWSY_TIME_SEC = 1.3              # Tempo de olhos fechados para disparar alerta de sono (segundos)
CRITICAL_SLEEP_SEC = 2.5           # Tempo para alerta crítico / micro-sono de emergência

# Boca (MAR - Mouth Aspect Ratio)
DEFAULT_MAR_THRESHOLD = 0.65       # Acima deste valor, indica boca amplamente aberta
YAWN_TIME_SEC = 1.8                # Duração da boca aberta para caracterizar bocejo genuíno
YAWN_COOLDOWN_SEC = 4.0            # Cooldown entre contagens consecutivas de bocejos

# Pose da Cabeça (Head Pose)
HEAD_NOD_PITCH_THRESHOLD = -8.0    # Ângulo abaixo do qual indica cabeça tombada para frente (graus)
HEAD_NOD_RATIO_THRESHOLD = 0.55    # Razão queixo/testa 2D comprimida (confirmação visual)
HEAD_NOD_DELTA_THRESHOLD = 12.0    # Queda angular relativa à posição de repouso
HEAD_NOD_TIME_SEC = 0.7            # Tempo de cabeça caída para disparar alarme de sono (0.7s)
HEAD_YAW_THRESHOLD = 28.0          # Rotação lateral da face (olhando para os lados)
DISTRACTION_TIME_SEC = 2.0         # Tempo com cabeça virada para alerta de desatenção

# Calibração Automática
CALIBRATION_DURATION_SEC = 3.0     # Duração do processo de auto-calibração do condutor

# ==========================================
# VÍDEO E CÂMERA
# ==========================================
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS_TARGET = 30

# ==========================================
# PALETA DE CORES HUD (BGR para OpenCV)
# ==========================================
COLOR_BACKGROUND_CARD = (25, 25, 30)      # Cinza escuro profundo
COLOR_CARD_BORDER = (55, 55, 65)          # Borda sutil
COLOR_TEXT_PRIMARY = (250, 250, 250)      # Branco puro
COLOR_TEXT_SECONDARY = (180, 180, 185)    # Cinza claro legível

COLOR_OK = (80, 220, 100)                 # Verde vibrante (Motorista Atento)
COLOR_WARNING = (30, 180, 255)            # Âmbar / Laranja (Bocejo / Atenção)
COLOR_DANGER = (50, 50, 240)              # Vermelho intenso (Sonolência / Emergência)
COLOR_ACCENT = (255, 180, 0)              # Azul ciano elétrico (Telemetria)
COLOR_CALIBRATING = (255, 215, 0)         # Amarelo elétrico
