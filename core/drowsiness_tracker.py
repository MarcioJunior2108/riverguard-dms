"""
core/drowsiness_tracker.py - Rastreador Temporal de Estados de Sono e Fadiga
Implementa uma Máquina de Estados Finita (FSM) invariante a FPS com persistência
de trajetória e tolerância a oclusões por queda de cabeça (Cochilo ao Volante).
"""

from enum import Enum
import time
import numpy as np

import config
from core.face_detector import FaceData


class DriverState(Enum):
    """Estados operacionais do condutor."""
    ATTENTIVE = "ATTENTIVE"             # Condutor atento e desperto
    DROWSY = "DROWSY"                   # Sonolência detectada (olhos fechados > limiar)
    CRITICAL_SLEEP = "CRITICAL_SLEEP"   # Micro-sono crítico (olhos fechados > 2.5s)
    HEAD_NODDING = "HEAD_NODDING"       # Cabeça caída para frente (cochilo/adormecimento ao volante)
    YAWNING = "YAWNING"                 # Bocejo prolongado detectado
    DISTRACTED = "DISTRACTED"           # Cabeça virada lateralmente (olhos fora da via)
    CALIBRATING = "CALIBRATING"         # Em processo de auto-calibração biométrica
    NO_FACE = "NO_FACE"                 # Rosto não detectado no campo visual


class DrowsinessTracker:
    """
    Rastreador de fadiga veicular baseado em janelas temporais contínuas (segundos).
    Independente de oscilações na taxa de quadros (FPS) da câmera.
    """

    def __init__(self):
        # Limiares biométricos
        self.ear_threshold = config.DEFAULT_EAR_THRESHOLD
        self.mar_threshold = config.DEFAULT_MAR_THRESHOLD
        self.drowsy_time_sec = config.DROWSY_TIME_SEC
        self.critical_sleep_sec = config.CRITICAL_SLEEP_SEC
        self.yawn_time_sec = config.YAWN_TIME_SEC
        self.distraction_time_sec = config.DISTRACTION_TIME_SEC
        self.head_nod_time_sec = config.HEAD_NOD_TIME_SEC

        # Referência de inclinação da cabeça em repouso
        self.baseline_pitch: float = 0.0

        # Timers e timestamps contínuos
        self.eye_closed_start_time: float | None = None
        self.mouth_open_start_time: float | None = None
        self.head_nod_start_time: float | None = None
        self.head_yaw_start_time: float | None = None
        self.head_down_open_eyes_start_time: float | None = None
        self.last_yawn_recorded_time: float = 0.0

        # Memória de persistência contra perda de tracking ao abaixar a cabeça
        self.last_face_seen_time: float = time.time()
        self.was_nodding_before_loss: bool = False

        # Contadores estatísticos da viagem
        self.blink_count: int = 0
        self.yawn_count: int = 0
        self.drowsy_events_count: int = 0
        self.is_currently_blinking: bool = False

        # Estado atual e durações ativas
        self.current_state = DriverState.ATTENTIVE
        self.eye_closed_duration: float = 0.0
        self.mouth_open_duration: float = 0.0
        self.head_nod_duration: float = 0.0
        self.head_yaw_duration: float = 0.0
        self.head_down_open_eyes_duration: float = 0.0
        self.is_head_dropped: bool = False

        # Calibração adaptativa do motorista
        self.is_calibrating: bool = False
        self.calibration_start_time: float = 0.0
        self.calibration_ear_samples: list[float] = []
        self.calibration_mar_samples: list[float] = []
        self.calibration_pitch_samples: list[float] = []
        self.calibration_message: str = ""

        # Índice de Fadiga Acumulado (0% a 100%)
        self.fatigue_index: float = 0.0

    def start_calibration(self) -> None:
        """Inicia a calibração de 3 segundos para personalizar os limiares ao condutor."""
        self.is_calibrating = True
        self.calibration_start_time = time.time()
        self.calibration_ear_samples.clear()
        self.calibration_mar_samples.clear()
        self.calibration_pitch_samples.clear()
        self.calibration_message = "Mantenha a cabeça ereta e olhos abertos..."

    def _process_calibration(self, face_data: FaceData, now: float) -> None:
        """Coleta amostras em repouso e ajusta limiares de olhos, boca e inclinação da cabeça."""
        elapsed = now - self.calibration_start_time
        self.calibration_ear_samples.append(face_data.ear_avg)
        self.calibration_mar_samples.append(face_data.mar)
        self.calibration_pitch_samples.append(face_data.head_pitch)

        if elapsed >= config.CALIBRATION_DURATION_SEC:
            if len(self.calibration_ear_samples) > 10:
                mean_ear = float(np.mean(self.calibration_ear_samples))
                mean_mar = float(np.mean(self.calibration_mar_samples))
                self.baseline_pitch = float(np.mean(self.calibration_pitch_samples))

                self.ear_threshold = max(0.16, min(0.28, mean_ear * 0.70))
                self.mar_threshold = max(0.50, min(0.85, mean_mar * 2.2))

                self.calibration_message = f"Calibrado! EAR: {self.ear_threshold:.2f} | Pitch: {self.baseline_pitch:+.0f}°"
            self.is_calibrating = False

    def update(self, face_data: FaceData | None) -> DriverState:
        """
        Atualiza a máquina de estados a cada frame.
        Possui tolerância a oclusões quando a cabeça cai para frente com olhos fechados.
        """
        now = time.time()

        # ==========================================
        # TRATAMENTO DE OCLUSÃO / PERDA MOMENTÂNEA
        # ==========================================
        if face_data is None:
            # Se a cabeça já estava em trajetória de queda com olhos fechados (cochilo real)
            # antes da câmera perder os pontos da face (queixo no peito),
            # o sistema mantém o temporizador de cochilo ativo e dispara o alarme!
            if self.head_nod_start_time is not None:
                self.head_nod_duration = now - self.head_nod_start_time
                if self.head_nod_duration >= self.head_nod_time_sec:
                    if self.current_state != DriverState.HEAD_NODDING:
                        self.drowsy_events_count += 1
                    self.current_state = DriverState.HEAD_NODDING
                    self._calculate_fatigue_index()
                    return self.current_state

            # Se não estava em cochilo e já se passaram mais de 1.5s sem rosto visível:
            if (now - self.last_face_seen_time) > 1.5:
                self._reset_active_timers()
                self.current_state = DriverState.NO_FACE
            return self.current_state

        # Face detectada normalmente: atualiza timestamp de presença
        self.last_face_seen_time = now

        # Modo de Calibração
        if self.is_calibrating:
            self._process_calibration(face_data, now)
            self.current_state = DriverState.CALIBRATING
            return self.current_state

        # ==========================================
        # 1. ANÁLISE DOS OLHOS (EAR + BLENDSHAPES)
        # ==========================================
        is_eyes_closed = face_data.ear_avg < self.ear_threshold
        if "eyeBlinkLeft" in face_data.blendshapes and "eyeBlinkRight" in face_data.blendshapes:
            blink_confidence = (face_data.blendshapes["eyeBlinkLeft"] + face_data.blendshapes["eyeBlinkRight"]) / 2.0
            if blink_confidence > 0.65:
                is_eyes_closed = True

        if is_eyes_closed:
            if self.eye_closed_start_time is None:
                self.eye_closed_start_time = now
            self.eye_closed_duration = now - self.eye_closed_start_time
            self.is_currently_blinking = True
        else:
            if self.is_currently_blinking:
                self.blink_count += 1
                self.is_currently_blinking = False
            self.eye_closed_start_time = None
            self.eye_closed_duration = 0.0

        # ==========================================
        # 2. ANÁLISE DA BOCA (MAR + BOCEJO)
        # ==========================================
        is_mouth_open = face_data.mar > self.mar_threshold
        if "jawOpen" in face_data.blendshapes and face_data.blendshapes["jawOpen"] > 0.45:
            is_mouth_open = True

        if is_mouth_open:
            if self.mouth_open_start_time is None:
                self.mouth_open_start_time = now
            self.mouth_open_duration = now - self.mouth_open_start_time

            if self.mouth_open_duration >= self.yawn_time_sec:
                if (now - self.last_yawn_recorded_time) > config.YAWN_COOLDOWN_SEC:
                    self.yawn_count += 1
                    self.last_yawn_recorded_time = now
        else:
            self.mouth_open_start_time = None
            self.mouth_open_duration = 0.0

        # ==========================================
        # 3. ANÁLISE DE QUEDA DE CABEÇA E DISTRAÇÃO VERTICAL
        # ==========================================
        # Validação geométrica dupla:
        # 1) Ângulo de Pitch 3D abaixo do limiar (-8°) OU queda relativa de 12° em relação ao repouso
        # 2) E/OU Compressão da razão queixo/testa em perspectiva 2D (< 0.55)
        is_pitch_down = (face_data.head_pitch < config.HEAD_NOD_PITCH_THRESHOLD) or \
                        (face_data.head_pitch < (self.baseline_pitch - config.HEAD_NOD_DELTA_THRESHOLD))
        is_ratio_down = face_data.chin_ratio < config.HEAD_NOD_RATIO_THRESHOLD

        self.is_head_dropped = is_pitch_down or is_ratio_down

        # REGRA CRÍTICA DE DMS AUTOMOTIVO (FUSÃO MULTI-MODAL):
        # A cabeça abaixada isoladamente NÃO caracteriza sono, pois o motorista
        # pode estar checando o velocímetro, painel, ar ou marcha.
        # O COCHILO COM CABEÇA CAÍDA (HEAD NODDING) exige OBRIGATORIAMENTE:
        # CABEÇA INCLINADA PARA BAIXO + OLHOS FECHADOS!
        is_sleep_nodding = self.is_head_dropped and is_eyes_closed

        if is_sleep_nodding:
            if self.head_nod_start_time is None:
                self.head_nod_start_time = now
            self.head_nod_duration = now - self.head_nod_start_time
            self.was_nodding_before_loss = True
            # Em situação de cochilo, anula distração de olhos abertos
            self.head_down_open_eyes_start_time = None
            self.head_down_open_eyes_duration = 0.0
        else:
            self.head_nod_start_time = None
            self.head_nod_duration = 0.0
            self.was_nodding_before_loss = False

            # Se a cabeça está abaixada com os OLHOS ABERTOS:
            # O motorista está olhando para o painel / celular.
            # Se for rápido (< 2.0s), comportamento normal de direção (ATTENTIVE).
            # Se for prolongado (> 2.0s), alerta de DISTRAÇÃO (olhos fora da via), NUNCA falso alarme de sono.
            if self.is_head_dropped and not is_eyes_closed:
                if self.head_down_open_eyes_start_time is None:
                    self.head_down_open_eyes_start_time = now
                self.head_down_open_eyes_duration = now - self.head_down_open_eyes_start_time
            else:
                self.head_down_open_eyes_start_time = None
                self.head_down_open_eyes_duration = 0.0

        # ==========================================
        # 4. ANÁLISE DE DESVIO LATERAL (DISTRAÇÃO YAW)
        # ==========================================
        is_looking_away = abs(face_data.head_yaw) > config.HEAD_YAW_THRESHOLD
        if is_looking_away:
            if self.head_yaw_start_time is None:
                self.head_yaw_start_time = now
            self.head_yaw_duration = now - self.head_yaw_start_time
        else:
            self.head_yaw_start_time = None
            self.head_yaw_duration = 0.0

        # ==========================================
        # 5. MÁQUINA DE ESTADOS POR PRIORIDADE DE RISCO
        # ==========================================
        # Prioridade 1: Sono Crítico de Olhos Fechados (> 2.5s)
        if self.eye_closed_duration >= self.critical_sleep_sec:
            if self.current_state != DriverState.CRITICAL_SLEEP:
                self.drowsy_events_count += 1
            self.current_state = DriverState.CRITICAL_SLEEP

        # Prioridade 2: Cochilo com Queda de Cabeça (Cabeça Caída + Olhos Fechados > 0.7s)
        elif self.head_nod_duration >= self.head_nod_time_sec:
            if self.current_state != DriverState.HEAD_NODDING:
                self.drowsy_events_count += 1
            self.current_state = DriverState.HEAD_NODDING

        # Prioridade 3: Olhos Fechados na Postura Normal (Sonolência > 1.3s)
        elif self.eye_closed_duration >= self.drowsy_time_sec:
            if self.current_state not in (DriverState.DROWSY, DriverState.CRITICAL_SLEEP):
                self.drowsy_events_count += 1
            self.current_state = DriverState.DROWSY

        # Prioridade 4: Desatenção / Olhos Fora da Via (> 2.0s)
        # Seja por cabeça virada lateralmente (Yaw) OU cabeça abaixada com olhos abertos
        elif (self.head_yaw_duration >= self.distraction_time_sec) or \
             (self.head_down_open_eyes_duration >= self.distraction_time_sec):
            self.current_state = DriverState.DISTRACTED

        # Prioridade 5: Bocejo Prolongado (> 1.8s)
        elif self.mouth_open_duration >= self.yawn_time_sec:
            self.current_state = DriverState.YAWNING

        # Normal: Condutor Atento
        else:
            self.current_state = DriverState.ATTENTIVE

        # Atualização do Índice de Fadiga Acumulado
        self._calculate_fatigue_index()

        return self.current_state

    def _reset_active_timers(self) -> None:
        """Reinicia timers quando a face do motorista não está visível."""
        self.eye_closed_start_time = None
        self.mouth_open_start_time = None
        self.head_nod_start_time = None
        self.head_yaw_start_time = None
        self.head_down_open_eyes_start_time = None
        self.eye_closed_duration = 0.0
        self.mouth_open_duration = 0.0
        self.head_nod_duration = 0.0
        self.head_yaw_duration = 0.0
        self.head_down_open_eyes_duration = 0.0
        self.was_nodding_before_loss = False
        self.is_head_dropped = False

    def _calculate_fatigue_index(self) -> None:
        """Calcula o índice de fadiga progressivo do motorista (0 a 100%)."""
        score = (self.yawn_count * 15.0) + (self.drowsy_events_count * 25.0)

        if self.eye_closed_duration > 0.5:
            score += min(30.0, (self.eye_closed_duration / self.drowsy_time_sec) * 30.0)

        if self.head_nod_duration > 0.3:
            score += min(30.0, (self.head_nod_duration / self.head_nod_time_sec) * 30.0)

        self.fatigue_index = float(np.clip(score, 0.0, 100.0))

    def reset_stats(self) -> None:
        """Reseta contadores de estatísticas da sessão."""
        self.blink_count = 0
        self.yawn_count = 0
        self.drowsy_events_count = 0
        self.fatigue_index = 0.0
        self._reset_active_timers()
        self.current_state = DriverState.ATTENTIVE
