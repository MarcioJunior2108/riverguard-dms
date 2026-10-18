"""
ui/hud.py - Interface Gráfica Automotiva Heads-Up Display (HUD)
Desenha indicadores telemétricos, barras dinâmicas, molduras translúcidas e
banners de alerta em tempo real sobre os frames de vídeo do OpenCV.
Totalmente responsivo a diferentes resoluções (640x480, 1280x720, etc).
"""

import numpy as np
import cv2

import config
from core.face_detector import FaceData
from core.drowsiness_tracker import DriverState, DrowsinessTracker
from core.metrics import LEFT_EYE_INDICES, RIGHT_EYE_INDICES, MOUTH_INDICES


def draw_translucent_rect(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    alpha: float = 0.65
) -> None:
    """Desenha um retângulo semitransparente sobre a imagem."""
    overlay = img.copy()
    cv2.rectangle(overlay, pt1, pt2, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_progress_bar(
    img: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    value: float,
    max_value: float,
    threshold: float,
    bar_color_normal: tuple[int, int, int],
    bar_color_alert: tuple[int, int, int],
    is_alert_above: bool = False
) -> None:
    """
    Desenha uma barra de progresso telemétrica horizontal com marcador de limiar.
    """
    cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 45), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (70, 70, 80), 1)

    fill_ratio = min(1.0, max(0.0, value / max_value))
    fill_w = int(w * fill_ratio)

    is_alert = (value > threshold) if is_alert_above else (value < threshold)
    fill_color = bar_color_alert if is_alert else bar_color_normal

    if fill_w > 0:
        cv2.rectangle(img, (x + 1, y + 1), (x + fill_w - 1, y + h - 1), fill_color, -1)

    # Linha marcadora do limiar
    thresh_x = int(x + w * min(1.0, threshold / max_value))
    cv2.line(img, (thresh_x, y - 2), (thresh_x, y + h + 2), (255, 255, 255), 1)


class AutomotiveHUD:
    """Renderizador do painel Heads-Up Display (estilo ADAS automotivo)."""

    def __init__(self):
        # Linhas de olhos/boca ocultas por padrão (para não tapar o rosto)
        self.show_landmarks: bool = False
        # Moldura delimitadora da face
        self.show_face_box: bool = True
        # Painel lateral de telemetria visível (pode ser ocultado com a tecla 'T')
        self.show_telemetry: bool = True

    def toggle_landmarks(self) -> bool:
        """Alterna a exibição das marcações sobrepostas de olhos e boca."""
        self.show_landmarks = not self.show_landmarks
        return self.show_landmarks

    def toggle_mesh(self) -> bool:
        """Alias para compatibilidade."""
        return self.toggle_landmarks()

    def toggle_telemetry(self) -> bool:
        """Alterna a exibição do painel de telemetria biométrica."""
        self.show_telemetry = not self.show_telemetry
        return self.show_telemetry

    def render(
        self,
        frame: np.ndarray,
        face_data: FaceData | None,
        tracker: DrowsinessTracker,
        fps: float,
        is_flashing: bool,
        is_muted: bool
    ) -> np.ndarray:
        """Renderiza a interface HUD de forma adaptativa à resolução do vídeo."""
        h, w, _ = frame.shape
        state = tracker.current_state
        is_compact = w < 850

        # 1. Borda de Alarme de Emergência (Flashing em vermelho)
        if state in (DriverState.DROWSY, DriverState.CRITICAL_SLEEP, DriverState.HEAD_NODDING) and is_flashing:
            border_thick = 10 if is_compact else 14
            cv2.rectangle(frame, (0, 0), (w, h), config.COLOR_DANGER, border_thick)

        # 2. Moldura delimitadora quadrada da face
        if face_data is not None and self.show_face_box:
            self._draw_face_bounding_box(frame, face_data, tracker, is_compact)

        # 3. Marcações anatômicas sobrepostas (opcional, oculto por padrão)
        if face_data is not None and self.show_landmarks:
            self._draw_facial_features(frame, face_data, tracker)

        # 4. Barra Superior de Cabeçalho (Header)
        header_h = 36 if is_compact else 46
        draw_translucent_rect(frame, (0, 0), (w, header_h), config.COLOR_BACKGROUND_CARD, alpha=0.88)
        cv2.line(frame, (0, header_h), (w, header_h), config.COLOR_CARD_BORDER, 1)

        # Título à esquerda
        title_scale = 0.55 if is_compact else 0.72
        title_y = int(header_h * 0.68)
        cv2.putText(
            frame, "DRIVER GUARD", (12, title_y),
            cv2.FONT_HERSHEY_DUPLEX, title_scale, (255, 200, 50), 2, cv2.LINE_AA
        )

        if not is_compact:
            cv2.putText(
                frame, "|  MONITORAMENTO DE FADIGA VEICULAR", (205, title_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, config.COLOR_TEXT_SECONDARY, 1, cv2.LINE_AA
            )

        # Indicadores de Áudio e FPS à direita (posicionamento dinâmico sem sobreposição)
        fps_text = f"FPS: {fps:04.1f}"
        audio_text = "[ MUTED ]" if is_muted else "[ AUDIO ON ]"
        audio_color = config.COLOR_WARNING if is_muted else config.COLOR_OK

        right_scale = 0.42 if is_compact else 0.48
        (fps_w, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, right_scale, 1)
        (aud_w, _), _ = cv2.getTextSize(audio_text, cv2.FONT_HERSHEY_SIMPLEX, right_scale, 1)

        fps_x = w - fps_w - 10
        aud_x = fps_x - aud_w - 12

        cv2.putText(
            frame, audio_text, (aud_x, title_y),
            cv2.FONT_HERSHEY_SIMPLEX, right_scale, audio_color, 1, cv2.LINE_AA
        )
        cv2.putText(
            frame, fps_text, (fps_x, title_y),
            cv2.FONT_HERSHEY_SIMPLEX, right_scale, config.COLOR_TEXT_PRIMARY, 1, cv2.LINE_AA
        )

        # 5. Banner Central de Alerta do Condutor
        self._draw_state_banner(frame, tracker, w, h, header_h, is_compact, is_flashing)

        # 6. Painel Lateral de Telemetria (Se ativo)
        if self.show_telemetry:
            self._draw_telemetry_card(frame, face_data, tracker, w, h, header_h, is_compact)

        # 7. Rodapé de Atalhos
        footer_h = 28 if is_compact else 32
        draw_translucent_rect(frame, (0, h - footer_h), (w, h), config.COLOR_BACKGROUND_CARD, alpha=0.88)
        cv2.line(frame, (0, h - footer_h), (w, h - footer_h), config.COLOR_CARD_BORDER, 1)

        tel_status = "ON" if self.show_telemetry else "OFF"
        mark_status = "ON" if self.show_landmarks else "OFF"
        if is_compact:
            shortcuts = f"[Q] Sair | [C] Calibrar | [M] Mute | [T] Telemetria: {tel_status} | [H] Marcos: {mark_status}"
            foot_scale = 0.35
        else:
            shortcuts = f"[Q] Sair  |  [C] Calibrar Perfil  |  [M] Silenciar  |  [R] Reset  |  [T] Telemetria: {tel_status}  |  [H] Marcos: {mark_status}"
            foot_scale = 0.42

        cv2.putText(
            frame, shortcuts, (12, h - 9),
            cv2.FONT_HERSHEY_SIMPLEX, foot_scale, config.COLOR_TEXT_SECONDARY, 1, cv2.LINE_AA
        )

        return frame

    def _draw_face_bounding_box(
        self,
        frame: np.ndarray,
        face_data: FaceData,
        tracker: DrowsinessTracker,
        is_compact: bool
    ) -> None:
        """Desenha a moldura de enquadramento da face do motorista."""
        h, w, _ = frame.shape
        x_min, y_min, x_max, y_max = face_data.bbox

        pad_x = int((x_max - x_min) * 0.08)
        pad_y = int((y_max - y_min) * 0.10)

        x1 = max(0, x_min - pad_x)
        y1 = max(0, y_min - pad_y)
        x2 = min(w - 1, x_max + pad_x)
        y2 = min(h - 1, y_max + pad_y)

        state = tracker.current_state
        if state == DriverState.ATTENTIVE:
            box_color = config.COLOR_OK
            label_text = "CONDUTOR ATENTO"
        elif state in (DriverState.YAWNING, DriverState.DISTRACTED):
            box_color = config.COLOR_WARNING
            label_text = "ALERTA"
        elif state in (DriverState.DROWSY, DriverState.CRITICAL_SLEEP, DriverState.HEAD_NODDING):
            box_color = config.COLOR_DANGER
            label_text = "PERIGO - SONO"
        elif state == DriverState.CALIBRATING:
            box_color = config.COLOR_CALIBRATING
            label_text = "CALIBRANDO"
        else:
            box_color = config.COLOR_CARD_BORDER
            label_text = "CONDUTOR"

        # Retângulo do rosto
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)

        # Cantos reforçados (L-brackets)
        corner_len = min(22 if is_compact else 28, (x2 - x1) // 4)
        thick = 3
        # Canto sup esq
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), box_color, thick)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), box_color, thick)
        # Canto sup dir
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), box_color, thick)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), box_color, thick)
        # Canto inf esq
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), box_color, thick)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), box_color, thick)
        # Canto inf dir
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), box_color, thick)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), box_color, thick)

    def _draw_facial_features(
        self,
        frame: np.ndarray,
        face_data: FaceData,
        tracker: DrowsinessTracker
    ) -> None:
        """Desenha contornos sobre olhos e boca (somente quando acionado via tecla 'H')."""
        is_eyes_closed = face_data.ear_avg < tracker.ear_threshold
        eye_color = config.COLOR_DANGER if is_eyes_closed else config.COLOR_OK

        left_eye_pts = face_data.landmarks_px[LEFT_EYE_INDICES]
        right_eye_pts = face_data.landmarks_px[RIGHT_EYE_INDICES]

        cv2.polylines(frame, [left_eye_pts], isClosed=True, color=eye_color, thickness=1, lineType=cv2.LINE_AA)
        cv2.polylines(frame, [right_eye_pts], isClosed=True, color=eye_color, thickness=1, lineType=cv2.LINE_AA)

        is_yawn = face_data.mar > tracker.mar_threshold
        mouth_color = config.COLOR_WARNING if is_yawn else config.COLOR_ACCENT

        mouth_key_indices = [
            MOUTH_INDICES["left_corner"], MOUTH_INDICES["upper_left"],
            MOUTH_INDICES["upper_mid"], MOUTH_INDICES["upper_right"],
            MOUTH_INDICES["right_corner"], MOUTH_INDICES["lower_right"],
            MOUTH_INDICES["lower_mid"], MOUTH_INDICES["lower_left"]
        ]
        mouth_pts = face_data.landmarks_px[mouth_key_indices]
        cv2.polylines(frame, [mouth_pts], isClosed=True, color=mouth_color, thickness=1, lineType=cv2.LINE_AA)

    def _draw_state_banner(
        self,
        frame: np.ndarray,
        tracker: DrowsinessTracker,
        w: int,
        h: int,
        header_h: int,
        is_compact: bool,
        is_flashing: bool
    ) -> None:
        """Desenha o banner central com o estado do condutor, centralizado e sem colisões."""
        state = tracker.current_state

        if state == DriverState.ATTENTIVE:
            badge_text = "MOTORISTA ATENTO"
            badge_sub = "Monitoramento normal"
            bg_color = (25, 70, 30)
            text_color = config.COLOR_OK

        elif state == DriverState.DROWSY:
            badge_text = "ALERTA: SONOLENCIA"
            badge_sub = f"Olhos fechados: {tracker.eye_closed_duration:.1f}s"
            bg_color = config.COLOR_DANGER if is_flashing else (15, 15, 60)
            text_color = (255, 255, 255)

        elif state == DriverState.CRITICAL_SLEEP:
            badge_text = "PERIGO: MICRO-SONO!"
            badge_sub = f"Olhos fechados: {tracker.eye_closed_duration:.1f}s - ACORDE!"
            bg_color = (255, 255, 255) if is_flashing else config.COLOR_DANGER
            text_color = config.COLOR_DANGER if is_flashing else (255, 255, 255)

        elif state == DriverState.HEAD_NODDING:
            badge_text = "PERIGO: COCHILO AO VOLANTE!"
            badge_sub = "Cabeca inclinada + Olhos fechados - ACORDE!"
            bg_color = (255, 255, 255) if is_flashing else config.COLOR_DANGER
            text_color = config.COLOR_DANGER if is_flashing else (255, 255, 255)

        elif state == DriverState.YAWNING:
            badge_text = "AVISO: BOCEJO"
            badge_sub = f"Fadiga ({tracker.mouth_open_duration:.1f}s)"
            bg_color = (15, 50, 85)
            text_color = config.COLOR_WARNING

        elif state == DriverState.DISTRACTED:
            badge_text = "ATENCAO: OLHOS NA VIA"
            if getattr(tracker, "head_down_open_eyes_duration", 0.0) >= tracker.distraction_time_sec:
                badge_sub = "Cabeca baixa prolongada - Olhe para a pista"
            else:
                badge_sub = "Cabeca desviada da via"
            bg_color = (15, 50, 85)
            text_color = config.COLOR_WARNING

        elif state == DriverState.CALIBRATING:
            badge_text = "CALIBRANDO PERFIL..."
            badge_sub = tracker.calibration_message[:30]
            bg_color = (30, 60, 80)
            text_color = config.COLOR_CALIBRATING

        else: # NO_FACE
            badge_text = "SEM CONDUTOR"
            badge_sub = "Posicione-se em frente a camera"
            bg_color = (35, 35, 45)
            text_color = config.COLOR_TEXT_SECONDARY

        # Dimensões responsivas do banner
        if is_compact:
            bw, bh = min(280, int(w * 0.46)), 38
            by = header_h + 5
            bx = (w - bw) // 2 + 30  # Leve compensação para balancear com o painel lateral
            title_scale = 0.44
            sub_scale = 0.32
        else:
            bw, bh = 420, 52
            by = header_h + 8
            bx = (w - bw) // 2
            title_scale = 0.60
            sub_scale = 0.38

        draw_translucent_rect(frame, (bx, by), (bx + bw, by + bh), bg_color, alpha=0.88)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), text_color, 2)

        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_DUPLEX, title_scale, 1)
        cv2.putText(
            frame, badge_text, (bx + (bw - tw) // 2, by + int(bh * 0.48)),
            cv2.FONT_HERSHEY_DUPLEX, title_scale, text_color, 1, cv2.LINE_AA
        )

        (stw, _), _ = cv2.getTextSize(badge_sub, cv2.FONT_HERSHEY_SIMPLEX, sub_scale, 1)
        cv2.putText(
            frame, badge_sub, (bx + (bw - stw) // 2, by + int(bh * 0.86)),
            cv2.FONT_HERSHEY_SIMPLEX, sub_scale, config.COLOR_TEXT_PRIMARY, 1, cv2.LINE_AA
        )

    def _draw_telemetry_card(
        self,
        frame: np.ndarray,
        face_data: FaceData | None,
        tracker: DrowsinessTracker,
        w: int,
        h: int,
        header_h: int,
        is_compact: bool
    ) -> None:
        """Desenha o painel lateral de telemetria sem obstruir o condutor."""
        current_ear = face_data.ear_avg if face_data else 0.0
        current_mar = face_data.mar if face_data else 0.0
        pitch = face_data.head_pitch if face_data else 0.0
        yaw = face_data.head_yaw if face_data else 0.0

        if is_compact:
            cw, ch = 150, 195
            cx = 8
            cy = header_h + 5

            draw_translucent_rect(frame, (cx, cy), (cx + cw, cy + ch), config.COLOR_BACKGROUND_CARD, alpha=0.80)
            cv2.rectangle(frame, (cx, cy), (cx + cw, cy + ch), config.COLOR_CARD_BORDER, 1)

            # Cabeçalho
            cv2.putText(
                frame, "TELEMETRIA", (cx + 8, cy + 16),
                cv2.FONT_HERSHEY_DUPLEX, 0.38, config.COLOR_ACCENT, 1, cv2.LINE_AA
            )
            cv2.line(frame, (cx + 8, cy + 22), (cx + cw - 8, cy + 22), config.COLOR_CARD_BORDER, 1)

            bar_w = cw - 16
            # 1. EAR
            cv2.putText(
                frame, f"EAR: {current_ear:.2f} [{tracker.ear_threshold:.2f}]",
                (cx + 8, cy + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.31, config.COLOR_TEXT_PRIMARY, 1, cv2.LINE_AA
            )
            draw_progress_bar(frame, cx + 8, cy + 42, bar_w, 6, current_ear, 0.45, tracker.ear_threshold, config.COLOR_OK, config.COLOR_DANGER)

            # 2. MAR
            cv2.putText(
                frame, f"MAR: {current_mar:.2f} [{tracker.mar_threshold:.2f}]",
                (cx + 8, cy + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.31, config.COLOR_TEXT_PRIMARY, 1, cv2.LINE_AA
            )
            draw_progress_bar(frame, cx + 8, cy + 68, bar_w, 6, current_mar, 1.0, tracker.mar_threshold, config.COLOR_ACCENT, config.COLOR_WARNING, is_alert_above=True)

            # 3. Cabeça
            cv2.putText(
                frame, f"Cabeca: P{pitch:+.0f} Y{yaw:+.0f}",
                (cx + 8, cy + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.30, config.COLOR_TEXT_SECONDARY, 1, cv2.LINE_AA
            )
            cv2.line(frame, (cx + 8, cy + 98), (cx + cw - 8, cy + 98), config.COLOR_CARD_BORDER, 1)

            # 4. Fadiga
            fatigue_col = config.COLOR_OK if tracker.fatigue_index <= 40 else (config.COLOR_WARNING if tracker.fatigue_index <= 70 else config.COLOR_DANGER)
            cv2.putText(
                frame, f"FADIGA: {tracker.fatigue_index:.0f}%",
                (cx + 8, cy + 114), cv2.FONT_HERSHEY_DUPLEX, 0.34, fatigue_col, 1, cv2.LINE_AA
            )
            draw_progress_bar(frame, cx + 8, cy + 120, bar_w, 7, tracker.fatigue_index, 100.0, 70.0, config.COLOR_OK if tracker.fatigue_index <= 40 else config.COLOR_WARNING, config.COLOR_DANGER, is_alert_above=True)

            # 5. Contadores
            cv2.line(frame, (cx + 8, cy + 137), (cx + cw - 8, cy + 137), config.COLOR_CARD_BORDER, 1)
            cv2.putText(frame, f"Piscadas: {tracker.blink_count}", (cx + 8, cy + 152), cv2.FONT_HERSHEY_SIMPLEX, 0.30, config.COLOR_TEXT_PRIMARY, 1, cv2.LINE_AA)
            cv2.putText(frame, f"Bocejos:  {tracker.yawn_count}", (cx + 8, cy + 167), cv2.FONT_HERSHEY_SIMPLEX, 0.30, config.COLOR_TEXT_PRIMARY, 1, cv2.LINE_AA)
            cv2.putText(frame, f"Sono:     {tracker.drowsy_events_count}", (cx + 8, cy + 182), cv2.FONT_HERSHEY_SIMPLEX, 0.30, config.COLOR_DANGER if tracker.drowsy_events_count > 0 else config.COLOR_OK, 1, cv2.LINE_AA)

        else:
            # Layout expandido para 720p ou superior
            cw, ch = 260, 270
            cx = 18
            cy = header_h + 8

            draw_translucent_rect(frame, (cx, cy), (cx + cw, cy + ch), config.COLOR_BACKGROUND_CARD, alpha=0.85)
            cv2.rectangle(frame, (cx, cy), (cx + cw, cy + ch), config.COLOR_CARD_BORDER, 1)

            cv2.putText(
                frame, "TELEMETRIA BIOMETRICA", (cx + 12, cy + 22),
                cv2.FONT_HERSHEY_DUPLEX, 0.46, config.COLOR_ACCENT, 1, cv2.LINE_AA
            )
            cv2.line(frame, (cx + 12, cy + 30), (cx + cw - 12, cy + 30), config.COLOR_CARD_BORDER, 1)

            bar_w = cw - 24
            cv2.putText(
                frame, f"EAR (Olhos): {current_ear:.2f}  [Limiar: {tracker.ear_threshold:.2f}]",
                (cx + 12, cy + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.38, config.COLOR_TEXT_PRIMARY, 1, cv2.LINE_AA
            )
            draw_progress_bar(frame, cx + 12, cy + 56, bar_w, 10, current_ear, 0.45, tracker.ear_threshold, config.COLOR_OK, config.COLOR_DANGER)

            cv2.putText(
                frame, f"MAR (Boca):  {current_mar:.2f}  [Limiar: {tracker.mar_threshold:.2f}]",
                (cx + 12, cy + 86), cv2.FONT_HERSHEY_SIMPLEX, 0.38, config.COLOR_TEXT_PRIMARY, 1, cv2.LINE_AA
            )
            draw_progress_bar(frame, cx + 12, cy + 92, bar_w, 10, current_mar, 1.0, tracker.mar_threshold, config.COLOR_ACCENT, config.COLOR_WARNING, is_alert_above=True)

            cv2.putText(
                frame, f"Cabeca: Pitch {pitch:+.1f} deg | Yaw {yaw:+.1f} deg",
                (cx + 12, cy + 122), cv2.FONT_HERSHEY_SIMPLEX, 0.36, config.COLOR_TEXT_SECONDARY, 1, cv2.LINE_AA
            )
            cv2.line(frame, (cx + 12, cy + 134), (cx + cw - 12, cy + 134), config.COLOR_CARD_BORDER, 1)

            fatigue_col = config.COLOR_OK if tracker.fatigue_index <= 40 else (config.COLOR_WARNING if tracker.fatigue_index <= 70 else config.COLOR_DANGER)
            cv2.putText(
                frame, f"INDICE DE FADIGA: {tracker.fatigue_index:.0f}%",
                (cx + 12, cy + 156), cv2.FONT_HERSHEY_DUPLEX, 0.44, fatigue_col, 1, cv2.LINE_AA
            )
            draw_progress_bar(frame, cx + 12, cy + 165, bar_w, 12, tracker.fatigue_index, 100.0, 70.0, config.COLOR_OK if tracker.fatigue_index <= 40 else config.COLOR_WARNING, config.COLOR_DANGER, is_alert_above=True)

            cv2.line(frame, (cx + 12, cy + 190), (cx + cw - 12, cy + 190), config.COLOR_CARD_BORDER, 1)
            cv2.putText(frame, f"Piscadas Registradas: {tracker.blink_count}", (cx + 12, cy + 212), cv2.FONT_HERSHEY_SIMPLEX, 0.38, config.COLOR_TEXT_PRIMARY, 1, cv2.LINE_AA)
            cv2.putText(frame, f"Bocejos Registrados:  {tracker.yawn_count}", (cx + 12, cy + 232), cv2.FONT_HERSHEY_SIMPLEX, 0.38, config.COLOR_TEXT_PRIMARY, 1, cv2.LINE_AA)
            cv2.putText(frame, f"Episodios de Sono:   {tracker.drowsy_events_count}", (cx + 12, cy + 252), cv2.FONT_HERSHEY_SIMPLEX, 0.38, config.COLOR_DANGER if tracker.drowsy_events_count > 0 else config.COLOR_OK, 1, cv2.LINE_AA)
