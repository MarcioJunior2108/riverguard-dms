"""
test_pipeline.py - Suite de Testes e Validação Completa do Pipeline
Valida sem necessidade de intervenção humana:
1. Imports e dependências
2. Modelo neural MediaPipe FaceLandmarker
3. Fórmulas biométricas (EAR, MAR, Head Pose)
4. Máquina de estados (Atenção, Sonolência, Bocejo, Micro-sono)
5. Sistema de áudio e sintetizador WAV
6. Renderização do Heads-Up Display (HUD)
"""

import sys
import time
import numpy as np
import cv2

import config
from core.metrics import calculate_ear, calculate_mar, rotation_matrix_to_euler_angles
from core.face_detector import FaceDetector, FaceData
from core.drowsiness_tracker import DrowsinessTracker, DriverState
from core.alert_system import AlertSystem
from ui.audio_generator import ensure_alert_sounds
from ui.hud import AutomotiveHUD


def test_audio():
    print("[TESTE 1/6] Validando síntese de áudio...")
    ensure_alert_sounds()
    assert config.SOUND_DROWSY.exists(), "SOUND_DROWSY não encontrado!"
    assert config.SOUND_YAWN.exists(), "SOUND_YAWN não encontrado!"
    assert config.SOUND_DISTRACTED.exists(), "SOUND_DISTRACTED não encontrado!"
    assert config.SOUND_DROWSY.stat().st_size > 1000, "Arquivo de áudio corrompido!"
    print("  -> Sons sintetizados e verificados com sucesso.")


def test_metrics_math():
    print("[TESTE 2/6] Validando fórmulas biométricas...")
    # Teste EAR (Olho aberto)
    open_eye = np.array([
        [10, 50],   # p1 (canto externo)
        [20, 40],   # p2 (superior esq)
        [30, 40],   # p3 (superior dir)
        [40, 50],   # p4 (canto interno)
        [30, 60],   # p5 (inferior dir)
        [20, 60]    # p6 (inferior esq)
    ], dtype=np.float32)
    ear_open = calculate_ear(open_eye)
    assert ear_open > 0.30, f"EAR aberto inesperado: {ear_open}"

    # Teste EAR (Olho fechado - pálpebras quase coladas)
    closed_eye = np.array([
        [10, 50],
        [20, 49],
        [30, 49],
        [40, 50],
        [30, 51],
        [20, 51]
    ], dtype=np.float32)
    ear_closed = calculate_ear(closed_eye)
    assert ear_closed < 0.15, f"EAR fechado inesperado: {ear_closed}"

    # Teste Head Pose (Matriz Identidade = 0 deg)
    R_identity = np.eye(3, dtype=np.float64)
    pitch, yaw, roll = rotation_matrix_to_euler_angles(R_identity)
    assert abs(pitch) < 1e-4 and abs(yaw) < 1e-4 and abs(roll) < 1e-4, "Erro na decomposição de Euler"
    print(f"  -> EAR Aberto: {ear_open:.3f} | EAR Fechado: {ear_closed:.3f} | Euler: OK")


def test_face_detector():
    print("[TESTE 3/6] Validando detector neural FaceLandmarker...")
    detector = FaceDetector()
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.process_frame(dummy_img)
    # Com imagem preta sem rosto, deve retornar None sem exceção
    assert result is None, "Resultado para imagem vazia deveria ser None"
    detector.close()
    print("  -> Inicialização e inferência protegida verificadas com sucesso.")


def test_drowsiness_tracker():
    print("[TESTE 4/6] Validando Máquina de Estados e Detecção de Fadiga...")
    tracker = DrowsinessTracker()

    # Criação de um FaceData sintético com olhos abertos
    normal_face = FaceData(
        landmarks_norm=np.zeros((478, 3), dtype=np.float32),
        landmarks_px=np.zeros((478, 2), dtype=np.int32),
        bbox=(100, 100, 300, 300),
        ear_left=0.32,
        ear_right=0.32,
        ear_avg=0.32,
        mar=0.25,
        head_pitch=0.0,
        head_yaw=0.0,
        head_roll=0.0,
        blendshapes={"eyeBlinkLeft": 0.05, "eyeBlinkRight": 0.05, "jawOpen": 0.05}
    )

    # Estado normal
    state = tracker.update(normal_face)
    assert state == DriverState.ATTENTIVE, f"Estado esperado ATTENTIVE, obtido {state}"

    # Simulação de olhos fechando (Sonolência)
    drowsy_face = FaceData(
        landmarks_norm=np.zeros((478, 3), dtype=np.float32),
        landmarks_px=np.zeros((478, 2), dtype=np.int32),
        bbox=(100, 100, 300, 300),
        ear_left=0.12,
        ear_right=0.12,
        ear_avg=0.12,
        mar=0.25,
        head_pitch=0.0,
        head_yaw=0.0,
        head_roll=0.0,
        blendshapes={"eyeBlinkLeft": 0.95, "eyeBlinkRight": 0.95, "jawOpen": 0.05}
    )

    # Primeiro frame com olhos fechados -> Ainda ATTENTIVE (iniciou timer)
    tracker.update(drowsy_face)
    # Força avanço artificial de tempo no timer para simular 1.5s fechado
    tracker.eye_closed_start_time = time.time() - 1.5
    state_drowsy = tracker.update(drowsy_face)
    assert state_drowsy == DriverState.DROWSY, f"Esperado DROWSY, obtido {state_drowsy}"

    # Força avanço de tempo para 2.6s fechado -> MICRO-SONO CRÍTICO
    tracker.eye_closed_start_time = time.time() - 2.6
    state_critical = tracker.update(drowsy_face)
    assert state_critical == DriverState.CRITICAL_SLEEP, f"Esperado CRITICAL_SLEEP, obtido {state_critical}"

    # 1. Teste: Cabeça abaixada com OLHOS ABERTOS (olhando velocímetro / painel por 1.0s)
    # NÃO deve disparar alarme de sono (deve permanecer ATTENTIVE)
    head_down_open_eyes = FaceData(
        landmarks_norm=np.zeros((478, 3), dtype=np.float32),
        landmarks_px=np.zeros((478, 2), dtype=np.int32),
        bbox=(100, 100, 300, 300),
        ear_left=0.32, ear_right=0.32, ear_avg=0.32,
        mar=0.25,
        head_pitch=-12.0, # Cabeça inclinada para baixo
        head_yaw=0.0, head_roll=0.0,
        blendshapes={"eyeBlinkLeft": 0.05, "eyeBlinkRight": 0.05}
    )
    tracker.update(head_down_open_eyes)
    tracker.head_down_open_eyes_start_time = time.time() - 1.0 # 1.0s olhando para o painel
    state_looking_down = tracker.update(head_down_open_eyes)
    assert state_looking_down == DriverState.ATTENTIVE, (
        f"Cabeça baixa com olhos abertos por 1.0s não deve ser sono! Esperado ATTENTIVE, obtido {state_looking_down}"
    )

    # 2. Teste: Cabeça abaixada prolongada com OLHOS ABERTOS (> 2.0s no celular/painel)
    # Deve acionar alerta de DISTRAÇÃO (olhos fora da via), NÃO sono
    tracker.head_down_open_eyes_start_time = time.time() - 2.5
    state_distracted = tracker.update(head_down_open_eyes)
    assert state_distracted == DriverState.DISTRACTED, (
        f"Cabeça baixa prolongada (> 2.0s) deve ser DISTRACTED, obtido {state_distracted}"
    )

    # 3. Teste: COCHILO AO VOLANTE (Cabeça Inclinada + OLHOS FECHADOS por > 0.7s)
    # DEVE disparar HEAD_NODDING!
    sleep_nod_face = FaceData(
        landmarks_norm=np.zeros((478, 3), dtype=np.float32),
        landmarks_px=np.zeros((478, 2), dtype=np.int32),
        bbox=(100, 100, 300, 300),
        ear_left=0.10, ear_right=0.10, ear_avg=0.10, # Olhos fechados!
        mar=0.25,
        head_pitch=-12.0, # Cabeça caída para frente!
        head_yaw=0.0, head_roll=0.0,
        blendshapes={"eyeBlinkLeft": 0.95, "eyeBlinkRight": 0.95}
    )
    tracker.update(sleep_nod_face)
    tracker.head_nod_start_time = time.time() - 1.0 # 1.0s com cabeça caída E olhos fechados
    state_nod = tracker.update(sleep_nod_face)
    assert state_nod == DriverState.HEAD_NODDING, f"Esperado HEAD_NODDING, obtido {state_nod}"

    # 4. Teste de Tolerância a Oclusão: face perdida temporariamente durante o cochilo
    state_nod_occluded = tracker.update(None)
    assert state_nod_occluded == DriverState.HEAD_NODDING, (
        f"Esperado manter HEAD_NODDING sob oclusão, obtido {state_nod_occluded}"
    )

    print(f"  -> Transições de estado validadas: {state.name} -> {state_drowsy.name} -> {state_critical.name} -> {state_looking_down.name} (olhos abertos) -> {state_distracted.name} -> {state_nod.name} (cabeça caída + olhos fechados)")


def test_alert_system():
    print("[TESTE 5/6] Validando sistema de alertas e thread de som...")
    alerts = AlertSystem(enable_sound=False)
    alerts.update(DriverState.DROWSY)
    muted = alerts.toggle_mute()
    alerts.stop_all_sounds()
    print("  -> AlertSystem funcional e seguro.")


def test_hud_rendering():
    print("[TESTE 6/6] Validando renderização gráfica do HUD...")
    hud = AutomotiveHUD()
    tracker = DrowsinessTracker()

    # Criação de frame sintético HD
    test_frame = np.full((720, 1280, 3), 40, dtype=np.uint8)

    sample_face = FaceData(
        landmarks_norm=np.zeros((478, 3), dtype=np.float32),
        landmarks_px=np.full((478, 2), 200, dtype=np.int32),
        bbox=(450, 200, 830, 600),
        ear_left=0.29,
        ear_right=0.30,
        ear_avg=0.295,
        mar=0.32,
        head_pitch=-2.5,
        head_yaw=1.8,
        head_roll=0.0,
        blendshapes={"eyeBlinkLeft": 0.08, "eyeBlinkRight": 0.08, "jawOpen": 0.12}
    )

    rendered = hud.render(
        frame=test_frame,
        face_data=sample_face,
        tracker=tracker,
        fps=32.4,
        is_flashing=False,
        is_muted=False
    )

    assert rendered.shape == (720, 1280, 3), "Dimensões corrompidas no render"

    # Salva uma amostra da interface para visualização do usuário
    preview_path = config.ASSETS_DIR / "sample_hud_preview.png"
    cv2.imwrite(str(preview_path), rendered)
    print(f"  -> HUD renderizado e salvo como demonstração visual em: {preview_path}")


def run_all_tests():
    print("=" * 60)
    print("    INICIANDO VALIDAÇÃO DO DRIVERGUARD DMS PIPELINE    ")
    print("=" * 60)
    test_audio()
    test_metrics_math()
    test_face_detector()
    test_drowsiness_tracker()
    test_alert_system()
    test_hud_rendering()
    print("=" * 60)
    print("  TODOS OS 6 TESTES DE INTEGRAÇÃO FORAM APROVADOS! (100%)")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
