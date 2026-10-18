"""
main.py - Ponto de Entrada do DriverGuard DMS (Driver Monitoring System)
Sistema de Visão Computacional para Detecção de Fadiga e Sono ao Volante
"""

import argparse
import sys
import time
import cv2

import config
from core.face_detector import FaceDetector
from core.drowsiness_tracker import DrowsinessTracker, DriverState
from core.alert_system import AlertSystem
from ui.hud import AutomotiveHUD


def parse_arguments() -> argparse.Namespace:
    """Configura os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="DriverGuard - Sistema de Monitoramento e Reconhecimento de Sono e Fadiga Veicular (DMS)"
    )
    parser.add_argument(
        "--camera", type=int, default=config.CAMERA_INDEX,
        help="Índice da câmera/webcam conectada (padrão: 0)"
    )
    parser.add_argument(
        "--video", type=str, default=None,
        help="Caminho para arquivo de vídeo gravado para testes (opcional)"
    )
    parser.add_argument(
        "--width", type=int, default=config.FRAME_WIDTH,
        help=f"Largura desejada da captura de vídeo (padrão: {config.FRAME_WIDTH})"
    )
    parser.add_argument(
        "--height", type=int, default=config.FRAME_HEIGHT,
        help=f"Altura desejada da captura de vídeo (padrão: {config.FRAME_HEIGHT})"
    )
    parser.add_argument(
        "--no-sound", action="store_true", default=False,
        help="Executa o sistema em modo silencioso (sem alarmes sonoros)"
    )
    parser.add_argument(
        "--debug", action="store_true", default=False,
        help="Exibe telemetria detalhada no terminal em tempo real"
    )
    return parser.parse_args()


def print_banner() -> None:
    """Exibe o cabeçalho estético de inicialização no terminal."""
    banner = """
    ==================================================================
        DRIVERGUARD DMS - MONITORAMENTO DE FADIGA VEICULAR
    ==================================================================
    * Tecnologias: Python 3.12, MediaPipe FaceLandmarker, OpenCV 5
    * Métricas: EAR (Olhos), MAR (Bocejos), Head Pose (Tombamento/Desvio)
    
    [ATALHOS DO TECLADO NA JANELA DE VÍDEO]
      - Q ou ESC : Finalizar e exibir relatório
      - C        : Iniciar Calibração Biométrica (3 segundos)
      - M        : Silenciar / Ativar Alarme Sonoro
      - R        : Resetar Contadores da Viagem
      - T        : Ocultar / Exibir Painel de Telemetria
      - H        : Ocultar / Exibir Marcos Faciais
    ==================================================================
    """
    print(banner)


def main():
    args = parse_arguments()
    print_banner()

    # Inicialização dos subsistemas modulares
    print("[1/4] Inicializando detector neural facial...")
    detector = FaceDetector()

    print("[2/4] Inicializando rastreador biométrico de fadiga...")
    tracker = DrowsinessTracker()

    print("[3/4] Inicializando sistema de alertas e sintetizador de áudio...")
    alert_system = AlertSystem(enable_sound=not args.no_sound)

    print("[4/4] Inicializando interface automotiva Heads-Up Display (HUD)...")
    hud = AutomotiveHUD()

    # Inicialização do fluxo de vídeo
    video_source = args.video if args.video else args.camera
    print(f"[DriverGuard] Conectando à fonte de vídeo: {video_source}...")

    # No Windows, cv2.CAP_DSHOW garante abertura ultrarrápida da webcam
    if isinstance(video_source, int) and sys.platform == "win32":
        cap = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"\n[ERRO CRÍTICO] Não foi possível acessar a câmera no índice {video_source}.")
        print("Verifique se a câmera não está em uso por outro aplicativo ou teste com '--camera 1'.")
        detector.close()
        return

    # Ajuste de resolução
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    window_name = "DriverGuard - Driver Monitoring System (DMS)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.width, args.height)

    # Variáveis para cálculo de FPS estável
    prev_time = time.time()
    fps = 30.0

    print("\n>>> Sistema ativo e monitorando! Pressione 'Q' na janela para sair.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                # Se for arquivo de vídeo e chegou ao fim, reinicia o loop
                if args.video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    print("[Aviso] Falha ao capturar quadro da câmera.")
                    break

            # Espelhamento horizontal para visão natural de espelho retrovisor
            frame = cv2.flip(frame, 1)

            # 1. Visão Computacional e Extração Biométrica
            face_data = detector.process_frame(frame)

            # 2. Atualização da Máquina de Estados de Sono
            driver_state = tracker.update(face_data)

            # 3. Disparo de Alertas Sonoros e Piscamento
            alert_system.update(driver_state)

            # 4. Cálculo da taxa de FPS (Média Móvel Suavizada)
            current_time = time.time()
            time_diff = current_time - prev_time
            prev_time = current_time
            if time_diff > 0:
                current_fps = 1.0 / time_diff
                fps = 0.9 * fps + 0.1 * current_fps

            # 5. Renderização do Painel Heads-Up Display
            rendered_frame = hud.render(
                frame=frame,
                face_data=face_data,
                tracker=tracker,
                fps=fps,
                is_flashing=alert_system.is_flashing(),
                is_muted=alert_system.is_muted
            )

            # 6. Exibição na janela OpenCV
            cv2.imshow(window_name, rendered_frame)

            # 7. Telemetria opcional de debug no console
            if args.debug and face_data is not None:
                print(
                    f"\r[STATUS: {driver_state.name:<14}] "
                    f"EAR: {face_data.ear_avg:.2f} | "
                    f"MAR: {face_data.mar:.2f} | "
                    f"Pitch: {face_data.head_pitch:+5.1f}° | "
                    f"Fadiga: {tracker.fatigue_index:3.0f}%",
                    end="", flush=True
                )

            # 8. Tratamento de Teclas de Atalho
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):  # Q ou ESC
                break
            elif key in (ord('c'), ord('C')):    # C: Calibrar
                tracker.start_calibration()
                print("\n[DriverGuard] Calibração de perfil iniciada (3 segundos)...")
            elif key in (ord('m'), ord('M')):    # M: Mute
                muted = alert_system.toggle_mute()
                print(f"\n[DriverGuard] Áudio: {'SILENCIADO' if muted else 'ATIVADO'}")
            elif key in (ord('r'), ord('R')):    # R: Reset
                tracker.reset_stats()
                print("\n[DriverGuard] Contadores da viagem reiniciados.")
            elif key in (ord('t'), ord('T')):    # T: Alternar painel de telemetria
                is_tel = hud.toggle_telemetry()
                status = "VISÍVEL" if is_tel else "OCULTO"
                print(f"\n[DriverGuard] Painel de telemetria: {status}")
            elif key in (ord('h'), ord('H')):    # H: Alternar marcos faciais
                is_vis = hud.toggle_landmarks()
                status = "VISÍVEIS" if is_vis else "OCULTOS"
                print(f"\n[DriverGuard] Marcações sobre olhos e boca: {status}")

    except KeyboardInterrupt:
        print("\n[DriverGuard] Interrompido pelo usuário.")

    finally:
        # Liberação e encerramento limpo e seguro de todos os recursos
        print("\n\nEncerrando sistema...")
        try:
            alert_system.stop_all_sounds()
        except (Exception, KeyboardInterrupt):
            pass
        try:
            cap.release()
        except (Exception, KeyboardInterrupt):
            pass
        try:
            cv2.destroyAllWindows()
        except (Exception, KeyboardInterrupt):
            pass
        try:
            detector.close()
        except (Exception, KeyboardInterrupt):
            pass

        # Relatório Final da Sessão
        print("\n" + "=" * 54)
        print("          RELATÓRIO DA JORNADA DO CONDUTOR        ")
        print("=" * 54)
        print(f"  * Total de Piscadas Registradas  : {tracker.blink_count}")
        print(f"  * Total de Bocejos Registrados   : {tracker.yawn_count}")
        print(f"  * Eventos de Sonolência Crítica  : {tracker.drowsy_events_count}")
        print(f"  * Índice de Fadiga Final         : {tracker.fatigue_index:.1f}%")
        if tracker.fatigue_index >= 70 or tracker.drowsy_events_count > 0:
            print("  [RECOMENDAÇÃO] Alto nível de fadiga detectado. Recomenda-se descanso!")
        else:
            print("  [AVALIAÇÃO] Nível de atenção estável ao longo da condução.")
        print("=" * 54 + "\n")


if __name__ == "__main__":
    main()
