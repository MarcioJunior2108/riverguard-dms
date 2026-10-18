"""
core/alert_system.py - Gerenciador de Alertas Sonoros e Visuais
Execução multithread não-bloqueante para áudio automotivo e controle de alarmes visuais.
"""

import sys
import threading
import time
from pathlib import Path

import config
from core.drowsiness_tracker import DriverState
from ui.audio_generator import ensure_alert_sounds

# Suporte nativo a winsound no Windows
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


class AlertSystem:
    """Sistema central de alertas sonoros e indicadores visuais de perigo."""

    def __init__(self, enable_sound: bool = True):
        self.enable_sound = enable_sound
        self.is_muted: bool = not enable_sound
        self.lock = threading.Lock()

        # Garante que os arquivos WAV existem
        ensure_alert_sounds()

        # Controle de estado de áudio ativo
        self.active_alarm_state: DriverState | None = None
        self.last_sound_play_time: float = 0.0
        self.sound_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Controle de piscamento visual (Flashing effect)
        self.flash_state: bool = False
        self.last_flash_toggle_time: float = 0.0

    def toggle_mute(self) -> bool:
        """Alterna estado de mudo (Mute/Unmute). Retorna o novo estado de silenciamento."""
        with self.lock:
            self.is_muted = not self.is_muted
            if self.is_muted:
                self.stop_all_sounds()
            return self.is_muted

    def update(self, driver_state: DriverState) -> None:
        """
        Atualiza os alertas sonoros e visuais conforme o estado do condutor.
        Chamado a cada ciclo de frame da aplicação.
        """
        now = time.time()

        # 1. Controle de piscamento visual para estados críticos (a cada 200ms)
        if (now - self.last_flash_toggle_time) > 0.20:
            self.flash_state = not self.flash_state
            self.last_flash_toggle_time = now

        # Se áudio estiver desligado ou silenciado, nada a emitir
        if self.is_muted:
            return

        # 2. Transições de áudio
        if driver_state in (DriverState.DROWSY, DriverState.CRITICAL_SLEEP, DriverState.HEAD_NODDING):
            self._trigger_sound(config.SOUND_DROWSY, cooldown=0.8, is_continuous=True)
            self.active_alarm_state = driver_state

        elif driver_state == DriverState.YAWNING:
            self._trigger_sound(config.SOUND_YAWN, cooldown=2.5, is_continuous=False)
            self.active_alarm_state = driver_state

        elif driver_state == DriverState.DISTRACTED:
            self._trigger_sound(config.SOUND_DISTRACTED, cooldown=2.0, is_continuous=False)
            self.active_alarm_state = driver_state

        else:
            # Motorista voltou ao estado normal ou levantou a cabeça -> interrompe alarme contínuo
            if self.active_alarm_state in (DriverState.DROWSY, DriverState.CRITICAL_SLEEP, DriverState.HEAD_NODDING):
                self.stop_all_sounds()
            self.active_alarm_state = None

    def _trigger_sound(self, sound_path: Path, cooldown: float, is_continuous: bool = False) -> None:
        """Dispara a execução de um arquivo de som em thread assíncrona respeitando o cooldown."""
        now = time.time()
        if (now - self.last_sound_play_time) < cooldown:
            return

        self.last_sound_play_time = now

        # Executa em thread desacoplada para não reduzir a taxa de FPS
        threading.Thread(
            target=self._play_sound_worker,
            args=(sound_path, is_continuous),
            daemon=True
        ).start()

    def _play_sound_worker(self, sound_path: Path, is_continuous: bool) -> None:
        """Worker que aciona a API de som do sistema operacional."""
        try:
            if HAS_WINSOUND and sys.platform == "win32":
                flags = winsound.SND_FILENAME | winsound.SND_ASYNC
                if is_continuous:
                    flags |= winsound.SND_LOOP
                winsound.PlaySound(str(sound_path), flags)
            else:
                # Fallback em sistemas sem winsound (beep do terminal)
                print("\a", end="", flush=True)
        except Exception as e:
            # Evita travamento da aplicação caso o dispositivo de som esteja ocupado
            pass

    def stop_all_sounds(self) -> None:
        """Para imediatamente qualquer som em loop que esteja tocando."""
        if HAS_WINSOUND and sys.platform == "win32":
            try:
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:
                pass

    def is_flashing(self) -> bool:
        """Retorna True se o ciclo atual de piscar visual estiver ativo (para bordas vermelhas)."""
        return self.flash_state
