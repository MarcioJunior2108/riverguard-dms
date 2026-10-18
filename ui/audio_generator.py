"""
ui/audio_generator.py - Gerador Autônomo de Áudio de Alerta Veicular
Sintetiza sinais sonoros no formato WAV nativo (sem dependências externas)
"""

import math
import struct
import wave
from pathlib import Path
import config


def generate_tone_wav(
    file_path: Path,
    frequencies: list[float],
    duration_per_tone: float = 0.15,
    sample_rate: int = 44100,
    volume: float = 0.7,
    repeat: int = 2
) -> None:
    """
    Gera um arquivo de áudio WAV sintético com tons especificados.
    
    :param file_path: Caminho de saída do arquivo WAV
    :param frequencies: Lista de frequências em Hertz
    :param duration_per_tone: Duração de cada tom em segundos
    :param sample_rate: Taxa de amostragem de áudio
    :param volume: Volume de 0.0 a 1.0
    :param repeat: Número de ciclos de repetição
    """
    if file_path.exists() and file_path.stat().st_size > 500:
        return

    frames = bytearray()
    max_amplitude = int(32767 * volume)

    for _ in range(repeat):
        for freq in frequencies:
            num_samples = int(sample_rate * duration_per_tone)
            for i in range(num_samples):
                # Envelope suave para evitar estalidos (cliques) no início e fim do tom
                envelope = 1.0
                attack = int(sample_rate * 0.015)
                if i < attack:
                    envelope = i / attack
                elif i > (num_samples - attack):
                    envelope = (num_samples - i) / attack

                # Onda senoidal
                t = float(i) / sample_rate
                sine_val = math.sin(2.0 * math.pi * freq * t)
                sample = int(max_amplitude * envelope * sine_val)
                frames.extend(struct.pack("<h", sample))

            # Pequeno silêncio entre tons (20ms)
            silence_samples = int(sample_rate * 0.02)
            for _ in range(silence_samples):
                frames.extend(struct.pack("<h", 0))

    with wave.open(str(file_path), "wb") as wav_file:
        wav_file.setnchannels(1)        # Mono
        wav_file.setsampwidth(2)        # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(frames)


def ensure_alert_sounds() -> None:
    """
    Garante que todos os arquivos de áudio necessários para os alertas existam.
    Se algum não existir, ele é gerado instantaneamente na primeira execução.
    """
    config.SOUNDS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Alarme de Sono Crítico (Tom alternado de alta frequência: 880Hz e 1320Hz)
    generate_tone_wav(
        file_path=config.SOUND_DROWSY,
        frequencies=[880.0, 1320.0],
        duration_per_tone=0.18,
        repeat=3,
        volume=0.85
    )

    # 2. Alerta de Bocejo (Tom suave duplo: 550Hz e 660Hz)
    generate_tone_wav(
        file_path=config.SOUND_YAWN,
        frequencies=[550.0, 660.0],
        duration_per_tone=0.14,
        repeat=2,
        volume=0.65
    )

    # 3. Alerta de Distração / Cabeça Inclinada (Tom de atenção: 720Hz)
    generate_tone_wav(
        file_path=config.SOUND_DISTRACTED,
        frequencies=[720.0, 720.0],
        duration_per_tone=0.12,
        repeat=2,
        volume=0.70
    )


if __name__ == "__main__":
    ensure_alert_sounds()
    print("Sons de alerta gerados com sucesso em:", config.SOUNDS_DIR)
