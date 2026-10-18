"""
core/face_detector.py - Detector Facial e Extrator de Características Biométricas
Encapsula o Google MediaPipe FaceLandmarker com suporte a aceleração XNNPACK,
cálculo de coordenadas em pixels e extração de Blendshapes neurais.
"""

from dataclasses import dataclass, field
import urllib.request
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import config
from core.metrics import (
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
    calculate_ear,
    calculate_mar,
    calculate_head_pose_3d,
    calculate_chin_forehead_ratio
)


@dataclass
class FaceData:
    """Estrutura com todas as características extraídas da face do condutor."""
    landmarks_norm: np.ndarray             # (478, 3) normalizados [0.0, 1.0]
    landmarks_px: np.ndarray               # (478, 2) coordenadas de pixels (x, y)
    bbox: tuple[int, int, int, int]        # (x_min, y_min, x_max, y_max)
    ear_left: float                        # EAR olho esquerdo
    ear_right: float                       # EAR olho direito
    ear_avg: float                         # Média do EAR dos dois olhos
    mar: float                             # Mouth Aspect Ratio (abertura da boca)
    head_pitch: float                      # Inclinação vertical (graus: negativo = abaixando cabeça)
    head_yaw: float                        # Rotação horizontal (graus: esquerda/direita)
    head_roll: float                       # Inclinação lateral (graus)
    chin_ratio: float = 1.0                # Razão queixo/testa (compressão perspectiva de queda)
    blendshapes: dict[str, float] = field(default_factory=dict) # Escore de contrações musculares


class FaceDetector:
    """Gerenciador do modelo de malha facial e inferência em tempo real."""

    def __init__(self, model_path=None):
        self.model_path = model_path or config.MODEL_PATH
        self._ensure_model_exists()

        base_options = python.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            running_mode=vision.RunningMode.IMAGE
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def _ensure_model_exists(self) -> None:
        """Garante que o modelo neural do MediaPipe esteja presente localmente."""
        if not self.model_path.exists() or self.model_path.stat().st_size < 1000:
            print(f"[DMS] Baixando modelo neural FaceLandmarker para {self.model_path}...")
            urllib.request.urlretrieve(config.MODEL_URL, str(self.model_path))
            print("[DMS] Download do modelo concluído com sucesso!")

    def process_frame(self, frame_bgr: np.ndarray) -> FaceData | None:
        """
        Processa um frame de vídeo BGR e retorna os dados biométricos da face.
        
        :param frame_bgr: Imagem capturada da câmera (formato OpenCV BGR)
        :return: Instância de FaceData ou None se nenhuma face for detectada
        """
        h, w, _ = frame_bgr.shape
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        result = self.detector.detect(mp_image)

        if not result.face_landmarks or len(result.face_landmarks) == 0:
            return None

        # Coordenadas normalizadas (478 pontos)
        raw_landmarks = result.face_landmarks[0]
        landmarks_norm = np.array([
            [lm.x, lm.y, lm.z] for lm in raw_landmarks
        ], dtype=np.float32)

        # Conversão para pixels
        landmarks_px = np.zeros((len(raw_landmarks), 2), dtype=np.int32)
        landmarks_px[:, 0] = np.clip(landmarks_norm[:, 0] * w, 0, w - 1).astype(np.int32)
        landmarks_px[:, 1] = np.clip(landmarks_norm[:, 1] * h, 0, h - 1).astype(np.int32)

        # Bounding box delimitadora da face
        x_min = int(np.min(landmarks_px[:, 0]))
        y_min = int(np.min(landmarks_px[:, 1]))
        x_max = int(np.max(landmarks_px[:, 0]))
        y_max = int(np.max(landmarks_px[:, 1]))
        bbox = (x_min, y_min, x_max, y_max)

        # Extração de pontos dos olhos
        left_eye_pts = landmarks_px[LEFT_EYE_INDICES]
        right_eye_pts = landmarks_px[RIGHT_EYE_INDICES]

        # Cálculo de EAR
        ear_left = calculate_ear(left_eye_pts)
        ear_right = calculate_ear(right_eye_pts)
        ear_avg = (ear_left + ear_right) / 2.0

        # Cálculo de MAR (Boca)
        mar = calculate_mar(landmarks_px)

        # Orientação da cabeça 3D anatômica e invariante
        pitch, yaw, roll = calculate_head_pose_3d(landmarks_norm)
        chin_ratio = calculate_chin_forehead_ratio(landmarks_norm)

        # Extração de Blendshapes (Expressões e contrações musculares)
        blendshapes = {}
        if result.face_blendshapes and len(result.face_blendshapes) > 0:
            for category in result.face_blendshapes[0]:
                blendshapes[category.category_name] = category.score

        return FaceData(
            landmarks_norm=landmarks_norm,
            landmarks_px=landmarks_px,
            bbox=bbox,
            ear_left=ear_left,
            ear_right=ear_right,
            ear_avg=ear_avg,
            mar=mar,
            head_pitch=pitch,
            head_yaw=yaw,
            head_roll=roll,
            chin_ratio=chin_ratio,
            blendshapes=blendshapes
        )

    def close(self) -> None:
        """Libera recursos do detector."""
        if hasattr(self, "detector") and self.detector:
            self.detector.close()
