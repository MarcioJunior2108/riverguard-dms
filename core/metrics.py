"""
core/metrics.py - Métricas Matemáticas e Biométricas de Fadiga
Implementa os algoritmos de Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR)
e Estimativa de Orientação da Cabeça (Head Pose - Pitch/Yaw/Roll).
"""

import math
import numpy as np
import cv2

# Índices canônicos da malha facial do MediaPipe (FaceMesh 478 pontos)
# Olho Esquerdo (da perspectiva do observador)
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

# Olho Direito (da perspectiva do observador)
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# Contorno interno da boca para cálculo de abertura (bocejo)
# [Canto Esq, Canto Dir, Sup Central, Inf Central, Sup Esq, Inf Esq, Sup Dir, Inf Dir]
MOUTH_INDICES = {
    "left_corner": 61,
    "right_corner": 291,
    "upper_mid": 13,
    "lower_mid": 14,
    "upper_left": 82,
    "lower_left": 87,
    "upper_right": 312,
    "lower_right": 317,
}

# Pontos de referência faciais 3D para cv2.solvePnP (Modelo antropométrico padrão)
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),          # Ponta do nariz (Nose tip) - marco 1
    (0.0, -330.0, -65.0),     # Queixo (Chin) - marco 152
    (-225.0, 170.0, -135.0),  # Canto externo olho esquerdo - marco 33
    (225.0, 170.0, -135.0),   # Canto externo olho direito - marco 263
    (-150.0, -150.0, -125.0), # Canto esquerdo da boca - marco 61
    (150.0, -150.0, -125.0)   # Canto direito da boca - marco 291
], dtype=np.float64)

HEAD_POSE_LANDMARK_IDS = [1, 152, 33, 263, 61, 291]


def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Calcula a distância euclidiana entre dois pontos 2D ou 3D."""
    return float(np.linalg.norm(p1 - p2))


def calculate_ear(eye_points: np.ndarray) -> float:
    """
    Calcula a Razão de Aspecto do Olho (Eye Aspect Ratio - EAR)
    baseado na fórmula de Soukupová e Čech (2016).
    
    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    
    :param eye_points: Array de 6 pontos correspondentes ao contorno do olho
    :return: Valor numérico do EAR (normalmente ~0.28-0.35 para olho aberto, <0.20 fechado)
    """
    if len(eye_points) < 6:
        return 0.0

    # Distâncias verticais
    v1 = euclidean_distance(eye_points[1], eye_points[5])
    v2 = euclidean_distance(eye_points[2], eye_points[4])

    # Distância horizontal
    horizontal = euclidean_distance(eye_points[0], eye_points[3])

    if horizontal < 1e-6:
        return 0.0

    ear = (v1 + v2) / (2.0 * horizontal)
    return float(ear)


def calculate_mar(landmarks: np.ndarray) -> float:
    """
    Calcula a Razão de Aspecto da Boca (Mouth Aspect Ratio - MAR)
    para detecção precisa de bocejos.
    
    :param landmarks: Array de coordenadas de todos os marcos faciais (pixel coordinates)
    :return: Valor do MAR (~0.20-0.30 repouso, >0.65 bocejo profundo)
    """
    p_left = landmarks[MOUTH_INDICES["left_corner"]]
    p_right = landmarks[MOUTH_INDICES["right_corner"]]
    p_up_mid = landmarks[MOUTH_INDICES["upper_mid"]]
    p_low_mid = landmarks[MOUTH_INDICES["lower_mid"]]
    p_up_left = landmarks[MOUTH_INDICES["upper_left"]]
    p_low_left = landmarks[MOUTH_INDICES["lower_left"]]
    p_up_right = landmarks[MOUTH_INDICES["upper_right"]]
    p_low_right = landmarks[MOUTH_INDICES["lower_right"]]

    # Distâncias verticais
    v_mid = euclidean_distance(p_up_mid, p_low_mid)
    v_left = euclidean_distance(p_up_left, p_low_left)
    v_right = euclidean_distance(p_up_right, p_low_right)

    # Distância horizontal
    horizontal = euclidean_distance(p_left, p_right)

    if horizontal < 1e-6:
        return 0.0

    mar = (v_mid + v_left + v_right) / (2.0 * horizontal)
    return float(mar)


def rotation_matrix_to_euler_angles(R: np.ndarray) -> tuple[float, float, float]:
    """
    Converte uma matriz de rotação 3x3 em ângulos de Euler (Pitch, Yaw, Roll) em graus.
    
    Convenção:
    - Pitch (Graus): Inclinação vertical (positivo = cima, negativo = baixo / cochilo)
    - Yaw (Graus): Rotação lateral (positivo = direita, negativo = esquerda / distração)
    - Roll (Graus): Inclinação sobre os ombros
    """
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0.0

    pitch = math.degrees(x)
    yaw = math.degrees(y)
    roll = math.degrees(z)
    return pitch, yaw, roll


def estimate_head_pose_pnp(
    landmarks_px: np.ndarray,
    img_width: int,
    img_height: int
) -> tuple[float, float, float]:
    """
    Calcula os ângulos de rotação da cabeça utilizando cv2.solvePnP.
    Altamente estável e não depende de recursos proprietários.
    
    :param landmarks_px: Marcos faciais em coordenadas de pixel (N x 2 ou N x 3)
    :param img_width: Largura do frame da câmera
    :param img_height: Altura do frame da câmera
    :return: (pitch, yaw, roll) em graus
    """
    image_points = np.array([
        landmarks_px[idx][:2] for idx in HEAD_POSE_LANDMARK_IDS
    ], dtype=np.float64)

    focal_length = img_width
    center = (img_width / 2.0, img_height / 2.0)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rotation_vector, _ = cv2.solvePnP(
        MODEL_POINTS_3D,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return 0.0, 0.0, 0.0

    rot_matrix, _ = cv2.Rodrigues(rotation_vector)
    pitch, yaw, roll = rotation_matrix_to_euler_angles(rot_matrix)
    return pitch, yaw, roll


def calculate_head_pose_3d(landmarks_norm: np.ndarray) -> tuple[float, float, float]:
    """
    Calcula a orientação tridimensional da cabeça (Pitch, Yaw, Roll) diretamente
    a partir das coordenadas espaciais 3D dos marcos faciais anatômicos (MediaPipe 478).
    - Pitch: negativo = cabeça inclinada/caída para frente; positivo = cabeça erguida
    - Yaw: positivo = virando à direita; negativo = virando à esquerda
    - Roll: inclinação lateral sobre os ombros
    """
    p10 = landmarks_norm[10]    # Testa / Glabella
    p152 = landmarks_norm[152]  # Queixo
    p33 = landmarks_norm[33]    # Canto olho esquerdo
    p263 = landmarks_norm[263]  # Canto olho direito

    # Pitch: relação Z e Y entre Queixo (152) e Testa (10)
    dz_pitch = float(p152[2] - p10[2])
    dy_pitch = float(p152[1] - p10[1])
    pitch = -math.degrees(math.atan2(dz_pitch, max(1e-4, abs(dy_pitch))))

    # Yaw: relação Z e X entre os olhos
    dx_yaw = float(p263[0] - p33[0])
    dz_yaw = float(p263[2] - p33[2])
    yaw = math.degrees(math.atan2(dz_yaw, max(1e-4, abs(dx_yaw))))

    # Roll: inclinação Y e X entre os olhos
    dy_roll = float(p263[1] - p33[1])
    dx_roll = float(p263[0] - p33[0])
    roll = math.degrees(math.atan2(dy_roll, max(1e-4, abs(dx_roll))))

    return pitch, yaw, roll


def calculate_chin_forehead_ratio(landmarks_norm: np.ndarray) -> float:
    """
    Calcula a razão de compressão vertical da face em perspectiva 2D.
    d_inferior = y_queixo - y_nariz
    d_superior = y_nariz - y_testa
    Quando a cabeça cai para frente (cochilo), o queixo recua e a razão cai abaixo de 0.55.
    """
    p10 = landmarks_norm[10]
    p1 = landmarks_norm[1]
    p152 = landmarks_norm[152]

    d_upper = max(1e-4, float(p1[1] - p10[1]))
    d_lower = max(0.0, float(p152[1] - p1[1]))
    return float(d_lower / d_upper)
