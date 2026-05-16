#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════
# vision_node.py
# Nodo de visión del PuzzleBot — detecta la línea negra y publica
# el error lateral para el controlador PID.
#
# Pasos implementados:
#   Paso 2 — Preprocesamiento: gris, ROI, blur, Otsu, morfología
#   Paso 3 — Detección: connected components, score espacial
#   Paso 4 — Errores: centro entre bordes + lookahead
#   Paso 5 — Línea perdida: timeout con último error conocido
#   Paso 6 — Máscara triangular: filtra distracciones laterales
#
# Tópicos:
#   Suscribe:  /video_source/raw            (sensor_msgs/Image)
#   Publica:   /perception/line_error       (std_msgs/Float32MultiArray)
#              /perception/line_detected    (std_msgs/Bool)
#              /vision_debug               (sensor_msgs/Image)
# ═══════════════════════════════════════════════════════════════════

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, Bool
from cv_bridge import CvBridge
import cv2
import numpy as np


class VisionNode(Node):

    # ───────────────────────────────────────────────────────────────
    # INICIALIZACIÓN
    # ───────────────────────────────────────────────────────────────
    def __init__(self):
        super().__init__('vision_node')

        # ── Parámetros Paso 2 (preprocesamiento) ──────────────────
        # roi_top: fracción desde arriba donde empieza el ROI
        #          0.4 = toma el 60% inferior de la imagen
        self.declare_parameter('roi_top',      0.4)

        # blur_kernel: tamaño del GaussianBlur (debe ser impar)
        self.declare_parameter('blur_kernel',  5)

        # morph_kernel: tamaño del kernel morfológico (debe ser impar)
        self.declare_parameter('morph_kernel', 5)

        # debug: activa /vision_debug — desactivar en producción
        self.declare_parameter('debug', True)

        # ── Parámetros Paso 3 (detección) ─────────────────────────
        # min_area: área mínima del blob — rechaza manchas de ruido
        self.declare_parameter('min_area', 500)

        # max_area: área máxima — rechaza si toda la imagen es blanca
        self.declare_parameter('max_area', 100000)

        # score_distance_weight: penalización por distancia al centro
        # score = area - weight * distancia
        # Subir si reflejos laterales ganan sobre la línea central
        self.declare_parameter('score_distance_weight', 2.0)

        # ── Parámetros Paso 4 (errores) ───────────────────────────
        # lookahead_row: posición del punto de anticipación en el ROI
        # 0.0 = arriba del ROI (más lejos), 1.0 = abajo (más cerca)
        self.declare_parameter('lookahead_row', 0.2)

        # ── Parámetros Paso 5 (línea perdida) ─────────────────────
        # lost_timeout: segundos con último error antes de publicar 0
        self.declare_parameter('lost_timeout', 0.5)

        # ── Parámetros Paso 6 (máscara triangular) ────────────────
        # trap_top_frac: qué tan estrecho es el triángulo arriba
        #   0.0 = punta perfecta, 0.05 = casi punta, 0.15 = más ancho
        self.declare_parameter('trap_top_frac', 0.05)

        # trap_bottom_frac: qué tan ancho es el triángulo abajo
        #   1.0 = ancho completo, 0.6 = 60% del ancho, 0.4 = estrecho
        self.declare_parameter('trap_bottom_frac', 0.6)

        # ── Publishers ────────────────────────────────────────────
        self.error_pub = self.create_publisher(
            Float32MultiArray, '/perception/line_error', 10
        )
        self.detected_pub = self.create_publisher(
            Bool, '/perception/line_detected', 10
        )
        self.debug_pub = self.create_publisher(Image, '/vision_debug', 10)

        # ── Subscriber ────────────────────────────────────────────
        self.create_subscription(
            Image, '/video_source/raw', self.image_callback, 10
        )

        # ── Variables de estado interno ───────────────────────────
        self.bridge      = CvBridge()
        self.last_time   = self.get_clock().now()
        self.frame_count = 0

        # Paso 5: memoria del último error conocido
        self.last_error_main = 0.0
        self.last_error_look = 0.0
        self.last_line_time  = self.get_clock().now()

        self.get_logger().info('VisionNode iniciado — Pasos 2, 3, 4, 5 y 6')

    # ───────────────────────────────────────────────────────────────
    # PASO 2 + PASO 6 — PREPROCESAMIENTO CON MÁSCARA TRIANGULAR
    #
    # Pipeline:
    #   1. Resize  → 320x240, 4x más rápido todo lo siguiente
    #   2. Gris    → elimina tinte rojizo, color no aporta nada
    #   3. ROI     → solo parte inferior donde está la línea
    #   4. Blur    → suaviza ruido del sensor
    #   5. Otsu    → umbral adaptativo automático por frame
    #   6. OPEN    → elimina manchas pequeñas de ruido
    #   7. CLOSE   → rellena huecos en zonas desgastadas
    #   8. Triángulo → filtra distracciones laterales
    #
    # El triángulo tiene ancho controlable arriba (trap_top_frac)
    # y abajo (trap_bottom_frac). Ambos parámetros son ajustables
    # desde YAML sin recompilar.
    # ───────────────────────────────────────────────────────────────
    def _preprocess(self, frame):
        roi_top     = self.get_parameter('roi_top').value
        blur_k      = self.get_parameter('blur_kernel').value
        morph_k     = self.get_parameter('morph_kernel').value
        top_frac    = self.get_parameter('trap_top_frac').value
        bottom_frac = self.get_parameter('trap_bottom_frac').value

        # Kernels deben ser impares
        blur_k  = blur_k  if blur_k  % 2 == 1 else blur_k  + 1
        morph_k = morph_k if morph_k % 2 == 1 else morph_k + 1

        # 1. Resize — 4x menos píxeles → todo más rápido
        #    w=320 SIEMPRE después del resize
        frame = cv2.resize(frame, (320, 240))
        h, w  = frame.shape[:2]

        # 2. Gris — elimina el tinte rojizo de la cámara del Jetson
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 3. ROI — solo la parte inferior donde está la línea
        roi_y  = int(h * roi_top)
        roi    = gray[roi_y:h, :]
        rh, rw = roi.shape[:2]

        # 4. Blur — suaviza ruido antes de umbralizar
        blurred = cv2.GaussianBlur(roi, (blur_k, blur_k), 0)

        # 5. Otsu — umbral adaptativo, se adapta a cambios de luz
        #    BINARY_INV: línea negra → blob blanco
        _, binary = cv2.threshold(
            blurred, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # 6. MORPH_OPEN — elimina manchas pequeñas de ruido
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_k, morph_k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)

        # 7. MORPH_CLOSE — rellena huecos en zonas desgastadas
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 8. Máscara triangular con control de ancho arriba Y abajo
        #    Arriba: estrecho en el centro (punta del triángulo)
        #    Abajo: ancho controlable — más estrecho filtra más ruido
        #           más ancho cubre mejor las curvas cerradas

        # Ancho arriba — punta del triángulo
        top_w  = int(rw * top_frac)
        top_x1 = (rw - top_w) // 2
        top_x2 = top_x1 + top_w

        # Ancho abajo — base del triángulo
        bot_w  = int(rw * bottom_frac)
        bot_x1 = (rw - bot_w) // 2
        bot_x2 = bot_x1 + bot_w

        triangle = np.array([
            [top_x1, 0     ],   # esquina superior izquierda (estrecha)
            [top_x2, 0     ],   # esquina superior derecha (estrecha)
            [bot_x2, rh - 1],   # esquina inferior derecha (ancha)
            [bot_x1, rh - 1],   # esquina inferior izquierda (ancha)
        ], dtype=np.int32)

        mask   = np.zeros((rh, rw), dtype=np.uint8)
        cv2.fillPoly(mask, [triangle], 255)

        # Solo procesar píxeles dentro del triángulo
        binary = cv2.bitwise_and(binary, binary, mask=mask)

        return binary, roi, roi_y, w

    # ───────────────────────────────────────────────────────────────
    # PASO 3 — DETECCIÓN DE LÍNEA CON SCORE ESPACIAL
    #
    # Score = area - weight * distancia_al_centro
    # El triángulo ya filtra laterales, pero el score agrega
    # una segunda capa de robustez para lo que pase dentro.
    # ───────────────────────────────────────────────────────────────
    def _detect_line(self, binary, w):
        min_area = self.get_parameter('min_area').value
        max_area = self.get_parameter('max_area').value
        weight   = self.get_parameter('score_distance_weight').value

        center_x = w / 2.0

        num_labels, labels, stats, centroids = \
            cv2.connectedComponentsWithStats(binary, connectivity=8)

        best_label = -1
        best_score = -float('inf')

        for i in range(1, num_labels):  # 0 = fondo, ignorar siempre
            area = stats[i, cv2.CC_STAT_AREA]

            if not (min_area <= area <= max_area):
                continue

            distance = abs(centroids[i][0] - center_x)
            score    = area - weight * distance

            if score > best_score:
                best_score = score
                best_label = i

        return best_label, labels, centroids, stats

    # ───────────────────────────────────────────────────────────────
    # PASO 4 — CÁLCULO DE ERRORES
    #
    # Centro entre bordes en vez de centroide puro:
    #   En curvas el blob es grande y torcido → el centroide queda
    #   en el medio geométrico, no sobre la línea real.
    #   Centro entre bordes por fila: siempre sobre la línea real.
    #
    # error_main → donde está la línea AHORA (fila del centroide)
    # error_look → hacia dónde VA la línea (fila superior del ROI)
    # Normalizados a [-1, 1]
    # ───────────────────────────────────────────────────────────────
    def _compute_errors(self, best_label, labels, centroids, binary_w):
        lookahead_row = self.get_parameter('lookahead_row').value
        center        = binary_w / 2.0
        roi_h         = labels.shape[0]

        line_detected = best_label != -1
        cx_main = cx_look = cy_main = None

        if line_detected:
            # Fila de referencia: centroide vertical del blob
            row_main = int(centroids[best_label][1])
            row_main = np.clip(row_main, 0, roi_h - 1)
            cy_main  = row_main

            # Centro entre bordes en la fila principal
            cols_main = np.where(labels[row_main, :] == best_label)[0]
            if cols_main.size > 0:
                cx_main = float(cols_main[0] + cols_main[-1]) / 2.0

            # Lookahead: misma técnica en fila más alta del ROI
            row_look  = int(roi_h * lookahead_row)
            row_look  = np.clip(row_look, 0, roi_h - 1)
            cols_look = np.where(labels[row_look, :] == best_label)[0]
            if cols_look.size > 0:
                cx_look = float(cols_look[0] + cols_look[-1]) / 2.0

        error_main = (cx_main - center) / center if cx_main is not None else None
        error_look = (cx_look - center) / center if cx_look is not None else None

        return line_detected, cx_main, cx_look, cy_main, error_main, error_look

    # ───────────────────────────────────────────────────────────────
    # PASO 5 — PUBLICACIÓN CON MANEJO DE LÍNEA PERDIDA
    #
    # Caso 1: línea visible → publicar error actual y guardarlo
    # Caso 2: perdida < timeout → último error (cubre cruce punteado)
    # Caso 3: perdida > timeout → publicar 0 para frenar
    # ───────────────────────────────────────────────────────────────
    def _publish(self, line_detected, error_main, error_look):
        now          = self.get_clock().now()
        lost_timeout = self.get_parameter('lost_timeout').value

        if line_detected and error_main is not None:
            self.last_error_main = error_main
            self.last_error_look = error_look if error_look is not None else error_main
            self.last_line_time  = now
            out_main = self.last_error_main
            out_look = self.last_error_look

        else:
            elapsed = (now - self.last_line_time).nanoseconds / 1e9

            if elapsed < lost_timeout:
                out_main = self.last_error_main
                out_look = self.last_error_look
            else:
                out_main = 0.0
                out_look = 0.0
                self.get_logger().warn(
                    f'Linea perdida {elapsed:.2f}s → publicando error 0'
                )

        error_msg      = Float32MultiArray()
        error_msg.data = [float(out_main), float(out_look)]
        self.error_pub.publish(error_msg)

        detected_msg      = Bool()
        detected_msg.data = line_detected
        self.detected_pub.publish(detected_msg)

    # ───────────────────────────────────────────────────────────────
    # DEBUG VISUAL
    #
    # Cian    → contorno del triángulo de búsqueda
    # Verde   → centro de referencia (error=0)
    # Amarillo→ bounding box del blob detectado
    # Rojo    → centroide principal + línea de error
    # Naranja → punto lookahead
    # ───────────────────────────────────────────────────────────────
    def _publish_debug(self, roi, best_label, stats,
                       cx_main, cx_look, cy_main, w):
        lookahead_row = self.get_parameter('lookahead_row').value
        top_frac      = self.get_parameter('trap_top_frac').value
        bottom_frac   = self.get_parameter('trap_bottom_frac').value
        roi_h         = roi.shape[0]

        vis = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

        # Triángulo cian — zona de búsqueda activa
        top_w  = int(w * top_frac)
        top_x1 = (w - top_w) // 2
        top_x2 = top_x1 + top_w

        bot_w  = int(w * bottom_frac)
        bot_x1 = (w - bot_w) // 2
        bot_x2 = bot_x1 + bot_w

        triangle_pts = np.array([
            [top_x1, 0        ],
            [top_x2, 0        ],
            [bot_x2, roi_h - 1],
            [bot_x1, roi_h - 1],
        ], dtype=np.int32)

        cv2.polylines(vis, [triangle_pts], True, (255, 255, 0), 1)

        # Línea verde — centro de referencia
        cv2.line(vis, (w//2, 0), (w//2, roi_h), (0, 255, 0), 1)

        if best_label != -1:
            # Bounding box amarillo
            bx = stats[best_label, cv2.CC_STAT_LEFT]
            by = stats[best_label, cv2.CC_STAT_TOP]
            bw = stats[best_label, cv2.CC_STAT_WIDTH]
            bh = stats[best_label, cv2.CC_STAT_HEIGHT]
            cv2.rectangle(vis, (bx, by), (bx+bw, by+bh), (0, 255, 255), 1)

            if cx_main is not None and cy_main is not None:
                # Círculo rojo — centroide principal
                cv2.circle(vis, (int(cx_main), cy_main), 8, (0, 0, 255), -1)
                # Línea roja — magnitud del error
                cv2.line(vis,
                         (int(cx_main), cy_main),
                         (w//2, cy_main),
                         (0, 0, 255), 2)

            if cx_look is not None:
                # Círculo naranja — lookahead
                lh_y = int(roi_h * lookahead_row)
                cv2.circle(vis, (int(cx_look), lh_y), 8, (0, 128, 255), -1)

        debug_msg = self.bridge.cv2_to_imgmsg(vis, encoding='bgr8')
        self.debug_pub.publish(debug_msg)

    # ───────────────────────────────────────────────────────────────
    # CALLBACK PRINCIPAL
    # ───────────────────────────────────────────────────────────────
    def image_callback(self, msg):
        debug = self.get_parameter('debug').value

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')
            return

        binary, roi, roi_y, w = self._preprocess(frame)
        best_label, labels, centroids, stats = self._detect_line(binary, w)
        line_detected, cx_main, cx_look, cy_main, error_main, error_look = \
            self._compute_errors(best_label, labels, centroids, w)

        self._publish(line_detected, error_main, error_look)

        # FPS — medir cada frame, loggear cada 30
        self.frame_count += 1
        now = self.get_clock().now()
        dt  = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now

        if self.frame_count % 30 == 0 and dt > 0:
            if error_main is not None and error_look is not None:
                self.get_logger().info(
                    f'FPS: {1.0/dt:.1f} | '
                    f'e_main: {error_main:.3f} | '
                    f'e_look: {error_look:.3f} | '
                    f'detected: {line_detected}'
                )
            else:
                self.get_logger().info(
                    f'FPS: {1.0/dt:.1f} | linea no detectada'
                )

        if debug:
            self._publish_debug(
                roi, best_label, stats,
                cx_main, cx_look, cy_main, w
            )


def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()