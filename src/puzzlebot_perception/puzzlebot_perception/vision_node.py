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
#   Paso 6 — Máscara pentagonal: mismo ancho que triángulo anterior,
#             más chaparra (70% de pantalla), vértices medios en 0.5
#
# Tópicos:
#   Suscribe:  /video_source/raw            (sensor_msgs/Image)
#   Publica:   /perception/line_error       (std_msgs/Float32MultiArray)
#              /perception/line_detected    (std_msgs/Bool)
#              /vision_debug               (sensor_msgs/Image)
#
# CORRECCIÓN [signo]:
#   error = (center - cx) / center   ← antes era (cx - center) / center
# ═══════════════════════════════════════════════════════════════════

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, Bool
from cv_bridge import CvBridge
import cv2
import numpy as np


class VisionNode(Node):

    def __init__(self):
        super().__init__('vision_node')

        self.declare_parameter('roi_top',               0.4)
        self.declare_parameter('blur_kernel',           7)
        self.declare_parameter('morph_kernel',          5)
        self.declare_parameter('debug',                 True)
        self.declare_parameter('min_area',              500)
        self.declare_parameter('max_area',              100000)
        self.declare_parameter('score_distance_weight', 8.0)
        self.declare_parameter('lookahead_row',         0.3)
        self.declare_parameter('lost_timeout',          2.22)

        # ── Máscara pentagonal ────────────────────────────────────
        # trap_top_frac:    ancho de la punta (arriba del ROI)
        # trap_mid_frac:    ancho a media altura — igual que base anterior
        # trap_mid_row:     fila del punto medio (0=arriba, 1=abajo)
        # trap_bottom_frac: ancho de la base (abajo del ROI)
        self.declare_parameter('trap_top_frac',         0.05)
        self.declare_parameter('trap_mid_frac',         0.6)
        self.declare_parameter('trap_mid_row',          0.5)
        self.declare_parameter('trap_bottom_frac',      0.6)

        self.error_pub    = self.create_publisher(Float32MultiArray, '/perception/line_error',    10)
        self.detected_pub = self.create_publisher(Bool,              '/perception/line_detected', 10)
        self.debug_pub    = self.create_publisher(Image,             '/vision_debug',             10)

        self.create_subscription(Image, '/video_source/raw', self.image_callback, 10)

        self.bridge          = CvBridge()
        self.last_time       = self.get_clock().now()
        self.frame_count     = 0
        self.last_error_main = 0.0
        self.last_error_look = 0.0
        self.last_line_time  = self.get_clock().now()

        self.get_logger().info('VisionNode iniciado — pentágono + lost_timeout 1.8s')

    # ───────────────────────────────────────────────────────────────
    # PASO 2 + PASO 6 — PREPROCESAMIENTO CON MÁSCARA PENTAGONAL
    # ───────────────────────────────────────────────────────────────
    def _preprocess(self, frame):
        roi_top     = self.get_parameter('roi_top').value
        blur_k      = self.get_parameter('blur_kernel').value
        morph_k     = self.get_parameter('morph_kernel').value
        top_frac    = self.get_parameter('trap_top_frac').value
        mid_frac    = self.get_parameter('trap_mid_frac').value
        mid_row     = self.get_parameter('trap_mid_row').value
        bottom_frac = self.get_parameter('trap_bottom_frac').value

        blur_k  = blur_k  if blur_k  % 2 == 1 else blur_k  + 1
        morph_k = morph_k if morph_k % 2 == 1 else morph_k + 1

        frame = cv2.resize(frame, (320, 240))
        h, w  = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        roi_y  = int(h * roi_top)
        roi    = gray[roi_y:h, :]
        rh, rw = roi.shape[:2]

        blurred = cv2.GaussianBlur(roi, (blur_k, blur_k), 0)

        _, binary = cv2.threshold(
            blurred, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_k, morph_k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # ── Vértices del pentágono ────────────────────────────────
        top_w  = int(rw * top_frac)
        top_x1 = (rw - top_w) // 2
        top_x2 = top_x1 + top_w

        mid_w  = int(rw * mid_frac)
        mid_x1 = (rw - mid_w) // 2
        mid_x2 = mid_x1 + mid_w
        mid_y  = int(rh * mid_row)

        bot_w  = int(rw * bottom_frac)
        bot_x1 = (rw - bot_w) // 2
        bot_x2 = bot_x1 + bot_w

        pentagon = np.array([
            [top_x1, 0      ],   # punta izquierda — arriba
            [top_x2, 0      ],   # punta derecha   — arriba
            [mid_x2, mid_y  ],   # vértice derecho — media altura
            [bot_x2, rh - 1 ],   # base derecha    — abajo
            [bot_x1, rh - 1 ],   # base izquierda  — abajo
            [mid_x1, mid_y  ],   # vértice izquierdo — media altura
        ], dtype=np.int32)

        mask   = np.zeros((rh, rw), dtype=np.uint8)
        cv2.fillPoly(mask, [pentagon], 255)
        binary = cv2.bitwise_and(binary, binary, mask=mask)

        return binary, roi, roi_y, w

    # ───────────────────────────────────────────────────────────────
    # PASO 3 — DETECCIÓN CON SCORE ESPACIAL
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

        for i in range(1, num_labels):
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
    # error = (center - cx) / center  →  convención setpoint - medición
    # ───────────────────────────────────────────────────────────────
    def _compute_errors(self, best_label, labels, centroids, binary_w):
        lookahead_row = self.get_parameter('lookahead_row').value
        center        = binary_w / 2.0
        roi_h         = labels.shape[0]

        line_detected = best_label != -1
        cx_main = cx_look = cy_main = None

        if line_detected:
            row_main = int(centroids[best_label][1])
            row_main = np.clip(row_main, 0, roi_h - 1)
            cy_main  = row_main

            cols_main = np.where(labels[row_main, :] == best_label)[0]
            if cols_main.size > 0:
                cx_main = float(cols_main[0] + cols_main[-1]) / 2.0

            row_look  = int(roi_h * lookahead_row)
            row_look  = np.clip(row_look, 0, roi_h - 1)
            cols_look = np.where(labels[row_look, :] == best_label)[0]
            if cols_look.size > 0:
                cx_look = float(cols_look[0] + cols_look[-1]) / 2.0

        error_main = (center - cx_main) / center if cx_main is not None else None
        error_look = (center - cx_look) / center if cx_look is not None else None

        return line_detected, cx_main, cx_look, cy_main, error_main, error_look

    # ───────────────────────────────────────────────────────────────
    # PASO 5 — PUBLICACIÓN CON MANEJO DE LÍNEA PERDIDA
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
    # DEBUG VISUAL — dibuja el pentágono cian
    # ───────────────────────────────────────────────────────────────
    def _publish_debug(self, roi, best_label, stats,
                       cx_main, cx_look, cy_main, w):
        lookahead_row = self.get_parameter('lookahead_row').value
        top_frac      = self.get_parameter('trap_top_frac').value
        mid_frac      = self.get_parameter('trap_mid_frac').value
        mid_row       = self.get_parameter('trap_mid_row').value
        bottom_frac   = self.get_parameter('trap_bottom_frac').value
        roi_h         = roi.shape[0]

        vis = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

        top_w  = int(w * top_frac)
        top_x1 = (w - top_w) // 2
        top_x2 = top_x1 + top_w

        mid_w  = int(w * mid_frac)
        mid_x1 = (w - mid_w) // 2
        mid_x2 = mid_x1 + mid_w
        mid_y  = int(roi_h * mid_row)

        bot_w  = int(w * bottom_frac)
        bot_x1 = (w - bot_w) // 2
        bot_x2 = bot_x1 + bot_w

        pentagon_pts = np.array([
            [top_x1, 0        ],
            [top_x2, 0        ],
            [mid_x2, mid_y    ],
            [bot_x2, roi_h - 1],
            [bot_x1, roi_h - 1],
            [mid_x1, mid_y    ],
        ], dtype=np.int32)

        cv2.polylines(vis, [pentagon_pts], True, (255, 255, 0), 1)
        cv2.line(vis, (w//2, 0), (w//2, roi_h), (0, 255, 0), 1)

        if best_label != -1:
            bx = stats[best_label, cv2.CC_STAT_LEFT]
            by = stats[best_label, cv2.CC_STAT_TOP]
            bw = stats[best_label, cv2.CC_STAT_WIDTH]
            bh = stats[best_label, cv2.CC_STAT_HEIGHT]
            cv2.rectangle(vis, (bx, by), (bx+bw, by+bh), (0, 255, 255), 1)

            if cx_main is not None and cy_main is not None:
                cv2.circle(vis, (int(cx_main), cy_main), 8, (0, 0, 255), -1)
                cv2.line(vis,
                         (int(cx_main), cy_main),
                         (w//2, cy_main),
                         (0, 0, 255), 2)

            if cx_look is not None:
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
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')
            return

        binary, roi, roi_y, w = self._preprocess(frame)
        best_label, labels, centroids, stats = self._detect_line(binary, w)
        line_detected, cx_main, cx_look, cy_main, error_main, error_look = \
            self._compute_errors(best_label, labels, centroids, w)

        self._publish(line_detected, error_main, error_look)

        self.frame_count += 1
        now = self.get_clock().now()
        dt  = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now

        if self.frame_count % 30 == 0 and dt > 0:
            if error_main is not None:
                look_str = f'{error_look:.3f}' if error_look is not None else 'N/A'
                self.get_logger().info(
                    f'FPS: {1.0/dt:.1f} | '
                    f'e_main: {error_main:.3f} | '
                    f'e_look: {look_str} | '
                    f'detected: {line_detected}'
                )
            else:
                self.get_logger().info(
                    f'FPS: {1.0/dt:.1f} | linea perdida'
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