#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════
# line_controller_node.py
# Controlador PID para el seguidor de línea del PuzzleBot.
#
# Recibe el error lateral del perception_node y publica velocidades
# al robot. Integra:
#   - semáforo de la Rubik Pi
#   - detector YOLO de señales de tráfico
#
# Tópicos:
#   Suscribe:
#       /perception/line_error       (Float32MultiArray)
#       /perception/line_detected    (Bool)
#       /traffic_light_state         (String)
#       /detecciones                 (String JSON desde YOLO)
#
#   Publica:
#       /cmd_vel                     (Twist)
#
# Formato esperado de /detecciones:
# [
#   {
#     "clase": "STOP",
#     "confianza": 0.87,
#     "bbox": [x1, y1, x2, y2]
#   }
# ]
# ═══════════════════════════════════════════════════════════════════

import json
import rclpy
import numpy as np

from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray, Bool, String


class LineControllerNode(Node):

    # ───────────────────────────────────────────────────────────────
    # INICIALIZACIÓN
    # ───────────────────────────────────────────────────────────────
    def __init__(self):
        super().__init__('line_controller_node')

        # ── Parámetros PID ────────────────────────────────────────
        self.declare_parameter('kp', 0.55)
        self.declare_parameter('ki', 0.0)
        self.declare_parameter('kd', 0.10)

        # Anti-windup
        self.declare_parameter('integral_max', 0.15)

        # ── Parámetros de velocidad ───────────────────────────────
        self.declare_parameter('v_base', 0.05)
        self.declare_parameter('v_min', 0.0)
        self.declare_parameter('w_max', 0.55)

        # ── Parámetros de mezcla lookahead ────────────────────────
        self.declare_parameter('alpha_look', 0.20)

        # ── Watchdogs ─────────────────────────────────────────────
        self.declare_parameter('lost_timeout', 0.25)
        self.declare_parameter('traffic_timeout', 1.0)
        self.declare_parameter('control_rate', 20.0)

        # ── Señales YOLO ─────────────────────────────────────────
        self.declare_parameter('sign_topic', '/detecciones')
        self.declare_parameter('sign_confidence_min', 0.45)
        self.declare_parameter('sign_timeout', 0.8)

        # Tiempo que el robot se queda detenido al ver STOP
        self.declare_parameter('stop_hold_seconds', 3.0)

        # Escalas de velocidad por señales
        self.declare_parameter('roadwork_speed_scale', 0.50)
        self.declare_parameter('give_way_speed_scale', 0.40)

        # ── Publishers ────────────────────────────────────────────
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # ── Subscribers ───────────────────────────────────────────
        self.create_subscription(
            Float32MultiArray,
            '/perception/line_error',
            self.line_error_callback,
            10
        )

        self.create_subscription(
            Bool,
            '/perception/line_detected',
            self.line_detected_callback,
            10
        )

        self.create_subscription(
            String,
            '/traffic_light_state',
            self.traffic_callback,
            10
        )

        sign_topic = self.get_parameter('sign_topic').value
        self.create_subscription(
            String,
            sign_topic,
            self.traffic_sign_callback,
            10
        )

        # ── Variables PID ─────────────────────────────────────────
        self.integral = 0.0
        self.prev_error = 0.0
        self.last_time = self.get_clock().now()

        # ── Estado de línea ───────────────────────────────────────
        self.error_main = 0.0
        self.error_look = 0.0
        self.line_detected = False
        self.last_line_time = self.get_clock().now()

        # ── Estado de semáforo ────────────────────────────────────
        self.traffic_state = 'UNKNOWN'
        self.red_latched = False
        self.last_traffic_time = self.get_clock().now()
        self.last_traffic_log = ''

        # ── Estado de señales YOLO ────────────────────────────────
        self.sign_state = 'NONE'
        self.sign_confidence = 0.0
        self.last_sign_time = self.get_clock().now()
        self.last_sign_log = ''

        # STOP hold:
        # stop_hold_remaining mantiene el robot detenido.
        # stop_sign_latched evita que el mismo STOP dispare hold infinitamente.
        self.stop_hold_remaining = 0.0
        self.stop_sign_latched = False

        # Tiempo para logs/experimentos
        self.start_time = self.get_clock().now()

        # ── Timer de control ──────────────────────────────────────
        rate = float(self.get_parameter('control_rate').value)
        self.create_timer(1.0 / max(rate, 1.0), self.timer_cb)

        self.get_logger().info('LineControllerNode iniciado — PID line follower')
        self.get_logger().info(f'Escuchando señales YOLO en: {sign_topic}')

    # ───────────────────────────────────────────────────────────────
    # CALLBACKS
    # ───────────────────────────────────────────────────────────────
    def line_error_callback(self, msg):
        """Guarda el error lateral publicado por perception_node."""
        if len(msg.data) >= 2:
            self.error_main = float(msg.data[0])
            self.error_look = float(msg.data[1])

    def line_detected_callback(self, msg):
        """
        Guarda si la línea está visible.
        Actualiza el tiempo del último frame con línea detectada.
        """
        self.line_detected = bool(msg.data)

        if self.line_detected:
            self.last_line_time = self.get_clock().now()

    def traffic_callback(self, msg):
        """
        Maneja el estado del semáforo de la Rubik Pi.
        RED activa red_latched.
        GREEN después de RED libera el latch.
        YELLOW reduce velocidad en timer_cb.
        """
        state = msg.data.upper().strip()

        if state not in ['RED', 'YELLOW', 'GREEN', 'UNKNOWN']:
            state = 'UNKNOWN'

        self.last_traffic_time = self.get_clock().now()

        prev = self.traffic_state
        self.traffic_state = state

        if state == 'RED':
            self.red_latched = True

        elif state == 'GREEN' and self.red_latched:
            self.red_latched = False
            self.integral = 0.0
            self.prev_error = 0.0
            self.last_time = self.get_clock().now()

        log = f'Semaforo: {prev} → {state}'
        if log != self.last_traffic_log:
            self.get_logger().info(log)
            self.last_traffic_log = log

    def traffic_sign_callback(self, msg):
        """
        Recibe detecciones del detector YOLO.

        Espera JSON tipo:
        [
            {
                "clase": "STOP",
                "confianza": 0.87,
                "bbox": [x1, y1, x2, y2]
            }
        ]

        Si no hay detecciones, el detector debe publicar [].
        """
        try:
            detections = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f'No pude leer JSON de /detecciones: {e}')
            self.sign_state = 'NONE'
            self.sign_confidence = 0.0
            return

        if not isinstance(detections, list) or len(detections) == 0:
            self.sign_state = 'NONE'
            self.sign_confidence = 0.0
            self.stop_sign_latched = False
            return

        best_detection = None
        best_confidence = -1.0

        for detection in detections:
            if not isinstance(detection, dict):
                continue

            confidence = float(detection.get('confianza', 0.0))

            if confidence > best_confidence:
                best_confidence = confidence
                best_detection = detection

        if best_detection is None:
            self.sign_state = 'NONE'
            self.sign_confidence = 0.0
            return

        min_conf = float(self.get_parameter('sign_confidence_min').value)

        if best_confidence < min_conf:
            self.sign_state = 'NONE'
            self.sign_confidence = best_confidence
            return

        raw_class = str(best_detection.get('clase', 'NONE'))
        sign = self._normalize_sign_name(raw_class)

        self.sign_state = sign
        self.sign_confidence = best_confidence
        self.last_sign_time = self.get_clock().now()

        log = f'Señal YOLO: {sign} conf={best_confidence:.2f}'
        if log != self.last_sign_log:
            self.get_logger().info(log)
            self.last_sign_log = log

    # ───────────────────────────────────────────────────────────────
    # NORMALIZACIÓN DE NOMBRES DE SEÑALES
    # ───────────────────────────────────────────────────────────────
    def _normalize_sign_name(self, name: str) -> str:
        """
        Convierte nombres de clase del modelo a nombres estándar.

        Ajusta aquí los nombres exactos de tus clases YOLO si son distintos.
        """
        s = name.upper().strip()
        s = s.replace(' ', '_')
        s = s.replace('-', '_')

        stop_aliases = [
            'STOP',
            'STOP_SIGN',
            'ALTO',
            'SIGN_STOP'
        ]

        roadwork_aliases = [
            'ROADWORK',
            'ROADWORK_AHEAD',
            'WORK_AHEAD',
            'CONSTRUCTION',
            'OBRAS',
            'OBRA'
        ]

        give_way_aliases = [
            'GIVE_WAY',
            'YIELD',
            'CEDA',
            'CEDA_EL_PASO'
        ]

        if s in stop_aliases:
            return 'STOP'

        if s in roadwork_aliases:
            return 'ROADWORK_AHEAD'

        if s in give_way_aliases:
            return 'GIVE_WAY'

        return s

    # ───────────────────────────────────────────────────────────────
    # PID ANGULAR
    # ───────────────────────────────────────────────────────────────
    def _compute_pid(self, error: float, dt: float) -> float:
        kp = float(self.get_parameter('kp').value)
        ki = float(self.get_parameter('ki').value)
        kd = float(self.get_parameter('kd').value)
        integral_max = float(self.get_parameter('integral_max').value)

        if error * self.prev_error < 0:
            self.integral = 0.0

        self.integral += error * dt
        self.integral = float(
            np.clip(self.integral, -integral_max, integral_max)
        )

        derivative = (error - self.prev_error) / dt if dt > 1e-6 else 0.0
        self.prev_error = error

        omega = kp * error + ki * self.integral + kd * derivative

        w_max = float(self.get_parameter('w_max').value)
        omega = float(np.clip(omega, -w_max, w_max))

        return omega

    # ───────────────────────────────────────────────────────────────
    # VELOCIDAD LINEAL
    # ───────────────────────────────────────────────────────────────
    def _compute_velocity(self, omega: float) -> float:
        v_base = float(self.get_parameter('v_base').value)
        v_min = float(self.get_parameter('v_min').value)
        w_max = float(self.get_parameter('w_max').value)

        if w_max <= 1e-6:
            return v_min

        curvature_scale = 1.0 - min(abs(omega) / w_max, 0.65)
        v = v_base * curvature_scale

        return float(np.clip(v, v_min, v_base))

    # ───────────────────────────────────────────────────────────────
    # LOOP DE CONTROL
    #
    # Prioridad:
    #   1. STOP hold por señal STOP
    #   2. Semáforo rojo
    #   3. Watchdog semáforo
    #   4. Watchdog línea
    #   5. Control normal
    # ───────────────────────────────────────────────────────────────
    def timer_cb(self):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now

        if dt <= 0.0:
            return

        # ── Prioridad 1: mantener STOP por tiempo definido ─────────
        if self.stop_hold_remaining > 0.0:
            self.stop_hold_remaining -= dt
            self._publish_stop()
            return

        # ── Señales viejas expiran ────────────────────────────────
        sign_timeout = float(self.get_parameter('sign_timeout').value)
        sign_age = (now - self.last_sign_time).nanoseconds / 1e9

        if sign_age > sign_timeout:
            self.sign_state = 'NONE'
            self.sign_confidence = 0.0
            self.stop_sign_latched = False

        # ── Prioridad 1.5: detectar nuevo STOP ────────────────────
        if self.sign_state == 'STOP' and not self.stop_sign_latched:
            hold_time = float(self.get_parameter('stop_hold_seconds').value)
            self.stop_hold_remaining = hold_time
            self.stop_sign_latched = True

            self.integral = 0.0
            self.prev_error = 0.0

            self.get_logger().warn(
                f'STOP detectado → frenando {hold_time:.1f}s'
            )

            self._publish_stop()
            return

        if self.sign_state != 'STOP':
            self.stop_sign_latched = False

        # ── Prioridad 2: semáforo rojo ────────────────────────────
        if self.red_latched:
            self._publish_stop()
            return

        # ── Prioridad 3: watchdog semáforo ────────────────────────
        traffic_timeout = float(self.get_parameter('traffic_timeout').value)
        traffic_age = (now - self.last_traffic_time).nanoseconds / 1e9

        if traffic_age > traffic_timeout and self.traffic_state != 'UNKNOWN':
            self.get_logger().warn(
                f'Rubik Pi semaforo sin mensajes ({traffic_age:.1f}s) '
                f'→ asumiendo UNKNOWN'
            )
            self.traffic_state = 'UNKNOWN'

        # ── Prioridad 4: watchdog línea perdida ───────────────────
        lost_timeout = float(self.get_parameter('lost_timeout').value)
        line_age = (now - self.last_line_time).nanoseconds / 1e9

        if not self.line_detected and line_age > lost_timeout:
            self.get_logger().warn(
                f'Linea perdida {line_age:.2f}s → frenando'
            )
            self._publish_stop()
            self.integral = 0.0
            self.prev_error = 0.0
            return

        # ── Prioridad 5: control normal ───────────────────────────
        alpha_look = float(self.get_parameter('alpha_look').value)
        alpha_look = float(np.clip(alpha_look, 0.0, 1.0))

        error = (1.0 - alpha_look) * self.error_main + \
                alpha_look * self.error_look

        omega = self._compute_pid(error, dt)
        v = self._compute_velocity(omega)

        # Semáforo amarillo: reducir velocidad y giro
        if self.traffic_state == 'YELLOW':
            v *= 0.5
            omega *= 0.5

        # Señales de tráfico
        if self.sign_state == 'ROADWORK_AHEAD':
            scale = float(self.get_parameter('roadwork_speed_scale').value)
            v *= scale

        elif self.sign_state == 'GIVE_WAY':
            scale = float(self.get_parameter('give_way_speed_scale').value)
            v *= scale

        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(omega)

        self.cmd_pub.publish(cmd)

        t_elapsed = (now - self.start_time).nanoseconds / 1e9

        self.get_logger().info(
            f't={t_elapsed:.3f} | '
            f'SEM={self.traffic_state} | '
            f'SIGN={self.sign_state} conf={self.sign_confidence:.2f} | '
            f'e={error:.4f} '
            f'(main={self.error_main:.3f} look={self.error_look:.3f}) | '
            f'v={v:.3f} w={omega:.3f}'
        )

    # ───────────────────────────────────────────────────────────────
    # UTILIDADES
    # ───────────────────────────────────────────────────────────────
    def _publish_stop(self):
        self.cmd_pub.publish(Twist())

    def destroy_node(self):
        self._publish_stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LineControllerNode()

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
