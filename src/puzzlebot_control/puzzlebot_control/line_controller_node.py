#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════
# line_controller_node.py
# Controlador PID para el seguidor de línea del PuzzleBot.
#
# Recibe el error lateral del vision_node y publica velocidades
# al robot. Integra el semáforo de la Rubik Pi.
#
# Tópicos:
#   Suscribe:  /perception/line_error    (Float32MultiArray)
#              /perception/line_detected (Bool)
#              /traffic_light_state      (String)
#   Publica:   /cmd_vel                  (Twist)
# ═══════════════════════════════════════════════════════════════════

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray, Bool, String
import numpy as np


class LineControllerNode(Node):

    # ───────────────────────────────────────────────────────────────
    # INICIALIZACIÓN
    # ───────────────────────────────────────────────────────────────
    def __init__(self):
        super().__init__('line_controller_node')

        # ── Parámetros PID ────────────────────────────────────────
        # El error viene normalizado [-1, 1] del vision_node.
        # Las ganancias iniciales son estimadas — afinar con MATLAB
        # después del experimento de identificación.
        # kp=0.8 equivale a kp=0.005 con error en píxeles (×160)
        self.declare_parameter('kp', 1.0)
        self.declare_parameter('ki', 0.005)
        self.declare_parameter('kd', 0.485)

        # integral_max: límite del anti-windup
        # Evita que el integral crezca sin límite en curvas largas
        self.declare_parameter('integral_max', 0.25)

        # ── Parámetros de velocidad ───────────────────────────────
        # v_base: velocidad en recta (m/s)
        self.declare_parameter('v_base', 0.15)

        # v_min: velocidad mínima en curva cerrada (m/s)
        # El robot nunca para aunque el error sea máximo
        self.declare_parameter('v_min', 0.04)

        # w_max: omega máximo (rad/s)
        # Limita el giro para no desestabilizar el robot
        self.declare_parameter('w_max', 1.8)

        # ── Parámetros de mezcla lookahead ────────────────────────
        # alpha_look: peso del error lookahead en la mezcla
        # error = (1 - alpha_look) * error_main + alpha_look * error_look
        # 0.0 = solo error inmediato (reactivo)
        # 1.0 = solo lookahead (anticipativo)
        # 0.4 = balance recomendado para curvas de 180°
        self.declare_parameter('alpha_look', 0.70)

        # ── Parámetros de watchdog ────────────────────────────────
        # lost_timeout: segundos sin línea detectada antes de frenar
        # El vision_node ya maneja 0.5s internamente, pero el
        # controlador necesita su propia capa de seguridad para
        # distinguir error=0 por línea centrada vs línea perdida
        self.declare_parameter('lost_timeout', 2.2)

        # traffic_timeout: segundos sin mensaje de semáforo
        # Si la Rubik Pi se desconecta → asumir UNKNOWN y continuar
        # El robot no debe pararse por un fallo de red
        self.declare_parameter('traffic_timeout', 1.0)

        # control_rate: frecuencia del loop de control (Hz)
        # Independiente de los fps de la cámara
        self.declare_parameter('control_rate', 20.0)

        # ── Publishers ────────────────────────────────────────────
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # ── Subscribers ───────────────────────────────────────────
        self.create_subscription(
            Float32MultiArray, '/perception/line_error',
            self.line_error_callback, 10
        )
        self.create_subscription(
            Bool, '/perception/line_detected',
            self.line_detected_callback, 10
        )
        self.create_subscription(
            String, '/traffic_light_state',
            self.traffic_callback, 10
        )

        # ── Variables de estado PID ───────────────────────────────
        self.integral  = 0.0
        self.prev_error = 0.0
        self.last_time  = self.get_clock().now()

        # ── Variables de estado de visión ─────────────────────────
        self.error_main    = 0.0
        self.error_look    = 0.0
        self.line_detected = False
        self.last_line_time = self.get_clock().now()

        # ── Variables de estado de semáforo ───────────────────────
        self.traffic_state    = 'UNKNOWN'
        self.red_latched      = False
        self.last_traffic_time = self.get_clock().now()
        self.last_traffic_log  = ''


        # obtencion de parametros para matlab
        self.start_time = self.get_clock().now()

        # ── Timer del loop de control ─────────────────────────────
        rate = self.get_parameter('control_rate').value
        self.create_timer(1.0 / rate, self.timer_cb)

        self.get_logger().info('LineControllerNode iniciado — PID line follower')

    # ───────────────────────────────────────────────────────────────
    # CALLBACKS DE SUSCRIPCIÓN
    # Solo guardan el último valor recibido.
    # El procesamiento ocurre en el timer_cb a 20 Hz.
    # ───────────────────────────────────────────────────────────────
    def line_error_callback(self, msg):
        """Guarda el error lateral publicado por vision_node."""
        if len(msg.data) >= 2:
            self.error_main = float(msg.data[0])
            self.error_look = float(msg.data[1])

    def line_detected_callback(self, msg):
        """
        Guarda si la línea está visible.
        Actualiza el tiempo del último frame con línea detectada.
        Esto permite al watchdog distinguir entre:
          - error=0 porque la línea está centrada (detected=True)
          - error=0 porque se perdió la línea (detected=False)
        """
        self.line_detected = msg.data
        if msg.data:
            self.last_line_time = self.get_clock().now()

    def traffic_callback(self, msg):
        """
        Maneja el estado del semáforo de la Rubik Pi.
        RED activa red_latched — el robot no avanza hasta GREEN.
        YELLOW escala la velocidad a la mitad.
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
            # Verde después de rojo: resetear integral para arranque limpio
            self.red_latched = False
            self.integral    = 0.0
            self.prev_error  = 0.0
            self.last_time   = self.get_clock().now()

        log = f'Semaforo: {prev} → {state}'
        if log != self.last_traffic_log:
            self.get_logger().info(log)
            self.last_traffic_log = log

    # ───────────────────────────────────────────────────────────────
    # PID ANGULAR
    #
    # Calcula omega a partir del error normalizado [-1, 1].
    #
    # Anti-windup: clip del integral entre ±integral_max
    # Reset en cruce de cero: si el error cambia de signo,
    # el integral se resetea para evitar sobreimpulso
    # ───────────────────────────────────────────────────────────────
    def _compute_pid(self, error: float, dt: float) -> float:
        kp = self.get_parameter('kp').value
        ki = self.get_parameter('ki').value
        kd = self.get_parameter('kd').value
        integral_max = self.get_parameter('integral_max').value

        # Reset integral cuando el error cruza cero
        # Evita sobreimpulso al salir de una curva
        if error * self.prev_error < 0:
            self.integral = 0.0

        # Acumular integral con anti-windup
        self.integral += error * dt
        self.integral  = float(np.clip(self.integral, -integral_max, integral_max))

        # Término derivativo
        derivative = (error - self.prev_error) / dt if dt > 1e-6 else 0.0
        self.prev_error = error

        omega = kp * error + ki * self.integral + kd * derivative
        w_max = self.get_parameter('w_max').value
        return float(np.clip(omega, -w_max, w_max))

    # ───────────────────────────────────────────────────────────────
    # VELOCIDAD LINEAL CON CURVATURE SCALE
    #
    # En vez de velocidad fija, reduce proporcional al giro.
    # Más elegante que curve_brake fijo:
    #   curvature_scale = 1 - |omega| / w_max * factor
    # El robot frena suavemente en curvas y acelera en rectas.
    # ───────────────────────────────────────────────────────────────
    def _compute_velocity(self, omega: float) -> float:
        v_base = self.get_parameter('v_base').value
        v_min  = self.get_parameter('v_min').value
        w_max  = self.get_parameter('w_max').value

        curvature_scale = 1.0 - min(abs(omega) / w_max, 0.65)
        v = v_base * curvature_scale
        return float(np.clip(v, v_min, v_base))

    # ───────────────────────────────────────────────────────────────
    # LOOP DE CONTROL — 20 Hz
    #
    # Orden de prioridad:
    #   1. Semáforo rojo → frenar siempre
    #   2. Watchdog semáforo → si Rubik Pi se cae, asumir UNKNOWN
    #   3. Watchdog línea perdida → si no hay línea, frenar
    #   4. Control normal → mezclar errores, PID, publicar
    # ───────────────────────────────────────────────────────────────
    def timer_cb(self):
        now = self.get_clock().now()
        dt  = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now

        if dt <= 0.0:
            return

        # ── Prioridad 1: semáforo rojo ─────────────────────────
        if self.red_latched:
            self._publish_stop()
            return

        # ── Prioridad 2: watchdog semáforo ─────────────────────
        # Si no llega mensaje de la Rubik Pi por más de traffic_timeout
        # asumir UNKNOWN y continuar — fallo de red no paraliza el robot
        traffic_timeout = self.get_parameter('traffic_timeout').value
        traffic_age = (now - self.last_traffic_time).nanoseconds / 1e9

        if traffic_age > traffic_timeout and self.traffic_state != 'UNKNOWN':
            self.get_logger().warn(
                f'Rubik Pi desconectada ({traffic_age:.1f}s) → asumiendo UNKNOWN'
            )
            self.traffic_state = 'UNKNOWN'

        # ── Prioridad 3: watchdog línea perdida ────────────────
        # Si el vision_node no detecta línea por más de lost_timeout
        # frenar y resetear el integral
        lost_timeout = self.get_parameter('lost_timeout').value
        line_age = (now - self.last_line_time).nanoseconds / 1e9

        if not self.line_detected and line_age > lost_timeout:
            self.get_logger().warn(
                f'Linea perdida {line_age:.2f}s → frenando'
            )
            self._publish_stop()
            self.integral   = 0.0
            self.prev_error = 0.0
            return

        # ── Prioridad 4: control normal ────────────────────────
        alpha_look = self.get_parameter('alpha_look').value

        # Mezclar error inmediato y lookahead
        # alpha_look=0.4 → 60% reactivo + 40% anticipativo
        error = (1.0 - alpha_look) * self.error_main + \
                 alpha_look        * self.error_look

        # PID angular
        omega = self._compute_pid(error, dt)

        # Velocidad con curvature scale
        v = self._compute_velocity(omega)

        # Escalar por semáforo amarillo
        if self.traffic_state == 'YELLOW':
            v     *= 0.5
            omega *= 0.5

        cmd = Twist()
        cmd.linear.x  = v
        cmd.angular.z = omega
        self.cmd_pub.publish(cmd)

        t_elapsed = (now - self.start_time).nanoseconds / 1e9

        self.get_logger().info(
            f't={t_elapsed:.4f} '
            f'SEM={self.traffic_state} | '
            f'e={error:.6f} '
            f'(main={self.error_main:.3f} look={self.error_look:.3f}) | '
            f'v={v:.3f} w={omega:.6f}'
        )

    def _publish_stop(self):
        self.cmd_pub.publish(Twist())

    def destroy_node(self):
        # Al morir el nodo, frenar el robot
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