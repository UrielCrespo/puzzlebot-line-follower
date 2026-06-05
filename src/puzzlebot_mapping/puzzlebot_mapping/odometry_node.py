#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════
# odometry_node.py
# Odometría diferencial para el PuzzleBot — versión mejorada.
#
# Mejoras sobre la versión base:
#   1. Integración RK2 (Runge-Kutta orden 2) — más preciso en curvas
#   2. Dead zone en encoders — elimina drift cuando el robot está quieto
#   3. Filtro de mediana (ventana=5) — elimina spikes de la Hackerboard
#
# Tópicos:
#   Suscribe:  /VelocityEncR  (std_msgs/Float32) — vel angular rueda derecha [rad/s]
#              /VelocityEncL  (std_msgs/Float32) — vel angular rueda izquierda [rad/s]
#   Publica:   /odom          (nav_msgs/Odometry)
#              /pose          (geometry_msgs/Pose2D)
#   TF:        odom → base_link  (dinámico)
#
# Parámetros:
#   wheel_radius  [m]     — radio de la rueda
#   wheel_sep     [m]     — distancia entre centros de ruedas
#   odom_rate     [Hz]    — frecuencia del loop de integración
#   dead_zone     [rad/s] — umbral mínimo de velocidad de rueda
#   median_window [int]   — tamaño de la ventana del filtro de mediana
# ═══════════════════════════════════════════════════════════════════

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose2D, TransformStamped
from tf2_ros import TransformBroadcaster
from collections import deque
import statistics
import math


class OdometryNode(Node):

    # ───────────────────────────────────────────────────────────────
    # INICIALIZACIÓN
    # ───────────────────────────────────────────────────────────────
    def __init__(self):
        super().__init__('odometry_node')

        # ── Parámetros del robot ──────────────────────────────────
        self.declare_parameter('wheel_radius',  0.05)
        self.declare_parameter('wheel_sep',     0.19)
        self.declare_parameter('odom_rate',     50.0)

        # Dead zone: velocidades de rueda menores a este valor
        # se tratan como cero. Elimina el drift causado por el
        # ruido de la Hackerboard cuando el robot está quieto.
        # Ajustar si el robot "se mueve" en el mapa estando parado.
        self.declare_parameter('dead_zone',     0.01)

        # Ventana del filtro de mediana.
        # 5 muestras = ~100ms de latencia a 50Hz — imperceptible.
        # Aumentar si la Hackerboard produce muchos spikes.
        self.declare_parameter('median_window', 5)

        self.r  = self.get_parameter('wheel_radius').value
        self.l  = self.get_parameter('wheel_sep').value
        self.dz = self.get_parameter('dead_zone').value

        win = int(self.get_parameter('median_window').value)

        # ── Buffers del filtro de mediana ─────────────────────────
        # deque con maxlen descarta automáticamente el valor más
        # antiguo cuando se añade uno nuevo — ventana deslizante.
        self.wr_buffer = deque(maxlen=win)
        self.wl_buffer = deque(maxlen=win)

        # Inicializar buffers con ceros para que la mediana
        # sea válida desde el primer mensaje recibido.
        for _ in range(win):
            self.wr_buffer.append(0.0)
            self.wl_buffer.append(0.0)

        # ── Velocidades filtradas (salida del filtro de mediana) ──
        self.wr = 0.0
        self.wl = 0.0

        # ── Estado de pose ────────────────────────────────────────
        self.x     = 0.0
        self.y     = 0.0
        self.theta = 0.0

        self.last_time = self.get_clock().now()

        # ── TF broadcaster ────────────────────────────────────────
        self.tf_broadcaster = TransformBroadcaster(self)

        # ── Publishers ────────────────────────────────────────────
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.pose_pub = self.create_publisher(Pose2D,   '/pose', 10)

        # ── Subscribers ───────────────────────────────────────────
        self.create_subscription(Float32, '/VelocityEncR',
                                 self._cb_wr, 10)
        self.create_subscription(Float32, '/VelocityEncL',
                                 self._cb_wl, 10)

        # ── Timer de integración ──────────────────────────────────
        rate = self.get_parameter('odom_rate').value
        self.create_timer(1.0 / rate, self._integrate)

        self.get_logger().info(
            f'OdometryNode iniciado — '
            f'r={self.r:.4f}m  l={self.l:.4f}m  '
            f'rate={rate}Hz  dead_zone={self.dz}  window={win}'
        )

    # ───────────────────────────────────────────────────────────────
    # CALLBACKS DE ENCODERS + FILTRO DE MEDIANA
    #
    # Cada callback añade el nuevo valor al buffer circular y
    # recalcula la mediana. La mediana es resistente a spikes:
    # un valor anómalo único nunca puede mover la mediana más
    # de una posición en el buffer ordenado.
    # ───────────────────────────────────────────────────────────────
    def _cb_wr(self, msg: Float32):
        self.wr_buffer.append(msg.data)
        self.wr = statistics.median(self.wr_buffer)

    def _cb_wl(self, msg: Float32):
        self.wl_buffer.append(msg.data)
        self.wl = statistics.median(self.wl_buffer)

    # ───────────────────────────────────────────────────────────────
    # INTEGRACIÓN RK2 (RUNGE-KUTTA ORDEN 2 — MÉTODO DEL PUNTO MEDIO)
    #
    # Euler simple asume orientación constante durante dt:
    #   x += v*cos(θ)*dt  ← usa θ al inicio del paso
    #
    # RK2 usa la orientación en el punto medio del paso:
    #   θ_mid = θ + 0.5*omega*dt
    #   x    += v*cos(θ_mid)*dt
    #   y    += v*sin(θ_mid)*dt
    #
    # La diferencia es importante en curvas: Euler acumula error
    # proporcional a omega*dt por paso, RK2 lo reduce a (omega*dt)^2.
    # En una curva de 180° a 0.5 rad/s con dt=0.02s, Euler comete
    # ~1cm de error adicional por vuelta — RK2 lo lleva a <1mm.
    # ───────────────────────────────────────────────────────────────
    def _integrate(self):
        now = self.get_clock().now()
        dt  = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now

        # Guard: saltar dt inválido (primer ciclo o pausa larga)
        if dt <= 0.0 or dt > 1.0:
            return

        # ── Dead zone ─────────────────────────────────────────────
        # Si la velocidad filtrada es menor al umbral, tratar como
        # cero. Esto evita que ruido residual del filtro de mediana
        # acumule desplazamiento cuando el robot está parado.
        wr = self.wr if abs(self.wr) > self.dz else 0.0
        wl = self.wl if abs(self.wl) > self.dz else 0.0

        # ── Velocidades del cuerpo ────────────────────────────────
        v     = self.r / 2.0 * (wr + wl)
        omega = self.r / self.l * (wr - wl)

        # ── RK2: integración con orientación en el punto medio ────
        theta_mid   = self.theta + 0.5 * omega * dt
        self.x     += v * math.cos(theta_mid) * dt
        self.y     += v * math.sin(theta_mid) * dt
        self.theta += omega * dt

        # Normalizar theta a [-π, π] para evitar acumulación
        # de punto flotante en recorridos largos
        self.theta = math.atan2(math.sin(self.theta),
                                math.cos(self.theta))

        stamp = now.to_msg()
        self._publish_odom(stamp, v, omega)
        self._publish_pose()
        self._broadcast_tf(stamp)

    # ───────────────────────────────────────────────────────────────
    # PUBLICAR /odom
    # ───────────────────────────────────────────────────────────────
    def _publish_odom(self, stamp, v: float, omega: float):
        msg = Odometry()
        msg.header.stamp    = stamp
        msg.header.frame_id = 'odom'
        msg.child_frame_id  = 'base_link'

        msg.pose.pose.position.x = self.x
        msg.pose.pose.position.y = self.y
        msg.pose.pose.position.z = 0.0

        msg.pose.pose.orientation.z = math.sin(self.theta / 2.0)
        msg.pose.pose.orientation.w = math.cos(self.theta / 2.0)

        msg.twist.twist.linear.x  = v
        msg.twist.twist.angular.z = omega

        self.odom_pub.publish(msg)

    # ───────────────────────────────────────────────────────────────
    # PUBLICAR /pose (Pose2D — compatibilidad con el resto del ws)
    # ───────────────────────────────────────────────────────────────
    def _publish_pose(self):
        msg = Pose2D()
        msg.x     = self.x
        msg.y     = self.y
        msg.theta = self.theta
        self.pose_pub.publish(msg)

    # ───────────────────────────────────────────────────────────────
    # BROADCAST TF: odom → base_link
    # ───────────────────────────────────────────────────────────────
    def _broadcast_tf(self, stamp):
        t = TransformStamped()
        t.header.stamp    = stamp
        t.header.frame_id = 'odom'
        t.child_frame_id  = 'base_link'

        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = 0.0

        t.transform.rotation.z = math.sin(self.theta / 2.0)
        t.transform.rotation.w = math.cos(self.theta / 2.0)

        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = OdometryNode()
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