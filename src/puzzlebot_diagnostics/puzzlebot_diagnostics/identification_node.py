#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════
# identification_node.py
# Nodo de identificación de planta para sintonización PID con MATLAB.
#
# Mueve el robot con escalones de omega predefinidos mientras el
# vision_node graba el error lateral. Con esos datos, MATLAB puede
# identificar la función de transferencia omega → error_lateral.
#
# IMPORTANTE: Correr SOLO en una recta larga (~1m).
#             NO correr junto con line_controller_node.
#
# Tópicos:
#   Suscribe:  /perception/line_error  (Float32MultiArray)
#   Publica:   /cmd_vel                (Twist)
#   Guarda:    ~/puzzlebot_ws/logs/identification.csv
# ═══════════════════════════════════════════════════════════════════

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray
import csv
import os


class IdentificationNode(Node):

    def __init__(self):
        super().__init__('identification_node')

        # ── Publisher y subscriber ────────────────────────────────
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.create_subscription(
            Float32MultiArray, '/perception/line_error',
            self.error_callback, 10
        )

        # ── Estado interno ────────────────────────────────────────
        self.error_main = 0.0
        self.error_look = 0.0
        self.log        = []
        self.start_time = self.get_clock().now()

        # ── Secuencia de escalones ────────────────────────────────
        # Diseñada para 1 metro de recta a v=0.08 m/s (~12 segundos)
        # Formato: (duración_segundos, omega_rad_s)
        # Escalones pequeños para mantener la línea en el ROI
        self.sequence = [
            (2.0,  0.0 ),   # quieto al inicio — verificar error≈0
            (1.0,  0.3 ),   # escalón positivo suave
            (1.0, -0.3 ),   # escalón negativo suave
            (1.0,  0.5 ),   # escalón positivo medio
            (1.0, -0.5 ),   # escalón negativo medio
            (1.0,  0.3 ),   # escalón positivo suave de nuevo
            (1.0, -0.3 ),   # escalón negativo suave de nuevo
            (2.0,  0.0 ),   # quieto al final
        ]

        self.seq_idx   = 0
        self.seq_time  = 0.0
        self.finished  = False

        # ── Timer a 20 Hz ─────────────────────────────────────────
        self.create_timer(0.05, self.timer_cb)

        self.get_logger().info(
            'IdentificationNode iniciado\n'
            'ADVERTENCIA: Colocar robot al inicio de una recta de ~1m\n'
            'El robot avanzará automáticamente en 2 segundos'
        )

    def error_callback(self, msg):
        if len(msg.data) >= 2:
            self.error_main = float(msg.data[0])
            self.error_look = float(msg.data[1])

    def timer_cb(self):
        if self.finished:
            return

        now     = self.get_clock().now()
        elapsed = (now - self.start_time).nanoseconds / 1e9

        # Secuencia terminada
        if self.seq_idx >= len(self.sequence):
            self.get_logger().info(
                f'Secuencia completa — {len(self.log)} muestras grabadas'
            )
            self._save_csv()
            self._publish_stop()
            self.finished = True
            return

        # Ejecutar escalón actual
        duration, omega = self.sequence[self.seq_idx]
        self.seq_time  += 0.05

        if self.seq_time >= duration:
            self.seq_idx  += 1
            self.seq_time  = 0.0
            if self.seq_idx < len(self.sequence):
                next_dur, next_w = self.sequence[self.seq_idx]
                self.get_logger().info(
                    f'Escalon {self.seq_idx}/{len(self.sequence)} '
                    f'→ omega={next_w:.1f} rad/s por {next_dur:.1f}s'
                )

        # Publicar comando
        cmd = Twist()
        cmd.linear.x  = 0.08   # velocidad fija baja — no cambia durante el experimento
        cmd.angular.z = float(omega)
        self.cmd_pub.publish(cmd)

        # Grabar muestra
        self.log.append({
            't':          round(elapsed, 4),
            'omega_cmd':  round(omega, 4),
            'error_main': round(self.error_main, 6),
            'error_look': round(self.error_look, 6),
        })

    def _publish_stop(self):
        self.cmd_pub.publish(Twist())

    def _save_csv(self):
        # Crear directorio si no existe
        log_dir = os.path.expanduser('~/puzzlebot_ws/logs')
        os.makedirs(log_dir, exist_ok=True)

        path = os.path.join(log_dir, 'identification.csv')

        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(
                f, fieldnames=['t', 'omega_cmd', 'error_main', 'error_look']
            )
            writer.writeheader()
            writer.writerows(self.log)

        self.get_logger().info(f'CSV guardado en: {path}')
        self.get_logger().info(
            'Copiar a laptop con:\n'
            'scp puzzlebot@172.20.10.2:~/puzzlebot_ws/logs/identification.csv ~/Desktop/'
        )

    def destroy_node(self):
        self._publish_stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = IdentificationNode()
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