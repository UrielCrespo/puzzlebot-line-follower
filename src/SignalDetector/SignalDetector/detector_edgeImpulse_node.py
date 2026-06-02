import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from edge_impulse_linux.image import ImageImpulseRunner
import json
import cv2

class DetectorEINode(Node):
    def __init__(self):
        super().__init__('detector_ei')
        
        self.bridge = CvBridge()
        self.frame_actual = None
        self.runner = None
        
        # Carga el modelo de Edge Impulse
        model_path = '/home/skqchs/Documents/Sexto/Challenge7/EdgeImpulse_model/trafficsigns-linux-x86_64-v1-impulse-#1.eim'
        self.runner = ImageImpulseRunner(model_path)
        model_info = self.runner.init()
        self.get_logger().info(f'Modelo EI cargado: {model_info["project"]["name"]}')
        
        self.labels = model_info['model_parameters']['labels']
        self.get_logger().info(f'Clases: {self.labels}')
        
        # Suscripción a la cámara
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            10
        )
        
        # Publicación de resultados
        self.publisher = self.create_publisher(String, '/detecciones_ei', 10)
        
        # Timer para visualización
        self.timer = self.create_timer(1/30, self.visualizar)
        
        self.get_logger().info('Nodo Edge Impulse iniciado correctamente')

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # Edge Impulse espera RGB, no BGR
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Correr inferencia
        features, cropped = self.runner.get_features_from_image(frame_rgb)
        result = self.runner.classify(features)
        
        detecciones = []
        if 'bounding_boxes' in result['result']:
            for bb in result['result']['bounding_boxes']:
                if bb['value'] > 0.5:  # umbral de confianza
                    detecciones.append({
                        'clase': bb['label'],
                        'confianza': round(bb['value'], 2),
                        'bbox': [bb['x'], bb['y'], 
                                bb['x'] + bb['width'], 
                                bb['y'] + bb['height']]
                    })
                    
                    # Dibujar bounding box
                    cv2.rectangle(frame,
                                (bb['x'], bb['y']),
                                (bb['x'] + bb['width'], bb['y'] + bb['height']),
                                (255, 0, 0), 2)
                    etiqueta = f"{bb['label']} {round(bb['value'], 2)}"
                    cv2.putText(frame, etiqueta,
                               (bb['x'], bb['y'] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        self.frame_actual = frame
        self.publisher.publish(String(data=json.dumps(detecciones)))
        
        if detecciones:
            self.get_logger().info(f'EI Detectado: {[d["clase"] for d in detecciones]}')

    def visualizar(self):
        if self.frame_actual is not None:
            cv2.imshow('Detector EI', self.frame_actual)
            cv2.waitKey(1)

    def destroy_node(self):
        if self.runner:
            self.runner.stop()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = DetectorEINode()
    rclpy.spin(node)
    cv2.destroyAllWindows()
    rclpy.shutdown()
