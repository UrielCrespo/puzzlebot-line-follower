import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO
import json
import cv2

class DetectorNode(Node):
    def __init__(self):
        super().__init__('detector_senales')
        
        self.model = YOLO('/home/skqchs/Documents/Sexto/Challenge7/Debloated_model/my_model.pt')
        self.bridge = CvBridge()
        # self.frame_actual = None
        
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            10
        )
        
        self.publisher = self.create_publisher(String, '/detecciones', 10)
        self.sign_publisher = self.create_publisher(String, '/traffic_sign_state', 10)
        
        # Timer para la visualización — refresca la ventana 30 veces por segundo
        # self.timer = self.create_timer(1/30, self.visualizar)
        
        self.get_logger().info('Nodo detector iniciado correctamente')

    def image_callback(self, msg):
        # Convierte la imagen de mensaje de ROS -> OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # Inferencia del modelo
        results = self.model(frame, verbose=False)
        
        detecciones = []
        for r in results:
            for box in r.boxes:
                confianza = round(float(box.conf), 2)

                if confianza < 0.70:
                    continue
                clase = self.model.names[int(box.cls)]
                confianza = round(float(box.conf), 2)
                
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                
                detecciones.append({
                    'clase': clase,
                    'confianza': confianza,
                    'bbox': [x1, y1, x2, y2]
                })
                
                '''
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                etiqueta = f'{clase} {confianza}'
                cv2.putText(frame, etiqueta, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                '''
        
        # self.frame_actual = frame
        # Publicar la señal de mayor confianza (si hay alguna)
        if detecciones:
            mejor = max(detecciones, key=lambda d: d['confianza'])
            self.sign_publisher.publish(String(data=mejor['clase']))
        else:
            self.sign_publisher.publish(String(data='NONE'))

        self.publisher.publish(String(data=json.dumps(detecciones)))
        
        if detecciones:
            self.get_logger().info(f'Detectado: {[d["clase"] for d in detecciones]}')
    '''
    def visualizar(self):
        if self.frame_actual is not None:
            cv2.imshow('Detector de Senales', self.frame_actual)
            cv2.waitKey(1)

    '''
def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode()
    rclpy.spin(node)
    cv2.destroyAllWindows()
    rclpy.shutdown()
