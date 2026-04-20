import rclpy
from rclpy.node import Node
import math
import numpy as np
import cv2 

from sensor_msgs.msg import LaserScan, Image, CompressedImage
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge  
from ultralytics import YOLO  # <--- IMPORTANTE: Asegúrate de haber instalado ultralytics

class PersonFollower(Node):

    def __init__(self):
        super().__init__('person_follower')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.bridge = CvBridge()
        self.latest_depth_img = None
        
        # 1. Cargar el modelo YOLO (n es el más ligero para robots)
        # Se descargará automáticamente la primera vez que lo corras
        self.model = YOLO('yolov8n.pt') 
        self.person_class_id = 0 # ID 0 en COCO es 'person'

        qos_policy = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT, 
            history=rclpy.qos.HistoryPolicy.KEEP_LAST, 
            depth=1
        )

        self.color_sub = self.create_subscription(
            CompressedImage,
            '/astra/color/image_raw/compressed', 
            self.color_callback, 
            qos_profile=qos_policy)

        self.depth_sub = self.create_subscription(
            CompressedImage, 
            '/astra/depth/image_raw/compressedDepth', 
            self.depth_callback, 
            qos_profile=qos_policy)
        
        self.get_logger().info("Nodo de seguimiento con YOLO iniciado")

    def color_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if cv_image is None or self.latest_depth_img is None:
            return

        cv_image = cv2.resize(cv_image, (320, 240))
        results = self.model.predict(source=cv_image, classes=[0], verbose=False, imgsz=320)

        vx = 0.0
        wz = 0.0
        
        if len(results[0].boxes) > 0:
            box = results[0].boxes[0]
            xywh = box.xywh[0]
            x_center = int(xywh[0].item())
            y_center = int(xywh[1].item())

            # --- CÁLCULO DE DISTANCIA ---
            # 1. Obtenemos el valor de profundidad en el centro de la persona
            # Ojo: la matriz de profundidad se accede como [fila, columna] -> [y, x]
            distancia_mm = self.latest_depth_img[y_center, x_center]

            # 2. Convertimos a metros (la Astra suele dar mm)
            distancia_m = distancia_mm / 1000.0

            if distancia_m > 0: # 0 suele significar "fuera de rango" o error
                self.get_logger().info(f"Persona a {distancia_m:.2f} metros")
                
                # --- LÓGICA DE MOVIMIENTO ---
                # Giro (Angular)
                img_width = cv_image.shape[1]
                center_error = (img_width / 2) - x_center
                wz = 0.005 * center_error
                
                # Avance (Lineal)
                target_dist = 1.0  # Queremos estar a 1 metro
                dist_error = distancia_m - target_dist
                vx = 0.5 * dist_error # Ganancia de velocidad
                
                # Limitar velocidad por seguridad
                vx = max(min(vx, 0.4), -0.4) 
            
            # Dibujar distancia en la pantalla
            cv2.putText(cv_image, f"{distancia_m:.2f}m", (x_center, y_center), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv_image = results[0].plot()

        # Publicar y mostrar
        move_msg = Twist()
        move_msg.linear.x = vx
        move_msg.angular.z = wz
        self.publisher_.publish(move_msg)

        cv2.imshow("Deteccion y Seguimiento", cv_image)
        cv2.waitKey(1)

    def depth_callback(self, msg):
        try:
            depth_header_size = 12
            raw_data = msg.data[depth_header_size:]
            np_arr = np.frombuffer(raw_data, np.uint8)
            # Decodificamos en UNCHANGED para mantener los valores reales (mm)
            depth_img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

            if depth_img is not None:
                self.latest_depth_img = depth_img  # <--- Guardamos la referencia
                
                # (Opcional) El resto de tu código de visualización...
        except Exception as e:
            self.get_logger().error(f"Error en profundidad: {e}")

def main(args=None):
    rclpy.init(args=args)
    person_follower = PersonFollower()
    rclpy.spin(person_follower)
    person_follower.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
