import rclpy
from rclpy.node import Node
import numpy as np
import cv2
import time

from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from ultralytics import YOLO


class DistanceMeasurer(Node):

    def __init__(self):
        super().__init__('distance_measurer')
        self.bridge = CvBridge()
        self.latest_depth_img = None

        # --- Configuración de rendimiento ---
        self.frame_count = 0
        self.skip_frames = 3
        self.yolo_imgsz = 160
        self.show_image = True
        self.use_gpu = False

        device = 'cuda' if self.use_gpu else 'cpu'
        try:
            self.model = YOLO('yolov8n-pose.pt')
            self.model.to(device)
        except Exception as e:
            self.get_logger().warn(f"No se pudo usar {device}, usando CPU: {e}")
            self.model = YOLO('yolov8n-pose.pt')

        self.img_width = 320
        self.img_height = 240

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

        self.get_logger().info("Nodo de medición (ID más bajo) iniciado")

        if self.show_image:
            cv2.namedWindow("Distancia persona (Astra)", cv2.WINDOW_NORMAL)

    def depth_callback(self, msg):
        try:
            depth_header_size = 12
            raw_data = msg.data[depth_header_size:]
            np_arr = np.frombuffer(raw_data, np.uint8)
            depth_img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
            if depth_img is not None:
                self.latest_depth_img = depth_img
        except Exception as e:
            self.get_logger().error(f"Error en profundidad: {e}")

    def color_callback(self, msg):
        self.frame_count += 1
        if self.frame_count % (self.skip_frames + 1) != 0:
            return

        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if cv_image is None or self.latest_depth_img is None:
            return

        cv_image = cv2.resize(cv_image, (self.img_width, self.img_height))

        # Inferencia de pose
        start_time = time.time()
        results = self.model.predict(cv_image,
                                     verbose=False,
                                     imgsz=self.yolo_imgsz,
                                     conf=0.5)
        inference_time = time.time() - start_time
        self.get_logger().debug(f"Inferencia en {inference_time:.3f}s")

        # Lista para almacenar info de cada persona: (id, distancia, keypoints_xy)
        personas = []

        if results[0].keypoints is not None and len(results[0].keypoints) > 0:
            # results[0].keypoints es un tensor con shape (N_personas, N_keypoints, 2/3)
            for person_id, (kpts, kpts_conf) in enumerate(zip(results[0].keypoints.xy,
                                                             results[0].keypoints.conf)):
                kpts = kpts.cpu().numpy()        # (N_kpts, 2)
                kpts_conf = kpts_conf.cpu().numpy()  # (N_kpts,)

                conf_mask = kpts_conf > 0.5
                if not np.any(conf_mask):
                    continue   # ninguna articulación fiable

                valid_kpts = kpts[conf_mask]
                # Redimensionar profundidad (si no se ha hecho ya para esta frame)
                depth_resized = cv2.resize(self.latest_depth_img,
                                           (self.img_width, self.img_height),
                                           interpolation=cv2.INTER_NEAREST)

                profundidades = []
                puntos_visibles = []

                for (x, y) in valid_kpts:
                    xi = int(round(x))
                    yi = int(round(y))
                    if 0 <= xi < self.img_width and 0 <= yi < self.img_height:
                        d = depth_resized[yi, xi]
                        if d > 0:
                            profundidades.append(d)
                            puntos_visibles.append((xi, yi))

                if len(profundidades) > 0:
                    distancia_media = np.mean(profundidades) / 1000.0
                    personas.append({
                        'id': person_id,
                        'distancia': distancia_media,
                        'keypoints': puntos_visibles
                    })

        # Seleccionar solo la persona con el ID más bajo
        persona_seleccionada = None
        if personas:
            persona_seleccionada = personas[0]   # ya está ordenado por ID (0,1,2...)

        # Dibujar todas las poses y resaltar la seleccionada
        frame_draw = results[0].plot() if results[0].keypoints is not None else cv_image

        if persona_seleccionada is not None:
            dist = persona_seleccionada['distancia']
            kps = persona_seleccionada['keypoints']
            pid = persona_seleccionada['id']

            self.get_logger().info(f"[ID:{pid}] Distancia media: {dist:.2f} m")

            # Resaltar keypoints de esta persona (amarillo)
            for (x, y) in kps:
                cv2.circle(frame_draw, (x, y), 5, (0, 255, 255), -1)

            # Texto con ID y distancia en la esquina superior izquierda
            cv2.putText(frame_draw, f"ID {pid}: {dist:.2f}m",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        if self.show_image:
            cv2.imshow("Distancia persona (Astra)", frame_draw)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = DistanceMeasurer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
