import rclpy
from rclpy.node import Node
import numpy as np
import cv2
import time

from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from ultralytics import YOLO


class DistanceMeasurer(Node):

    def __init__(self):
        super().__init__('distance_measurer')
        self.bridge = CvBridge()
        self.latest_depth_img = None
        self.depth_ready = False          # <-- nueva bandera
        self.color_ready = False          # <-- nueva bandera

        # Configuración de rendimiento
        self.frame_count = 0
        self.skip_frames = 3              # procesa 1 de cada 4
        self.yolo_imgsz = 160
        self.show_image = True
        self.use_gpu = False

        # Parámetros de control
        self.target_distance = 0.8
        self.k_linear = 0.4
        self.k_angular = 2.5
        self.camera_hfov = 1.047

        # Variables de estado (sin lock)
        self.has_target = False
        self.target_id = -1
        self.target_distance_m = 0.0
        self.target_center_x = 0

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

        # Publicador de comandos de velocidad
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Temporizador para el control (10 Hz)
        self.control_timer = self.create_timer(0.1, self.control_callback)

        self.get_logger().info("Nodo de medición con control por timer iniciado (sin locks)")

        if self.show_image:
            cv2.namedWindow("Person Follower - Solo cámara", cv2.WINDOW_NORMAL)

    def depth_callback(self, msg):
        start_time = time.time()
        try:
            depth_header_size = 12
            raw_data = msg.data[depth_header_size:]
            np_arr = np.frombuffer(raw_data, np.uint8)
            depth_img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
            if depth_img is not None:
                self.latest_depth_img = depth_img.copy()
                self.depth_ready = True          # ya tenemos profundidad
        except Exception as e:
            self.get_logger().error(f"Error en profundidad: {e}")
        elapsed_ms = (time.time() - start_time) * 1000.0
        if self.frame_count % 30 == 0:           # log cada 30 frames
            self.get_logger().info(f"[TIMING] depth_callback tardó {elapsed_ms:.2f} ms")

    def color_callback(self, msg):
        start_time = time.time()
        self.frame_count += 1
        # Saltamos frames según skip_frames
        if self.frame_count % (self.skip_frames + 1) != 0:
            return

        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if cv_image is None or self.latest_depth_img is None:
            # No hay datos suficientes, deshabilitar control
            self.has_target = False
            self.color_ready = True   # al menos sabemos que no hay persona
            return

        cv_image = cv2.resize(cv_image, (self.img_width, self.img_height))

        results = self.model.track(cv_image, persist=True, verbose=False,
                                   imgsz=self.yolo_imgsz, conf=0.5)

        track_ids = []
        if results[0].boxes is not None and results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()

        keypoints_data = results[0].keypoints.xy.cpu().numpy() if results[0].keypoints is not None else []
        keypoints_conf_data = results[0].keypoints.conf.cpu().numpy() if results[0].keypoints is not None else []

        personas = []

        if len(keypoints_data) > 0:
            # Redimensionar profundidad una sola vez
            depth_resized = cv2.resize(self.latest_depth_img,
                                       (self.img_width, self.img_height),
                                       interpolation=cv2.INTER_NEAREST)

            for person_idx, (kpts, kpts_conf) in enumerate(zip(keypoints_data, keypoints_conf_data)):
                person_track_id = track_ids[person_idx] if person_idx < len(track_ids) else -1
                if person_track_id == -1:
                    continue

                conf_mask = kpts_conf > 0.5
                if not np.any(conf_mask):
                    continue

                valid_kpts = kpts[conf_mask]
                cx = int(np.mean(valid_kpts[:, 0]))
                cy = int(np.mean(valid_kpts[:, 1]))

                # Calcular distancia desde profundidad
                profundidades = []
                for (x, y) in valid_kpts:
                    xi = int(round(x))
                    yi = int(round(y))
                    if 0 <= xi < self.img_width and 0 <= yi < self.img_height:
                        d = depth_resized[yi, xi]
                        if d > 0:
                            profundidades.append(d)
                if len(profundidades) > 0:
                    distancia_m = np.mean(profundidades) / 1000.0
                else:
                    continue

                personas.append({
                    'id': person_track_id,
                    'distancia': distancia_m,
                    'centro_x': cx,
                    'centro_y': cy,
                    'keypoints': [(int(x), int(y)) for (x, y) in valid_kpts]
                })

        # --- Seleccionar la persona más cercana ---
        selected = None
        if personas:
            personas.sort(key=lambda p: p['distancia'])
            selected = personas[0]

        # Actualizar variables de estado
        if selected is not None:
            self.has_target = True
            self.target_id = selected['id']
            self.target_distance_m = selected['distancia']
            self.target_center_x = selected['centro_x']
        else:
            self.has_target = False

        self.color_ready = True   # ya tenemos un estado actualizado

        # --- Dibujo (opcional) ---
        frame_draw = results[0].plot() if results[0].keypoints is not None else cv_image
        if selected is not None:
            dist = selected['distancia']
            pid = selected['id']
            cx = selected['centro_x']
            cy = selected['centro_y']
            self.get_logger().info(f"[ID:{pid}] Distancia media: {dist:.2f} m")
            cv2.circle(frame_draw, (cx, cy), 6, (0, 0, 255), -1)
            cv2.putText(frame_draw, f"Target ID {pid}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            for (x, y) in selected['keypoints']:
                cv2.circle(frame_draw, (x, y), 3, (0, 255, 255), -1)
        else:
            cv2.putText(frame_draw, "No persons detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        if self.show_image:
            cv2.imshow("Person Follower - Solo cámara", frame_draw)
            cv2.waitKey(1)

        elapsed_ms = (time.time() - start_time) * 1000.0
        if self.frame_count % 30 == 0:
            self.get_logger().info(f"[TIMING] color_callback tardó {elapsed_ms:.1f} ms")

    # ------------------------------------------------------------------
    # Callback del temporizador (control)
    # ------------------------------------------------------------------
    def control_callback(self):
        # Esperar a que ambos sensores hayan entregado al menos un dato
        if not (self.depth_ready and self.color_ready):
            self.publicar_velocidad(0.0, 0.0)
            return

        if not self.has_target:
            self.publicar_velocidad(0.0, 0.0)
            return

        dist = self.target_distance_m
        cx = self.target_center_x

        # Calcular errores
        error_distancia = dist - self.target_distance
        error_angular = (cx - self.img_width/2) * (self.camera_hfov / self.img_width)

        # Control proporcional
        vx = self.k_linear * error_distancia
        wz = -self.k_angular * error_angular

        # Límites de seguridad
        vx = max(min(vx, 0.4), -0.2)
        wz = max(min(wz, 2.5), -2.5)

        self.publicar_velocidad(vx, wz)

        # Log opcional (cada 50 ciclos)
        
        self.get_logger().debug(f"Control: dist={dist:.2f} err_d={error_distancia:.2f} ang_err={error_angular:.2f} -> vx={vx:.2f} wz={wz:.2f}")

    def publicar_velocidad(self, vx, wz):
        twist = Twist()
        twist.linear.x = vx
        twist.angular.z = wz
        self.cmd_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = DistanceMeasurer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
