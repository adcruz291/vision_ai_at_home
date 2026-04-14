import json
from enum import Enum

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO


class State(Enum):
    IDLE               = 'IDLE'
    WAITING_NAVIGATION = 'WAITING_NAVIGATION'
    DETECTING          = 'DETECTING'


class ObjectRecognitionNode(Node):
    def __init__(self):
        super().__init__('object_recognition_node')

        # --- Parámetros ---
        self.declare_parameter('model_path', 'yolov8n.pt')
        self.declare_parameter('confidence', 0.5)

        model_path       = self.get_parameter('model_path').value
        self.confidence  = self.get_parameter('confidence').value

        # --- Cargar modelo YOLO una sola vez ---
        self.get_logger().info(f'Cargando modelo YOLO: {model_path}')
        self.model = YOLO(model_path)
        self.get_logger().info('Modelo YOLO cargado correctamente')

        # --- Estado interno ---
        self.state          = State.IDLE
        self.target_object  = None   # objeto pedido por speech
        self.latest_frame   = None   # frame mas reciente de la camara
        self.bridge         = CvBridge()

        # --- Subscripciones ---
        self.create_subscription(
            String, '/target_object', self.cb_target_object, 10)
        self.create_subscription(
            Bool,   '/got_target',    self.cb_got_target,    10)
        self.create_subscription(
            Image,  '/camera/image_raw', self.cb_camera,     10)

        # --- Publicador ---
        # Publica JSON: {"target_object": "key", "objects": ["telefono","key"], "target_index": 1, "confidence_scores": [0.9, 0.87]}
        self.pub = self.create_publisher(String, '/detected_objects', 10)

        self.get_logger().info('ObjectRecognitionNode listo — Estado: IDLE')

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def cb_camera(self, msg: Image):
        """Actualiza silenciosamente el frame mas reciente."""
        self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def cb_target_object(self, msg: String):
        """Recibe el objeto a buscar desde speech_one."""
        if self.state != State.IDLE:
            self.get_logger().warn(
                f'/target_object recibido en estado {self.state.value} — ignorando. '
                f'(El sistema ya esta procesando "{self.target_object}")'
            )
            return

        self.target_object = msg.data.strip().lower()
        self.state = State.WAITING_NAVIGATION
        self.get_logger().info(
            f'Objeto solicitado: "{self.target_object}" — '
            f'Esperando confirmacion de navegacion (/got_target)...'
        )

    def cb_got_target(self, msg: Bool):
        """Recibe confirmacion del sistema de navegacion."""
        if not msg.data:
            return  # ignorar False

        if self.state != State.WAITING_NAVIGATION:
            self.get_logger().warn(
                f'/got_target=True recibido en estado {self.state.value} — ignorando.'
            )
            return

        self.get_logger().info(
            'Confirmacion de navegacion recibida — Ejecutando deteccion YOLO...'
        )
        self.state = State.DETECTING
        self._run_detection()

    # ------------------------------------------------------------------
    # Detección y publicación
    # ------------------------------------------------------------------

    def _run_detection(self):
        if self.latest_frame is None:
            self.get_logger().error(
                'No hay frame de camara disponible. Regresando a IDLE.'
            )
            self._reset()
            return

        frame = self.latest_frame.copy()
        results = self.model(frame, conf=self.confidence, verbose=False)

        # Extraer detecciones y ordenar de izquierda a derecha por x_center
        detections = []
        for box in results[0].boxes:
            x1, _, x2, _ = box.xyxy[0].tolist()
            class_id   = int(box.cls[0])
            class_name = self.model.names[class_id]
            conf       = float(box.conf[0])
            x_center   = (x1 + x2) / 2.0
            detections.append((x_center, class_name, conf))

        detections.sort(key=lambda d: d[0])  # izquierda → derecha

        objects     = [d[1] for d in detections]
        confidences = [d[2] for d in detections]

        # Buscar el índice del objeto solicitado
        target_index = -1
        for i, name in enumerate(objects):
            if name.lower() == self.target_object:
                target_index = i
                break

        # Log resultado
        if objects:
            self.get_logger().info(
                f'Objetos detectados (izq→der): {objects}'
            )
            if target_index >= 0:
                self.get_logger().info(
                    f'"{self.target_object}" encontrado en casilla {target_index}'
                )
            else:
                self.get_logger().warn(
                    f'"{self.target_object}" NO fue encontrado entre los objetos detectados'
                )
        else:
            self.get_logger().warn('No se detecto ningun objeto en el frame')

        # Publicar resultado como JSON en std_msgs/String
        payload = {
            'target_object':     self.target_object,
            'objects':           objects,           # ordenados izq→der
            'target_index':      target_index,      # -1 si no se encontró
            'confidence_scores': confidences,
        }
        msg = String()
        msg.data = json.dumps(payload)
        self.pub.publish(msg)

        self._reset()

    def _reset(self):
        self.state         = State.IDLE
        self.target_object = None
        self.get_logger().info('Estado: IDLE — Listo para nuevo ciclo.')


def main(args=None):
    rclpy.init(args=args)
    node = ObjectRecognitionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
