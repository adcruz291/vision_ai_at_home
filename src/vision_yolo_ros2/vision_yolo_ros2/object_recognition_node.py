import json
from enum import Enum

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image
from ultralytics import YOLO


def imgmsg_to_frame(msg: Image) -> np.ndarray:
    """Convierte sensor_msgs/Image a numpy array (BGR) sin usar cv_bridge."""
    frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
    return frame.copy()


def asignar_casilla(x_center: float, frame_width: int, num_casillas: int) -> int:
    """
    Divide el frame en num_casillas zonas iguales y devuelve
    en cual zona cae x_center (0 = izquierda, num_casillas-1 = derecha).
    """
    zona = int(x_center / frame_width * num_casillas)
    return min(zona, num_casillas - 1)  # clamp por si x_center == frame_width


class State(Enum):
    IDLE               = 'IDLE'
    WAITING_NAVIGATION = 'WAITING_NAVIGATION'
    DETECTING          = 'DETECTING'


class ObjectRecognitionNode(Node):
    def __init__(self):
        super().__init__('object_recognition_node')

        # --- Parámetros ---
        self.declare_parameter('model_path',   'yolov8n.pt')
        self.declare_parameter('confidence',   0.5)
        self.declare_parameter('num_casillas', 3)

        model_path          = self.get_parameter('model_path').value
        self.confidence     = self.get_parameter('confidence').value
        self.num_casillas   = self.get_parameter('num_casillas').value

        # --- Cargar modelo YOLO una sola vez ---
        self.get_logger().info(f'Cargando modelo YOLO: {model_path}')
        self.model = YOLO(model_path)
        self.get_logger().info('Modelo YOLO cargado correctamente')
        self.get_logger().info(f'Casillas configuradas: {self.num_casillas}')

        # --- Estado interno ---
        self.state         = State.IDLE
        self.target_object = None
        self.latest_frame  = None
        self.frame_width   = None   # se detecta del primer frame recibido

        # --- Subscripciones ---
        self.create_subscription(
            String, '/target_object',    self.cb_target_object, 10)
        self.create_subscription(
            Bool,   '/got_target',       self.cb_got_target,    10)
        self.create_subscription(
            Image,  '/camera/image_raw', self.cb_camera,        10)

        # --- Publicador ---
        # JSON: {
        #   "target_object": "key",
        #   "slots": ["telefono", "", "key"],   <- índice = casilla absoluta
        #   "target_index": 2,                  <- casilla donde está el target (-1 si no se encontró)
        #   "confidence_scores": {"0": 0.92, "2": 0.87}
        # }
        self.pub = self.create_publisher(String, '/detected_objects', 10)

        self.get_logger().info('ObjectRecognitionNode listo — Estado: IDLE')

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def cb_camera(self, msg: Image):
        """Actualiza silenciosamente el frame mas reciente."""
        self.latest_frame = imgmsg_to_frame(msg)
        if self.frame_width is None:
            self.frame_width = msg.width
            self.get_logger().info(
                f'Frame detectado: {msg.width}x{msg.height} — '
                f'cada casilla abarca {msg.width // self.num_casillas}px'
            )

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
            return

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
        if self.latest_frame is None or self.frame_width is None:
            self.get_logger().error(
                'No hay frame de camara disponible. Regresando a IDLE.'
            )
            self._reset()
            return

        results = self.model(self.latest_frame, conf=self.confidence, verbose=False)

        # Construir array de casillas fijas (vacías por defecto)
        slots       = [''] * self.num_casillas   # casilla vacía = ""
        confs       = [0.0] * self.num_casillas
        target_index = -1

        for box in results[0].boxes:
            x1, _, x2, _ = box.xyxy[0].tolist()
            class_id   = int(box.cls[0])
            class_name = self.model.names[class_id]
            conf       = float(box.conf[0])
            x_center   = (x1 + x2) / 2.0

            casilla = asignar_casilla(x_center, self.frame_width, self.num_casillas)

            # Si dos objetos caen en la misma casilla, gana el de mayor confianza
            if slots[casilla] == '' or conf > confs[casilla]:
                slots[casilla] = class_name
                confs[casilla] = conf

        # Buscar casilla del target
        for i, nombre in enumerate(slots):
            if nombre.lower() == self.target_object:
                target_index = i
                break

        # Log resultado — mostrar todas las casillas
        self.get_logger().info(
            f'--- Deteccion completa ({self.num_casillas} casillas) ---'
        )
        for i in range(self.num_casillas):
            if slots[i]:
                marker = ' ← TARGET' if i == target_index else ''
                self.get_logger().info(
                    f'  Casilla {i}: "{slots[i]}" (conf: {confs[i]:.2f}){marker}'
                )
            else:
                self.get_logger().info(f'  Casilla {i}: (vacia)')

        if target_index >= 0:
            self.get_logger().info(
                f'"{self.target_object}" encontrado en casilla {target_index}'
            )
        else:
            self.get_logger().warn(
                f'"{self.target_object}" NO encontrado en ninguna casilla'
            )

        # Publicar resultado
        payload = {
            'target_object': self.target_object,
            'slots':         slots,         # lista de num_casillas, "" = vacia
            'target_index':  target_index,  # -1 si no se encontró
            'confidence_scores': confs,
        }
        out = String()
        out.data = json.dumps(payload)
        self.pub.publish(out)

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
