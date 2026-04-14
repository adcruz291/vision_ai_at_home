import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2


def frame_to_imgmsg(frame, stamp):
    """Convierte un frame OpenCV (BGR) a sensor_msgs/Image sin usar cv_bridge."""
    msg = Image()
    msg.header.stamp = stamp
    msg.height       = frame.shape[0]
    msg.width        = frame.shape[1]
    msg.encoding     = 'bgr8'
    msg.is_bigendian = 0
    msg.step         = frame.shape[1] * frame.shape[2]
    msg.data         = frame.tobytes()
    return msg


class CameraPublisherNode(Node):
    def __init__(self):
        super().__init__('camera_publisher_node')

        self.declare_parameter('camera_index', 0)
        self.declare_parameter('fps', 30)
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)

        cam_idx = self.get_parameter('camera_index').value
        fps     = self.get_parameter('fps').value
        width   = self.get_parameter('width').value
        height  = self.get_parameter('height').value

        self.publisher = self.create_publisher(Image, '/camera/image_raw', 10)

        self.cap = cv2.VideoCapture(cam_idx)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self.cap.isOpened():
            self.get_logger().error(f'No se pudo abrir la camara indice {cam_idx}')
            return

        self.timer = self.create_timer(1.0 / fps, self.publish_frame)
        self.get_logger().info(
            f'CameraPublisherNode iniciado — camara {cam_idx} @ {fps}fps '
            f'({width}x{height}) → /camera/image_raw'
        )

    def publish_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('No se pudo leer frame de la camara')
            return

        msg = frame_to_imgmsg(frame, self.get_clock().now().to_msg())
        self.publisher.publish(msg)

    def destroy_node(self):
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
