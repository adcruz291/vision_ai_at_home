cmake_minimum_required(VERSION 3.8)
project(vision_yolo_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/DetectedObjects.msg"
  DEPENDENCIES builtin_interfaces
)

ament_package()
