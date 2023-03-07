cmake_minimum_required(VERSION 3.5)
project(spl_lio_sam)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb -Wno-unused-parameter")  

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
find_package(GTSAM REQUIRED)
find_package(Eigen REQUIRED)
find_package(pcl_msgs REQUIRED)
find_package(OpenMP REQUIRED)


include_directories(
  include
)

rosidl_generate_interfaces(${PROJECT_NAME} "msg/CloudInfo.msg" "srv/SaveMap.srv" DEPENDENCIES std_msgs sensor_msgs)
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")

add_executable(${PROJECT_NAME}_imageProjection src/imageProjection.cpp)
ament_target_dependencies(${PROJECT_NAME}_imageProjection rclcpp rclpy std_msgs sensor_msgs geometry_msgs nav_msgs pcl_msgs visualization_msgs pcl_msgs tf2 tf2_ros tf2_eigen tf2_sensor_msgs tf2_geometry_msgs OpenCV PCL)
target_link_libraries(${PROJECT_NAME}_imageProjection ${cpp_typesupport_target})

add_executable(${PROJECT_NAME}_featureExtraction src/featureExtraction.cpp)
ament_target_dependencies(${PROJECT_NAME}_featureExtraction rclcpp rclpy std_msgs sensor_msgs geometry_msgs nav_msgs pcl_msgs visualization_msgs pcl_msgs tf2 tf2_ros tf2_eigen tf2_sensor_msgs tf2_geometry_msgs OpenCV PCL)
target_link_libraries(${PROJECT_NAME}_featureExtraction ${cpp_typesupport_target})

add_executable(${PROJECT_NAME}_imuPreintegration src/imuPreintegration.cpp)
ament_target_dependencies(${PROJECT_NAME}_imuPreintegration rclcpp rclpy std_msgs sensor_msgs geometry_msgs nav_msgs pcl_msgs visualization_msgs pcl_msgs tf2 tf2_ros tf2_eigen tf2_sensor_msgs tf2_geometry_msgs OpenCV PCL GTSAM Eigen pcl_msgs)
target_link_libraries(${PROJECT_NAME}_imuPreintegration gtsam ${cpp_typesupport_target})

add_executable(${PROJECT_NAME}_mapOptimization src/mapOptimization.cpp)
ament_target_dependencies(${PROJECT_NAME}_mapOptimization rclcpp rclpy std_msgs sensor_msgs geometry_msgs nav_msgs pcl_msgs visualization_msgs pcl_msgs tf2 tf2_ros tf2_eigen tf2_sensor_msgs tf2_geometry_msgs OpenCV PCL GTSAM Eigen pcl_msgs)
target_link_libraries(${PROJECT_NAME}_mapOptimization gtsam ${cpp_typesupport_target} OpenMP::OpenMP_CXX)

add_executable(${PROJECT_NAME}_transformFusion src/transformFusion.cpp)
ament_target_dependencies(${PROJECT_NAME}_transformFusion rclcpp rclpy std_msgs sensor_msgs geometry_msgs nav_msgs pcl_msgs visualization_msgs pcl_msgs tf2 tf2_ros tf2_eigen tf2_sensor_msgs tf2_geometry_msgs OpenCV PCL)
target_link_libraries(${PROJECT_NAME}_transformFusion ${cpp_typesupport_target})


# add_executable(${PROJECT_NAME}_imuPreintegration src/imuPreintegration.cpp)
# ament_target_dependencies(${PROJECT_NAME}_imuPreintegration rclcpp rclpy std_msgs sensor_msgs geometry_msgs nav_msgs visualization_msgs tf2 tf2_ros tf2_eigen tf2_sensor_msgs tf2_geometry_msgs OpenCV PCL GTSAM Eigen pcl_msgs)
# target_link_libraries(${PROJECT_NAME}_imuPreintegration gtsam)
# rosidl_target_interfaces(${PROJECT_NAME}_imuPreintegration ${PROJECT_NAME} "rosidl_typesupport_cpp")

# add_executable(${PROJECT_NAME}_mapOptimization src/mapOptmization.cpp)
# ament_target_dependencies(${PROJECT_NAME}_mapOptimization rclcpp rclpy std_msgs sensor_msgs geometry_msgs nav_msgs visualization_msgs tf2 tf2_ros tf2_eigen tf2_sensor_msgs tf2_geometry_msgs OpenCV PCL GTSAM pcl_msgs)
# target_link_libraries(${PROJECT_NAME}_mapOptimization gtsam)
# rosidl_target_interfaces(${PROJECT_NAME}_mapOptimization ${PROJECT_NAME} "rosidl_typesupport_cpp")

# add_executable(${PROJECT_NAME}_transformFusion src/transformFusion.cpp)
# ament_target_dependencies(${PROJECT_NAME}_transformFusion rclcpp rclpy std_msgs sensor_msgs geometry_msgs nav_msgs visualization_msgs tf2 tf2_ros tf2_eigen tf2_sensor_msgs tf2_geometry_msgs OpenCV PCL Eigen pcl_msgs)
# rosidl_target_interfaces(${PROJECT_NAME}_transformFusion ${PROJECT_NAME} "rosidl_typesupport_cpp")




install(
  TARGETS 
  ${PROJECT_NAME}_imageProjection
  ${PROJECT_NAME}_featureExtraction
  ${PROJECT_NAME}_imuPreintegration
  ${PROJECT_NAME}_mapOptimization
  ${PROJECT_NAME}_transformFusion
  DESTINATION lib/${PROJECT_NAME}
)


install(
  DIRECTORY "include/"
  DESTINATION include/${PROJECT_NAME}
)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

ament_export_include_directories(include)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

