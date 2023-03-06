# Cloud Info
std_msgs/Header header 


# 在进行点云去畸变时，把range数据展开成一维向量
# ring代表第几条激光线数，比如16线的激光雷达有16个ring
# start_ring_index记录第一个ring在一维向量中的起始索引
# end_ring_index记录第一个ring在一维向量中的结束索引
int32[] start_ring_index
int32[] end_ring_index

# 记录一维的Range数据中每一个点在原始range图片中属于那一个列
int32[]  point_col_ind # point column index in range image
float32[] point_range # point range 

int64 imu_available
int64 odom_available

# Attitude for LOAM initialization
float32 imu_roll_init
float32 imu_pitch_init
float32 imu_yaw_init

# Initial guess from imu pre-integration
float32 initial_guess_x
float32 initial_guess_y
float32 initial_guess_z
float32 initial_guess_roll
float32 initial_guess_pitch
float32 initial_guess_yaw

# Point cloud messages
sensor_msgs/PointCloud2 cloud_deskewed  # original cloud deskewed
sensor_msgs/PointCloud2 cloud_corner    # extracted corner feature
sensor_msgs/PointCloud2 cloud_surface   # extracted surface feature
