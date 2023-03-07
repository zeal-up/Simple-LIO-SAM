/****************************************************************************
Project: 简化版LIO-SAM
Github: https://github.com/zeal-github/Simple-LIO-SAM
Author: zeal
EMail: 1156478780@qq.com
Date: 2023-02-14
-------------------------------------------------------
*transformFusion*
(此部分原先与imuPreintegration.cpp在同一个文件，与imuPreintegration共享一个进程启动。但是这部分的功能较为独立，为了方便理解，将这一部分
代码单独作为一个节点事项)

功能：
+ transformFusion对于算法的运行没有任何实际意义，完全是作为rviz可视化的一些中间话题发布者
+ 订阅雷达里程计、IMU里程计，以IMU里程计频率发布TF坐标系关系
+ 在两帧雷达里程计之间发布IMU里程计的信息作为轨迹显示

订阅：
1. IMU里程计
2. 雷达里程计

发布：
1. /tf，TF2的坐标系关系话题

流程：
1. 监听雷达里程计数据，实时更新最新的雷达里程计时间戳
2. 监听IMU里程计数据，将监听到的IMU里程计转换成TF2数据并发布
3. 发布从最新缓存的雷达里程计到现在为止的IMU里程计轨迹
*******************************************************************************/
#include "spl_lio_sam/utility.hpp"
#include <tf2/transform_datatypes.h>
#include <stdexcept>


class TransformFusion : public ParamServer
{
public:
    std::mutex mtx;

    // 订阅IMU里程计
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subImuOdometry;
    rclcpp::CallbackGroup::SharedPtr callbackGroupImuOdometry;
    // 订阅雷达里程计
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subLaserOdometry;  
    rclcpp::CallbackGroup::SharedPtr callbackGroupLaserOdometry;

    // 发布两帧雷达里程计之间的IMU轨迹
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubImuPath;

    // 时时更新的雷达里程计变量
    Eigen::Isometry3d lidarOdomAffine;

    // tf2的相关组建
    std::shared_ptr<tf2_ros::Buffer> tfBuffer;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tfBroadcaster;
    std::shared_ptr<tf2_ros::TransformListener> tfListener;

    double lidarOdomTime = -1;

    TransformFusion(const rclcpp::NodeOptions & options) : ParamServer("lio_sam_transformFusion", options)
    {
        tfBuffer = std::make_shared<tf2_ros::Buffer>(get_clock());
        tfListener = std::make_shared<tf2_ros::TransformListener>(*tfBuffer);

        // 订阅低频的雷达里程计信息
        callbackGroupImuOdometry = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        auto imuOdomOpt = rclcpp::SubscriptionOptions();
        imuOdomOpt.callback_group = callbackGroupImuOdometry;
        subImuOdometry = create_subscription<nav_msgs::msg::Odometry>(
            imuOdomTopic, qos_imu,
            std::bind(&TransformFusion::imuOdometryHandler, this, std::placeholders::_1),
            imuOdomOpt);
        
        // 订阅高频的IMU里程计信息
        callbackGroupLaserOdometry = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        auto laserOdomOpt = rclcpp::SubscriptionOptions();
        laserOdomOpt.callback_group = callbackGroupLaserOdometry;
        subLaserOdometry = create_subscription<nav_msgs::msg::Odometry>(
            lidarOdomTopic, qos,
            std::bind(&TransformFusion::lidarOdometryHandler, this, std::placeholders::_1),
            laserOdomOpt);

        // 发布两帧雷达里程计之间的IMU里程计信息
        pubImuPath = create_publisher<nav_msgs::msg::Path>("lio_sam/imu/path", qos);

        // TF2 坐标关系发布器
        tfBroadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(this);
    }

    Eigen::Isometry3d odom2affine(nav_msgs::msg::Odometry odom)
    {
        tf2::Transform t;
        tf2::fromMsg(odom.pose.pose, t);
        return tf2::transformToEigen(tf2::toMsg(t));
    }

    /**
     * @brief 雷达里程计回调函数。实际只是监听并实时更新时间戳
    */
    void lidarOdometryHandler(const nav_msgs::msg::Odometry::SharedPtr odomMsg)
    {
        std::lock_guard<std::mutex> lock(mtx);

        lidarOdomAffine = odom2affine(*odomMsg);

        lidarOdomTime = stamp2Sec(odomMsg->header.stamp);
    }

    /**
     * @brief IMU里程计回调函数。
     * 1. 将监听到的IMU里程计转换成TF2数据并发布
     * 2. 发布从最新缓存的雷达里程计到现在为止的IMU里程计轨迹
    */
    void imuOdometryHandler(const nav_msgs::msg::Odometry::SharedPtr odomMsg)
    {
        std::lock_guard<std::mutex> lock(mtx);

        // 获取
        Eigen::Isometry3d imuOdomAffineBack = odom2affine(*odomMsg);
        geometry_msgs::msg::TransformStamped tf_odom = tf2::eigenToTransform(imuOdomAffineBack);
        tf_odom.header = odomMsg->header;   // 设置消息头，时间戳
        tf_odom.header.frame_id = odomFrame;    // 设置父坐标系是地图坐标系
        tf_odom.child_frame_id = baseLinkFrame;     // 设置子坐标系是底盘坐标系
        tfBroadcaster->sendTransform(tf_odom);      // 发布tf

        // 如果雷达点云的坐标系字段不与baseLinkFrame字段相同，发布他们二者之间的关系
        // 这里主要是用来实时可视化点云用
        // 整个框架认为雷达坐标系与底盘坐标系等同，因此这里直接发布的是单位矩阵
        if (lidarFrame != baseLinkFrame)
        {
            geometry_msgs::msg::TransformStamped tf_baseLink2Lidar;
            tf2::convert(tf2::Transform::getIdentity(), tf_baseLink2Lidar.transform);
            tf_baseLink2Lidar.header = odomMsg->header;
            tf_baseLink2Lidar.header.frame_id = baseLinkFrame;
            tf_baseLink2Lidar.child_frame_id = lidarFrame;
            tfBroadcaster->sendTransform(tf_baseLink2Lidar);
        }


        // 发布IMU轨迹
        static nav_msgs::msg::Path imuPath;
        static double last_path_time = -1;
        double imuTime = stamp2Sec(odomMsg->header.stamp);
        if (imuTime - last_path_time > 0.1)
        {
            last_path_time = imuTime;
            geometry_msgs::msg::PoseStamped pose_stamped;
            pose_stamped.header.stamp = odomMsg->header.stamp;
            pose_stamped.header.frame_id = odomFrame;
            pose_stamped.pose = odomMsg->pose.pose;
            imuPath.poses.push_back(pose_stamped);
            while(!imuPath.poses.empty() && stamp2Sec(imuPath.poses.front().header.stamp) < lidarOdomTime - 1.0)
                imuPath.poses.erase(imuPath.poses.begin());
            if (pubImuPath->get_subscription_count() != 0)
            {
                imuPath.header.stamp = odomMsg->header.stamp;
                imuPath.header.frame_id = odomFrame;
                pubImuPath->publish(imuPath);
            }
        }
    }
};


int main(int argc, char** argv)
{   
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);
    rclcpp::executors::MultiThreadedExecutor e;

    auto TF = std::make_shared<TransformFusion>(options);
    e.add_node(TF);

    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;32m----> TransformFusion Started.\033[0m");

    e.spin();

    rclcpp::shutdown();
    return 0;
}