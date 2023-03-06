#pragma once
#ifndef _UTILITY_LIDAR_ODOMETRY_H_
#define _UTILITY_LIDAR_ODOMETRY_H_

#include <iostream>
#include <rclcpp/rclcpp.hpp>

#include <std_msgs/msg/header.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <opencv2/opencv.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h> 
#include <pcl_conversions/pcl_conversions.h>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
 
#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <deque>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cfloat>
#include <iterator>
#include <sstream>
#include <string>
#include <limits>
#include <iomanip>
#include <array>
#include <thread>
#include <mutex>

using namespace std;

typedef pcl::PointXYZI PointType;

enum class SensorType { VELODYNE, OUSTER };

class ParamServer : public rclcpp::Node
{
public:
    std::string robot_id;

    // Topics
    string pointCloudTopic;  // 原始点云数据话题（/points_raw）
    string imuTopic;         // 原始IMU数据话题（/imu_correct）
    string imuOdomTopic;     // IMU里程计，在imuPreintegration中对IMU做预积分得到（/lio_sam/imu/odometry）
    string lidarOdomTopic;   // 雷达里程计，在mapOptimization中得到（/lio_sam/mapping/odometry）
    string gpsTopic;         // 原始gps经过robot_localization包计算得到，暂未使用

    // Services
    string saveMapSrv;      // 保存地图service地址

    // Frames
    string imuFrame;        // IMU数据坐标系，如果IMU和激光雷达坐标系硬件对齐，可以认为IMU、Lidar、Chassis坐标系相同
    string lidarFrame;      // 激光雷达坐标系，点云数据坐标系，由激光雷达发布的数据指定。与lidarFrame相同，但是不同雷达有不同的名称
    string baseLinkFrame;   // 车辆底盘坐标系
    string odomFrame;       // 地图坐标系，在SLAM中一般也是世界坐标系，通常是车辆的起始坐标系

    // GPS Settings
    bool useImuHeadingInitialization;
    bool useGpsElevation;
    float gpsCovThreshold;
    float poseCovThreshold;

    // Save pcd
    bool savePCD;
    string savePCDDirectory;

    // Lidar Sensor Configuration
    SensorType sensor;
    int N_SCAN;
    int Horizon_SCAN;
    int downsampleRate;
    float lidarMinRange;
    float lidarMaxRange;

    // IMU
    float imuAccNoise;      // IMU加速度噪声协方差，可以用Allen方差标定；这里三个轴设为相同的方差
    float imuGyrNoise;      // IMU角速度噪声协方差，可以用Allen方差标定；这里三个轴设为相同的方差
    float imuAccBiasN;      // IMU加速度偏差，三轴统一
    float imuGyrBiasN;      // IMU角速度偏差，三轴统一
    float imuGravity;       // 重力加速度值
    float imuRPYWeight;     // 算法中使用IMU的roll、pitch角对激光里程计的结果加权融合
    vector<double> extRotV;         // IMU加速度向量到雷达坐标系的旋转
    vector<double> extRPYV;         // IMU角速度向量到雷达坐标系的旋转
    vector<double> extTransV;       // IMU向量到雷达坐标系的平移：P_{lidar} = T * P_{imu}
    Eigen::Matrix3d extRot;         // IMU加速度向量到雷达坐标系的旋转
    Eigen::Matrix3d extRPY;         // IMU角速度向量到雷达坐标系的旋转
    Eigen::Vector3d extTrans;       // IMU向量到雷达坐标系的平移：P_{lidar} = T * P_{imu}
    Eigen::Quaterniond extQRPY;     // IMU角速度向量到雷达坐标系的旋转（四元数形式）

    // LOAM
    float edgeThreshold;            // 边缘特征点提取阈值
    float surfThreshold;            // 平面特征点提取阈值
    int edgeFeatureMinValidNum;     // 边缘特征点数量阈值（default:10)
    int surfFeatureMinValidNum;     // 平面特征点数量阈值（default:100)

    // voxel filter paprams
    float odometrySurfLeafSize;
    float mappingCornerLeafSize;
    float mappingSurfLeafSize ;

    float z_tollerance;             // 限制z轴平移的大小
    float rotation_tollerance;      // 限制roll、pitch角的大小

    // CPU Params
    int numberOfCores;              // 在点云匹配中使用指令集并行加速（default:4）
    double mappingProcessInterval;  // 点云帧处理时间间隔（default:0.15s）

    // Surrounding map
    float surroundingkeyframeAddingDistThreshold;   // 当前帧需要与上一帧距离大于1米或者角度大于0.2度才有可能采纳为关键帧
    float surroundingkeyframeAddingAngleThreshold;  // 当前帧需要与上一帧距离大于1米或者角度大于0.2度才有可能采纳为关键帧
    float surroundingKeyframeDensity;               // 构建局部地图时对采用的关键帧数量做降采样
    float surroundingKeyframeSearchRadius;          // 构建局部地图时关键帧的检索半径
    
    // Loop closure
    bool  loopClosureEnableFlag;
    float loopClosureFrequency;                     // 回环检测独立线程的执行频率
    int   surroundingKeyframeSize;                  // 回环检测构建局部地图的最大关键帧数量
    float historyKeyframeSearchRadius;              // 执行回环检测时关键帧的检索半径
    float historyKeyframeSearchTimeDiff;            // 执行回环检测时关键帧的检索时间范围
    int   historyKeyframeSearchNum;                 // 执行回环检测时融合局部地图时对目标关键帧执行+-25帧的关键帧融合
    float historyKeyframeFitnessScore;              // 执行回环检测时使用ICP做点云匹配，阈值大于0.3认为匹配失败，不采纳当前回环检测

    // global map visualization radius
    float globalMapVisualizationSearchRadius;
    float globalMapVisualizationPoseDensity;
    float globalMapVisualizationLeafSize;

    ParamServer(std::string node_name, const rclcpp::NodeOptions & options) : Node(node_name, options)
    {
        declare_parameter("pointCloudTopic", "/points");
        get_parameter("pointCloudTopic", pointCloudTopic);
        declare_parameter("imuTopic", "/imu_correct");
        get_parameter("imuTopic", imuTopic);
        declare_parameter("imuOdomTopic", "/lio_sam/imu/odometry");
        get_parameter("imuOdomTopic", imuOdomTopic);
        declare_parameter("lidarOdomTopic", "/lio_sam/mapping/odometry");
        get_parameter("lidarOdomTopic", lidarOdomTopic);
        declare_parameter("gpsTopic", "lio_sam/odometry/gps");
        get_parameter("gpsTopic", gpsTopic);

        declare_parameter("saveMapSrv", "/lio_sam/save_map");
        get_parameter("saveMapSrv", saveMapSrv);

        declare_parameter("imuFrame", "imu_link");
        get_parameter("imuFrame", imuFrame);
        declare_parameter("lidarFrame", "lidar_link");
        get_parameter("lidarFrame", lidarFrame);
        declare_parameter("baseLinkFrame", "base_link");
        get_parameter("baseLinkFrame", baseLinkFrame);
        declare_parameter("odomFrame", "odom");
        get_parameter("odomFrame", odomFrame);

        declare_parameter("useImuHeadingInitialization", false);
        get_parameter("useImuHeadingInitialization", useImuHeadingInitialization);
        declare_parameter("useGpsElevation", false);
        get_parameter("useGpsElevation", useGpsElevation);
        declare_parameter("gpsCovThreshold", 2.0);
        get_parameter("gpsCovThreshold", gpsCovThreshold);
        declare_parameter("poseCovThreshold", 25.0);
        get_parameter("poseCovThreshold", poseCovThreshold);
        
        declare_parameter("savePCD", false);
        get_parameter("savePCD", savePCD);
        declare_parameter("savePCDDirectory", "/Downloads/LOAM/");
        get_parameter("savePCDDirectory", savePCDDirectory);

        declare_parameter("N_SCAN", 64);
        get_parameter("N_SCAN", N_SCAN);
        declare_parameter("Horizon_SCAN", 512);
        get_parameter("Horizon_SCAN", Horizon_SCAN);
        declare_parameter("downsampleRate", 1);
        get_parameter("downsampleRate", downsampleRate);
        declare_parameter("lidarMinRange", 5.5);
        get_parameter("lidarMinRange", lidarMinRange);
        declare_parameter("lidarMaxRange", 1000.0);
        get_parameter("lidarMaxRange", lidarMaxRange);

        declare_parameter("imuAccNoise", 9e-4);
        get_parameter("imuAccNoise", imuAccNoise);
        declare_parameter("imuGyrNoise", 1.6e-4);
        get_parameter("imuGyrNoise", imuGyrNoise);
        declare_parameter("imuAccBiasN", 5e-4);
        get_parameter("imuAccBiasN", imuAccBiasN);
        declare_parameter("imuGyrBiasN", 7e-5);
        get_parameter("imuGyrBiasN", imuGyrBiasN);
        declare_parameter("imuGravity", 9.80511);
        get_parameter("imuGravity", imuGravity);
        declare_parameter("imuRPYWeight", 0.01);
        get_parameter("imuRPYWeight", imuRPYWeight);

        double ida[] = { 1.0,  0.0,  0.0,
                         0.0,  1.0,  0.0,
                         0.0,  0.0,  1.0};
        std::vector < double > id(ida, std::end(ida));
        declare_parameter("extrinsicRot", id);
        get_parameter("extrinsicRot", extRotV);
        declare_parameter("extrinsicRPY", id);
        get_parameter("extrinsicRPY", extRPYV);
        double zea[] = {0.0, 0.0, 0.0};
        std::vector < double > ze(zea, std::end(zea));
        declare_parameter("extrinsicTrans", ze);
        get_parameter("extrinsicTrans", extTransV);

        /*
        使用Eigen::Map,重用extRotV,extRPYV,extTransV向量。等号左边又变成Eigen::Matrix3d格式
        */
        extRot = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRotV.data(), 3, 3);
        extRPY = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRPYV.data(), 3, 3);
        extTrans = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extTransV.data(), 3, 1);
        extQRPY = Eigen::Quaterniond(extRPY);

        declare_parameter("edgeThreshold", 1.0);
        get_parameter("edgeThreshold", edgeThreshold);
        declare_parameter("surfThreshold", 0.1);
        get_parameter("surfThreshold", surfThreshold);
        declare_parameter("edgeFeatureMinValidNum", 10);
        get_parameter("edgeFeatureMinValidNum", edgeFeatureMinValidNum);
        declare_parameter("surfFeatureMinValidNum", 100);
        get_parameter("surfFeatureMinValidNum", surfFeatureMinValidNum);

        declare_parameter("odometrySurfLeafSize", 0.4);
        get_parameter("odometrySurfLeafSize", odometrySurfLeafSize);
        declare_parameter("mappingCornerLeafSize", 0.2);
        get_parameter("mappingCornerLeafSize", mappingCornerLeafSize);
        declare_parameter("mappingSurfLeafSize", 0.4);
        get_parameter("mappingSurfLeafSize", mappingSurfLeafSize);

        declare_parameter("z_tollerance", 1000.0);
        get_parameter("z_tollerance", z_tollerance);
        declare_parameter("rotation_tollerance", 1000.0);
        get_parameter("rotation_tollerance", rotation_tollerance);
        
        declare_parameter("numberOfCores", 4);
        get_parameter("numberOfCores", numberOfCores);
        declare_parameter("mappingProcessInterval", 0.15);
        get_parameter("mappingProcessInterval", mappingProcessInterval);

        declare_parameter("surroundingkeyframeAddingDistThreshold", 1.0);
        get_parameter("surroundingkeyframeAddingDistThreshold", surroundingkeyframeAddingDistThreshold);
        declare_parameter("surroundingkeyframeAddingAngleThreshold", 0.2);
        get_parameter("surroundingkeyframeAddingAngleThreshold", surroundingkeyframeAddingAngleThreshold);
        declare_parameter("surroundingKeyframeDensity", 2.0);
        get_parameter("surroundingKeyframeDensity", surroundingKeyframeDensity);
        declare_parameter("surroundingKeyframeSearchRadius", 50.0);
        get_parameter("surroundingKeyframeSearchRadius", surroundingKeyframeSearchRadius);

        declare_parameter("loopClosureEnableFlag", true);
        get_parameter("loopClosureEnableFlag", loopClosureEnableFlag);
        declare_parameter("loopClosureFrequency", 1.0);
        get_parameter("loopClosureFrequency", loopClosureFrequency);
        declare_parameter("surroundingKeyframeSize", 50);
        get_parameter("surroundingKeyframeSize", surroundingKeyframeSize);
        declare_parameter("historyKeyframeSearchRadius", 15.0);
        get_parameter("historyKeyframeSearchRadius", historyKeyframeSearchRadius);
        declare_parameter("historyKeyframeSearchTimeDiff", 30.0);
        get_parameter("historyKeyframeSearchTimeDiff", historyKeyframeSearchTimeDiff);
        declare_parameter("historyKeyframeSearchNum", 25);
        get_parameter("historyKeyframeSearchNum", historyKeyframeSearchNum);
        declare_parameter("historyKeyframeFitnessScore", 0.3);
        get_parameter("historyKeyframeFitnessScore", historyKeyframeFitnessScore);

        declare_parameter("globalMapVisualizationSearchRadius", 1000.0);
        get_parameter("globalMapVisualizationSearchRadius", globalMapVisualizationSearchRadius);
        declare_parameter("globalMapVisualizationPoseDensity", 10.0);
        get_parameter("globalMapVisualizationPoseDensity", globalMapVisualizationPoseDensity);
        declare_parameter("globalMapVisualizationLeafSize", 1.0);
        get_parameter("globalMapVisualizationLeafSize", globalMapVisualizationLeafSize);

        declare_parameter("sensor", "");
        std::string sensorStr;
        get_parameter("sensor", sensorStr);
        if (sensorStr == "velodyne")
        {
            sensor = SensorType::VELODYNE;
        }
        else if (sensorStr == "ouster")
        {
            sensor = SensorType::OUSTER;
        }
        else
        {
            RCLCPP_ERROR(get_logger(), "Invalid sensor type (must be either 'velodyne' or 'ouster'): %s", sensorStr.c_str());
            rclcpp::shutdown();
        }

        usleep(100);
    }

    /*
    将原始IMU数据：三轴加速度、三轴角速度、三轴角度，与雷达坐标系进行旋转对齐
    + 对齐之后输出的加速度、角速度、角度的x，y，z就变成雷达坐标系的x，y，z
    + 这里的特殊之处在于允许IMU的加速度、角速度与角度的输出是两个不同的坐标系。但在算法中，角度的输出除了用来做第一帧的初始化和加权融合，似乎没有其他作用
    + 这里是将IMU的三个轴与雷达的三个轴在旋转上做对齐，不能加上平移
    + 对向量做坐标系变换，对多个变换的复合应该是右乘
    */
    sensor_msgs::msg::Imu imuConverter(const sensor_msgs::msg::Imu& imu_in)
    {
        sensor_msgs::msg::Imu imu_out = imu_in;
        /*
        对加速度向量做坐标系变换，注意这里要理解成坐标系变换，也就是同一个加速度在IMU坐标系和Lidar坐标系的不同表达。不能想象成对加速度做旋转
        */
        Eigen::Vector3d acc(imu_in.linear_acceleration.x, imu_in.linear_acceleration.y, imu_in.linear_acceleration.z);
        acc = extRot * acc;
        imu_out.linear_acceleration.x = acc.x();
        imu_out.linear_acceleration.y = acc.y();
        imu_out.linear_acceleration.z = acc.z();

        /*
        对角速度做坐标系变换。将IMU坐标系下的向量变换到雷达坐标系。
        */
        Eigen::Vector3d gyr(imu_in.angular_velocity.x, imu_in.angular_velocity.y, imu_in.angular_velocity.z);
        gyr = extRot * gyr;
        imu_out.angular_velocity.x = gyr.x();
        imu_out.angular_velocity.y = gyr.y();
        imu_out.angular_velocity.z = gyr.z();

        /*
        对角度做坐标系变换。
        + q_from是IMU在全局坐标系下的位姿，q_from: transformation_from_map_to_imu
        + extQRPY如果与extRot对应的话应该是lidar到imu的变换：transformation_from_lidar_to_imu
        + q_final是将雷达点云从雷达坐标系转换到map坐标系的变换，也是：transformation_from_map_to_lidar -> pcd_in_map = q_final * pcd_in_lidar
        + 这里原代码是q_final = q_from * extQRPY；似乎有点问题，还是按照我的推导修改成q_final = q_from * extQRPT.inverse()；由于这里的extQRPY是
        + 直接从配置文件里面读取的，所以这里加不加逆只需要在配置文件里改就行。认为这里有问题的假设是认为extQRPY和extRot的坐标系关系的定义是一致的，也就是
        + 将imu坐标系下的向量转换到雷达坐标系下。如果作者对这两者的定义刚好是相反的，那这里就没有问题。
        */
        Eigen::Quaterniond q_from(imu_in.orientation.w, imu_in.orientation.x, imu_in.orientation.y, imu_in.orientation.z);
        Eigen::Quaterniond q_final = q_from * extQRPY.inverse();
        imu_out.orientation.x = q_final.x();
        imu_out.orientation.y = q_final.y();
        imu_out.orientation.z = q_final.z();
        imu_out.orientation.w = q_final.w();

        if (sqrt(q_final.x()*q_final.x() + q_final.y()*q_final.y() + q_final.z()*q_final.z() + q_final.w()*q_final.w()) < 0.1)
        {
            RCLCPP_ERROR(get_logger(), "Invalid quaternion, please use a 9-axis IMU!");
            rclcpp::shutdown();
        }

        return imu_out;
    }
};

/// @brief 发布点云话题的工具函数
/// @param thisPub 点云topic的发布者
/// @param thisCloud pcl格式的点云
/// @param thisStamp ros时间戳
/// @param thisFrame 点云的坐标系
/// @return 将pcl点云转成ros格式，并加上时间戳和frame_id
sensor_msgs::msg::PointCloud2 publishCloud(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr thisPub, pcl::PointCloud<PointType>::Ptr thisCloud, rclcpp::Time thisStamp, std::string thisFrame)
{
    sensor_msgs::msg::PointCloud2 tempCloud;
    pcl::toROSMsg(*thisCloud, tempCloud);
    tempCloud.header.stamp = thisStamp;
    tempCloud.header.frame_id = thisFrame;
    if (thisPub->get_subscription_count() != 0)
        thisPub->publish(tempCloud);
    return tempCloud;
}

/// @brief 将PCL点云转为ROS2 PointCloud2 数据格式
/// @param inCloud 
/// @param outCloud 
/// @param stamp 
/// @param frameId 
void pclPointcloud2Ros(pcl::PointCloud<PointType>::Ptr inCloud, sensor_msgs::msg::PointCloud2& outCloud, rclcpp::Time stamp, std::string frameId)
{
    pcl::toROSMsg(*inCloud, outCloud);
    outCloud.header.stamp = stamp;
    outCloud.header.frame_id = frameId;
}

/// @brief 将ROS2 msg中header的时间戳转换成double的表示（秒）；原代码这里是一个模板函数，没有必要
/// @param stamp 消息头中的时间戳
/// @return 秒为单位的时间戳
double stamp2Sec(const builtin_interfaces::msg::Time& stamp)
{
    return rclcpp::Time(stamp).seconds();
}

/// @brief 提取IMU消息角速度
/// @tparam T 返回的角度格式
/// @param thisImuMsg 
/// @param angular_x 
/// @param angular_y 
/// @param angular_z 
template<typename T>
void imuAngular2rosAngular(sensor_msgs::msg::Imu *thisImuMsg, T *angular_x, T *angular_y, T *angular_z)
{
    *angular_x = thisImuMsg->angular_velocity.x;
    *angular_y = thisImuMsg->angular_velocity.y;
    *angular_z = thisImuMsg->angular_velocity.z;
}

/// @brief 提取IMU消息的加速度
/// @tparam T 
/// @param thisImuMsg 
/// @param acc_x 
/// @param acc_y 
/// @param acc_z 
template<typename T>
void imuAccel2rosAccel(sensor_msgs::msg::Imu *thisImuMsg, T *acc_x, T *acc_y, T *acc_z)
{
    *acc_x = thisImuMsg->linear_acceleration.x;
    *acc_y = thisImuMsg->linear_acceleration.y;
    *acc_z = thisImuMsg->linear_acceleration.z;
}

/// @brief 提取IMU消息的角度roll、pitch、yaw
/// @tparam T 
/// @param thisImuMsg 
/// @param rosRoll 
/// @param rosPitch 
/// @param rosYaw 
template<typename T>
void imuRPY2rosRPY(sensor_msgs::msg::Imu *thisImuMsg, T *rosRoll, T *rosPitch, T *rosYaw)
{
    double imuRoll, imuPitch, imuYaw;
    tf2::Quaternion orientation;
    tf2::fromMsg(thisImuMsg->orientation, orientation);
    tf2::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);

    *rosRoll = imuRoll;
    *rosPitch = imuPitch;
    *rosYaw = imuYaw;
}


float pointDistance(PointType p)
{
    return sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
}


float pointDistance(PointType p1, PointType p2)
{
    return sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y) + (p1.z-p2.z)*(p1.z-p2.z));
}

/// @brief 算法框架中默认的QOS，主要是depth=1和reliability=best_effort起作用。对于传输实时性有要求，不要求每个数据可接收的消息，一般
/// 设成best_effort。在ROS2中对于传感器数据，有一个内置的QOS叫rclcpp::SensorDataQoS()
rmw_qos_profile_t qos_profile{
  RMW_QOS_POLICY_HISTORY_KEEP_LAST,
  1,
  RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
  RMW_QOS_POLICY_DURABILITY_VOLATILE,
  RMW_QOS_DEADLINE_DEFAULT,
  RMW_QOS_LIFESPAN_DEFAULT,
  RMW_QOS_POLICY_LIVELINESS_SYSTEM_DEFAULT,
  RMW_QOS_LIVELINESS_LEASE_DURATION_DEFAULT,
  false
};

auto qos = rclcpp::QoS(
    rclcpp::QoSInitialization(
      qos_profile.history,
      qos_profile.depth
    ),
    qos_profile);

/// @brief 原始IMU数据的QOS，因为IMU数据较小，所以depth可以设成较大
rmw_qos_profile_t qos_profile_imu{
  RMW_QOS_POLICY_HISTORY_KEEP_LAST,
  2000,
  RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
  RMW_QOS_POLICY_DURABILITY_VOLATILE,
  RMW_QOS_DEADLINE_DEFAULT,
  RMW_QOS_LIFESPAN_DEFAULT,
  RMW_QOS_POLICY_LIVELINESS_SYSTEM_DEFAULT,
  RMW_QOS_LIVELINESS_LEASE_DURATION_DEFAULT,
  false
};
auto qos_imu = rclcpp::QoS(
    rclcpp::QoSInitialization(
      qos_profile_imu.history,
      qos_profile_imu.depth
    ),
    qos_profile_imu);

/// @brief 原始雷达数据topic的QOS，主要是best_effort和depth起作用
rmw_qos_profile_t qos_profile_lidar{
  RMW_QOS_POLICY_HISTORY_KEEP_LAST,
  5,
  RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
  RMW_QOS_POLICY_DURABILITY_VOLATILE,
  RMW_QOS_DEADLINE_DEFAULT,
  RMW_QOS_LIFESPAN_DEFAULT,
  RMW_QOS_POLICY_LIVELINESS_SYSTEM_DEFAULT,
  RMW_QOS_LIVELINESS_LEASE_DURATION_DEFAULT,
  false
};
auto qos_lidar = rclcpp::QoS(
    rclcpp::QoSInitialization(
      qos_profile_lidar.history,
      qos_profile_lidar.depth
    ),
    qos_profile_lidar);

#endif
