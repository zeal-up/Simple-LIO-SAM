/****************************************************************************
Project: 简化版LIO-SAM
Github: https://github.com/zeal-github/Simple-LIO-SAM
Author: zeal
EMail: 1156478780@qq.com
Date: 2023-02-14
-------------------------------------------------------
*mapOptimization*

功能：
+ 从特征提取模块获得提取特征后的点云信息，该点云信息中包括来自imageProjection模块提取出的该帧
点云的初始估计位姿，主要来自imuPreintegration模块。将当前帧的点云匹配到前几个关键帧构成的局部地图，
得到经过点云匹配校正后的位姿，将这些位姿加入因子图进行优化，得到更准确的位姿，即雷达里程计
+ 在执行过程中进行回环检测，并在检测到回环时更新历史关键帧的位姿（smoothing）
+ 同时，可以选择订阅GPS里程计，将将其加入因子图，构建更准确的位姿。

订阅：
1. 点云特征信息集合，来自featureExtraction模块

发布：
1. 雷达里程计
2. 地图点云
3. 历史轨迹
4. 回环信息

流程：

*****************************************************************************/
#include "spl_lio_sam/utility.hpp"
#include "spl_lio_sam/msg/cloud_info.hpp"
#include "spl_lio_sam/srv/save_map.hpp"
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/nonlinear/ISAM2.h>

using namespace gtsam;

using symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::G; // GPS pose

/*
    * A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity is time stamp)
    */
/**
 * 6D位姿点定义。为了保存位姿，作者将位姿点（x,y,z,roll,pitch,yaw)定义为PCL的点结构，可以直接使用PCL的接口
 * 保存批量的位姿点。
 * 对于3D位姿点（x,y,z），则直接使用PointType数据结构表达。
 * 这里的intensity字段存储的是该位姿点的顺序索引。
 * 关于PCL自定义点云格式的说明可以参照imageProjection文件
*/
struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;                  // preferred way of adding a XYZ+padding
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRPYT,
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
                                   (double, time, time))

typedef PointXYZIRPYT  PointTypePose;


class mapOptimization : public ParamServer
{

public:

    // 因子图
    NonlinearFactorGraph gtSAMgraph;
    // 因子图变量初始值
    Values initialEstimate;
    // 非线性优化器
    ISAM2 *isam;
    // 优化器当前优化结果
    Values isamCurrentEstimate;
    // 当前优化结果的位姿方差。该方差在GPS因子中用到，如果该方差较小，则说明优化结果较好，即使打开GPS开关也不会将GPS因子加入因子图。
    Eigen::MatrixXd poseCovariance;

    // 发布雷达里程计
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubLaserOdometryGlobal;
    // 发布轨迹（path）
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath;
    // 发布全局地图（低速，单独线程）
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubMapCloud;
    // 发布当前帧转换到odom坐标系下的点云（高速，每次都发送——
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubRegisteredCurCloud;
    // 发布回环
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pubLoopConstraintEdge;

    // 保存地图服务接口
    rclcpp::Service<spl_lio_sam::srv::SaveMap>::SharedPtr srvSaveMap;
    // 订阅从featureExtraction模块发布出来的点云信息集合
    rclcpp::Subscription<spl_lio_sam::msg::CloudInfo>::SharedPtr subCloudInfo;
    // 订阅GPS里程计（实际是由robot_localization包计算后的GPS位姿）
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subGPS;

    // GPS信息队列
    std::deque<nav_msgs::msg::Odometry> gpsQueue;
    // 当前点云信息
    spl_lio_sam::msg::CloudInfo cloudInfo;

    // 所有关键帧的角点点云
    vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames;
    // 所有关键帧的平面点点云
    vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;
    
    /**
     * cloudKeyPoses3D保存所有关键帧的三维位姿，x,y,z
     * cloudKeyPoses6D保存所有关键帧的六维位姿，x,y,z,roll,pitch,yaw
     * 带copy_前缀的两个位姿序列是在回环检测线程中使用的，只是为了不干扰主线程计算，实际内容一样。
    */
    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;
    pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D;

    // 当前帧的角点点云
    pcl::PointCloud<PointType>::Ptr laserCloudCornerCur;
    // 当前帧的平面点点云
    pcl::PointCloud<PointType>::Ptr laserCloudSurfCur;
    // 当前帧的角点点云降采样
    pcl::PointCloud<PointType>::Ptr laserCloudCornerCurDS;
    // 当前帧的平面点点云降采样
    pcl::PointCloud<PointType>::Ptr laserCloudSurfCurDS;

    // 在做点云匹配的过程中使用的中间变量
    pcl::PointCloud<PointType>::Ptr laserCloudOri;
    pcl::PointCloud<PointType>::Ptr coeffSel;
    std::vector<PointType> laserCloudOriCornerVec; // corner point holder for parallel computation
    std::vector<PointType> coeffSelCornerVec;
    std::vector<bool> laserCloudOriCornerFlag;
    std::vector<PointType> laserCloudOriSurfVec; // surf point holder for parallel computation
    std::vector<PointType> coeffSelSurfVec;
    std::vector<bool> laserCloudOriSurfFlag;

    // 变换到odom坐标系下的关键帧点云字典，为了加速缓存历史关键帧变换后的点云
    map<int, pair<pcl::PointCloud<PointType>, pcl::PointCloud<PointType>>> laserCloudMapContainer;
    // 局部地图角点点云（odom坐标系）
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;
    // 局部地图平面点点云（odom坐标系）
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;
    // 局部地图角点点云降采样
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS;
    // 局部地图平面点点云降采样
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;

    // 在做点云匹配时构建的角点kdtree
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;
    // 在做点云匹配时构建的平面点kdtree
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;

    // 在构建局部地图时挑选的周围关键帧的三维姿态（构建kdtree加速搜索）
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;
    // 在构建局部地图时挑选的邻近时间关键帧的三维姿态（构建kdtree加速搜索）
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;

    // 角点点云降采样器
    pcl::VoxelGrid<PointType> downSizeFilterCorner;
    // 平面点点云降采样器
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    // 做回环检测时使用ICP时的点云降采样器
    pcl::VoxelGrid<PointType> downSizeFilterICP;
    // 构建局部地图时都挑选的关键帧做降采样
    pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses;

    // 当前雷达帧的时间戳
    rclcpp::Time timeLaserInfoStamp;
    // 当前雷达帧的时间戳，秒
    double timeLaserInfoCur;

    /**
     * 注意注意注意！！这是一个非常重要的变量，transformTobeMapped[6]缓存的是当前帧
     * 的`最新`位姿x,y,z,roll,pitch,yaw。无论在哪个环节，对位姿的更新都会被缓存到这个
     * 变量供给下一个环节使用！！
    */
    float transformTobeMapped[6];

    // 点云信息回调函数锁
    std::mutex mtx;
    // 回环检测线程锁
    std::mutex mtxLoopInfo;

    // 标识点云匹配的结果是否较差，当isDegenerate为true的时候，标识本次的点云匹配结果较差，
    // 会在雷达里程计的协方差中置位，在imuPreintegration中会根据这个标志位选择因子图的噪声模型
    bool isDegenerate = false;
    Eigen::Matrix<float, 6, 6> matP;

    // 降采样后局部地图角点点云数量
    int laserCloudCornerFromMapDSNum = 0;
    // 降采样后局部地图平面点云数量
    int laserCloudSurfFromMapDSNum = 0;
    // 降采样后当前帧角点点云数量
    int laserCloudCornerCurDSNum = 0;
    // 降采样后当前帧平面点云数量
    int laserCloudSurfCurDSNum = 0;

    // 当新的回环节点出现或者GPS信息被加入校正位置，这个变量被置为true，
    // 因子图优化器会执行多次更新，然后将所有的历史帧位置都更新一遍
    bool aLoopGpsIsClosed = false;
    
    // 回环的索引字典，从当前帧到回环节点的索引
    map<int, int> loopIndexContainer;
    // 所有回环配对关系
    vector<pair<int, int>> loopIndexQueue;
    // 所有回环的姿态配对关系
    vector<gtsam::Pose3> loopPoseQueue;
    // 每个回环因子的噪声模型
    vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;

    // 全局关键帧轨迹
    nav_msgs::msg::Path globalPath;

    // 缓存雷达帧位姿用来做点云变换
    Eigen::Affine3f transPointAssociateToMap;

    // incrementalOdometryAffineFront在每次点云进来时缓存上一次的位姿
    Eigen::Affine3f incrementalOdometryAffineFront;
    // incrementalOdometryAffineBack是当前帧点云优化后的最终位姿，
    // incrementalOdometryAffineBack与Front可以算出一个增量，应用到上一次的雷达里程计
    // 计算出当前的雷达里程计。这一步似乎有点多余
    Eigen::Affine3f incrementalOdometryAffineBack;

    // TF发布器
    std::unique_ptr<tf2_ros::TransformBroadcaster> br;

    mapOptimization(const rclcpp::NodeOptions & options) : ParamServer("lio_sam_mapOptimization", options)
    {
        // isam 优化器参数
        ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.1;
        parameters.relinearizeSkip = 1;
        isam = new ISAM2(parameters);

        // 发布雷达里程计
        pubLaserOdometryGlobal = create_publisher<nav_msgs::msg::Odometry>(lidarOdomTopic, qos);
        // 发布全局轨迹
        pubPath = create_publisher<nav_msgs::msg::Path>("lio_sam/mapping/path", 1);
        // 发布回环约束
        pubLoopConstraintEdge = create_publisher<visualization_msgs::msg::MarkerArray>("/lio_sam/mapping/loop_closure_constraints", 1);
        // 发布当前帧变换到odom坐标系下的点云
        pubRegisteredCurCloud = create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/mapping/registered_cur_cloud", 1);
        // 发布全局地图
        pubMapCloud = create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/mapping/map_point_cloud", 1);

        // TF2数据广播器
        br = std::make_unique<tf2_ros::TransformBroadcaster>(this);

        // 监听从特征提取模块发布出来的点云特征信息集合
        subCloudInfo = create_subscription<spl_lio_sam::msg::CloudInfo>(
            "lio_sam/feature/cloud_info", qos,
            std::bind(&mapOptimization::laserCloudInfoHandler, this, std::placeholders::_1));
        // 监听从robot_localization模块发布出来的GPS里程计，只有在打开GPS约束的时候才会用到这个信息
        subGPS = create_subscription<nav_msgs::msg::Odometry>(
            gpsTopic, 200,
            std::bind(&mapOptimization::gpsHandler, this, std::placeholders::_1));

        // 后续把这个service写为单独的函数
        auto saveMapService = [this](const std::shared_ptr<rmw_request_id_t> request_header, const std::shared_ptr<spl_lio_sam::srv::SaveMap::Request> req, std::shared_ptr<spl_lio_sam::srv::SaveMap::Response> res) -> void {
            (void)request_header;
            string saveMapDirectory;
            cout << "****************************************************" << endl;
            cout << "Saving map to pcd files ..." << endl;
            // if(req->destination.empty()) saveMapDirectory = std::getenv("HOME") + savePCDDirectory;
            // else saveMapDirectory = std::getenv("HOME") + req->destination;
            if(req->destination.empty()) saveMapDirectory = savePCDDirectory;
            else saveMapDirectory = req->destination;
            cout << "Save destination: " << saveMapDirectory << endl;
            // create directory and remove old files;
            int unused = system((std::string("exec rm -r ") + saveMapDirectory).c_str());
            unused = system((std::string("mkdir -p ") + saveMapDirectory).c_str());
            if (!unused)
            {
                RCLCPP_ERROR(get_logger(), "Remove old map and create new map directory error!");
            }
            // save key frame transformations
            pcl::io::savePCDFileBinary(saveMapDirectory + "/trajectory.pcd", *cloudKeyPoses3D);
            pcl::io::savePCDFileBinary(saveMapDirectory + "/transformations.pcd", *cloudKeyPoses6D);
            // extract global point cloud map
            pcl::PointCloud<PointType>::Ptr globalCornerCloud(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr globalCornerCloudDS(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr globalSurfCloud(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr globalSurfCloudDS(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr globalMapCloud(new pcl::PointCloud<PointType>());
            for (int i = 0; i < (int)cloudKeyPoses3D->size(); i++) 
            {
                *globalCornerCloud += *transformPointCloud(cornerCloudKeyFrames[i],  &cloudKeyPoses6D->points[i]);
                *globalSurfCloud   += *transformPointCloud(surfCloudKeyFrames[i],    &cloudKeyPoses6D->points[i]);
                cout << "\r" << std::flush << "Processing feature cloud " << i << " of " << cloudKeyPoses6D->size() << " ...";
            }
            // save key frames's corner
            for (int i = 0; i < (int)cloudKeyPoses3D->size(); i++)
            {
                pcl::io::savePCDFileBinary(saveMapDirectory + "/kf_corner_" + std::to_string(i) + ".pcd", *cornerCloudKeyFrames[i]);
                pcl::io::savePCDFileBinary(saveMapDirectory + "/kf_surf_" + std::to_string(i) + ".pcd", *surfCloudKeyFrames[i]);
                cout << "\r" << std::flush << "Saving feature cloud " << i << " of " << cloudKeyPoses6D->size() << " ...";
            }
            if(req->resolution != 0)
            {
               cout << "\n\nSave resolution: " << req->resolution << endl;
               // down-sample and save corner cloud
               downSizeFilterCorner.setInputCloud(globalCornerCloud);
               downSizeFilterCorner.setLeafSize(req->resolution, req->resolution, req->resolution);
               downSizeFilterCorner.filter(*globalCornerCloudDS);
               pcl::io::savePCDFileBinary(saveMapDirectory + "/CornerMap.pcd", *globalCornerCloudDS);
               // down-sample and save surf cloud
               downSizeFilterSurf.setInputCloud(globalSurfCloud);
               downSizeFilterSurf.setLeafSize(req->resolution, req->resolution, req->resolution);
               downSizeFilterSurf.filter(*globalSurfCloudDS);
               pcl::io::savePCDFileBinary(saveMapDirectory + "/SurfMap.pcd", *globalSurfCloudDS);
            }
            else
            {
            // save corner cloud
               pcl::io::savePCDFileBinary(saveMapDirectory + "/CornerMap.pcd", *globalCornerCloud);
               // save surf cloud
               pcl::io::savePCDFileBinary(saveMapDirectory + "/SurfMap.pcd", *globalSurfCloud);
            }
            // save global point cloud map
            *globalMapCloud += *globalCornerCloud;
            *globalMapCloud += *globalSurfCloud;
            int ret = pcl::io::savePCDFileBinary(saveMapDirectory + "/GlobalMap.pcd", *globalMapCloud);
            res->success = ret == 0;
            downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
            downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
            cout << "****************************************************" << endl;
            cout << "Saving map to pcd files completed\n" << endl;
            return;
        };
        
        // 地图保存service
        srvSaveMap = create_service<spl_lio_sam::srv::SaveMap>("lio_sam/save_map", saveMapService);

        // 当前帧角点点云降采样器
        downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
        // 当前帧平面点点云降采样器
        downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        // 回环检测时ICP匹配前点云降采样器
        downSizeFilterICP.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        // 构建局部地图时对检索出的位姿点做降采样
        downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity, surroundingKeyframeDensity); // for surrounding key poses of scan-to-map optimization

        // 为变量和动态指针分配内存
        allocateMemory();
    }

    // 为变量和动态指针分配内存
    void allocateMemory()
    {
        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
        copy_cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        copy_cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

        kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

        laserCloudCornerCur.reset(new pcl::PointCloud<PointType>()); // corner feature set from odoOptimization
        laserCloudSurfCur.reset(new pcl::PointCloud<PointType>()); // surf feature set from odoOptimization
        laserCloudCornerCurDS.reset(new pcl::PointCloud<PointType>()); // downsampled corner featuer set from odoOptimization
        laserCloudSurfCurDS.reset(new pcl::PointCloud<PointType>()); // downsampled surf featuer set from odoOptimization

        laserCloudOri.reset(new pcl::PointCloud<PointType>());
        coeffSel.reset(new pcl::PointCloud<PointType>());

        laserCloudOriCornerVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelCornerVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriCornerFlag.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelSurfVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfFlag.resize(N_SCAN * Horizon_SCAN);

        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);

        laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

        kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

        for (int i = 0; i < 6; ++i){
            transformTobeMapped[i] = 0;
        }

        matP.setZero();
    }

    // 特征点云回调函数，也是整个模块核心的计算函数
    void laserCloudInfoHandler(const spl_lio_sam::msg::CloudInfo::SharedPtr msgIn)
    {
        // 提取当前点云时间戳
        timeLaserInfoStamp = msgIn->header.stamp;
        timeLaserInfoCur = stamp2Sec(msgIn->header.stamp);

        // 提取当前点云的特征角点和特征平面点
        cloudInfo = *msgIn;
        pcl::fromROSMsg(msgIn->cloud_corner,  *laserCloudCornerCur);
        pcl::fromROSMsg(msgIn->cloud_surface, *laserCloudSurfCur);

        /**
         * 一共三个线程使用到这把锁
         * 1. 雷达里程计线程，也就是当前线程
         * 2. 发布全局地图线程，执行关键帧点云拷贝转换操作
         * 3. 回环检测线程，执行关键帧姿态拷贝操作
        */
        std::lock_guard<std::mutex> lock(mtx);

        // 记录上一帧的时间戳，两帧之间时间间隔大于mappingProcessInterval才会进行处理
        static double timeLastProcessing = -1;
        if (timeLaserInfoCur - timeLastProcessing >= mappingProcessInterval)
        {
            timeLastProcessing = timeLaserInfoCur;

            // 当前帧位姿初始化
            RCLCPP_DEBUG(get_logger(), "updateInitialGuess");
            updateInitialGuess();

            // 构建局部地图
            RCLCPP_DEBUG(get_logger(), "extractSurroundingKeyFrames");
            extractSurroundingKeyFrames();

            // 对当前帧点云做降采样
            RCLCPP_DEBUG(get_logger(), "downsampleCurrentScan");
            downsampleCurrentScan();

            // 将当前帧点云匹配到构建的局部地图，优化当前位姿
            RCLCPP_DEBUG(get_logger(), "scan2MapOptimization");
            scan2MapOptimization();

            // 计算是否将当前帧采纳为关键帧，加入因子图优化
            RCLCPP_DEBUG(get_logger(), "saveKeyFramesAndFactor");
            saveKeyFramesAndFactor();

            // 当新的回环因子或者GPS因子加入因子图时，对历史帧执行位姿更新
            RCLCPP_DEBUG(get_logger(), "correctPoses");
            correctPoses();

            // 发布激光历程计
            RCLCPP_DEBUG(get_logger(), "publishOdometry");
            publishOdometry();

            // 发布当前帧对齐到地图坐标系的点云和完整轨迹
            RCLCPP_DEBUG(get_logger(), "publishFrames");
            publishFrames();
        }
    }

    /**
     * GPS数据回调函数
     * 1. liosam用的不是原始的GPS传感器出来的数据，而是经过robot_localization计算后
     * 的GPS里程计。要使用GPS里程计则必须使用9轴IMU（带地磁）
    */
    void gpsHandler(const nav_msgs::msg::Odometry::SharedPtr gpsMsg)
    {
        gpsQueue.push_back(*gpsMsg);
    }

    /**
     * 将雷达坐标系下的点云转到odom坐标系（map）
    */
    void pointAssociateToMap(PointType const * const pi, PointType * const po)
    {
        po->x = transPointAssociateToMap(0,0) * pi->x + transPointAssociateToMap(0,1) * pi->y + transPointAssociateToMap(0,2) * pi->z + transPointAssociateToMap(0,3);
        po->y = transPointAssociateToMap(1,0) * pi->x + transPointAssociateToMap(1,1) * pi->y + transPointAssociateToMap(1,2) * pi->z + transPointAssociateToMap(1,3);
        po->z = transPointAssociateToMap(2,0) * pi->x + transPointAssociateToMap(2,1) * pi->y + transPointAssociateToMap(2,2) * pi->z + transPointAssociateToMap(2,3);
        po->intensity = pi->intensity;
    }

    /**
     * 点云转换函数
     * PCL有自己的点云转换函数，这里用了CPU指令进行多线程执行。具体效率还需要实验分析
    */
    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose* transformIn)
    {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        int cloudSize = cloudIn->size();
        cloudOut->resize(cloudSize);

        Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);
        
        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < cloudSize; ++i)
        {
            const auto &pointFrom = cloudIn->points[i];
            cloudOut->points[i].x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3);
            cloudOut->points[i].y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3);
            cloudOut->points[i].z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3);
            cloudOut->points[i].intensity = pointFrom.intensity;
        }
        return cloudOut;
    }

    /**
     * 将PointTypePose（x,y,z,roll,pitch,yaw）表示的6D位姿转换成gtsam位姿的实用函数
     * 这里似乎放在utility中更为合适
    */
    gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint)
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                                  gtsam::Point3(double(thisPoint.x),    double(thisPoint.y),     double(thisPoint.z)));
    }

    /**
     * 将数组表示的6D位姿（roll,pitch,yaw,x,y,z)转换成gtsam位姿的实用函数
     * 这里似乎放在utility中更为合适
    */
    gtsam::Pose3 trans2gtsamPose(float transformIn[])
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]), 
                                  gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
    }

    /**
     * 将PointTypePose（x,y,z,roll,pitch,yaw）表示的6D位姿转换成Eigen的转换表达
     * 同样应该放在utility中
    */
    Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
    {
        return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
    }

    /**
     * 将数组表示的6D位姿（roll,pitch,yaw,x,y,z)转换成Eigen的表达
    */
    Eigen::Affine3f trans2Affine3f(float transformIn[])
    {
        return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
    }

    /**
     * 将数组表示的6D位姿（roll,pitch,yaw,x,y,z)转换成PointTypePose
    */
    PointTypePose trans2PointTypePose(float transformIn[])
    {
        PointTypePose thisPose6D;
        thisPose6D.x = transformIn[3];
        thisPose6D.y = transformIn[4];
        thisPose6D.z = transformIn[5];
        thisPose6D.roll  = transformIn[0];
        thisPose6D.pitch = transformIn[1];
        thisPose6D.yaw   = transformIn[2];
        return thisPose6D;
    }

    /**
     * 可视化全局地图的独立线程
     * @brief 由于全局地图的点数较多，且需要对多帧拼接，因此放在独立线程中运行。且频率较低（5HZ）
     * 1. 发布全局地图点云
     *      1）对所有关键帧3D位姿构建KD树
     *      2）以最后一帧关键帧为索引找出一定半径范围内的所有关键帧
     *      3）对找出的关键帧数量做降采样
     *      4）对所有关键帧的点云做拼接（投影到地图坐标系）
     *      5）对地图点云做降采样
     *      6）发布全局地图点云
     * 
     * 2. 保存全局地图及轨迹
     *      1）如果保存地图变量为False，直接返回
     *      2）保存关键帧序列的3D位姿为trajectory.pcd
     *      3）保存关键帧序列的6D位姿为transformations.pcd
     *      4）构建全局角点地图和全局平面点地图和两者的集合全局地图
     *      5）将全局角点地图和全局平面点地图和全局地图依次保存
    */
    void visualizeGlobalMapThread()
    {
        rclcpp::Rate rate(0.2);
        while (rclcpp::ok()){
            rate.sleep();
            publishGlobalMap();
        }
        if (savePCD == false)
            return;
        cout << "****************************************************" << endl;
        cout << "Saving map to pcd files ..." << endl;
        savePCDDirectory = std::getenv("HOME") + savePCDDirectory;
        int unused = system((std::string("exec rm -r ") + savePCDDirectory).c_str());
        unused = system((std::string("mkdir ") + savePCDDirectory).c_str());
        if (!unused)
        {
            RCLCPP_ERROR(get_logger(), "Remove old map dir and create new failed");
        }
        pcl::io::savePCDFileASCII(savePCDDirectory + "trajectory.pcd", *cloudKeyPoses3D);
        pcl::io::savePCDFileASCII(savePCDDirectory + "transformations.pcd", *cloudKeyPoses6D);
        pcl::PointCloud<PointType>::Ptr globalCornerCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalCornerCloudDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalSurfCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalSurfCloudDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapCloud(new pcl::PointCloud<PointType>());
        for (int i = 0; i < (int)cloudKeyPoses3D->size(); i++) {
            *globalCornerCloud += *transformPointCloud(cornerCloudKeyFrames[i],  &cloudKeyPoses6D->points[i]);
            *globalSurfCloud   += *transformPointCloud(surfCloudKeyFrames[i],    &cloudKeyPoses6D->points[i]);
            cout << "\r" << std::flush << "Processing feature cloud " << i << " of " << cloudKeyPoses6D->size() << " ...";
        }
        downSizeFilterCorner.setInputCloud(globalCornerCloud);
        downSizeFilterCorner.filter(*globalCornerCloudDS);
        pcl::io::savePCDFileASCII(savePCDDirectory + "cloudCorner.pcd", *globalCornerCloudDS);
        downSizeFilterSurf.setInputCloud(globalSurfCloud);
        downSizeFilterSurf.filter(*globalSurfCloudDS);
        pcl::io::savePCDFileASCII(savePCDDirectory + "cloudSurf.pcd", *globalSurfCloudDS);
        *globalMapCloud += *globalCornerCloud;
        *globalMapCloud += *globalSurfCloud;
        pcl::io::savePCDFileASCII(savePCDDirectory + "cloudGlobal.pcd", *globalMapCloud);
        cout << "****************************************************" << endl;
        cout << "Saving map to pcd files completed" << endl;
    }

    /**
     * 发布全局地图点云，在全局地图可视化线程中调用
     * 1. 对所有关键帧3D位姿构建KD树
     * 2. 以最后一帧关键帧为索引找出一定半径范围内的所有关键帧
     * 3. 对找出的关键帧数量做降采样
     * 4. 对所有关键帧的点云做拼接（投影到地图坐标系）
     * 5. 对地图点云做降采样
     * 6. 发布全局地图点云
    */
    void publishGlobalMap()
    {
        if (pubMapCloud->get_subscription_count() == 0)
            return;

        if (cloudKeyPoses3D->points.empty() == true)
            return;

        pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap(new pcl::KdTreeFLANN<PointType>());;
        pcl::PointCloud<PointType>::Ptr globalMapKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFrames(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS(new pcl::PointCloud<PointType>());

        // kd-tree to find near key frames to visualize
        std::vector<int> pointSearchIndGlobalMap;
        std::vector<float> pointSearchSqDisGlobalMap;
        // search near key frames to visualize
        mtx.lock();
        kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);
        kdtreeGlobalMap->radiusSearch(cloudKeyPoses3D->back(), globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
        mtx.unlock();

        for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i)
            globalMapKeyPoses->push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);
        // downsample near selected key frames
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses; // for global map visualization
        downSizeFilterGlobalMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity); // for global map visualization
        downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
        downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);
        for(auto& pt : globalMapKeyPosesDS->points)
        {
            kdtreeGlobalMap->nearestKSearch(pt, 1, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap);
            pt.intensity = cloudKeyPoses3D->points[pointSearchIndGlobalMap[0]].intensity;
        }

        // extract visualized and downsampled key frames
        for (int i = 0; i < (int)globalMapKeyPosesDS->size(); ++i){
            if (pointDistance(globalMapKeyPosesDS->points[i], cloudKeyPoses3D->back()) > globalMapVisualizationSearchRadius)
                continue;
            int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
            *globalMapKeyFrames += *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
            *globalMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);
        }
        // downsample visualized points
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames; // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setLeafSize(globalMapVisualizationLeafSize, globalMapVisualizationLeafSize, globalMapVisualizationLeafSize); // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
        downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);
        publishCloud(pubMapCloud, globalMapKeyFramesDS, timeLaserInfoStamp, odomFrame);
    }

    /**
     * 回环检测独立线程
     * 1. 由于回环检测中用到了点云匹配，较为耗时，所以独立为单独的线程运行
     * 2. 新的回环关系被检测出来时被主线程加入因子图中优化
    */
    void loopClosureThread()
    {
        if (loopClosureEnableFlag == false)
            return;

        rclcpp::Rate rate(loopClosureFrequency);
        while (rclcpp::ok())
        {
            rate.sleep();
            performLoopClosure();
            visualizeLoopClosure();
        }
    }

    /**
     * 回环检测函数
     * 1. 关键帧队列为空，直接返回
     * 2. 加锁拷贝关键帧3D和6D位姿，避免多线程干扰
     * 3. 将最后一帧关键帧作为当前帧，如果当前帧已经在回环对应关系中，则返回（已经处理过这一帧了）。回环关系用一个全局map缓存
     * 4. 对关键帧3D位姿构建kd树，并用当前帧位置从kd树寻找距离最近的几帧，挑选时间间隔最远的那一帧作为匹配帧
     * 5. 将当前帧转到Map坐标系并降采样
     * 6. 对匹配帧前后几帧转换到Map坐标系下，融合并降采样，构建局部地图
     * 7. 调用ICP降当前帧匹配到局部地图，得到当前帧位姿的偏差，将偏差应用到当前帧的位姿，得到修正后的当前帧位姿。
     * 8. 根据修正后的当前帧位姿和匹配帧的位姿，计算帧间相对位姿，这个位姿被用来作为回环因子。同时，将ICP的匹配分数当作因子的噪声模型
     * 9. 将回环索引、回环间相对位姿、回环噪声模型加入全局变量
    */
    void performLoopClosure()
    {
        // 1. 关键帧队列为空，直接返回
        if (cloudKeyPoses3D->points.empty() == true)
            return;

        // 2. 加锁拷贝关键帧3D和6D位姿，避免多线程干扰
        mtx.lock();
        *copy_cloudKeyPoses3D = *cloudKeyPoses3D;
        *copy_cloudKeyPoses6D = *cloudKeyPoses6D;
        mtx.unlock();

        // 3. 将最后一帧关键帧作为当前帧，如果当前帧已经在回环对应关系中，则返回（已经处理过这一帧了）。如果找到的回环对应帧相差时间过短也返回false。回环关系用一个全局map缓存
        // 4. 对关键帧3D位姿构建kd树，并用当前帧位置从kd树寻找距离最近的几帧，挑选时间间隔最远的那一帧作为匹配帧
        int loopKeyCur;
        int loopKeyPre;
        if (detectLoopClosureDistance(&loopKeyCur, &loopKeyPre) == false)
            return;

        // extract cloud
        pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());
        {
            // 5. 将当前帧转到Map坐标系并降采样，注意这里第三个参数是0, 也就是不加上前后其他帧
            loopFindNearKeyframes(cureKeyframeCloud, loopKeyCur, 0);
            // 6. 对匹配帧前后几帧转换到Map坐标系下，融合并降采样，构建局部地图
            loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum);
            if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
                return;
        }

        // 7. 调用ICP降当前帧匹配到局部地图，得到当前帧位姿的偏差，将偏差应用到当前帧的位姿，得到修正后的当前帧位姿。
        static pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius*2);
        icp.setMaximumIterations(100);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);

        // Align clouds
        icp.setInputSource(cureKeyframeCloud);
        icp.setInputTarget(prevKeyframeCloud);
        pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
        icp.align(*unused_result);

        if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)
            return;

        // 8. 根据修正后的当前帧位姿和匹配帧的位姿，计算帧间相对位姿，这个位姿被用来作为回环因子。同时，将ICP的匹配分数当作因子的噪声模型
        float x, y, z, roll, pitch, yaw;
        Eigen::Affine3f correctionLidarFrame;
        correctionLidarFrame = icp.getFinalTransformation();
        // transform from world origin to wrong pose
        Eigen::Affine3f tWrong = pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]);
        // transform from world origin to corrected pose
        Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;// pre-multiplying -> successive rotation about a fixed frame
        pcl::getTranslationAndEulerAngles (tCorrect, x, y, z, roll, pitch, yaw);
        gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
        gtsam::Pose3 poseTo = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]);
        gtsam::Vector Vector6(6);
        float noiseScore = icp.getFitnessScore();
        Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
        noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);

        // 9. 将回环索引、回环间相对位姿、回环噪声模型加入全局变量
        mtx.lock();
        loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
        loopPoseQueue.push_back(poseFrom.between(poseTo));
        loopNoiseQueue.push_back(constraintNoise);
        mtx.unlock();
        loopIndexContainer[loopKeyCur] = loopKeyPre;
    }

    /**
     * @brief 根据位置关系寻找当前帧与对应帧的索引
     * 1. 将最后一帧关键帧作为当前帧，如果当前帧已经在回环对应关系中，则返回（已经处理过这一帧了）。如果找到的回环对应帧相差时间过短也返回false。回环关系用一个全局map缓存
     * 2. 对关键帧3D位姿构建kd树，并用当前帧位置从kd树寻找距离最近的几帧，挑选时间间隔最远的那一帧作为匹配帧
     * 
     * @param latestID 传出参数，找到的当前帧索引，实际就是用最后一帧关键帧
     * @param closestID 传出参数，找到的当前帧对应的匹配帧
    */
    bool detectLoopClosureDistance(int *latestID, int *closestID)
    {
        int loopKeyCur = copy_cloudKeyPoses3D->size() - 1;
        int loopKeyPre = -1;

        // 确认最后一帧关键帧没有被加入过回环关系中
        auto it = loopIndexContainer.find(loopKeyCur);
        if (it != loopIndexContainer.end())
            return false;

        // 将关键帧的3D位置构建kdtree，并检索空间位置相近的关键帧
        std::vector<int> pointSearchIndLoop;
        std::vector<float> pointSearchSqDisLoop;
        kdtreeHistoryKeyPoses->setInputCloud(copy_cloudKeyPoses3D);
        // 寻找空间距离相近的关键帧
        kdtreeHistoryKeyPoses->radiusSearch(copy_cloudKeyPoses3D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);
        
        // 确保空间距离相近的帧是较久前采集的，排除是前面几个关键帧
        for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i)
        {
            int id = pointSearchIndLoop[i];
            if (abs(copy_cloudKeyPoses6D->points[id].time - timeLaserInfoCur) > historyKeyframeSearchTimeDiff)
            {
                loopKeyPre = id;
                break;
            }
        }

        // 如果没有找到位置关系、时间关系都符合要求的关键帧，则返回false
        if (loopKeyPre == -1 || loopKeyCur == loopKeyPre)
            return false;

        *latestID = loopKeyCur;
        *closestID = loopKeyPre;

        return true;
    }

    /**
     * @brief 根据当前帧索引key，从前后多帧（searchNum）构建局部地图
     * @param nearKeyframes 传出参数，构建出的局部地图
     * @param key 当前帧的索引
     * @param searchNum 从当前帧的前后各searchNum个关键帧构建局部点云地图
    */
    void loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr& nearKeyframes, const int& key, const int& searchNum)
    {
        // extract near keyframes
        nearKeyframes->clear();
        int cloudSize = copy_cloudKeyPoses6D->size();
        for (int i = -searchNum; i <= searchNum; ++i)
        {
            int keyNear = key + i;
            if (keyNear < 0 || keyNear >= cloudSize )
                continue;
            *nearKeyframes += *transformPointCloud(cornerCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[keyNear]);
            *nearKeyframes += *transformPointCloud(surfCloudKeyFrames[keyNear],   &copy_cloudKeyPoses6D->points[keyNear]);
        }

        if (nearKeyframes->empty())
            return;

        // downsample near keyframes
        pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
        downSizeFilterICP.setInputCloud(nearKeyframes);
        downSizeFilterICP.filter(*cloud_temp);
        *nearKeyframes = *cloud_temp;
    }

    /**
     * 可视化回环关系，主要是根据回环关系的构建Rivz可以直接显示的MarkerArray
    */
    void visualizeLoopClosure()
    {
        if (loopIndexContainer.empty())
            return;

        visualization_msgs::msg::MarkerArray markerArray;
        // loop nodes
        visualization_msgs::msg::Marker markerNode;
        markerNode.header.frame_id = odomFrame;
        markerNode.header.stamp = timeLaserInfoStamp;
        markerNode.action = visualization_msgs::msg::Marker::ADD;
        markerNode.type = visualization_msgs::msg::Marker::SPHERE_LIST;
        markerNode.ns = "loop_nodes";
        markerNode.id = 0;
        markerNode.pose.orientation.w = 1;
        markerNode.scale.x = 0.3; markerNode.scale.y = 0.3; markerNode.scale.z = 0.3; 
        markerNode.color.r = 0; markerNode.color.g = 0.8; markerNode.color.b = 1;
        markerNode.color.a = 1;
        // loop edges
        visualization_msgs::msg::Marker markerEdge;
        markerEdge.header.frame_id = odomFrame;
        markerEdge.header.stamp = timeLaserInfoStamp;
        markerEdge.action = visualization_msgs::msg::Marker::ADD;
        markerEdge.type = visualization_msgs::msg::Marker::LINE_LIST;
        markerEdge.ns = "loop_edges";
        markerEdge.id = 1;
        markerEdge.pose.orientation.w = 1;
        markerEdge.scale.x = 0.1;
        markerEdge.color.r = 0.9; markerEdge.color.g = 0.9; markerEdge.color.b = 0;
        markerEdge.color.a = 1;

        for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end(); ++it)
        {
            int key_cur = it->first;
            int key_pre = it->second;
            geometry_msgs::msg::Point p;
            p.x = copy_cloudKeyPoses6D->points[key_cur].x;
            p.y = copy_cloudKeyPoses6D->points[key_cur].y;
            p.z = copy_cloudKeyPoses6D->points[key_cur].z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
            p.x = copy_cloudKeyPoses6D->points[key_pre].x;
            p.y = copy_cloudKeyPoses6D->points[key_pre].y;
            p.z = copy_cloudKeyPoses6D->points[key_pre].z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
        }

        markerArray.markers.push_back(markerNode);
        markerArray.markers.push_back(markerEdge);
        pubLoopConstraintEdge->publish(markerArray);
    }

    /**
     * 当前帧位姿初始化
     * 1、如果是第一帧，用原始imu数据的RPY初始化当前帧位姿（旋转部分）
     * 2、后续帧，用imu里程计计算两帧之间的增量位姿变换，作用于前一帧的激光位姿，得到当前帧激光位姿
    */
    void updateInitialGuess()
    {
        // 把上一次激光里程计的结果缓存到incrementalOdometryAffineFront
        // 在当前帧处理结束的时候会使用结束后的位姿与incrementalOdometryAffineFront计算一个位姿增量
        // 在增加到上一次的里程计结果。（似乎是没有必要的操作）
        // save current transformation before any processing
        incrementalOdometryAffineFront = trans2Affine3f(transformTobeMapped);

        // lastImuTransformation缓存上一帧的IMU输出的roll、pitch、yaw。
        // 当IMU里程计不能使用的时候，使用当前的roll、pitch、yaw与lastImuTransformation
        // 计算角度增量，增加到上一帧的里程计结果，作为当前帧的初始位姿

        static Eigen::Affine3f lastImuTransformation;
        // 第一帧做初始化，直接使用IMU输出roll、pitch、yaw作为当前帧的初始位姿。如果不使用9轴IMU（即yaw角与地磁角关系绑定）
        // 的话，则yaw角强制设为0
        if (cloudKeyPoses3D->points.empty())
        {
            RCLCPP_INFO(get_logger(), "cloudKeyPoses3D is empty, set transform init to imu rotate");
            transformTobeMapped[0] = cloudInfo.imu_roll_init;
            transformTobeMapped[1] = cloudInfo.imu_pitch_init;
            transformTobeMapped[2] = cloudInfo.imu_yaw_init;

            if (!useImuHeadingInitialization)
                transformTobeMapped[2] = 0;

            lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imu_roll_init, cloudInfo.imu_pitch_init, cloudInfo.imu_yaw_init); // save imu before return;
            return;
        }

        // 后续帧做帧位姿初始化
        // 1. 如果IMU里程计可用（已经融合了激光里程计结果），使用其作为6D位姿初始化
        // 2. 如果IMU里程计不可用但IMU原始角度可用，则在上一帧的位姿上叠加IMU角度的变化作为初始化
        static bool lastImuPreTransAvailable = false;
        static Eigen::Affine3f lastImuPreTransformation;
        if (cloudInfo.odom_available == true)
        {
            Eigen::Affine3f transBack = pcl::getTransformation(
                cloudInfo.initial_guess_x, cloudInfo.initial_guess_y, cloudInfo.initial_guess_z,
                cloudInfo.initial_guess_roll, cloudInfo.initial_guess_pitch, cloudInfo.initial_guess_yaw);
            if (lastImuPreTransAvailable == false)
            {
                lastImuPreTransformation = transBack;
                lastImuPreTransAvailable = true;
            } else {
                Eigen::Affine3f transIncre = lastImuPreTransformation.inverse() * transBack;
                Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
                Eigen::Affine3f transFinal = transTobe * transIncre;
                pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                              transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

                lastImuPreTransformation = transBack;

                lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imu_roll_init, cloudInfo.imu_pitch_init, cloudInfo.imu_yaw_init); // save imu before return;
                return;
            }
        }

        // 使用IMU输出的角度作为初始化，只采用角度
        // 注意，当IMU里程计可用的情况下直接返回，不会进入下面这个函数
        // 如果当IMU原始角度和IMU里程计不可用的情况下，transformTobeMapped没有被改变，也就是使用了上一帧的结果作为当前帧的初始值
        if (cloudInfo.imu_available == true)
        {
            Eigen::Affine3f transBack = pcl::getTransformation(0, 0, 0, cloudInfo.imu_roll_init, cloudInfo.imu_pitch_init, cloudInfo.imu_yaw_init);
            Eigen::Affine3f transIncre = lastImuTransformation.inverse() * transBack;

            Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
            Eigen::Affine3f transFinal = transTobe * transIncre;
            pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                          transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

            lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imu_roll_init, cloudInfo.imu_pitch_init, cloudInfo.imu_yaw_init); // save imu before return;
            return;
        }
    }


    /**
     * 提取最后一个关键帧空间、时间邻近的帧并提取点云构建局部地图
     * 1. 对所有关键帧3D位姿构建KD树
     * 2. 使用最后一个关键帧位姿作为索引，从KD树中找到指定半径范围内的其他关键帧
     * 3. 对找出的关键帧数量做降采样，避免关键帧位姿太过靠近
     * 4. 加上时间上相邻的关键帧
     * 5. 对所有挑选出的关键帧数量再做一次降采样，避免位置过近
     * 6. 将挑选出的关键帧点云转换到odom坐标系。（这里使用一个map缓存坐标变换后的点云，避免重复计算）
     * 7. 对局部地图的角点、平面点点云做降采样
    */
    void extractNearby()
    {
        pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS(new pcl::PointCloud<PointType>());
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;

        // extract all the nearby key poses and downsample them
        kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D); // create kd-tree
        kdtreeSurroundingKeyPoses->radiusSearch(cloudKeyPoses3D->back(), (double)surroundingKeyframeSearchRadius, pointSearchInd, pointSearchSqDis);
        for (int i = 0; i < (int)pointSearchInd.size(); ++i)
        {
            int id = pointSearchInd[i];
            surroundingKeyPoses->push_back(cloudKeyPoses3D->points[id]);
        }

        downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
        downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);
        for(auto& pt : surroundingKeyPosesDS->points)
        {
            kdtreeSurroundingKeyPoses->nearestKSearch(pt, 1, pointSearchInd, pointSearchSqDis);
            pt.intensity = cloudKeyPoses3D->points[pointSearchInd[0]].intensity;
        }

        // also extract some latest key frames in case the robot rotates in one position
        int numPoses = cloudKeyPoses3D->size();
        for (int i = numPoses-1; i >= 0; --i)
        {
            if (timeLaserInfoCur - cloudKeyPoses6D->points[i].time < 10.0)
                surroundingKeyPosesDS->push_back(cloudKeyPoses3D->points[i]);
            else
                break;
        }

        extractCloud(surroundingKeyPosesDS);
    }

    /**
     * @brief 根据传入的关键帧姿态集合，提取点云，转化到Map坐标系，融合，降采样
     * 这个函数在里程计线程和回环检测线程中都被使用到
     * 
     * @param cloudToExtract 要提取的关键帧序列的姿态集合
    */
    void extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract)
    {
        // fuse the map
        laserCloudCornerFromMap->clear();
        laserCloudSurfFromMap->clear(); 
        for (int i = 0; i < (int)cloudToExtract->size(); ++i)
        {
            if (pointDistance(cloudToExtract->points[i], cloudKeyPoses3D->back()) > surroundingKeyframeSearchRadius)
            {
                RCLCPP_INFO(get_logger(), "pointDistance too large! Empty laserCloudCornerFromMap and laserCloudSurfFromMap");
                continue;
            }
                

            int thisKeyInd = (int)cloudToExtract->points[i].intensity;
            if (laserCloudMapContainer.find(thisKeyInd) != laserCloudMapContainer.end()) 
            {
                // transformed cloud available
                *laserCloudCornerFromMap += laserCloudMapContainer[thisKeyInd].first;
                *laserCloudSurfFromMap   += laserCloudMapContainer[thisKeyInd].second;
            } else {
                // transformed cloud not available
                pcl::PointCloud<PointType> laserCloudCornerTemp = *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
                pcl::PointCloud<PointType> laserCloudSurfTemp = *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);
                *laserCloudCornerFromMap += laserCloudCornerTemp;
                *laserCloudSurfFromMap   += laserCloudSurfTemp;
                laserCloudMapContainer[thisKeyInd] = make_pair(laserCloudCornerTemp, laserCloudSurfTemp);
            }
            
        }

        // Downsample the surrounding corner key frames (or map)
        downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
        downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
        laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->size();
        // Downsample the surrounding surf key frames (or map)
        downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
        downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
        laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->size();

        // clear map cache if too large
        if (laserCloudMapContainer.size() > 1000)
            laserCloudMapContainer.clear();
    }

    /**
     * 构建局部地图
     * 1. 对所有关键帧3D位姿构建KD树
     * 2. 使用最后一个关键帧位姿作为索引，从KD树中找到指定半径范围内的其他关键帧
     * 3. 对找出的关键帧数量做降采样，避免关键帧位姿太过靠近
     * 4. 加上时间上相邻的关键帧
     * 5. 对所有挑选出的关键帧数量再做一次降采样，避免位置过近
     * 6. 将挑选出的关键帧点云转换到odom坐标系。（这里使用一个map缓存坐标变换后的点云，避免重复计算）
     * 7. 对局部地图的角点、平面点点云做降采样
    */
    void extractSurroundingKeyFrames()
    {
        if (cloudKeyPoses3D->points.empty() == true)
            return; 
        
        extractNearby();
    }

    /**
     * 对当前帧点云做降采样
     * 1. 对当前帧角点点云做降采样
     * 2. 对当前帧平面点点云做降采样
    */
    void downsampleCurrentScan()
    {
        // 降采样当前帧角点点云
        laserCloudCornerCurDS->clear();
        downSizeFilterCorner.setInputCloud(laserCloudCornerCur);
        downSizeFilterCorner.filter(*laserCloudCornerCurDS);
        laserCloudCornerCurDSNum = laserCloudCornerCurDS->size();

        // 降采样当前帧平面点点云
        laserCloudSurfCurDS->clear();
        downSizeFilterSurf.setInputCloud(laserCloudSurfCur);
        downSizeFilterSurf.filter(*laserCloudSurfCurDS);
        laserCloudSurfCurDSNum = laserCloudSurfCurDS->size();
    }

    /**
     * 从transformToBeMapped更新当前姿态到transPointAssociateToMap变量
     * transPointAssociateToMap变量被用来做点云匹配优化
     * */ 
    void updatePointAssociateToMap()
    {
        transPointAssociateToMap = trans2Affine3f(transformTobeMapped);
    }

    /**
     * @brief 计算边缘点点集中每一个点到局部地图中匹配到的直线的距离和法向量;
    */
    void cornerOptimization()
    {
        // 将transformTobeMapped存储到transPointAssociateToMap转换矩阵，
        // 方便后面用旋转平移关系对选中的点转换到地图坐标系
        updatePointAssociateToMap();

        // omp指令集进行并行计算，打开后在某些电脑上似乎有比较奇怪的表现
        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < laserCloudCornerCurDSNum; i++)
        {
            // pointOri是雷达坐标系下的边缘点；pointSel是转换到地图坐标系下的点
            // coeff存储的是经过距离加权后的点到平面向量
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            pointOri = laserCloudCornerCurDS->points[i];
            // 将雷达坐标系下的点pointOri转换到地图坐标系pointSel
            pointAssociateToMap(&pointOri, &pointSel);
            // 从局部地图（已经提前设置好kdtree）中找到最近的5个点
            // pointSel为检索点
            // pointSearchInd存储检索结果的5个点在原始点云中的索引
            // pointSearchSqDis存储检索出的5个点与检索点的距离的平方
            kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            // matA1是检索出的5个点的协方差矩阵
            cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));
            // matD1是协方差矩阵的特征值
            cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
            // matV1是协方差矩阵的特征向量
            cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));
            
            // 如果找到的第5个点（距离最大）的点也小与1米，认为检索结果有效，否则跳过当前的pointOri
            if (pointSearchSqDis[4] < 1.0) {
                // cx,cy,cz是检索出的5个点的中心坐标
                float cx = 0, cy = 0, cz = 0;
                for (int j = 0; j < 5; j++) {
                    cx += laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
                    cy += laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
                    cz += laserCloudCornerFromMapDS->points[pointSearchInd[j]].z;
                }
                cx /= 5; cy /= 5;  cz /= 5;

                // 协方差矩阵是对称矩阵
                float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
                for (int j = 0; j < 5; j++) {
                    float ax = laserCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
                    float ay = laserCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
                    float az = laserCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;

                    a11 += ax * ax; a12 += ax * ay; a13 += ax * az;
                    a22 += ay * ay; a23 += ay * az;
                    a33 += az * az;
                }
                a11 /= 5; a12 /= 5; a13 /= 5; a22 /= 5; a23 /= 5; a33 /= 5;

                // 存储协方差的值到matA1
                matA1.at<float>(0, 0) = a11; matA1.at<float>(0, 1) = a12; matA1.at<float>(0, 2) = a13;
                matA1.at<float>(1, 0) = a12; matA1.at<float>(1, 1) = a22; matA1.at<float>(1, 2) = a23;
                matA1.at<float>(2, 0) = a13; matA1.at<float>(2, 1) = a23; matA1.at<float>(2, 2) = a33;

                // 对协方差矩阵做特征值分解，最大特征值对应的特征向量是这5个点的主方向
                cv::eigen(matA1, matD1, matV1);

                // 如果最大的特征值要远大于第二个特征值，则认为则5个点能够构成一条直线
                // 类似PCA主成分分析的原理，数据协方差的最大特征值对应的特征向量为主方向
                if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1)) {
                    // 以下部分是在计算当前点pointSel到检索出的直线的距离和方向，如果距离够近，则认为匹配成功，否则认为匹配失败
                    // x0,y0,z0是直线外一点
                    float x0 = pointSel.x;
                    float y0 = pointSel.y;
                    float z0 = pointSel.z;
                    // matV1的第一行就是5个点形成的直线的方向，cx,cy,cz是5个点的中心点
                    // 因此，x1,y1,z1和x2,y2,z2是经过中心点的直线上的另外两个点，两点之间的距离是0.2米
                    float x1 = cx + 0.1 * matV1.at<float>(0, 0);
                    float y1 = cy + 0.1 * matV1.at<float>(0, 1);
                    float z1 = cz + 0.1 * matV1.at<float>(0, 2);
                    float x2 = cx - 0.1 * matV1.at<float>(0, 0);
                    float y2 = cy - 0.1 * matV1.at<float>(0, 1);
                    float z2 = cz - 0.1 * matV1.at<float>(0, 2);

                    // 这边是在求[(x0-x1),(y0-y1),(z0-z1)]与[(x0-x2),(y0-y2),(z0-z2)]叉乘得到的向量的模长
                    // 这个模长是由0.2*V1[0]和点[x0,y0,z0]构成的平行四边形的面积
                    // 垂直于0,1,2三点构成的平面的向量[XXX,YYY,ZZZ] = [(y0-y1)(z0-z2)-(y0-y2)(z0-z1),-(x0-x1)(z0-z2)+(x0-x2)(z0-z1),(x0-x1)(y0-y2)-(x0-x2)(y0-y1)]
                    float a012 = sqrt(((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) * ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                                    + ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) * ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                                    + ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)) * ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)));

                    // l12表示的是0.2*(||V1[0]||)
                    // 点x1,y1,z1到点x2,y2,z2的距离
                    float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));

                    // 求叉乘结果[la',lb',lc']=[(x1-x2),(y1-y2),(z1-z2)]x[XXX,YYY,ZZZ]
                    // [la,lb,lc]=[la',lb',lc']/a012/l12
                    // LLL=[la,lb,lc]是0.2*V1[0]这条高上的单位法向量。||LLL||=1；
                    float la = ((y1 - y2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                              + (z1 - z2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))) / a012 / l12;

                    float lb = -((x1 - x2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                               - (z1 - z2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    float lc = -((x1 - x2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                               + (y1 - y2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    // ld2就是点pointSel(x0,y0,z0)到直线的距离
                    float ld2 = a012 / l12;

                    // 如果点pointSel刚好在直线上，则ld2=0,s=1；
                    // 点到直线的距离越远，s越小，则赋予的比重越低
                    float s = 1 - 0.9 * fabs(ld2);

                    // 使用系数对法向量加权，实际上相当于对导数（雅克比矩阵加权了）
                    coeff.x = s * la;
                    coeff.y = s * lb;
                    coeff.z = s * lc;
                    coeff.intensity = s * ld2;

                    // 经验阈值，判断点到直线的距离是否够近，足够近才采纳为优化目标点
                    if (s > 0.1) {
                        laserCloudOriCornerVec[i] = pointOri;
                        coeffSelCornerVec[i] = coeff;
                        laserCloudOriCornerFlag[i] = true;
                    }
                }
            }
        }
    }

    /**
     * @brief 计算平面点点集中每一个点到局部地图中匹配到的平面的距离和法向量
    */
    void surfOptimization()
    {
        updatePointAssociateToMap();

        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < laserCloudSurfCurDSNum; i++)
        {
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            pointOri = laserCloudSurfCurDS->points[i];
            pointAssociateToMap(&pointOri, &pointSel);
            // 与边缘点找直线一样，从局部地图的平面点集中找到与pointSel距离最近的5个点
            kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            // 下面的过程要求解Ax+By+Cz+1=0的平面方程
            // 由于有5个点，因此是求解超定方程
            // 假设5个点都在平面上，则matA0是系数矩阵，matB0是等号右边的值（都是-1）；matX0是求出来的A，B，C
            Eigen::Matrix<float, 5, 3> matA0;
            Eigen::Matrix<float, 5, 1> matB0;
            Eigen::Vector3f matX0;

            matA0.setZero();
            matB0.fill(-1);
            matX0.setZero();

            if (pointSearchSqDis[4] < 1.0) {
                for (int j = 0; j < 5; j++) {
                    matA0(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
                    matA0(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
                    matA0(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
                }

                // 这里是求解matA0XmatX0 = matB0方程
                matX0 = matA0.colPivHouseholderQr().solve(matB0);

                float pa = matX0(0, 0);
                float pb = matX0(1, 0);
                float pc = matX0(2, 0);
                float pd = 1;

                // （pa,pb,pc)是平面的法向量，这里是对法向量规一化，变成单位法向量
                float ps = sqrt(pa * pa + pb * pb + pc * pc);
                pa /= ps; pb /= ps; pc /= ps; pd /= ps;

                // 下面是判定检索出来的5个点是否能够构成一个合格的法向量
                // 点到平面（Ax+By+Cz+D=0)的距离为 |Ax+By+Cz+D|/\sqrt(A^2+B^2+C^2)
                // 由于这里法向量已经归一化成为单位法向量，因此这里|Ax+By+Cz+D|就等于点到平面的距离
                // 这里只有当5个点到拟合的平面距离都小于0.2米才认为拟合的平面合格，否则跳过这个点
                bool planeValid = true;
                for (int j = 0; j < 5; j++) {
                    if (fabs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                             pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                             pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z + pd) > 0.2) {
                        planeValid = false;
                        break;
                    }
                }

                // 如果由检索出的5个点拟合的平面是合格的平面，计算点到平面的距离
                if (planeValid) {
                    // pd2是点到平面的距离（注意这里pd2是由正负号的）
                    float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

                    // 与cornerOptimization中类似，使用距离计算一个权重
                    // keep in mind, 后面部分（0.9*fabs.....）越小越好。因此，这里可以理解为对点到平面距离的加权
                    // 越远的平面对匹配具有更好的约束性，因此要赋予更大的比重。
                    float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointSel.x * pointSel.x
                            + pointSel.y * pointSel.y + pointSel.z * pointSel.z));

                    coeff.x = s * pa;
                    coeff.y = s * pb;
                    coeff.z = s * pc;
                    coeff.intensity = s * pd2;

                    // 如果点到平面的距离够小，则采纳为优化目标点，否则跳过
                    if (s > 0.1) {
                        laserCloudOriSurfVec[i] = pointOri;
                        coeffSelSurfVec[i] = coeff;
                        laserCloudOriSurfFlag[i] = true;
                    }
                }
            }
        }
    }

    /**
     * @brief 将cornerOptimization和surfOptimization两个函数计算出来的边缘点、平面点到局部地图的
     * 距离、法向量集合在一起
    */
    void combineOptimizationCoeffs()
    {
        // combine corner coeffs
        for (int i = 0; i < laserCloudCornerCurDSNum; ++i){
            if (laserCloudOriCornerFlag[i] == true){
                laserCloudOri->push_back(laserCloudOriCornerVec[i]);
                coeffSel->push_back(coeffSelCornerVec[i]);
            }
        }
        // combine surf coeffs
        for (int i = 0; i < laserCloudSurfCurDSNum; ++i){
            if (laserCloudOriSurfFlag[i] == true){
                laserCloudOri->push_back(laserCloudOriSurfVec[i]);
                coeffSel->push_back(coeffSelSurfVec[i]);
            }
        }
        // reset flag for next iteration
        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);
    }

    /**
     * @brief 这部分代码是基于高斯牛顿法的优化，不是LOAM论文里面提到的LM优化。目标函数（点到线、点到平面的距离）对位姿（这里
     * 使用的是欧拉角、tx,ty,tz的表达）求导，计算高斯-牛顿法更新方向和步长，然后对transformTobeMapped（存放当前雷达点云位姿）进行
     * 更新。这里相比于LOAM，多了坐标轴的转换，但实际上这部分转换是没有必要的。这部分代码可以直接阅读LeGO-LOAM的代码：
     * https://github.com/RobustFieldAutonomyLab/LeGO-LOAM/blob/896a7a95a8bc510b76819d4cc48707e344bad621/LeGO-LOAM/src/mapOptmization.cpp#L1229
     * 
     * @param iterCount 迭代更新次数，这个函数在scan2MapOptimization中被调用，默认最大迭代次数是30
    */
    bool LMOptimization(int iterCount)
    {
        // This optimization is from the original loam_velodyne by Ji Zhang, need to cope with coordinate transformation
        // lidar <- camera      ---     camera <- lidar
        // x = z                ---     x = y
        // y = x                ---     y = z
        // z = y                ---     z = x
        // roll = yaw           ---     roll = pitch
        // pitch = roll         ---     pitch = yaw
        // yaw = pitch          ---     yaw = roll

        // 计算三轴欧拉角的sin、cos，后面使用旋转矩阵对欧拉角求导中会使用到
        // lidar -> camera
        float srx = sin(transformTobeMapped[1]);
        float crx = cos(transformTobeMapped[1]);
        float sry = sin(transformTobeMapped[2]);
        float cry = cos(transformTobeMapped[2]);
        float srz = sin(transformTobeMapped[0]);
        float crz = cos(transformTobeMapped[0]);

        // laserCloudOri是在cornerOptimization、surfOptimization两个函数中找到的有匹配关系的
        // 角点和平面点，如果找到的可供优化的点数太少，则跳过此次优化
        int laserCloudSelNum = laserCloudOri->size();
        if (laserCloudSelNum < 50) {
            return false;
        }

        // matA是Jacobians矩阵J
        cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
        // matB是目标函数，也就是距离
        cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
        // matX是高斯-牛顿法计算出的更新向量
        cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matP(6, 6, CV_32F, cv::Scalar::all(0));

        PointType pointOri, coeff;

        for (int i = 0; i < laserCloudSelNum; i++) {
            // 坐标系转换这部分可以不用看，没有什么作用
            // lidar -> camera
            pointOri.x = laserCloudOri->points[i].y;
            pointOri.y = laserCloudOri->points[i].z;
            pointOri.z = laserCloudOri->points[i].x;
            // lidar -> camera
            coeff.x = coeffSel->points[i].y;
            coeff.y = coeffSel->points[i].z;
            coeff.z = coeffSel->points[i].x;
            coeff.intensity = coeffSel->points[i].intensity;
            // in camera
            // 求雅克比矩阵的值，也就是求目标函数（点到线、平面的距离）相对于tx,ty,tz,rx,ry,rz的导数
            // 具体的公式推导看仓库README中本项目博客，高斯牛顿法方程：J^{T}J\Delta{x} = -Jf(x)，\Delta{x}就是要求解的更新向量matX
            // arx是目标函数相对于roll的导数
            float arx = (crx*sry*srz*pointOri.x + crx*crz*sry*pointOri.y - srx*sry*pointOri.z) * coeff.x
                      + (-srx*srz*pointOri.x - crz*srx*pointOri.y - crx*pointOri.z) * coeff.y
                      + (crx*cry*srz*pointOri.x + crx*cry*crz*pointOri.y - cry*srx*pointOri.z) * coeff.z;
            // ary是目标函数相对于pitch的导数
            float ary = ((cry*srx*srz - crz*sry)*pointOri.x 
                      + (sry*srz + cry*crz*srx)*pointOri.y + crx*cry*pointOri.z) * coeff.x
                      + ((-cry*crz - srx*sry*srz)*pointOri.x 
                      + (cry*srz - crz*srx*sry)*pointOri.y - crx*sry*pointOri.z) * coeff.z;
            // arz是目标函数相对于yaw的导数
            float arz = ((crz*srx*sry - cry*srz)*pointOri.x + (-cry*crz-srx*sry*srz)*pointOri.y)*coeff.x
                      + (crx*crz*pointOri.x - crx*srz*pointOri.y) * coeff.y
                      + ((sry*srz + cry*crz*srx)*pointOri.x + (crz*sry-cry*srx*srz)*pointOri.y)*coeff.z;

            /*
            在求点到直线的距离时，coeff表示的是如下内容
            [la,lb,lc]表示的是点到直线的垂直连线方向，s是长度
            coeff.x = s * la;
            coeff.y = s * lb;
            coeff.z = s * lc;
            coeff.intensity = s * ld2;

            在求点到平面的距离时，coeff表示的是
            [pa,pb,pc]表示过外点的平面的法向量，s是线的长度
            coeff.x = s * pa;
            coeff.y = s * pb;
            coeff.z = s * pc;
            coeff.intensity = s * pd2;
            */

            // lidar -> camera
            matA.at<float>(i, 0) = arz;
            matA.at<float>(i, 1) = arx;
            matA.at<float>(i, 2) = ary;
            // 目标函数相对于tx的导数等于法向量的x
            matA.at<float>(i, 3) = coeff.z;
            // 目标函数相对于ty的导数等于法向量的y
            matA.at<float>(i, 4) = coeff.x;
            // 目标函数相对于tz的导数等于法向量的z
            matA.at<float>(i, 5) = coeff.y;

            // matB存储的是目标函数（距离）的负值，因为：J^{T}J\Delta{x} = -Jf(x)
            matB.at<float>(i, 0) = -coeff.intensity;
        }

        cv::transpose(matA, matAt);
        matAtA = matAt * matA;
        matAtB = matAt * matB;
        // 求解高斯-牛顿法中的增量方程：J^{T}J\Delta{x} = -Jf(x)，这里解出来的matX就是更新向量
        // matA是雅克比矩阵J
        // matAtB是上面等式中等号的右边，负号在matB赋值的时候已经加入
        cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

        // 如果是第一次迭代，判断求解出来的近似Hessian矩阵，也就是J^{T}J:=matAtA是否退化
        /**
            * 这部分的计算说实话没有找到很好的理论出处，这里只能大概说一下这段代码想要做的事情
            * 这里用matAtA也就是高斯-牛顿中的近似海瑟（Hessian）矩阵H。求解增量方程：J^{T}J\Delta{x} = -Jf(x)
            * 要求H:=J^{T}J可逆，但H不一定可逆。下面的代码通过H的特征值判断H是否退化，并将退化的方向清零matV2。而后又根据
            * matV.inv()*matV2作为更新向量的权重系数，matV是H的特征向量矩阵。
        */
        if (iterCount == 0) {

            cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

            // 对近似Hessian矩阵做特征值分解，matE是特征值，matV是特征向量。opencv的matV中每一行是一个特征向量
            cv::eigen(matAtA, matE, matV);
            matV.copyTo(matV2);

            isDegenerate = false;
            float eignThre[6] = {100, 100, 100, 100, 100, 100};
            for (int i = 5; i >= 0; i--) {
                if (matE.at<float>(0, i) < eignThre[i]) {
                    for (int j = 0; j < 6; j++) {
                        matV2.at<float>(i, j) = 0;
                    }
                    isDegenerate = true;
                } else {
                    break;
                }
            }
            matP = matV.inv() * matV2;
        }

        // 当第一次迭代判断到海瑟矩阵退化，后面会使用计算出来的权重matP对增量matX做加权组合
        if (isDegenerate)
        {
            cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2;
        }

        // 将增量matX叠加到变量（位姿）transformTobeMapped中
        transformTobeMapped[0] += matX.at<float>(0, 0);
        transformTobeMapped[1] += matX.at<float>(1, 0);
        transformTobeMapped[2] += matX.at<float>(2, 0);
        transformTobeMapped[3] += matX.at<float>(3, 0);
        transformTobeMapped[4] += matX.at<float>(4, 0);
        transformTobeMapped[5] += matX.at<float>(5, 0);

        // 计算roll、pitch、yaw的迭代步长
        float deltaR = sqrt(
                            pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
        // 计算tx，ty，tz的迭代步长
        float deltaT = sqrt(
                            pow(matX.at<float>(3, 0) * 100, 2) +
                            pow(matX.at<float>(4, 0) * 100, 2) +
                            pow(matX.at<float>(5, 0) * 100, 2));

        // 如果迭代的步长达到设定阈值，则认为已经收敛
        if (deltaR < 0.05 && deltaT < 0.05) {
            return true; // converged
        }
        return false; // keep optimizing
    }

    /**
     * 将当前帧点云匹配到局部地图，并优化位姿
     * 1. 当前帧角点、平面点数量过少，直接返回
     * 2. 迭代30次（上限）优化
     *    1) 当前激光帧角点寻找局部map匹配点
     *       a.更新当前帧位姿，将当前帧角点坐标变换到map系下，在局部map中查找5个最近点，距离小于1m，且5个点构成直线（用距离中心点的协方差矩阵，特征值进行判断），则认为匹配上了
     *       b.计算当前帧角点到直线的距离、垂线的单位向量，存储为角点参数
     *    2) 当前激光帧平面点寻找局部map匹配点
     *       a.更新当前帧位姿，将当前帧平面点坐标变换到map系下，在局部map中查找5个最近点，距离小于1m，且5个点构成平面（最小二乘拟合平面），则认为匹配上了
     *       b.计算当前帧平面点到平面的距离、垂线的单位向量，存储为平面点参数
     *    3) 提取当前帧中与局部map匹配上了的角点、平面点，加入同一集合
     *    4) 对匹配特征点计算Jacobian矩阵，观测值为特征点到直线、平面的距离，构建高斯牛顿方程，迭代优化当前位姿，存transformTobeMapped
     * 3、用imu原始RPY数据与scan-to-map优化后的位姿进行加权融合（只加权融合roll、pitch角），更新当前帧位姿的roll、pitch，约束z坐标
    */
    void scan2MapOptimization()
    {
        if (cloudKeyPoses3D->points.empty())
            return;

        if (laserCloudCornerCurDSNum > edgeFeatureMinValidNum && laserCloudSurfCurDSNum > surfFeatureMinValidNum)
        {
            kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);
            kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);

            for (int iterCount = 0; iterCount < 30; iterCount++)
            {
                laserCloudOri->clear();
                coeffSel->clear();

                cornerOptimization();
                surfOptimization();

                combineOptimizationCoeffs();

                if (LMOptimization(iterCount) == true)
                    break;              
            }

            transformUpdate();
        } else {
            RCLCPP_WARN(get_logger(), "Not enough features! Only %d edge and %d planar features available.", laserCloudCornerCurDSNum, laserCloudSurfCurDSNum);
        }
    }

    /**
     * 使用IMU的原始输出roll、pitch与当前估计的roll、pitch加权融合
     * 注意这里只对roll、pitch加权融合。同时有一个权重控制IMU的比重（默认0.01）
    */
    void transformUpdate()
    {
        if (cloudInfo.imu_available == true)
        {
            if (std::abs(cloudInfo.imu_pitch_init) < 1.4)
            {
                double imuWeight = imuRPYWeight;
                tf2::Quaternion imuQuaternion;
                tf2::Quaternion transformQuaternion;
                double rollMid, pitchMid, yawMid;

                // slerp roll
                transformQuaternion.setRPY(transformTobeMapped[0], 0, 0);
                imuQuaternion.setRPY(cloudInfo.imu_roll_init, 0, 0);
                tf2::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                transformTobeMapped[0] = rollMid;

                // slerp pitch
                transformQuaternion.setRPY(0, transformTobeMapped[1], 0);
                imuQuaternion.setRPY(0, cloudInfo.imu_pitch_init, 0);
                tf2::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                transformTobeMapped[1] = pitchMid;
            }
        }

        transformTobeMapped[0] = constraintTransformation(transformTobeMapped[0], rotation_tollerance);
        transformTobeMapped[1] = constraintTransformation(transformTobeMapped[1], rotation_tollerance);
        transformTobeMapped[5] = constraintTransformation(transformTobeMapped[5], z_tollerance);

        // 这里就已经记录了incrementalOdometryAffineBack，后续被用来计算odometry_incremental
        // 也就是说，incremental的激光里程计没有使用到因子图优化的结果，因此更加光滑，没有跳变，可以给IMU预积分模块使用
        incrementalOdometryAffineBack = trans2Affine3f(transformTobeMapped);
    }

    /**
     * 角度限制函数。
     * 配置文件中可以选择对roll、pitch角做限制。实际感觉不需要
    */
    float constraintTransformation(float value, float limit)
    {
        if (value < -limit)
            value = -limit;
        if (value > limit)
            value = limit;

        return value;
    }
    
    /**
     * 是否将当前帧选择为关键帧。
     * 当距离不够且角度不够时，不会将当前帧选择为关键帧。
     * 对于非关键帧的点云帧，只做点云匹配校准里程计。对于关键帧，则会加入因子图进行优化
    */
    bool saveFrame()
    {
        if (cloudKeyPoses3D->points.empty())
            return true;

        Eigen::Affine3f transStart = pclPointToAffine3f(cloudKeyPoses6D->back());
        Eigen::Affine3f transFinal = pcl::getTransformation(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                            transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);

        if (abs(roll)  < surroundingkeyframeAddingAngleThreshold &&
            abs(pitch) < surroundingkeyframeAddingAngleThreshold && 
            abs(yaw)   < surroundingkeyframeAddingAngleThreshold &&
            sqrt(x*x + y*y + z*z) < surroundingkeyframeAddingDistThreshold)
            return false;

        return true;
    }

    /**
     * 2. 添加激光里程计因子
     *   1）若是第一帧，则构建prior因子，赋予较大的方差
     *   2）后续帧，根据当前的位姿估计，以及上一个关键帧的位姿，计算位姿增量，添加间隔因子（BetweenFactor);
     *      同时，将当前帧当前的位姿估计作为因子图当前变量的初始值。
    */
    void addOdomFactor()
    {
        if (cloudKeyPoses3D->points.empty())
        {
            noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-2, 1e-2, M_PI*M_PI, 1e8, 1e8, 1e8).finished()); // rad*rad, meter*meter
            gtSAMgraph.add(PriorFactor<Pose3>(0, trans2gtsamPose(transformTobeMapped), priorNoise));
            initialEstimate.insert(0, trans2gtsamPose(transformTobeMapped));
        }else{
            noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
            gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back());
            gtsam::Pose3 poseTo   = trans2gtsamPose(transformTobeMapped);
            gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->size()-1, cloudKeyPoses3D->size(), poseFrom.between(poseTo), odometryNoise));
            initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);
        }
    }

    /**
     * 3. 添加GPS因子
     *   1）GPS队列为空，或者关键帧序列为空，或者前后关键帧距离小于5m，或者位姿协方差较小，直接返回，认为不需要加入GPS校正
     *   2）从GPS队列中找到与当前帧时间最接近的GPS数据
     *   3）GPS数据方差大于阈值或者与上一次采用的GPS位置小于5m，直接返回
     *   4）从GPS数据中提取x,y,z和协方差，构建GPS因子加入因子图。其中的z坐标可以设置为不使用GPS的输出（GPS的z坐标较为不准）
     *   5）设置aLoopGpsIsClosed标志位为true，后面因子图优化时会多次迭代且更新所有历史关键帧位姿
    */
    void addGPSFactor()
    {
        if (gpsQueue.empty())
            return;

        // wait for system initialized and settles down
        if (cloudKeyPoses3D->points.empty())
            return;
        else
        {
            if (pointDistance(cloudKeyPoses3D->front(), cloudKeyPoses3D->back()) < 5.0)
                return;
        }

        // pose covariance small, no need to correct
        if (poseCovariance(3,3) < poseCovThreshold && poseCovariance(4,4) < poseCovThreshold)
            return;

        // last gps position
        static PointType lastGPSPoint;

        while (!gpsQueue.empty())
        {
            if (stamp2Sec(gpsQueue.front().header.stamp) < timeLaserInfoCur - 0.2)
            {
                // message too old
                gpsQueue.pop_front();
            }
            else if (stamp2Sec(gpsQueue.front().header.stamp) > timeLaserInfoCur + 0.2)
            {
                // message too new
                break;
            }
            else
            {
                nav_msgs::msg::Odometry thisGPS = gpsQueue.front();
                gpsQueue.pop_front();

                // GPS too noisy, skip
                float noise_x = thisGPS.pose.covariance[0];
                float noise_y = thisGPS.pose.covariance[7];
                float noise_z = thisGPS.pose.covariance[14];
                if (noise_x > gpsCovThreshold || noise_y > gpsCovThreshold)
                    continue;
                float gps_x = thisGPS.pose.pose.position.x;
                float gps_y = thisGPS.pose.pose.position.y;
                float gps_z = thisGPS.pose.pose.position.z;
                if (!useGpsElevation)
                {
                    gps_z = transformTobeMapped[5];
                    noise_z = 0.01;
                }

                // GPS not properly initialized (0,0,0)
                if (abs(gps_x) < 1e-6 && abs(gps_y) < 1e-6)
                    continue;

                // Add GPS every a few meters
                PointType curGPSPoint;
                curGPSPoint.x = gps_x;
                curGPSPoint.y = gps_y;
                curGPSPoint.z = gps_z;
                if (pointDistance(curGPSPoint, lastGPSPoint) < 5.0)
                    continue;
                else
                    lastGPSPoint = curGPSPoint;

                gtsam::Vector Vector3(3);
                Vector3 << max(noise_x, 1.0f), max(noise_y, 1.0f), max(noise_z, 1.0f);
                noiseModel::Diagonal::shared_ptr gps_noise = noiseModel::Diagonal::Variances(Vector3);
                gtsam::GPSFactor gps_factor(cloudKeyPoses3D->size(), gtsam::Point3(gps_x, gps_y, gps_z), gps_noise);
                gtSAMgraph.add(gps_factor);

                aLoopGpsIsClosed = true;
                break;
            }
        }
    }

    /**
     * 4. 添加回环因子（回环信息由独立线程提供）
     *   1）回环队列为空，直接返回
     *   2）遍历回环关系，将所有回环关系加入因子图
     *   3）清空回环关系
     *   4）设置aLoopGpsIsClosed标志位为true，后面因子图优化时会多次迭代且更新所有历史关键帧位姿
    */
    void addLoopFactor()
    {
        if (loopIndexQueue.empty())
            return;

        for (int i = 0; i < (int)loopIndexQueue.size(); ++i)
        {
            int indexFrom = loopIndexQueue[i].first;
            int indexTo = loopIndexQueue[i].second;
            gtsam::Pose3 poseBetween = loopPoseQueue[i];
            gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue[i];
            gtSAMgraph.add(BetweenFactor<Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));
        }

        loopIndexQueue.clear();
        loopPoseQueue.clear();
        loopNoiseQueue.clear();
        aLoopGpsIsClosed = true;
    }

    /**
     * 添加因子并执行图优化，更新当前位姿
     * 1. 只有在当前帧距离上一帧足够远（距离、角度）时，才纳入为关键帧，并加入因子图
     * 2. 添加激光里程计因子
     *      1）若是第一帧，则构建prior因子，赋予较大的方差
     *      2）后续帧，根据当前的位姿估计，以及上一个关键帧的位姿，计算位姿增量，添加间隔因子（BetweenFactor);
     *          同时，将当前帧当前的位姿估计作为因子图当前变量的初始值。
     * 3. 添加GPS因子
     *      1）GPS队列为空，或者关键帧序列为空，或者前后关键帧距离小于5m，或者位姿协方差较小，直接返回，认为不需要加入GPS校正
     *      2）从GPS队列中找到与当前帧时间最接近的GPS数据
     *      3）GPS数据方差大于阈值或者与上一次采用的GPS位置小于5m，直接返回
     *      4）从GPS数据中提取x,y,z和协方差，构建GPS因子加入因子图。其中的z坐标可以设置为不使用GPS的输出（GPS的z坐标较为不准）
     *      5）设置aLoopGpsIsClosed标志位为true，后面因子图优化时会多次迭代且更新所有历史关键帧位姿
     * 4. 添加回环因子（回环信息由独立线程提供）
     *      1）回环队列为空，直接返回
     *      2）遍历回环关系，将所有回环关系加入因子图
     *      3）清空回环关系
     *      4）设置aLoopGpsIsClosed标志位为true，后面因子图优化时会多次迭代且更新所有历史关键帧位姿
     * 5. 因子图优化
     *      1）将当前因子图加入优化器
     *      2）对优化器执行一次迭代更新
     *      3）如果aLoopGpsIsClosed为真，额外执行5次优化器迭代
     *      4）清空因子图和初始值（ISAM优化器已经记录这些信息）
     *      5）将当前帧位姿加入关键帧队列
     *      6）将优化后的结果更新为当前位姿
     *      7）保存当前帧特征点到特征点集合
     *      8）当当前帧位姿更新到轨迹缓存变量
    */
    void saveKeyFramesAndFactor()
    {
        // 是否将当前帧采纳为关键帧
        // 如果距离（小于1米）和角度同时不符合要求，不采纳为关键帧
        if (saveFrame() == false)
            return;

        // 添加激光里程计因子
        addOdomFactor();

        // 添加GPS因子
        addGPSFactor();

        // 添加回环因子
        addLoopFactor();

        // cout << "****************************************************" << endl;
        // gtSAMgraph.print("GTSAM Graph:\n");

        // 迭代一次优化器
        isam->update(gtSAMgraph, initialEstimate);
        isam->update();

        // 如果当前帧有新的GPS因子或者回环因子加入，执行多次迭代更新，且后面会更新所有历史帧位姿
        if (aLoopGpsIsClosed == true)
        {
            isam->update();
            isam->update();
            isam->update();
            isam->update();
            isam->update();
        }

        // 清空因子图和初始值（标准做法），因子已经加入了优化器
        gtSAMgraph.resize(0);
        initialEstimate.clear();

        // 从优化器拿出当前帧的位姿存入关键帧位姿序列
        PointType thisPose3D;
        PointTypePose thisPose6D;
        Pose3 latestEstimate;

        // 从优化器中拿出最近一帧的优化结果
        isamCurrentEstimate = isam->calculateEstimate();
        latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size()-1);
        // cout << "****************************************************" << endl;
        // isamCurrentEstimate.print("Current estimate: ");

        // 将当前帧经过优化的结果存入关键帧位姿序列（3D/6D）
        thisPose3D.x = latestEstimate.translation().x();
        thisPose3D.y = latestEstimate.translation().y();
        thisPose3D.z = latestEstimate.translation().z();
        thisPose3D.intensity = cloudKeyPoses3D->size(); // this can be used as index
        cloudKeyPoses3D->push_back(thisPose3D);

        thisPose6D.x = thisPose3D.x;
        thisPose6D.y = thisPose3D.y;
        thisPose6D.z = thisPose3D.z;
        thisPose6D.intensity = thisPose3D.intensity ; // this can be used as index
        thisPose6D.roll  = latestEstimate.rotation().roll();
        thisPose6D.pitch = latestEstimate.rotation().pitch();
        thisPose6D.yaw   = latestEstimate.rotation().yaw();
        thisPose6D.time = timeLaserInfoCur;
        cloudKeyPoses6D->push_back(thisPose6D);

        // cout << "****************************************************" << endl;
        // cout << "Pose covariance:" << endl;
        // cout << isam->marginalCovariance(isamCurrentEstimate.size()-1) << endl << endl;
        poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size()-1);

        // 将当前帧经过优化的结果更新到全局缓存变量
        transformTobeMapped[0] = latestEstimate.rotation().roll();
        transformTobeMapped[1] = latestEstimate.rotation().pitch();
        transformTobeMapped[2] = latestEstimate.rotation().yaw();
        transformTobeMapped[3] = latestEstimate.translation().x();
        transformTobeMapped[4] = latestEstimate.translation().y();
        transformTobeMapped[5] = latestEstimate.translation().z();

        // 将当前帧的角点、平面点点云存入关键帧点云缓存队列
        pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
        pcl::copyPointCloud(*laserCloudCornerCurDS,  *thisCornerKeyFrame);
        pcl::copyPointCloud(*laserCloudSurfCurDS,    *thisSurfKeyFrame);
        // save key frame cloud
        cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
        surfCloudKeyFrames.push_back(thisSurfKeyFrame);

        // 将当前帧的位姿更新到全局轨迹
        updatePath(thisPose6D);
    }

    /**
     * 更新所有历史关键帧位姿
     * 1. 只当aLoopGpsIsClosed标志位为真时才执行历史关键帧位姿更新
     * 2. 从因子图优化器中拿出所有关键帧的位姿（优化结果）
     * 3. 清空全局路径变量，替换成当前的关键帧位姿序列
     * 4. 将优化后的位姿更新为当前的位姿
    */
    void correctPoses()
    {
        if (cloudKeyPoses3D->points.empty())
            return;

        // 这个标志位在新的回环因子加入或者GPS因子加入会被置为True
        if (aLoopGpsIsClosed == true)
        {
            // laserCloudMapContainer缓存的是转换到map坐标系的点云
            // 在更新历史轨迹之后需要清空
            laserCloudMapContainer.clear();

            // 把缓存的轨迹清空
            globalPath.poses.clear();
            // 从优化器中拿出历史所有时刻的位姿并更新到位姿序列
            int numPoses = isamCurrentEstimate.size();
            for (int i = 0; i < numPoses; ++i)
            {
                cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<Pose3>(i).translation().x();
                cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<Pose3>(i).translation().y();
                cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<Pose3>(i).translation().z();

                cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
                cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
                cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
                cloudKeyPoses6D->points[i].roll  = isamCurrentEstimate.at<Pose3>(i).rotation().roll();
                cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<Pose3>(i).rotation().pitch();
                cloudKeyPoses6D->points[i].yaw   = isamCurrentEstimate.at<Pose3>(i).rotation().yaw();

                updatePath(cloudKeyPoses6D->points[i]);
            }

            aLoopGpsIsClosed = false;
        }
    }

    /**
     * @brief 将传入的位姿加入轨迹
     * 
     * @param pose_in 当前关键帧的6D位姿
     * 
    */
    void updatePath(const PointTypePose& pose_in)
    {
        geometry_msgs::msg::PoseStamped pose_stamped;
        pose_stamped.header.stamp = rclcpp::Time(pose_in.time * 1e9);
        pose_stamped.header.frame_id = odomFrame;
        pose_stamped.pose.position.x = pose_in.x;
        pose_stamped.pose.position.y = pose_in.y;
        pose_stamped.pose.position.z = pose_in.z;
        tf2::Quaternion q;
        q.setRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
        pose_stamped.pose.orientation.x = q.x();
        pose_stamped.pose.orientation.y = q.y();
        pose_stamped.pose.orientation.z = q.z();
        pose_stamped.pose.orientation.w = q.w();

        globalPath.poses.push_back(pose_stamped);
    }

    /**
     * 发布激光里程计
     * 1. 发布当前帧位姿（激光里程计）
     * 2. 发布TF坐标系，从odom坐标系到雷达坐标系的变换
    */
    void publishOdometry()
    {
        // 发布全局最优的激光里程计结果（mapping/odometry)
        nav_msgs::msg::Odometry laserOdometryROS;
        laserOdometryROS.header.stamp = timeLaserInfoStamp;
        laserOdometryROS.header.frame_id = odomFrame;
        laserOdometryROS.child_frame_id = imuFrame;     // twist所在坐标系，这里是IMU坐标系
        laserOdometryROS.pose.pose.position.x = transformTobeMapped[3];
        laserOdometryROS.pose.pose.position.y = transformTobeMapped[4];
        laserOdometryROS.pose.pose.position.z = transformTobeMapped[5];
        tf2::Quaternion quat_tf;
        quat_tf.setRPY(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        geometry_msgs::msg::Quaternion quat_msg;
        tf2::convert(quat_tf, quat_msg);
        laserOdometryROS.pose.pose.orientation = quat_msg;
        if (isDegenerate)
            laserOdometryROS.pose.covariance[0] = 1;
        else
            laserOdometryROS.pose.covariance[0] = 0;
        pubLaserOdometryGlobal->publish(laserOdometryROS);

        // 发布光滑的激光里程计结果（mapping/odometry_incremental）
        /**
         * mapping/odometry_incremental里程计是只使用了点云匹配而没有使用因子图优化的里程计
         * liosam作者TixiaoShan在github回复中（https://github.com/TixiaoShan/LIO-SAM/issues/92）提到了这一点
         * 下面这部分计算incremental里程计中，incrementalOdometryAffineFront是上一帧经过因子图优化后的结果，
         * incrementalOdometryAffineBack是在点云匹配之后、因子图优化之前的缓存结果。
         * 因此，odometry_incremental是间接使用了因子图优化，相比odometry应该有一定延迟和平滑。但是根据实验的结果
         * 来看，似乎差别不大，但是为了体现作者的工作和思考，下面这部分代码依旧保留。
        */
        static bool lastIncreOdomPubFlag = false;
        static nav_msgs::msg::Odometry laserOdomIncremental; // incremental odometry msg
        static Eigen::Affine3f increOdomAffine; // incremental odometry in affine
        if (lastIncreOdomPubFlag == false)
        {
            lastIncreOdomPubFlag = true;
            laserOdomIncremental = laserOdometryROS;
            increOdomAffine = trans2Affine3f(transformTobeMapped);
        } else {
            /**
             * incrementalOdometryAffineBack在执行完scan2MapOptimization后就被记录下来
             * 也就是说，incremental的激光里程计没有使用到因子图优化的结果，因此更加光滑，没有跳变，可以给IMU预积分模块使用
            */
            Eigen::Affine3f affineIncre = incrementalOdometryAffineFront.inverse() * incrementalOdometryAffineBack;
            increOdomAffine = increOdomAffine * affineIncre;
            float x, y, z, roll, pitch, yaw;
            pcl::getTranslationAndEulerAngles (increOdomAffine, x, y, z, roll, pitch, yaw);
            if (cloudInfo.imu_available == true)
            {
                if (std::abs(cloudInfo.imu_pitch_init) < 1.4)
                {
                    double imuWeight = 0.1;
                    tf2::Quaternion imuQuaternion;
                    tf2::Quaternion transformQuaternion;
                    double rollMid, pitchMid, yawMid;

                    // slerp roll
                    transformQuaternion.setRPY(roll, 0, 0);
                    imuQuaternion.setRPY(cloudInfo.imu_roll_init, 0, 0);
                    tf2::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                    roll = rollMid;

                    // slerp pitch
                    transformQuaternion.setRPY(0, pitch, 0);
                    imuQuaternion.setRPY(0, cloudInfo.imu_pitch_init, 0);
                    tf2::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                    pitch = pitchMid;
                }
            }
            laserOdomIncremental.header.stamp = timeLaserInfoStamp;
            laserOdomIncremental.header.frame_id = odomFrame;
            laserOdomIncremental.child_frame_id = imuFrame;
            laserOdomIncremental.pose.pose.position.x = x;
            laserOdomIncremental.pose.pose.position.y = y;
            laserOdomIncremental.pose.pose.position.z = z;
            tf2::Quaternion quat_tf;
            quat_tf.setRPY(roll, pitch, yaw);
            geometry_msgs::msg::Quaternion quat_msg;
            tf2::convert(quat_tf, quat_msg);
            laserOdomIncremental.pose.pose.orientation = quat_msg;
            if (isDegenerate)
                laserOdomIncremental.pose.covariance[0] = 1;
            else
                laserOdomIncremental.pose.covariance[0] = 0;
        }
        // pubLaserOdometryGlobal->publish(laserOdomIncremental);
    }

    /**
     * 发布点云及轨迹信息
     * 1. 发布当前帧转换到Odom坐标系下的点云
     * 2. 发布关键帧轨迹
    */
    void publishFrames()
    {
        if (cloudKeyPoses3D->points.empty())
            return;
        
        // publish registered key frame
        if (pubRegisteredCurCloud->get_subscription_count() != 0)
        {
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            *cloudOut += *transformPointCloud(laserCloudCornerCurDS,  &thisPose6D);
            *cloudOut += *transformPointCloud(laserCloudSurfCurDS,    &thisPose6D);
            publishCloud(pubRegisteredCurCloud, cloudOut, timeLaserInfoStamp, odomFrame);
        }

        // publish path
        if (pubPath->get_subscription_count() != 0)
        {
            globalPath.header.stamp = timeLaserInfoStamp;
            globalPath.header.frame_id = odomFrame;
            pubPath->publish(globalPath);
        }
    }
};


int main(int argc, char** argv)
{   
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);
    rclcpp::executors::SingleThreadedExecutor exec;

    auto MO = std::make_shared<mapOptimization>(options);
    exec.add_node(MO);

    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;32m----> Map Optimization Started.\033[0m");

    // 回环检测独立线程
    std::thread loopthread(&mapOptimization::loopClosureThread, MO);

    // 全局地图可视化独立线程
    std::thread visualizeMapThread(&mapOptimization::visualizeGlobalMapThread, MO);

    exec.spin();

    rclcpp::shutdown();

    loopthread.join();
    visualizeMapThread.join();

    return 0;
}
