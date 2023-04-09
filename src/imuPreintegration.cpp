/****************************************************************************
Project: 简化版LIO-SAM
Github: https://github.com/zeal-github/Simple-LIO-SAM
Author: zeal
EMail: 1156478780@qq.com
Date: 2023-02-14
-------------------------------------------------------
*imuPreintegration*

功能：
+ 主要功能是订阅激光里程计（低频里程计，来自mapOptimization)，对两帧激光里程计中间过程使用IMU信号做积分，得到高频的IMU里程计（与IMU同频）

功能要点：
+ IMU信号积分使用gtsam的IMU预积分模块。里面共使用了两个队列，imuQueImu和imuQueOpt，以及两个预积分器imuPreintegratorImu和imuPreintegratorOpt;
imuQueOpt和imuPreintegratorOpt主要是根据历史信息计算IMU数据bias给真正的IMU里程计预积分器使用。imuQueImu和imuPreintegratorImu是真正用来做IMU里程计的优化。
+ IMU里程计主要是在imageProjection中被塞入cloudInfo数据结构，被当作每一帧雷达点云的初始估计位姿
+ 模块中有两个handler，分别处理雷达里程计和IMU原始数据。雷达里程计的handler中，主要是将新到来的雷达里程计之前的IMU做积分，得出bias；
IMU的handler中主要是对当前里程计之后、下一时刻雷达里程计到来之前的时刻对IMU数据积分，并发布IMU里程计。

订阅：
1. IMU原始数据
2. Lidar里程计（来自mapOptimization）

发布：
1. IMU里程计（/lio_sam/imu/odometry)

流程：
1. 订阅雷达里程计
*  1.1 如果系统没有初始化，则初始化系统，包括因子图、优化器、预积分器等
*  1.2 每100帧雷达里程计之后重置优化器。清空因子图优化器，用优化出的结果作为先验
*  1.3 将imuQueOpt队列中，所有早于当前雷达里程计的数据进行积分，获取最新的IMU bias
*  1.4 使用预积分器构造ImuFactor，并加入因子图
*  1.5 添加Imu的BetweenFactor（偏差的相对差别）
*  1.6 将雷达里程计平移对齐到IMU（只做平移）
*  1.7 构建雷达里程计因子，并加入因子图
*  1.8 使用IMU预积分器的预测作为当前因子图的变量初始值
*  1.9 将新的因子图和变量初始值加入优化器，并更新
*  1.10 清空因子图和变量初始值缓存，为下一次加入因子准备
*  1.11 从因子图中获取当前时刻优化后的各个变量
*  1.12 重置预积分器
*  1.13 检查优化结果，优化结果有问题时重置优化器
*  1.14 将偏差优化器的结果传递到里程计优化器
*  1.15 对里程计队列中剩余的数据进行积分

2. 订阅IMU原始数据
*  2.1 加锁，对新来的IMU数据放入两个队列（imuQueOpt,imuQueImu)
*  2.2 对IMU数据直接使用imuIntegratorImu_进行积分，并使用上一时刻的状态预测当前状态（积分结果）
*  2.3 构建发布数据包，发布IMU里程计数据

备注：
1. 关于递增式因子图用法可以看gtsam仓库：gtsam/examples/VisualSAM2Example.cpp
2. 关于IMU预积分的用法可以看gtsam仓库:gtsam/examples/ImuFactorExample.cpp


*******************************************************************************/
#include "spl_lio_sam/utility.hpp"

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
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

// 生成ImuFactor中Key的快速符号，这里使用的ImuFactor是5-way因子，
// 连接上一时刻的pose、velocity,当前时刻的pose、velocity和bias
using gtsam::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using gtsam::symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)

class IMUPreintegration : public ParamServer
{
public:

    std::mutex mtx;

    // 订阅IMU原始数据
    rclcpp::CallbackGroup::SharedPtr callbackGroupImu;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr subImu;
    // 订阅Lidar里程计（来自mapOptimization)
    rclcpp::CallbackGroup::SharedPtr callbackGroupOdom;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subOdometry;
    // 发布IMU里程计
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubImuOdometry;

    // 标志系统初始化，主要是用来初始化gtsam
    bool systemInitialized = false;

    // priorXXXNoise是因子图中添加第一个因子时指定的噪声
    gtsam::noiseModel::Diagonal::shared_ptr priorPoseNoise;
    gtsam::noiseModel::Diagonal::shared_ptr priorVelNoise;
    gtsam::noiseModel::Diagonal::shared_ptr priorBiasNoise;
    // correctionNoise是小噪声，correctionNoise2是大噪声。在雷达里程计方差大的情况下使用大噪声，反之使用小噪声
    gtsam::noiseModel::Diagonal::shared_ptr correctionNoise;
    gtsam::noiseModel::Diagonal::shared_ptr correctionNoise2;
    // IMU的Bias噪声模型
    gtsam::Vector noiseModelBetweenBias;

    // imuIntegratorOpt_负责预积分两帧激光里程计之间的IMU数据，计算IMU的bias
    gtsam::PreintegratedImuMeasurements *imuIntegratorOpt_;
    // imuIntegratorImu_根据最新的激光里程计，以及后续到到的IMU数据，预测从当前激光里程计往后的位姿（IMU里程计）
    gtsam::PreintegratedImuMeasurements *imuIntegratorImu_;

    // imuQueOpt用来给imuIntegratorOpt_提供数据来源。将当前激光里程计之前的数据统统做积分，优化出bias
    std::deque<sensor_msgs::msg::Imu> imuQueOpt;
    // imuQueImu用来给imuIntegratorImu_提供数据来源。优化当前激光里程计之后，下一帧激光里程计到来之前的位姿
    std::deque<sensor_msgs::msg::Imu> imuQueImu;

    // 因子图优化过程中的状态变量
    gtsam::Pose3 prevPose_;
    gtsam::Vector3 prevVel_;
    gtsam::NavState prevState_;
    gtsam::imuBias::ConstantBias prevBias_;

    gtsam::NavState prevStateOdom;
    gtsam::imuBias::ConstantBias prevBiasOdom;

    // 第一帧初始化标签
    bool doneFirstOpt = false;
    // 上一个IMU数据的时间(在IMU的handler中使用)
    double lastImuT_imu = -1;
    // 上一个IMU数据的时间（在雷达里程计的handler中使用）
    double lastImuT_opt = -1;

    // 因子图优化器
    gtsam::ISAM2 optimizer;
    // 非线性因子图
    gtsam::NonlinearFactorGraph graphFactors;
    // 因子图变量的值
    gtsam::Values graphValues;

    // 在做IMU数据和雷达里程计同步过程中的时间间隔
    const double sync_t = 0;

    int key = 1;

    // imu坐标系和lidar坐标系的平移转换关系
    // 注意，由于在处理Imu的数据时，已经先通过utility.hpp中的imuConverter把imu和lidar的坐标系对齐
    // 这里的平移转换是用来对最后的IMU积分出的姿态转换到lidar坐标系下
    gtsam::Pose3 imu2Lidar = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(-extTrans.x(), -extTrans.y(), -extTrans.z()));
    gtsam::Pose3 lidar2Imu = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(extTrans.x(), extTrans.y(), extTrans.z()));

    IMUPreintegration(const rclcpp::NodeOptions & options) :
            ParamServer("lio_sam_imu_preintegration", options)
    {

        // 订阅原始IMU数据，MutuallyExclusive指明使用单线程执行器执行回调函数
        callbackGroupImu = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        auto imuOpt = rclcpp::SubscriptionOptions();
        imuOpt.callback_group = callbackGroupImu;
        subImu = create_subscription<sensor_msgs::msg::Imu>(
            imuTopic, qos_imu,
            std::bind(&IMUPreintegration::imuHandler, this, std::placeholders::_1),
            imuOpt);

        // 订阅雷达里程计数据，MutuallyExclusive指明使用单线程执行器执行回调函数
        callbackGroupOdom = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        auto odomOpt = rclcpp::SubscriptionOptions();
        odomOpt.callback_group = callbackGroupOdom;        
        subOdometry = create_subscription<nav_msgs::msg::Odometry>(
            lidarOdomTopic, qos,
            std::bind(&IMUPreintegration::odometryHandler, this, std::placeholders::_1),
            odomOpt);

        // 发布IMU里程计数据
        pubImuOdometry = create_publisher<nav_msgs::msg::Odometry>(imuOdomTopic, qos_imu);

        // IMU预积分器的初始参数
        // 这里对加速度、角速度、积分结果中，三个轴的初始协方差都设为一样
        // 同时将imu积分器的初始偏差设为0
        std::shared_ptr<gtsam::PreintegrationParams> p = gtsam::PreintegrationParams::MakeSharedU(imuGravity);
        p->accelerometerCovariance  = gtsam::Matrix33::Identity(3,3) * pow(imuAccNoise, 2); // acc white noise in continuous
        p->gyroscopeCovariance      = gtsam::Matrix33::Identity(3,3) * pow(imuGyrNoise, 2); // gyro white noise in continuous
        p->integrationCovariance    = gtsam::Matrix33::Identity(3,3) * pow(1e-4, 2); // error committed in integrating position from velocities
        gtsam::imuBias::ConstantBias prior_imu_bias((gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished());; // assume zero initial bias

        // 因子图中先验变量的噪声模型，以及IMU偏差的噪声模型（额外标定）
        priorPoseNoise  = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2).finished()); // rad,rad,rad,m, m, m
        priorVelNoise   = gtsam::noiseModel::Isotropic::Sigma(3, 1e4); // m/s
        priorBiasNoise  = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3); // 1e-2 ~ 1e-3 seems to be good
        correctionNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1).finished()); // rad,rad,rad,m, m, m
        correctionNoise2 = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1, 1, 1, 1, 1, 1).finished()); // rad,rad,rad,m, m, m
        noiseModelBetweenBias = (gtsam::Vector(6) << imuAccBiasN, imuAccBiasN, imuAccBiasN, imuGyrBiasN, imuGyrBiasN, imuGyrBiasN).finished();
        
        // IMU预积分器，在IMU里程计中使用
        imuIntegratorImu_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias);
        // IMU偏差预积分器，在雷达里程计线程中使用
        imuIntegratorOpt_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias);     
    }

    /**
     * @brief 重置优化器和因子图。在第一帧雷达里程计到来的时候重置。
     * 每100帧激光里程计也重置一次参数
    */
    void resetOptimization()
    {
        // 重置ISAM2优化器
        gtsam::ISAM2Params optParameters;
        optParameters.relinearizeThreshold = 0.1;
        optParameters.relinearizeSkip = 1;
        optimizer = gtsam::ISAM2(optParameters);

        // 重置因子图
        gtsam::NonlinearFactorGraph newGraphFactors;
        graphFactors = newGraphFactors;

        // 重置图变量值
        gtsam::Values NewGraphValues;
        graphValues = NewGraphValues;
    }

    /**
     * @brief 重置中间变量。在优化失败的时候会进行一次重置，让整个系统重新初始化。
     * 这一点也可以体现作者很优秀的工程功底阿。
    */
    void resetParams()
    {
        lastImuT_imu = -1;
        doneFirstOpt = false;
        systemInitialized = false;
    }

    /**
     * @brief 
     * 1.雷达里程计的回调函数
     *  1.1 如果系统没有初始化，则初始化系统，包括因子图、优化器、预积分器等
     *  1.2 每100帧雷达里程计之后重置优化器。清空因子图优化器，用优化出的结果作为先验
     *  1.3 将imuQueOpt队列中，所有早于当前雷达里程计的数据进行积分，获取最新的IMU bias
     *  1.4 使用预积分器构造ImuFactor，并加入因子图
     *  1.5 添加Imu的BetweenFactor（偏差的相对差别）
     *  1.6 将雷达里程计平移对齐到IMU（只做平移）
     *  1.7 构建雷达里程计因子，并加入因子图
     *  1.8 使用IMU预积分器的预测作为当前因子图的变量初始值
     *  1.9 将新的因子图和变量初始值加入优化器，并更新
     *  1.10 清空因子图和变量初始值缓存，为下一次加入因子准备
     *  1.11 从因子图中获取当前时刻优化后的各个变量
     *  1.12 重置预积分器
     *  1.13 检查优化结果，优化结果有问题时重置优化器
     *  1.14 将偏差优化器的结果传递到里程计优化器
     *  1.15 对里程计对立中剩余的数据进行积分
    */
    void odometryHandler(const nav_msgs::msg::Odometry::SharedPtr odomMsg)
    {
        // 这里加锁主要是防止imuQueOpt队列被imuHandler线程修改
        std::lock_guard<std::mutex> lock(mtx);

        RCLCPP_DEBUG(get_logger(), "Receive lidar odom in %f", stamp2Sec(odomMsg->header.stamp));

        // 当前雷达里程计时间
        double currentLidarOdomTime = stamp2Sec(odomMsg->header.stamp);

        // 确保imuQueOpt有值
        if (imuQueOpt.empty())
            return;

        // 提取lidar里程计的位姿
        float p_x = odomMsg->pose.pose.position.x;
        float p_y = odomMsg->pose.pose.position.y;
        float p_z = odomMsg->pose.pose.position.z;
        float r_x = odomMsg->pose.pose.orientation.x;
        float r_y = odomMsg->pose.pose.orientation.y;
        float r_z = odomMsg->pose.pose.orientation.z;
        float r_w = odomMsg->pose.pose.orientation.w;
        gtsam::Pose3 lidarPose = gtsam::Pose3(gtsam::Rot3::Quaternion(r_w, r_x, r_y, r_z), gtsam::Point3(p_x, p_y, p_z));
        // 检查雷达里程计是否准确。这个covariance的第一位在mapOptimization中被置位
        // 可以理解为如果degenerate为true，则雷达里程计的不准确，使用的大的噪声模型
        bool degenerate = (int)odomMsg->pose.covariance[0] == 1 ? true : false;


        // 初始化系统。在第一帧或者优化出错的情况下重新初始化
        if (systemInitialized == false)
        {
            // 重置优化器和因子图
            resetOptimization();

            // 同步IMU队列和雷达里程计时间。使队列中第一帧与雷达里程计时间戳同步
            while (!imuQueOpt.empty())
            {
                if (stamp2Sec(imuQueOpt.front().header.stamp) < currentLidarOdomTime - sync_t)
                {
                    lastImuT_opt = stamp2Sec(imuQueOpt.front().header.stamp);
                    imuQueOpt.pop_front();
                }
                else
                    break;
            }
            // 初始化因子图的prior状态
            // 将雷达里程计位姿平移到IMU坐标系，只是做了平移
            prevPose_ = lidarPose.compose(lidar2Imu);
            // 加入priorPose因子
            gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, priorPoseNoise);
            graphFactors.add(priorPose);
            // 加入priorVel因子
            prevVel_ = gtsam::Vector3(0, 0, 0);
            gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, priorVelNoise);
            graphFactors.add(priorVel);
            // 加入priorBias因子
            prevBias_ = gtsam::imuBias::ConstantBias();
            gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, priorBiasNoise);
            graphFactors.add(priorBias);

            // 将初始状态设置为因子图变量的初始值
            graphValues.insert(X(0), prevPose_);
            graphValues.insert(V(0), prevVel_);
            graphValues.insert(B(0), prevBias_);

            // 加入新的因子，并使用优化器对因子图做一次优化
            // 可以多次调用update()对当前的因子图进行多次更新
            optimizer.update(graphFactors, graphValues);

            // 打印因子图和优化器
            // graphFactors.print("GTSAM GraphFactors in systemInitialized:\n");
            // graphValues.print("GTSAM GraphValues in systemInitialized:\n");
            // optimizer.print("GTSAM optimizer in systemInitialized:\n");

            // 清空因子图。因子已经已经被记录到优化器中。这是在gtsam的官方example的递增式优化流程中的用法
            // example:gtsam/examples/VisualSAM2Example.cpp
            graphFactors.resize(0);
            graphValues.clear();

            // 重置两个预积分器。
            imuIntegratorImu_->resetIntegrationAndSetBias(prevBias_);
            imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);
            
            // 重置起始帧的key值
            key = 1;
            // 设置系统已经初始化标志位
            systemInitialized = true;
            return;
        }


        // 每100帧lidar里程计重置一下优化器
        // 删除旧因子图，加快优化速度
        if (key == 100)
        {
            // 从优化器中先缓存一下当前优化出来的变量的方差
            gtsam::noiseModel::Gaussian::shared_ptr updatedPoseNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(X(key-1)));
            gtsam::noiseModel::Gaussian::shared_ptr updatedVelNoise  = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(V(key-1)));
            gtsam::noiseModel::Gaussian::shared_ptr updatedBiasNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(B(key-1)));
            // 重置优化器和因子图
            resetOptimization();
            // 把上一次优化出的位姿作为重新初始化的priorPose
            gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, updatedPoseNoise);
            graphFactors.add(priorPose);
            // 把上一次优化出的速度作为重新初始化的priorVel
            gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, updatedVelNoise);
            graphFactors.add(priorVel);
            // 把上一次优化出的bias作为重新初始化的priorBias
            gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, updatedBiasNoise);
            graphFactors.add(priorBias);
            // 将prior状态设置成初始估计值
            graphValues.insert(X(0), prevPose_);
            graphValues.insert(V(0), prevVel_);
            graphValues.insert(B(0), prevBias_);
            // 进行一次迭代优化
            optimizer.update(graphFactors, graphValues);
            // 清空因子图和值（已经被保存进优化器里了）
            graphFactors.resize(0);
            graphValues.clear();

            // 重置因子索引
            key = 1;
        }

        // 将imuQueOpt队列中，所有早于当前雷达里程计的数据进行积分，获取最新的IMU bias
        while (!imuQueOpt.empty())
        {
            // pop and integrate imu data that is between two optimizations
            sensor_msgs::msg::Imu *thisImu = &imuQueOpt.front();
            double imuTime = stamp2Sec(thisImu->header.stamp);
            RCLCPP_DEBUG(get_logger(), "lidar_time = %f, imu_time = %f", currentLidarOdomTime, imuTime);
            if (imuTime < currentLidarOdomTime - sync_t)
            {
                double dt = (lastImuT_opt < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_opt);
                
                if(dt<=0){ 
                    imuQueOpt.pop_front();
                    continue;
                }
                // 这里是实际做积分的地方
                imuIntegratorOpt_->integrateMeasurement(
                        gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                        gtsam::Vector3(thisImu->angular_velocity.x,    thisImu->angular_velocity.y,    thisImu->angular_velocity.z), dt);
                
                lastImuT_opt = imuTime;
                imuQueOpt.pop_front();
            }
            // imu数据时间已经大于当前的雷达里程计，跳出积分
            else
                break;
        }

        /**
         * 下面这一大部分是使用IMU预积分的结果加入因子图中，同时，加入雷达里程计因子，
         * 使用优化器优化因子图得出当前IMU对应的Bias。
         * 流程：
         *  1. 使用预积分器构造ImuFactor，并加入因子图
         *  2. 添加Imu的BetweenFactor（偏差的相对差别）
         *  3. 将雷达里程计平移对齐到IMU（只做平移）
         *  4. 构建雷达里程计因子，并加入因子图
         *  5. 使用IMU预积分器的预测作为当前因子图的变量初始值
         *  6. 将新的因子图和变量初始值加入优化器，并更新
         *  7. 清空因子图和变量初始值缓存，为下一次加入因子准备
         *  8. 从因子图中获取当前时刻优化后的各个变量
         *  9. 重置预积分器
         *  10. 检查优化结果，优化结果有问题时重置优化器
        */
        // 使用IMU预积分的结果构建IMU因子，并加入因子图中
        const gtsam::PreintegratedImuMeasurements& preint_imu = dynamic_cast<const gtsam::PreintegratedImuMeasurements&>(*imuIntegratorOpt_);
        gtsam::ImuFactor imu_factor(X(key - 1), V(key - 1), X(key), V(key), B(key - 1), preint_imu);
        graphFactors.add(imu_factor);
        // 添加IMU的bias的BetweenFactor
        // 这里的deltaTij()获取的是IMU预积分器从上一次积分输入到当前输入的时间
        // noiseModelBetweenBias是我们提前标定的IMU的偏差的噪声
        // 两者的乘积就等于这两个时刻之间IMU偏差的漂移
        graphFactors.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(B(key - 1), B(key), gtsam::imuBias::ConstantBias(),
                         gtsam::noiseModel::Diagonal::Sigmas(sqrt(imuIntegratorOpt_->deltaTij()) * noiseModelBetweenBias)));

        // 将雷达的位姿平移到IMU坐标系（只做平移）
        gtsam::Pose3 curPose = lidarPose.compose(lidar2Imu);
        // 添加Pose因子。这里的degenerate调整噪声的大小。correctionNoise2是大噪声
        gtsam::PriorFactor<gtsam::Pose3> pose_factor(X(key), curPose, degenerate ? correctionNoise2 : correctionNoise);
        graphFactors.add(pose_factor);
        // imuIntegratorOpt_->predict输入之前时刻的状态和偏差，预测当前时刻的状态
        gtsam::NavState propState_ = imuIntegratorOpt_->predict(prevState_, prevBias_);
        // 将IMU预积分的结果作为当前时刻因子图变量的初值
        graphValues.insert(X(key), propState_.pose());
        graphValues.insert(V(key), propState_.v());
        graphValues.insert(B(key), prevBias_);
        // 打印因子图
        // graphFactors.print("GTSAM GraphFactors in update:\n");
        // graphValues.print("GTSAM GraphValues in update:\n");
        // 更新一次优化器
        optimizer.update(graphFactors, graphValues);
        optimizer.update();
        // 打印更新后的优化器状态
        // optimizer.print("GTSAM optimizer in update:");

        // 清空因子图和值（已经被保存进优化器里了）
        graphFactors.resize(0);
        graphValues.clear();
        // 从优化器中获取当前经过优化后估计值
        gtsam::Values result = optimizer.calculateEstimate();
        prevPose_  = result.at<gtsam::Pose3>(X(key));
        prevVel_   = result.at<gtsam::Vector3>(V(key));
        prevState_ = gtsam::NavState(prevPose_, prevVel_);
        prevBias_  = result.at<gtsam::imuBias::ConstantBias>(B(key));
        // 重置预积分器
        imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);
        // 检查优化器优化结果，当优化出的速度过快（大于30），或者Bias过大，认为优化或者某个环节出现问题，重置优化器
        if (failureDetection(prevVel_, prevBias_))
        {
            resetParams();
            return;
        }


        /**
         * 这一部分是将偏差优化器imuIntegratorOpt_/graphFactors优化出的结果传递到IMU里程计的预积分器。
         * 让IMU里程计预积分器使用最新估计出来的Bias进行积分
         *  1. 缓存最新的状态（作为IMU里程计预积分器预测时的上一时刻状态传入），和最新的偏差（重置IMU里程计预积分器时使用）
         *  2. 同步IMU数据队列和雷达里程计时间，去除当前雷达里程计时间之前的数据
         *  3. 对IMU队列中剩余的其他数据进行积分。（这样在新的IMU到来的时候就可以直接在这个基础上进行积分）
        */
        // 缓存当前的状态和偏差，preStateOdom在IMU里程计的预积分器进行预测时需要作为上一时刻的状态传入
        prevStateOdom = prevState_;
        prevBiasOdom  = prevBias_;
        // 同样先做IMU数据队列和雷达里程计的时间同步
        double lastImuQT = -1;
        while (!imuQueImu.empty() && stamp2Sec(imuQueImu.front().header.stamp) < currentLidarOdomTime - sync_t)
        {
            lastImuQT = stamp2Sec(imuQueImu.front().header.stamp);
            imuQueImu.pop_front();
        }
        // 对当前IMU队列中的数据全部积分（从
        if (!imuQueImu.empty())
        {
            // 将偏差优化器优化出的最新偏差重置到IMU里程计预积分器
            imuIntegratorImu_->resetIntegrationAndSetBias(prevBiasOdom);
            // 将IMU队列中从当前雷达里程计之后的IMU数据进行积分。后续每一个新到来的IMU数据（在ImuHandler中处理）可以直接进行积分
            for (int i = 0; i < (int)imuQueImu.size(); ++i)
            {
                sensor_msgs::msg::Imu *thisImu = &imuQueImu[i];
                double imuTime = stamp2Sec(thisImu->header.stamp);
                double dt = (lastImuQT < 0) ? (1.0 / 500.0) :(imuTime - lastImuQT);
                if(dt<=0){ 
                    continue;
                }

                imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                                                        gtsam::Vector3(thisImu->angular_velocity.x,    thisImu->angular_velocity.y,    thisImu->angular_velocity.z), dt);
                lastImuQT = imuTime;
            }
        }

        // 对因子图的当前帧索引加1
        ++key;
        // 设置首次优化标志位为真。如果这一段代码没有先执行，imuHandler无法对新到来的IMU数据进行积分
        doneFirstOpt = true;
    }

    /**
     * @brief 检查优化器优化结果，当优化出的速度过快（大于30），或者Bias过大，认为优化或者某个环节出现问题，重置优化器
     * 
     * @param velCur 当前优化出的速度值
     * @param biasCur 当前优化出的bias
    */
    bool failureDetection(const gtsam::Vector3& velCur, const gtsam::imuBias::ConstantBias& biasCur)
    {
        Eigen::Vector3f vel(velCur.x(), velCur.y(), velCur.z());
        if (vel.norm() > 30)
        {
            RCLCPP_WARN(get_logger(), "Large velocity, reset IMU-preintegration!");
            return true;
        }

        Eigen::Vector3f ba(biasCur.accelerometer().x(), biasCur.accelerometer().y(), biasCur.accelerometer().z());
        Eigen::Vector3f bg(biasCur.gyroscope().x(), biasCur.gyroscope().y(), biasCur.gyroscope().z());
        if (ba.norm() > 1.0 || bg.norm() > 1.0)
        {
            RCLCPP_WARN(get_logger(), "Large bias, reset IMU-preintegration!");
            return true;
        }

        return false;
    }

    /**
     * 2. Imu数据处理回调函数
     *  这里是实际做实时的IMU里程计计算的地方，但由于预积分器都在odomHandler中被处理好了，因此这里可以直接对每一个新到来的IMU数据进行积分
     *  2.1 加锁，对新来的IMU数据放入两个队列（imuQueOpt,imuQueImu)
     *  2.2 对IMU数据直接使用imuIntegratorImu_进行积分，并使用上一时刻的状态预测当前状态（积分结果）
     *  2.3 构建发布数据包，发布IMU里程计数据
     * 注意：
     *  1. 这个回调函数和odomHandler被同一把锁保护，保证了对预积分器做重置时不会出现数据错乱问题。
     *  2. 所有对积分的处理都是先将旋转对齐到lidar坐标系，但是平移到IMU中心
    */
    void imuHandler(const sensor_msgs::msg::Imu::SharedPtr imu_raw)
    {
        // 对两个IMU队列加锁
        std::lock_guard<std::mutex> lock(mtx);

        // 将IMU数据旋转到lidar的旋转朝向
        sensor_msgs::msg::Imu thisImu = imuConverter(*imu_raw);

        // 缓存IMU数据
        imuQueOpt.push_back(thisImu);
        imuQueImu.push_back(thisImu);

        // 如果预积分器没有被设置好，无法进行预积分，直接返回
        if (doneFirstOpt == false)
            return;

        // 对当前的IMU数据进行积分
        double imuTime = stamp2Sec(thisImu.header.stamp);
        double dt = (lastImuT_imu < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_imu);
        if(dt<=0){
            return;
        }
        lastImuT_imu = imuTime;
        imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu.linear_acceleration.x, thisImu.linear_acceleration.y, thisImu.linear_acceleration.z),
                                                gtsam::Vector3(thisImu.angular_velocity.x,    thisImu.angular_velocity.y,    thisImu.angular_velocity.z), dt);

        // 使用上一时刻在雷达里程计到来时优化出来的状态预测当前时刻的状态
        gtsam::NavState currentState = imuIntegratorImu_->predict(prevStateOdom, prevBiasOdom);

        // 发布IMU里程计
        auto odometry = nav_msgs::msg::Odometry();
        odometry.header.stamp = thisImu.header.stamp;
        odometry.header.frame_id = odomFrame;   // pose所在坐标系，这里已经对齐到odom坐标系
        odometry.child_frame_id = imuFrame;     // twist所在坐标系，这里是IMU坐标系

        // 将IMU里程计完全对齐到雷达（剩下一个平移关系）
        gtsam::Pose3 imuPose = gtsam::Pose3(currentState.quaternion(), currentState.position());
        gtsam::Pose3 lidarPose = imuPose.compose(imu2Lidar);

        // 将预积分结果放到ROS2数据包
        odometry.pose.pose.position.x = lidarPose.translation().x();
        odometry.pose.pose.position.y = lidarPose.translation().y();
        odometry.pose.pose.position.z = lidarPose.translation().z();
        odometry.pose.pose.orientation.x = lidarPose.rotation().toQuaternion().x();
        odometry.pose.pose.orientation.y = lidarPose.rotation().toQuaternion().y();
        odometry.pose.pose.orientation.z = lidarPose.rotation().toQuaternion().z();
        odometry.pose.pose.orientation.w = lidarPose.rotation().toQuaternion().w();
        
        odometry.twist.twist.linear.x = currentState.velocity().x();
        odometry.twist.twist.linear.y = currentState.velocity().y();
        odometry.twist.twist.linear.z = currentState.velocity().z();
        odometry.twist.twist.angular.x = thisImu.angular_velocity.x + prevBiasOdom.gyroscope().x();
        odometry.twist.twist.angular.y = thisImu.angular_velocity.y + prevBiasOdom.gyroscope().y();
        odometry.twist.twist.angular.z = thisImu.angular_velocity.z + prevBiasOdom.gyroscope().z();
        pubImuOdometry->publish(odometry);
    }
};


int main(int argc, char** argv)
{   
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);
    // 使用多线程执行器执行该节点
    rclcpp::executors::MultiThreadedExecutor e;

    auto ImuP = std::make_shared<IMUPreintegration>(options);
    e.add_node(ImuP);

    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;32m----> IMU Preintegration Started.\033[0m");

    e.spin();

    rclcpp::shutdown();
    return 0;
}
