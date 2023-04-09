/****************************************************************************
Project: 简化版LIO-SAM
Github: https://github.com/zeal-github/Simple-LIO-SAM
Author: zeal
EMail: 1156478780@qq.com
Date: 2023-02-14
-------------------------------------------------------
*featureExtraction*

功能：
+ 对经过运动畸变校正之后的当前帧激光点云，计算每个点的曲率，进而提取角点、平面点（用曲率的大小进行判定）

订阅：
1. 去畸变后的点云及附属信息，包括：原始点云、去畸变点云、该帧点云的初始旋转旋转角（来自IMU原始roll、pitch、yaw）、该帧点云的初始位姿（来自IMU里程计）

发布：
1. 包含提取完特征点的点云信息包。在订阅的信息包基础上塞进去提取出的特征角点和特征平面点

流程：
1. 接收到从imageProjection中发布出的一个去畸变点云信息cloudInfo(自定义格式)
2. 对每个点计算曲率。计算时是计算周围点的平均距离用来作为曲率的替代
3. 标记遮挡点和与激光平行的点，后续这些点不能采纳为特征点
4. 特征提取。分别做角点（曲率大）和平面点（曲率小）特征点提取
5. 整合信息，发布完整数据包
*******************************************************************************/
#include "spl_lio_sam/utility.hpp"
#include "spl_lio_sam/msg/cloud_info.hpp"

// 激光点曲率
struct smoothness_t{ 
    float value;
    size_t ind;
};

// 激光点曲率排序函数，从小到大排序
struct by_value{ 
    bool operator()(smoothness_t const &left, smoothness_t const &right) { 
        return left.value < right.value;
    }
};

class FeatureExtraction : public ParamServer
{

public:

    // 订阅从imageProjection发布出来的去畸变点云信息
    rclcpp::Subscription<spl_lio_sam::msg::CloudInfo>::SharedPtr subLaserCloudInfo;

    // 发布增加了角点、平面点的点云信息包
    rclcpp::Publisher<spl_lio_sam::msg::CloudInfo>::SharedPtr pubLaserCloudInfo;

    // 当前激光帧运动畸变校正后的有效点云
    pcl::PointCloud<PointType>::Ptr extractedCloud;
    // 当前激光帧提取的角点点云
    pcl::PointCloud<PointType>::Ptr cornerCloud;
    // 当前激光帧提取的平面点点云
    pcl::PointCloud<PointType>::Ptr surfaceCloud;

    // 用来做平面特征点点云降采样的过滤器
    pcl::VoxelGrid<PointType> downSizeFilter;

    // 最终要发布的数据包
    spl_lio_sam::msg::CloudInfo cloudInfo;
    // 接收到的当前帧信息的数据头
    std_msgs::msg::Header cloudHeader;

    // 点云曲率，cloudSmoothness可以用来做排序，所以里面有另一个字段存储index
    std::vector<smoothness_t> cloudSmoothness;
    // 点云曲率，原始数据，顺序不变。长度为N_SCAN*Horizon_SCAN的数组
    float *cloudCurvature;
    // 特征提取标志，1表示遮挡、平行，或者已经进行了特征提取的点，0表示未进行特征提取
    // 这是一个N_SCAN*Horizon_SCAN长的数组cloudNeighborPicked = new int[N_SCAN*Horizon_SCAN];
    int *cloudNeighborPicked;
    // 1表示角点，-1表示平面点，0表示没有被选择为特征点，同样是一个N_SCAN*Horizon_SCAN长的数组
    int *cloudLabel;

    FeatureExtraction(const rclcpp::NodeOptions & options) :
        ParamServer("lio_sam_featureExtraction", options)
    {
        // 订阅从imageProjection发布的点云信息（运动畸变校正后）
        subLaserCloudInfo = create_subscription<spl_lio_sam::msg::CloudInfo>(
            "lio_sam/deskew/cloud_info", qos,
            std::bind(&FeatureExtraction::laserCloudInfoHandler, this, std::placeholders::_1));

        // 发布经过特征提取后的点云信息
        pubLaserCloudInfo = create_publisher<spl_lio_sam::msg::CloudInfo>(
            "lio_sam/feature/cloud_info", qos);

        // 初始化向量、数组及标志位的值
        initializationValue();
    }

    void initializationValue()
    {
        cloudSmoothness.resize(N_SCAN*Horizon_SCAN);

        // 平面特征点云降采样器，默认voxel大小是0.4米
        downSizeFilter.setLeafSize(odometrySurfLeafSize, odometrySurfLeafSize, odometrySurfLeafSize);

        extractedCloud.reset(new pcl::PointCloud<PointType>());
        cornerCloud.reset(new pcl::PointCloud<PointType>());
        surfaceCloud.reset(new pcl::PointCloud<PointType>());

        cloudCurvature = new float[N_SCAN*Horizon_SCAN];
        cloudNeighborPicked = new int[N_SCAN*Horizon_SCAN];
        cloudLabel = new int[N_SCAN*Horizon_SCAN];
    }

    /**
     * @brief
     * cloudInfo话题的回调函数，这个模块的功能都是顺序进行
     * 1. 接收到从imageProjection中发布出的一个去畸变点云信息cloudInfo(自定义格式)
     * 2. 对每个点计算曲率。计算时是计算周围点的平均距离用来作为曲率的替代
     * 3. 标记遮挡点和与激光平行的点，后续这些点不能采纳为特征点
     * 4. 特征提取。分别做角点（曲率大）和平面点（曲率小）特征点提取
     * 5. 整合信息，发布完整数据包
     * 
     * @param msgIn 从去畸变模块接受的数据包
    */
    void laserCloudInfoHandler(const spl_lio_sam::msg::CloudInfo::SharedPtr msgIn)
    {
        // 缓存全局变量，后面的函数可以直接读取cloudInfo和cloudHeader进行处理
        cloudInfo = *msgIn;
        cloudHeader = msgIn->header;
        // 把ros2 PointCloud2转成PCL格式，方便后面处理
        pcl::fromROSMsg(msgIn->cloud_deskewed, *extractedCloud);

        // 计算点云每一个点曲率
        calculateSmoothness();

        //标记遮挡点和与激光平行的点，后续这些点不能采纳为特征点
        markOccludedPoints();

        // 特征提取。分别做角点（曲率大）和平面点（曲率小）特征点提取
        extractFeatures();

        // 整合信息，发布完整数据包
        publishFeatureCloud();
    }

    /**
     * @brief 2.计算每一个点曲率
     * 对于每一个点，计算其前后共10个点与它的range差距的平方作为曲率的近似。
     * 可以认为这个差值越大，曲率越大
    */
    void calculateSmoothness()
    {
        int cloudSize = extractedCloud->points.size();
        for (int i = 5; i < cloudSize - 5; i++)
        {
            float diffRange = cloudInfo.point_range[i-5] + cloudInfo.point_range[i-4]
                            + cloudInfo.point_range[i-3] + cloudInfo.point_range[i-2]
                            + cloudInfo.point_range[i-1] - cloudInfo.point_range[i] * 10
                            + cloudInfo.point_range[i+1] + cloudInfo.point_range[i+2]
                            + cloudInfo.point_range[i+3] + cloudInfo.point_range[i+4]
                            + cloudInfo.point_range[i+5];

            cloudCurvature[i] = diffRange*diffRange;//diffX * diffX + diffY * diffY + diffZ * diffZ;

            // 清空点云的各个标志位
            cloudNeighborPicked[i] = 0;
            cloudLabel[i] = 0;
            // 将曲率和点的索引保存到cloudSmoothness变量，后面用来做排序用
            cloudSmoothness[i].value = cloudCurvature[i];
            cloudSmoothness[i].ind = i;
        }
    }


    /**
     * @brief 3. 标记遮挡点和与激光线平行的点（将cloudNeighborPicked标志位设为1）
     * 对于遮挡的物体，会呈现一条类似边缘的线。对于遮挡的点，其与左边或者右边的点会呈现断层状态。因此可以通过range的差距判断
     * 对于与激光线较为平行的点，左右几个点之间的range差距应该都会比较大
    */
    void markOccludedPoints()
    {
        int cloudSize = extractedCloud->points.size();
        // 标记被遮挡的点和与激光束平行的点
        for (int i = 5; i < cloudSize - 6; ++i)
        {
            // 标记被遮挡的点
            float depth1 = cloudInfo.point_range[i];
            float depth2 = cloudInfo.point_range[i+1];
            int columnDiff = std::abs(int(cloudInfo.point_col_ind[i+1] - cloudInfo.point_col_ind[i]));
            // 两个点的列索引相差10个像素之内，认为是同一块区域
            if (columnDiff < 10){
                // 当前点距离大于右点距离0.3米，认为当前点及左边6个点无效
                if (depth1 - depth2 > 0.3){
                    cloudNeighborPicked[i - 5] = 1;
                    cloudNeighborPicked[i - 4] = 1;
                    cloudNeighborPicked[i - 3] = 1;
                    cloudNeighborPicked[i - 2] = 1;
                    cloudNeighborPicked[i - 1] = 1;
                    cloudNeighborPicked[i] = 1;
                // 当前点距离小于右边点距离0.3米，认为右边6个点无效
                }else if (depth2 - depth1 > 0.3){
                    cloudNeighborPicked[i + 1] = 1;
                    cloudNeighborPicked[i + 2] = 1;
                    cloudNeighborPicked[i + 3] = 1;
                    cloudNeighborPicked[i + 4] = 1;
                    cloudNeighborPicked[i + 5] = 1;
                    cloudNeighborPicked[i + 6] = 1;
                }
            }

            // 标记平行与激光束的点
            float diff1 = std::abs(float(cloudInfo.point_range[i-1] - cloudInfo.point_range[i]));
            float diff2 = std::abs(float(cloudInfo.point_range[i+1] - cloudInfo.point_range[i]));
            // 当前点与左右两点的距离均大于阈值，认为当前点是处于平行面的点
            if (diff1 > 0.02 * cloudInfo.point_range[i] && diff2 > 0.02 * cloudInfo.point_range[i])
                cloudNeighborPicked[i] = 1;
        }
    }

    /**
     * @brief
     * 特征点提取，遍历扫描线，每根扫描线扫描一周的点云划分为6段
     * 1. 提取角点
     *      1.1 针对每段提取最多20个角点
     *      1.2 设置cloudLabel标志，cloudLabel=1标志该点为角点
     *      1.3 cloudNeighborPicked=1标记该点已进行特征提取
     *      1.4 对该角点左右各5个点，如果两点之间的列索引差距小于10，则抛弃周围的点，避免重复对同一块区域提取角点
     *      1.5 将该角点加入角点点云集合
     * 2. 提取平面点
     *      2.1 不限制每段平面点的数量，后面会进行降采样
     *      2.2 设置cloudLabel标志，cloudLabel=-1标志该点为平面点
     *      2.3 cloudNeighborPicked=1标记该点已进行特征提取
     *      2.3 对该平面点左右各5个点，如果两点之间的列索引差距小于10，则抛弃周围的点，避免重复对同一块区域提取角点
     * 3. 将该段中除了角点之外的点加入平面点集合
     *      ！！！ 这点让步骤2感觉是多余的，最终的结果可能只是原始点云降采样，可能特征点提取也是没有必要的！！！
     * 4. 遍历结束
     * 5. 将平面特征点集合进行降采样
     * 
    */
    void extractFeatures()
    {
        cornerCloud->clear();
        surfaceCloud->clear();

        pcl::PointCloud<PointType>::Ptr surfaceCloudScan(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surfaceCloudScanDS(new pcl::PointCloud<PointType>());

        // 遍历每一条激光束
        for (int i = 0; i < N_SCAN; i++)
        {
            surfaceCloudScan->clear();
    
            // 将每一条激光线扫出来的点分成6段
            for (int j = 0; j < 6; j++)
            {
                
                // 计算该段的起始sp和结束ep索引
                // 这里计算有点绕，换一种方式会更好理解
                // sp = [(end-start)/6]*j + start
                // ep = [(end-start)/6]*(j+1) + start -1
                int sp = (cloudInfo.start_ring_index[i] * (6 - j) + cloudInfo.end_ring_index[i] * j) / 6;
                int ep = (cloudInfo.start_ring_index[i] * (5 - j) + cloudInfo.end_ring_index[i] * (j + 1)) / 6 - 1;

                if (sp >= ep)
                    continue;

                // 对当前段的雷达点云的曲率进行从小到大排序
                std::sort(cloudSmoothness.begin()+sp, cloudSmoothness.begin()+ep, by_value());

                /// 角点提取
                int largestPickedNum = 0;
                // 由于cloudSmoothness是从小到大排序，这里从ep（end point）到sp（start point）遍历，就是从大到小遍历
                // 曲率大的点更有可能是边缘点和角点
                for (int k = ep; k >= sp; k--)
                {
                    int ind = cloudSmoothness[k].ind;
                    // cloudNeighborPicked == 1表示是被遮挡、平行或者已经被当作特征点的点
                    // 曲率大于edgeThreshold则认为是角点，默认=1.0
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] > edgeThreshold)
                    {
                        largestPickedNum++;
                        // 对于每一段，最多只提取20个角点
                        if (largestPickedNum <= 20){
                            // cloudLabel = 1 标志为角点，-1标志为平面点
                            cloudLabel[ind] = 1;
                            cornerCloud->push_back(extractedCloud->points[ind]);
                        } else {
                            break;
                        }

                        cloudNeighborPicked[ind] = 1;
                        // 对该角点左右各5个点，如果两点之间的列索引差距小于10，则抛弃周围的点，避免重复对同一块区域提取角点
                        for (int l = 1; l <= 5; l++)
                        {
                            int columnDiff = std::abs(int(cloudInfo.point_col_ind[ind + l] - cloudInfo.point_col_ind[ind + l - 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--)
                        {
                            int columnDiff = std::abs(int(cloudInfo.point_col_ind[ind + l] - cloudInfo.point_col_ind[ind + l + 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                /// 平面点提取
                // 这里从sp开始遍历到ep，曲率由小到大
                // 这里与角点提取的区别在于不限制平面点的数量
                for (int k = sp; k <= ep; k++)
                {
                    int ind = cloudSmoothness[k].ind;
                    // cloudNeighborPicked == 1表示是被遮挡、平行或者已经被当作特征点的点
                    // 曲率小于surfThreshold认为是平面点，默认=0.1
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < surfThreshold)
                    {
                        cloudLabel[ind] = -1;
                        cloudNeighborPicked[ind] = 1;

                        // 对该平面点左右各5个点，如果两点之间的列索引差距小于10，则抛弃周围的点，避免重复对同一块区域提取角点
                        for (int l = 1; l <= 5; l++) {
                            int columnDiff = std::abs(int(cloudInfo.point_col_ind[ind + l] - cloudInfo.point_col_ind[ind + l - 1]));
                            if (columnDiff > 10)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--) {
                            int columnDiff = std::abs(int(cloudInfo.point_col_ind[ind + l] - cloudInfo.point_col_ind[ind + l + 1]));
                            if (columnDiff > 10)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                // 将该段中除了角点之外的点加入平面点集合
                // ！！！ 这点让步骤2感觉是多余的，最终的结果可能只是原始点云降采样，可能特征点提取也是没有必要的！！！
                for (int k = sp; k <= ep; k++)
                {
                    if (cloudLabel[k] <= 0){
                        surfaceCloudScan->push_back(extractedCloud->points[k]);
                    }
                }
            }

            // 对平面点集合做采样
            surfaceCloudScanDS->clear();
            downSizeFilter.setInputCloud(surfaceCloudScan);
            downSizeFilter.filter(*surfaceCloudScanDS);

            *surfaceCloud += *surfaceCloudScanDS;
        }
    }

    /**
     * 清空准备发布的点云信息中无效的字段，节省数据量
     * 这里清空的数据字段中point_col_ind和point_range的长度都是N_SCAN * N_COL，且下游任务均不使用
    */
    void freeCloudInfoMemory()
    {
        cloudInfo.start_ring_index.clear();
        cloudInfo.end_ring_index.clear();
        cloudInfo.point_col_ind.clear();
        cloudInfo.point_range.clear();
    }

    /**
     * 发布特征提取后的点云信息
    */
    void publishFeatureCloud()
    {
        // 清理准备发布的cloudInfo中一些无效的字段
        freeCloudInfoMemory();
        // 将角点和平面点集合转成PointCloud2格式并放入准备发布的cloudInfo数据包中
        pclPointcloud2Ros(cornerCloud, cloudInfo.cloud_corner, cloudHeader.stamp, lidarFrame);
        pclPointcloud2Ros(surfaceCloud, cloudInfo.cloud_surface, cloudHeader.stamp, lidarFrame);
        // 发布cloudInfo数据包
        pubLaserCloudInfo->publish(cloudInfo);
    }
};


int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);
    rclcpp::executors::SingleThreadedExecutor exec;

    auto FE = std::make_shared<FeatureExtraction>(options);

    exec.add_node(FE);

    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;32m----> Feature Extraction Started.\033[0m");

    exec.spin();

    rclcpp::shutdown();
    return 0;
}
