#include <iostream>
#include <string>

#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>

namespace {

using PointT = pcl::PointXYZ;
using CloudT = pcl::PointCloud<PointT>;

// 这个函数负责完成 Day 1 上午的全部任务：
// 1. 读取原始 Bunny 点云
// 2. 先做体素滤波，减少点数
// 3. 再做统计滤波，去掉离群噪声点
// 4. 把结果保存成后续粗配准更方便读取的 PCD 文件
bool preprocessCloud(const std::string& input_ply_path,
                     const std::string& output_pcd_path,
                     float voxel_leaf_size,
                     int mean_k,
                     double stddev_mul_thresh) {
  CloudT::Ptr raw_cloud(new CloudT);
  if (pcl::io::loadPLYFile<PointT>(input_ply_path, *raw_cloud) != 0) {
    std::cerr << "读取 PLY 文件失败: " << input_ply_path << std::endl;
    return false;
  }

  std::cout << "\n========== 开始处理: " << input_ply_path << " ==========" << std::endl;
  std::cout << "原始点云点数: " << raw_cloud->size() << std::endl;

  // 第一步：体素滤波（VoxelGrid）
  // 可以把空间切成很多小立方体，每个小立方体只保留一个代表点。
  // 这样做的好处是：大幅减少点数，但整体形状基本保持不变。
  CloudT::Ptr voxel_cloud(new CloudT);
  pcl::VoxelGrid<PointT> voxel_filter;
  voxel_filter.setInputCloud(raw_cloud);
  voxel_filter.setLeafSize(voxel_leaf_size, voxel_leaf_size, voxel_leaf_size);
  voxel_filter.filter(*voxel_cloud);

  std::cout << "体素滤波后点数: " << voxel_cloud->size() << std::endl;
  std::cout << "体素大小 leaf size: " << voxel_leaf_size << std::endl;

  // 第二步：统计滤波（StatisticalOutlierRemoval）
  // 这个滤波器会检查每个点与周围邻居之间的距离。
  // 如果某个点离周围点太远，它大概率是噪声，就会被删除。
  CloudT::Ptr filtered_cloud(new CloudT);
  pcl::StatisticalOutlierRemoval<PointT> sor;
  sor.setInputCloud(voxel_cloud);
  sor.setMeanK(mean_k);
  sor.setStddevMulThresh(stddev_mul_thresh);
  sor.filter(*filtered_cloud);

  std::cout << "统计滤波后点数: " << filtered_cloud->size() << std::endl;
  std::cout << "MeanK: " << mean_k
            << ", StddevMulThresh: " << stddev_mul_thresh << std::endl;

  if (pcl::io::savePCDFileBinary(output_pcd_path, *filtered_cloud) != 0) {
    std::cerr << "保存 PCD 文件失败: " << output_pcd_path << std::endl;
    return false;
  }

  std::cout << "预处理结果已保存到: " << output_pcd_path << std::endl;
  return true;
}

}  // namespace

int main() {
  // 学习计划中明确要求 Day 1 使用 bun000.ply 和 bun045.ply 两幅 Bunny 点云。
  const std::string input_cloud_a = "../bun000.ply";
  const std::string input_cloud_b = "../bun045.ply";

  // 预处理后的结果统一保存到 data/processed 目录，方便 Day 1 下午继续使用。
  const std::string output_cloud_a = "../data/processed/bun000_filtered.pcd";
  const std::string output_cloud_b = "../data/processed/bun045_filtered.pcd";

  // 这些参数直接对应学习计划中的建议值：
  // - 体素大小 0.005
  // - 统计滤波 K=50
  // - 标准差倍数 1.0
  const float voxel_leaf_size = 0.005f;
  const int mean_k = 50;
  const double stddev_mul_thresh = 1.0;

  const bool ok_a = preprocessCloud(
      input_cloud_a, output_cloud_a, voxel_leaf_size, mean_k, stddev_mul_thresh);
  const bool ok_b = preprocessCloud(
      input_cloud_b, output_cloud_b, voxel_leaf_size, mean_k, stddev_mul_thresh);

  if (!ok_a || !ok_b) {
    std::cerr << "\nDay 1 预处理失败，请先检查输入文件路径和 PCL 环境配置。" << std::endl;
    return 1;
  }

  std::cout << "\nDay 1 上午任务完成：两幅 Bunny 点云已经完成降采样和去噪。" << std::endl;
  std::cout << "下一步可以运行 day1_sac_ia.cpp 做粗配准。" << std::endl;
  return 0;
}
