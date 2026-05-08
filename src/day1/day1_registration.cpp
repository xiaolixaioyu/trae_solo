#include <filesystem>
#include <iostream>
#include <string>

#include <pcl/features/fpfh.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/search/kdtree.h>

namespace {

using PointT = pcl::PointXYZ;
using CloudT = pcl::PointCloud<PointT>;
using NormalT = pcl::Normal;
using NormalCloudT = pcl::PointCloud<NormalT>;
using FeatureT = pcl::FPFHSignature33;
using FeatureCloudT = pcl::PointCloud<FeatureT>;

// Day 1 的任务分成两部分：
// 1. 上午先把原始点云整理干净，让后续算法少处理一些点。
// 2. 下午再用 SAC-IA 做粗配准，先把两幅点云摆到大致正确的位置。
// 这样做的好处是：不仅逻辑清楚，而且在 Debug 模式下也更容易跑通。
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

  std::cout << "\n========== Day 1 预处理: " << input_ply_path << " ==========" << std::endl;
  std::cout << "原始点数: " << raw_cloud->size() << std::endl;

  // 第一步：体素滤波（VoxelGrid）。
  // 作用：把密集点云按小立方体分桶，每个桶只保留一个代表点。
  // 效果：点数明显变少，运算更快，但整体形状基本保留。
  CloudT::Ptr voxel_cloud(new CloudT);
  pcl::VoxelGrid<PointT> voxel_filter;
  voxel_filter.setInputCloud(raw_cloud);
  voxel_filter.setLeafSize(voxel_leaf_size, voxel_leaf_size, voxel_leaf_size);
  voxel_filter.filter(*voxel_cloud);

  std::cout << "体素滤波后点数: " << voxel_cloud->size()
            << "，leaf size = " << voxel_leaf_size << std::endl;

  // 第二步：统计滤波（StatisticalOutlierRemoval）。
  // 作用：检查每个点和邻居的平均距离，把离群点删掉。
  // 效果：噪声点减少，后面的法向量和配准更稳定。
  CloudT::Ptr filtered_cloud(new CloudT);
  pcl::StatisticalOutlierRemoval<PointT> sor;
  sor.setInputCloud(voxel_cloud);
  sor.setMeanK(mean_k);
  sor.setStddevMulThresh(stddev_mul_thresh);
  sor.filter(*filtered_cloud);

  std::cout << "统计滤波后点数: " << filtered_cloud->size() << std::endl;
  std::cout << "MeanK = " << mean_k
            << ", StddevMulThresh = " << stddev_mul_thresh << std::endl;

  std::filesystem::create_directories(std::filesystem::path(output_pcd_path).parent_path());
  if (pcl::io::savePCDFileBinary(output_pcd_path, *filtered_cloud) != 0) {
    std::cerr << "保存 PCD 文件失败: " << output_pcd_path << std::endl;
    return false;
  }

  std::cout << "预处理结果已保存到: " << output_pcd_path << std::endl;
  return true;
}

NormalCloudT::Ptr estimateNormals(const CloudT::Ptr& cloud, int k_search) {
  NormalCloudT::Ptr normals(new NormalCloudT);

  pcl::NormalEstimation<PointT, NormalT> normal_estimator;
  normal_estimator.setInputCloud(cloud);
  normal_estimator.setSearchMethod(
      pcl::search::KdTree<PointT>::Ptr(new pcl::search::KdTree<PointT>));
  normal_estimator.setKSearch(k_search);
  normal_estimator.compute(*normals);

  return normals;
}

FeatureCloudT::Ptr computeFPFH(const CloudT::Ptr& cloud,
                               const NormalCloudT::Ptr& normals,
                               float radius) {
  FeatureCloudT::Ptr features(new FeatureCloudT);

  pcl::FPFHEstimation<PointT, NormalT, FeatureT> fpfh_estimator;
  fpfh_estimator.setInputCloud(cloud);
  fpfh_estimator.setInputNormals(normals);
  fpfh_estimator.setSearchMethod(
      pcl::search::KdTree<PointT>::Ptr(new pcl::search::KdTree<PointT>));
  fpfh_estimator.setRadiusSearch(radius);
  fpfh_estimator.compute(*features);

  return features;
}

bool runSacIaRegistration(const std::string& source_path,
                          const std::string& target_path,
                          const std::string& registration_output) {
  CloudT::Ptr source_cloud(new CloudT);
  CloudT::Ptr target_cloud(new CloudT);

  if (pcl::io::loadPCDFile<PointT>(source_path, *source_cloud) != 0) {
    std::cerr << "读取源点云失败: " << source_path << std::endl;
    return false;
  }
  if (pcl::io::loadPCDFile<PointT>(target_path, *target_cloud) != 0) {
    std::cerr << "读取目标点云失败: " << target_path << std::endl;
    return false;
  }

  std::cout << "\n========== Day 1 SAC-IA 粗配准 ==========" << std::endl;
  std::cout << "源点云点数: " << source_cloud->size() << std::endl;
  std::cout << "目标点云点数: " << target_cloud->size() << std::endl;

  // 先计算法向量，再计算 FPFH 特征。
  // 这是 SAC-IA 的前置工作，因为它不是直接靠坐标硬配，而是靠局部几何特征找相似区域。
  const int normal_k_search = 15;
  const float feature_radius = 0.04f;
  NormalCloudT::Ptr source_normals = estimateNormals(source_cloud, normal_k_search);
  NormalCloudT::Ptr target_normals = estimateNormals(target_cloud, normal_k_search);
  FeatureCloudT::Ptr source_features =
      computeFPFH(source_cloud, source_normals, feature_radius);
  FeatureCloudT::Ptr target_features =
      computeFPFH(target_cloud, target_normals, feature_radius);

  std::cout << "法向量与 FPFH 计算完成。" << std::endl;

  // 为了照顾 Debug 模式，这里把迭代次数和采样距离调得更轻量。
  // 作用：先找到一个“差不多对齐”的初始位姿。
  // 效果：不追求像 ICP 那样严丝合缝，但要给 Day 2 足够好的起点。
  pcl::SampleConsensusInitialAlignment<PointT, PointT, FeatureT> sac_ia;
  sac_ia.setInputSource(source_cloud);
  sac_ia.setSourceFeatures(source_features);
  sac_ia.setInputTarget(target_cloud);
  sac_ia.setTargetFeatures(target_features);
  sac_ia.setMinSampleDistance(0.03f);
  sac_ia.setMaxCorrespondenceDistance(0.08f);
  sac_ia.setMaximumIterations(80);

  CloudT::Ptr aligned_cloud(new CloudT);
  sac_ia.align(*aligned_cloud);

  if (!sac_ia.hasConverged()) {
    std::cerr << "SAC-IA 粗配准没有收敛，请检查输入点云和参数。" << std::endl;
    return false;
  }

  std::filesystem::create_directories(
      std::filesystem::path(registration_output).parent_path());
  if (pcl::io::savePCDFileBinary(registration_output, *aligned_cloud) != 0) {
    std::cerr << "保存粗配准结果失败: " << registration_output << std::endl;
    return false;
  }

  std::cout << "粗配准完成，Fitness Score = " << sac_ia.getFitnessScore() << std::endl;
  std::cout << "最终变换矩阵:\n" << sac_ia.getFinalTransformation() << std::endl;
  std::cout << "粗配准结果已保存到: " << registration_output << std::endl;
  return true;
}

}  // namespace

int main() {
  // Day 1 学什么：
  // - 上午：预处理（降采样 + 去噪）
  // - 下午：粗配准（SAC-IA）
  // Day 1 的作用：让原始点云先变得更干净、更容易处理。
  // Day 1 的效果：得到两个过滤后的 PCD 文件，以及一个大致对齐的配准结果。
  const std::string input_cloud_a = "../bun000.ply";
  const std::string input_cloud_b = "../bun045.ply";
  const std::string output_cloud_a = "../data/processed/bun000_filtered.pcd";
  const std::string output_cloud_b = "../data/processed/bun045_filtered.pcd";
  const std::string registration_output = "../results/day1/bun000_to_bun045_sac_ia.pcd";

  // 这些参数特意比“完整实验版”更轻，目的是让 Debug 模式更快跑完。
  const float voxel_leaf_size = 0.008f;
  const int mean_k = 20;
  const double stddev_mul_thresh = 1.0;

  const bool ok_a = preprocessCloud(
      input_cloud_a, output_cloud_a, voxel_leaf_size, mean_k, stddev_mul_thresh);
  const bool ok_b = preprocessCloud(
      input_cloud_b, output_cloud_b, voxel_leaf_size, mean_k, stddev_mul_thresh);
  if (!ok_a || !ok_b) {
    std::cerr << "\nDay 1 预处理失败，请先检查输入文件路径和 PCL 环境。" << std::endl;
    return 1;
  }

  if (!runSacIaRegistration(output_cloud_a, output_cloud_b, registration_output)) {
    std::cerr << "\nDay 1 粗配准失败。" << std::endl;
    return 1;
  }

  std::cout << "\nDay 1 完成。" << std::endl;
  std::cout << "作用：先把点云整理干净，再做大致对齐。" << std::endl;
  std::cout << "效果：Day 2 可以在这个基础上继续做 ICP / GICP 精配准。" << std::endl;
  return 0;
}
