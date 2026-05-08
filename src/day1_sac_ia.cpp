#include <iostream>
#include <string>

#include <pcl/features/fpfh.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
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

}  // namespace

int main() {
  // Day 1 下午的输入，不再直接读取原始 PLY，而是使用上午预处理后的 PCD。
  const std::string source_path = "../data/processed/bun000_filtered.pcd";
  const std::string target_path = "../data/processed/bun045_filtered.pcd";
  const std::string registration_output = "../results/day1/bun000_to_bun045_sac_ia.pcd";

  CloudT::Ptr source_cloud(new CloudT);
  CloudT::Ptr target_cloud(new CloudT);

  if (pcl::io::loadPCDFile<PointT>(source_path, *source_cloud) != 0) {
    std::cerr << "读取源点云失败: " << source_path << std::endl;
    return 1;
  }
  if (pcl::io::loadPCDFile<PointT>(target_path, *target_cloud) != 0) {
    std::cerr << "读取目标点云失败: " << target_path << std::endl;
    return 1;
  }

  std::cout << "源点云点数: " << source_cloud->size() << std::endl;
  std::cout << "目标点云点数: " << target_cloud->size() << std::endl;

  // 第一步：计算法向量。
  // SAC-IA 自己并不直接“看”原始点坐标，而是借助后面的特征来理解局部几何形状。
  // 而 FPFH 特征的计算又依赖法向量，所以法向量是一个必要前置步骤。
  const int normal_k_search = 20;
  NormalCloudT::Ptr source_normals = estimateNormals(source_cloud, normal_k_search);
  NormalCloudT::Ptr target_normals = estimateNormals(target_cloud, normal_k_search);

  std::cout << "法向量计算完成，KSearch = " << normal_k_search << std::endl;

  // 第二步：计算 FPFH 特征。
  // 可以把 FPFH 理解成“每个点周围局部几何形状的数字签名”。
  // SAC-IA 会借助这些签名，在两个点云之间寻找相似区域。
  const float feature_radius = 0.05f;
  FeatureCloudT::Ptr source_features =
      computeFPFH(source_cloud, source_normals, feature_radius);
  FeatureCloudT::Ptr target_features =
      computeFPFH(target_cloud, target_normals, feature_radius);

  std::cout << "FPFH 特征计算完成，半径 = " << feature_radius << std::endl;

  // 第三步：执行 SAC-IA 粗配准。
  // 这里让 bun000 作为源点云，被变换到 bun045 的坐标系中。
  pcl::SampleConsensusInitialAlignment<PointT, PointT, FeatureT> sac_ia;
  sac_ia.setInputSource(source_cloud);
  sac_ia.setSourceFeatures(source_features);
  sac_ia.setInputTarget(target_cloud);
  sac_ia.setTargetFeatures(target_features);
  sac_ia.setMinSampleDistance(0.05f);
  sac_ia.setMaxCorrespondenceDistance(0.15f);
  sac_ia.setMaximumIterations(500);

  CloudT::Ptr aligned_cloud(new CloudT);
  sac_ia.align(*aligned_cloud);

  if (!sac_ia.hasConverged()) {
    std::cerr << "SAC-IA 粗配准没有收敛，请检查点云质量或参数设置。" << std::endl;
    return 1;
  }

  const Eigen::Matrix4f transformation = sac_ia.getFinalTransformation();
  const double fitness_score = sac_ia.getFitnessScore();

  std::cout << "\nSAC-IA 粗配准完成。" << std::endl;
  std::cout << "Fitness Score: " << fitness_score << std::endl;
  std::cout << "最终变换矩阵:\n" << transformation << std::endl;

  if (pcl::io::savePCDFileBinary(registration_output, *aligned_cloud) != 0) {
    std::cerr << "保存粗配准结果失败: " << registration_output << std::endl;
    return 1;
  }

  std::cout << "粗配准结果已保存到: " << registration_output << std::endl;
  std::cout << "这一步的目标是“大致对齐”，不要求像 ICP 那样完全贴合。" << std::endl;
  return 0;
}
