#include <filesystem>
#include <iostream>
#include <string>

#include <pcl/features/fpfh.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>

namespace {

using PointT = pcl::PointXYZ;
using CloudT = pcl::PointCloud<PointT>;
using NormalT = pcl::Normal;
using NormalCloudT = pcl::PointCloud<NormalT>;
using FeatureT = pcl::FPFHSignature33;
using FeatureCloudT = pcl::PointCloud<FeatureT>;

CloudT::Ptr loadAndDownsample(const std::string& path) {
  CloudT::Ptr cloud(new CloudT);
  if (pcl::io::loadPCDFile<PointT>(path, *cloud) != 0) {
    std::cerr << "读取点云失败: " << path << std::endl;
    return nullptr;
  }

  CloudT::Ptr sampled(new CloudT);
  pcl::VoxelGrid<PointT> voxel_filter;
  voxel_filter.setInputCloud(cloud);
  voxel_filter.setLeafSize(0.01f, 0.01f, 0.01f);
  voxel_filter.filter(*sampled);
  return sampled;
}

NormalCloudT::Ptr computeNormals(const CloudT::Ptr& cloud) {
  NormalCloudT::Ptr normals(new NormalCloudT);
  pcl::NormalEstimation<PointT, NormalT> estimator;
  estimator.setInputCloud(cloud);
  estimator.setSearchMethod(
      pcl::search::KdTree<PointT>::Ptr(new pcl::search::KdTree<PointT>));
  estimator.setKSearch(15);
  estimator.compute(*normals);
  return normals;
}

FeatureCloudT::Ptr computeFpfh(const CloudT::Ptr& cloud, const NormalCloudT::Ptr& normals) {
  FeatureCloudT::Ptr features(new FeatureCloudT);
  pcl::FPFHEstimation<PointT, NormalT, FeatureT> estimator;
  estimator.setInputCloud(cloud);
  estimator.setInputNormals(normals);
  estimator.setSearchMethod(
      pcl::search::KdTree<PointT>::Ptr(new pcl::search::KdTree<PointT>));
  estimator.setRadiusSearch(0.04f);
  estimator.compute(*features);
  return features;
}

CloudT::Ptr detectIssKeypoints(const CloudT::Ptr& cloud) {
  pcl::ISSKeypoint3D<PointT, PointT> iss_detector;
  iss_detector.setInputCloud(cloud);
  iss_detector.setSearchMethod(
      pcl::search::KdTree<PointT>::Ptr(new pcl::search::KdTree<PointT>));
  iss_detector.setSalientRadius(0.03);
  iss_detector.setNonMaxRadius(0.02);
  iss_detector.setThreshold21(0.975);
  iss_detector.setThreshold32(0.975);
  iss_detector.setMinNeighbors(5);
  iss_detector.setNumberOfThreads(1);

  CloudT::Ptr keypoints(new CloudT);
  iss_detector.compute(*keypoints);
  return keypoints;
}

}  // namespace

int main() {
  // Day 3 学什么：法向量、FPFH 和 ISS 关键点。
  // 作用：理解点云除了坐标之外，还能怎样描述“表面朝向”和“局部几何结构”。
  // 效果：输出法向量数量、FPFH 特征数量和关键点数量，为配准与识别打基础。
  const std::string input_path = "../data/processed/bun000_filtered.pcd";
  CloudT::Ptr cloud = loadAndDownsample(input_path);
  if (!cloud) {
    return 1;
  }

  std::cout << "Day 3 输入点数: " << cloud->size() << std::endl;

  NormalCloudT::Ptr normals = computeNormals(cloud);
  std::cout << "法向量计算完成，数量 = " << normals->size() << std::endl;

  FeatureCloudT::Ptr features = computeFpfh(cloud, normals);
  std::cout << "FPFH 特征计算完成，数量 = " << features->size() << std::endl;
  std::cout << "每个 FPFH 特征本质上是 33 维描述子。" << std::endl;

  CloudT::Ptr keypoints = detectIssKeypoints(cloud);
  std::cout << "ISS 关键点提取完成，数量 = " << keypoints->size() << std::endl;

  std::filesystem::create_directories("../results/day3");
  pcl::io::savePCDFileBinary("../results/day3/bun000_feature_cloud.pcd", *cloud);
  pcl::io::savePCDFileBinary("../results/day3/bun000_keypoints.pcd", *keypoints);

  std::cout << "\nDay 3 完成。" << std::endl;
  std::cout << "法向量的作用：告诉我们表面朝向。" << std::endl;
  std::cout << "FPFH 的作用：描述局部几何形状。" << std::endl;
  std::cout << "ISS 关键点的作用：只保留更有代表性的点，减少后续计算量。" << std::endl;
  return 0;
}
