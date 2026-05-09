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

const std::filesystem::path kDatasetRoot = R"(D:\PCL_Datasets\bunny1\bunny\data)";

std::string pathString(const std::filesystem::path& path) {
  return path.string();
}

CloudT::Ptr loadAndDownsample(const std::string& path) {
  CloudT::Ptr cloud(new CloudT);
  if (pcl::io::loadPCDFile<PointT>(path, *cloud) != 0) {
    std::cerr << "Failed to load cloud: " << path << std::endl;
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
  const std::string input_path =
      pathString(kDatasetRoot / "processed" / "bun000_filtered.pcd");
  CloudT::Ptr cloud = loadAndDownsample(input_path);
  if (!cloud) {
    return 1;
  }

  std::cout << "Day 3 input points: " << cloud->size() << std::endl;

  NormalCloudT::Ptr normals = computeNormals(cloud);
  std::cout << "Normal estimation complete, count = " << normals->size() << std::endl;

  FeatureCloudT::Ptr features = computeFpfh(cloud, normals);
  std::cout << "FPFH complete, count = " << features->size() << std::endl;
  std::cout << "Each FPFH descriptor is 33-dimensional." << std::endl;

  CloudT::Ptr keypoints = detectIssKeypoints(cloud);
  std::cout << "ISS keypoint extraction complete, count = " << keypoints->size() << std::endl;

  std::filesystem::create_directories(kDatasetRoot / "results" / "day3");
  pcl::io::savePCDFileBinary(
      pathString(kDatasetRoot / "results" / "day3" / "bun000_feature_cloud.pcd"), *cloud);
  pcl::io::savePCDFileBinary(
      pathString(kDatasetRoot / "results" / "day3" / "bun000_keypoints.pcd"), *keypoints);

  std::cout << "\nDay 3 completed." << std::endl;
  std::cout << "Normals describe local surface direction." << std::endl;
  std::cout << "FPFH describes local geometric shape." << std::endl;
  std::cout << "ISS keeps only more representative points for later stages." << std::endl;
  return 0;
}
