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

const std::filesystem::path kDatasetRoot = R"(D:\PCL_Datasets\bunny1\bunny\data)";

std::string pathString(const std::filesystem::path& path) {
  return path.string();
}

bool preprocessCloud(const std::string& input_ply_path,
                     const std::string& output_pcd_path,
                     float voxel_leaf_size,
                     int mean_k,
                     double stddev_mul_thresh) {
  CloudT::Ptr raw_cloud(new CloudT);
  if (pcl::io::loadPLYFile<PointT>(input_ply_path, *raw_cloud) != 0) {
    std::cerr << "Failed to load PLY file: " << input_ply_path << std::endl;
    return false;
  }

  std::cout << "\n========== Day 1 Preprocess: " << input_ply_path << " ==========" << std::endl;
  std::cout << "Raw point count: " << raw_cloud->size() << std::endl;

  // Step 1: VoxelGrid downsampling.
  // Purpose: reduce the point count before later stages.
  // Effect: debug runs finish faster while the overall shape is still preserved.
  CloudT::Ptr voxel_cloud(new CloudT);
  pcl::VoxelGrid<PointT> voxel_filter;
  voxel_filter.setInputCloud(raw_cloud);
  voxel_filter.setLeafSize(voxel_leaf_size, voxel_leaf_size, voxel_leaf_size);
  voxel_filter.filter(*voxel_cloud);

  std::cout << "After VoxelGrid: " << voxel_cloud->size()
            << " points, leaf size = " << voxel_leaf_size << std::endl;

  // Step 2: StatisticalOutlierRemoval denoising.
  // Purpose: remove points that are far away from their neighbors.
  // Effect: normal estimation and registration become more stable.
  CloudT::Ptr filtered_cloud(new CloudT);
  pcl::StatisticalOutlierRemoval<PointT> sor;
  sor.setInputCloud(voxel_cloud);
  sor.setMeanK(mean_k);
  sor.setStddevMulThresh(stddev_mul_thresh);
  sor.filter(*filtered_cloud);

  std::cout << "After StatisticalOutlierRemoval: " << filtered_cloud->size() << std::endl;
  std::cout << "MeanK = " << mean_k
            << ", StddevMulThresh = " << stddev_mul_thresh << std::endl;

  std::filesystem::create_directories(std::filesystem::path(output_pcd_path).parent_path());
  if (pcl::io::savePCDFileBinary(output_pcd_path, *filtered_cloud) != 0) {
    std::cerr << "Failed to save PCD file: " << output_pcd_path << std::endl;
    return false;
  }

  std::cout << "Saved preprocess result to: " << output_pcd_path << std::endl;
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
    std::cerr << "Failed to load source cloud: " << source_path << std::endl;
    return false;
  }
  if (pcl::io::loadPCDFile<PointT>(target_path, *target_cloud) != 0) {
    std::cerr << "Failed to load target cloud: " << target_path << std::endl;
    return false;
  }

  std::cout << "\n========== Day 1 SAC-IA ==========" << std::endl;
  std::cout << "Source points: " << source_cloud->size() << std::endl;
  std::cout << "Target points: " << target_cloud->size() << std::endl;

  const int normal_k_search = 15;
  const float feature_radius = 0.04f;
  NormalCloudT::Ptr source_normals = estimateNormals(source_cloud, normal_k_search);
  NormalCloudT::Ptr target_normals = estimateNormals(target_cloud, normal_k_search);
  FeatureCloudT::Ptr source_features =
      computeFPFH(source_cloud, source_normals, feature_radius);
  FeatureCloudT::Ptr target_features =
      computeFPFH(target_cloud, target_normals, feature_radius);

  std::cout << "Normals and FPFH features computed." << std::endl;

  // Step 3: SAC-IA coarse registration.
  // Purpose: find a reasonable initial alignment before ICP.
  // Effect: Day 2 starts from a much better pose estimate.
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
    std::cerr << "SAC-IA did not converge." << std::endl;
    return false;
  }

  std::filesystem::create_directories(
      std::filesystem::path(registration_output).parent_path());
  if (pcl::io::savePCDFileBinary(registration_output, *aligned_cloud) != 0) {
    std::cerr << "Failed to save registration result: " << registration_output << std::endl;
    return false;
  }

  std::cout << "SAC-IA fitness score = " << sac_ia.getFitnessScore() << std::endl;
  std::cout << "Final transformation:\n" << sac_ia.getFinalTransformation() << std::endl;
  std::cout << "Saved registration result to: " << registration_output << std::endl;
  return true;
}

}  // namespace

int main() {
  // Day 1 topic: preprocess raw Bunny scans and run SAC-IA coarse registration.
  // Day 1 purpose: clean the input cloud and produce a usable initial alignment.
  // Day 1 result: two filtered PCD files and one coarse registration output.
  const std::string input_cloud_a = pathString(kDatasetRoot / "bun000.ply");
  const std::string input_cloud_b = pathString(kDatasetRoot / "bun045.ply");
  const std::string output_cloud_a = pathString(kDatasetRoot / "processed" / "bun000_filtered.pcd");
  const std::string output_cloud_b = pathString(kDatasetRoot / "processed" / "bun045_filtered.pcd");
  const std::string registration_output =
      pathString(kDatasetRoot / "results" / "day1" / "bun000_to_bun045_sac_ia.pcd");

  // These values are intentionally lighter than a full experiment.
  // They are chosen so Debug mode is more likely to finish quickly.
  const float voxel_leaf_size = 0.008f;
  const int mean_k = 20;
  const double stddev_mul_thresh = 1.0;

  const bool ok_a = preprocessCloud(
      input_cloud_a, output_cloud_a, voxel_leaf_size, mean_k, stddev_mul_thresh);
  const bool ok_b = preprocessCloud(
      input_cloud_b, output_cloud_b, voxel_leaf_size, mean_k, stddev_mul_thresh);
  if (!ok_a || !ok_b) {
    std::cerr << "\nDay 1 preprocess failed. Check file paths and PCL setup." << std::endl;
    return 1;
  }

  if (!runSacIaRegistration(output_cloud_a, output_cloud_b, registration_output)) {
    std::cerr << "\nDay 1 coarse registration failed." << std::endl;
    return 1;
  }

  std::cout << "\nDay 1 completed." << std::endl;
  std::cout << "Purpose: clean the cloud first, then get a rough alignment." << std::endl;
  std::cout << "Effect: Day 2 can continue with ICP or GICP on this result." << std::endl;
  return 0;
}
