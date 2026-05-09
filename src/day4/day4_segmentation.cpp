#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <pcl/common/common.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>

namespace {

using PointT = pcl::PointXYZ;
using CloudT = pcl::PointCloud<PointT>;

const std::filesystem::path kDatasetRoot = R"(D:\PCL_Datasets\bunny1\bunny\data)";

std::string pathString(const std::filesystem::path& path) {
  return path.string();
}

CloudT::Ptr loadBaseCloud(const std::string& path) {
  CloudT::Ptr cloud(new CloudT);
  if (pcl::io::loadPCDFile<PointT>(path, *cloud) != 0) {
    std::cerr << "Failed to load cloud: " << path << std::endl;
    return nullptr;
  }
  return cloud;
}

CloudT::Ptr buildSegmentationDemoCloud(const CloudT::Ptr& bunny_cloud) {
  CloudT::Ptr demo_cloud(new CloudT(*bunny_cloud));

  // Bunny does not contain a large obvious plane by itself.
  // Add a small synthetic plane so SACSegmentation can show a clear result.
  for (float x = -0.08f; x <= 0.08f; x += 0.01f) {
    for (float y = -0.08f; y <= 0.08f; y += 0.01f) {
      demo_cloud->push_back(PointT(x, y, -0.03f));
    }
  }

  for (float t = 0.0f; t < 1.0f; t += 0.05f) {
    demo_cloud->push_back(PointT(0.12f + 0.01f * std::cos(6.28f * t),
                                 0.10f + 0.01f * std::sin(6.28f * t),
                                 0.02f));
  }

  demo_cloud->width = static_cast<std::uint32_t>(demo_cloud->size());
  demo_cloud->height = 1;
  demo_cloud->is_dense = true;
  return demo_cloud;
}

}  // namespace

int main() {
  const std::string input_path =
      pathString(kDatasetRoot / "processed" / "bun000_filtered.pcd");
  CloudT::Ptr bunny_cloud = loadBaseCloud(input_path);
  if (!bunny_cloud) {
    return 1;
  }

  CloudT::Ptr demo_cloud = buildSegmentationDemoCloud(bunny_cloud);
  std::cout << "Day 4 input points: " << demo_cloud->size() << std::endl;

  pcl::SACSegmentation<PointT> segmentation;
  segmentation.setOptimizeCoefficients(true);
  segmentation.setModelType(pcl::SACMODEL_PLANE);
  segmentation.setMethodType(pcl::SAC_RANSAC);
  segmentation.setDistanceThreshold(0.004);
  segmentation.setInputCloud(demo_cloud);

  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  segmentation.segment(*inliers, *coefficients);
  if (inliers->indices.empty()) {
    std::cerr << "No plane model was found." << std::endl;
    return 1;
  }

  pcl::ExtractIndices<PointT> extractor;
  CloudT::Ptr plane_cloud(new CloudT);
  CloudT::Ptr objects_cloud(new CloudT);
  extractor.setInputCloud(demo_cloud);
  extractor.setIndices(inliers);
  extractor.setNegative(false);
  extractor.filter(*plane_cloud);
  extractor.setNegative(true);
  extractor.filter(*objects_cloud);

  std::cout << "SACSegmentation complete." << std::endl;
  std::cout << "Plane points: " << plane_cloud->size() << std::endl;
  std::cout << "Remaining points: " << objects_cloud->size() << std::endl;

  pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
  tree->setInputCloud(objects_cloud);

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<PointT> clustering;
  clustering.setClusterTolerance(0.02);
  clustering.setMinClusterSize(20);
  clustering.setMaxClusterSize(50000);
  clustering.setSearchMethod(tree);
  clustering.setInputCloud(objects_cloud);
  clustering.extract(cluster_indices);

  std::cout << "EuclideanClusterExtraction complete, cluster count = "
            << cluster_indices.size() << std::endl;

  std::filesystem::create_directories(kDatasetRoot / "results" / "day4");
  pcl::io::savePCDFileBinary(
      pathString(kDatasetRoot / "results" / "day4" / "demo_plane_cloud.pcd"), *plane_cloud);
  pcl::io::savePCDFileBinary(
      pathString(kDatasetRoot / "results" / "day4" / "demo_objects_cloud.pcd"), *objects_cloud);

  std::cout << "\nDay 4 completed." << std::endl;
  std::cout << "Purpose: split a mixed cloud into model points and remaining points." << std::endl;
  std::cout << "Effect: you can inspect plane inliers and spatial clusters separately." << std::endl;
  return 0;
}
