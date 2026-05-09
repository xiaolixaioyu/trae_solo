#include <filesystem>
#include <iostream>
#include <string>

#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/icp.h>

namespace {

using PointT = pcl::PointXYZ;
using CloudT = pcl::PointCloud<PointT>;

const std::filesystem::path kDatasetRoot = R"(D:\PCL_Datasets\bunny1\bunny\data)";

std::string pathString(const std::filesystem::path& path) {
  return path.string();
}

CloudT::Ptr loadCloudOrExit(const std::string& path) {
  CloudT::Ptr cloud(new CloudT);
  if (pcl::io::loadPCDFile<PointT>(path, *cloud) != 0) {
    std::cerr << "Failed to load cloud: " << path << std::endl;
    return nullptr;
  }
  return cloud;
}

CloudT::Ptr downsampleForDebug(const CloudT::Ptr& input, float leaf_size) {
  CloudT::Ptr output(new CloudT);
  pcl::VoxelGrid<PointT> voxel_filter;
  voxel_filter.setInputCloud(input);
  voxel_filter.setLeafSize(leaf_size, leaf_size, leaf_size);
  voxel_filter.filter(*output);
  return output;
}

bool runPointToPointIcp(const CloudT::Ptr& source,
                        const CloudT::Ptr& target,
                        const std::string& output_path) {
  pcl::IterativeClosestPoint<PointT, PointT> icp;
  icp.setInputSource(source);
  icp.setInputTarget(target);
  icp.setMaximumIterations(30);
  icp.setMaxCorrespondenceDistance(0.05);
  icp.setTransformationEpsilon(1e-7);
  icp.setEuclideanFitnessEpsilon(1e-5);

  CloudT aligned_cloud;
  icp.align(aligned_cloud);

  if (!icp.hasConverged()) {
    std::cerr << "Point-to-point ICP did not converge." << std::endl;
    return false;
  }

  std::filesystem::create_directories(std::filesystem::path(output_path).parent_path());
  pcl::io::savePCDFileBinary(output_path, aligned_cloud);

  std::cout << "\nPoint-to-point ICP complete." << std::endl;
  std::cout << "Purpose: refine the Day 1 coarse alignment." << std::endl;
  std::cout << "Effect: overlap should be tighter than SAC-IA." << std::endl;
  std::cout << "Fitness Score: " << icp.getFitnessScore() << std::endl;
  std::cout << "Final transformation:\n" << icp.getFinalTransformation() << std::endl;
  return true;
}

bool runGicp(const CloudT::Ptr& source,
             const CloudT::Ptr& target,
             const std::string& output_path) {
  pcl::GeneralizedIterativeClosestPoint<PointT, PointT> gicp;
  gicp.setInputSource(source);
  gicp.setInputTarget(target);
  gicp.setMaximumIterations(20);
  gicp.setMaxCorrespondenceDistance(0.05);
  gicp.setTransformationEpsilon(1e-6);
  gicp.setEuclideanFitnessEpsilon(1e-5);

  CloudT aligned_cloud;
  gicp.align(aligned_cloud);

  if (!gicp.hasConverged()) {
    std::cerr << "GICP did not converge." << std::endl;
    return false;
  }

  std::filesystem::create_directories(std::filesystem::path(output_path).parent_path());
  pcl::io::savePCDFileBinary(output_path, aligned_cloud);

  std::cout << "\nGICP complete." << std::endl;
  std::cout << "Purpose: refine alignment with a more robust local model." << std::endl;
  std::cout << "Effect: registration is often more stable on curved surfaces." << std::endl;
  std::cout << "Fitness Score: " << gicp.getFitnessScore() << std::endl;
  std::cout << "Final transformation:\n" << gicp.getFinalTransformation() << std::endl;
  return true;
}

}  // namespace

int main() {
  const std::string source_path =
      pathString(kDatasetRoot / "results" / "day1" / "bun000_to_bun045_sac_ia.pcd");
  const std::string target_path =
      pathString(kDatasetRoot / "processed" / "bun045_filtered.pcd");
  const std::string icp_output =
      pathString(kDatasetRoot / "results" / "day2" / "bun000_to_bun045_icp.pcd");
  const std::string gicp_output =
      pathString(kDatasetRoot / "results" / "day2" / "bun000_to_bun045_gicp.pcd");

  CloudT::Ptr source_cloud = loadCloudOrExit(source_path);
  CloudT::Ptr target_cloud = loadCloudOrExit(target_path);
  if (!source_cloud || !target_cloud) {
    return 1;
  }

  CloudT::Ptr debug_source = downsampleForDebug(source_cloud, 0.01f);
  CloudT::Ptr debug_target = downsampleForDebug(target_cloud, 0.01f);

  std::cout << "Source points: " << debug_source->size() << std::endl;
  std::cout << "Target points: " << debug_target->size() << std::endl;

  const bool icp_ok = runPointToPointIcp(debug_source, debug_target, icp_output);
  const bool gicp_ok = runGicp(debug_source, debug_target, gicp_output);
  if (!icp_ok || !gicp_ok) {
    std::cerr << "Day 2 registration did not fully succeed." << std::endl;
    return 1;
  }

  std::cout << "\nDay 2 completed." << std::endl;
  std::cout << "ICP minimizes point-to-point distance directly." << std::endl;
  std::cout << "GICP also uses local structure information and is often more robust." << std::endl;
  return 0;
}
