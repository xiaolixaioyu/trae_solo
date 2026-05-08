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

CloudT::Ptr loadCloudOrExit(const std::string& path) {
  CloudT::Ptr cloud(new CloudT);
  if (pcl::io::loadPCDFile<PointT>(path, *cloud) != 0) {
    std::cerr << "读取点云失败: " << path << std::endl;
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
    std::cerr << "点到点 ICP 没有收敛。" << std::endl;
    return false;
  }

  std::filesystem::create_directories(std::filesystem::path(output_path).parent_path());
  pcl::io::savePCDFileBinary(output_path, aligned_cloud);

  std::cout << "\n点到点 ICP 完成。" << std::endl;
  std::cout << "作用：继续细调 Day 1 的粗对齐结果。" << std::endl;
  std::cout << "效果：重叠区域会比 SAC-IA 更贴合。" << std::endl;
  std::cout << "Fitness Score: " << icp.getFitnessScore() << std::endl;
  std::cout << "最终变换矩阵:\n" << icp.getFinalTransformation() << std::endl;
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
    std::cerr << "GICP 没有收敛。" << std::endl;
    return false;
  }

  std::filesystem::create_directories(std::filesystem::path(output_path).parent_path());
  pcl::io::savePCDFileBinary(output_path, aligned_cloud);

  std::cout << "\nGICP 完成。" << std::endl;
  std::cout << "作用：在 ICP 的思路上进一步利用局部几何统计信息。" << std::endl;
  std::cout << "效果：通常在噪声或局部曲面情况下更稳定。" << std::endl;
  std::cout << "Fitness Score: " << gicp.getFitnessScore() << std::endl;
  std::cout << "最终变换矩阵:\n" << gicp.getFinalTransformation() << std::endl;
  return true;
}

}  // namespace

int main() {
  // Day 2 学什么：精配准。
  // 作用：在 Day 1 粗配准的基础上继续优化，让两幅点云更精确地贴合。
  // 效果：输出 ICP 和 GICP 两种精配准结果，便于你观察谁更稳定。
  const std::string source_path = "../results/day1/bun000_to_bun045_sac_ia.pcd";
  const std::string target_path = "../data/processed/bun045_filtered.pcd";
  const std::string icp_output = "../results/day2/bun000_to_bun045_icp.pcd";
  const std::string gicp_output = "../results/day2/bun000_to_bun045_gicp.pcd";

  CloudT::Ptr source_cloud = loadCloudOrExit(source_path);
  CloudT::Ptr target_cloud = loadCloudOrExit(target_path);
  if (!source_cloud || !target_cloud) {
    return 1;
  }

  // 为了照顾 Debug 模式，这里额外做一次轻量降采样。
  // 这样可以让 IterativeClosestPoint 和 GeneralizedIterativeClosestPoint 更快结束。
  CloudT::Ptr debug_source = downsampleForDebug(source_cloud, 0.01f);
  CloudT::Ptr debug_target = downsampleForDebug(target_cloud, 0.01f);

  std::cout << "源点云点数: " << debug_source->size() << std::endl;
  std::cout << "目标点云点数: " << debug_target->size() << std::endl;

  const bool icp_ok = runPointToPointIcp(debug_source, debug_target, icp_output);
  const bool gicp_ok = runGicp(debug_source, debug_target, gicp_output);
  if (!icp_ok || !gicp_ok) {
    std::cerr << "Day 2 精配准未全部成功。" << std::endl;
    return 1;
  }

  std::cout << "\nDay 2 完成。" << std::endl;
  std::cout << "点到点 ICP 的特点：直接最小化对应点之间的距离。" << std::endl;
  std::cout << "GICP 的特点：把局部结构信息也考虑进去，通常更鲁棒。" << std::endl;
  return 0;
}
