#include <filesystem>
#include <iostream>
#include <string>

#include <pcl/features/normal_3d.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/surface/gp3.h>

namespace {

using PointT = pcl::PointXYZ;
using CloudT = pcl::PointCloud<PointT>;
using PointNormalT = pcl::PointNormal;
using PointNormalCloudT = pcl::PointCloud<PointNormalT>;
using NormalT = pcl::Normal;
using NormalCloudT = pcl::PointCloud<NormalT>;

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

PointNormalCloudT::Ptr combinePointsAndNormals(const CloudT::Ptr& cloud,
                                               const NormalCloudT::Ptr& normals) {
  PointNormalCloudT::Ptr combined(new PointNormalCloudT);
  combined->resize(cloud->size());
  for (std::size_t i = 0; i < cloud->size() && i < normals->size(); ++i) {
    (*combined)[i].x = (*cloud)[i].x;
    (*combined)[i].y = (*cloud)[i].y;
    (*combined)[i].z = (*cloud)[i].z;
    (*combined)[i].normal_x = (*normals)[i].normal_x;
    (*combined)[i].normal_y = (*normals)[i].normal_y;
    (*combined)[i].normal_z = (*normals)[i].normal_z;
  }
  return combined;
}

}  // namespace

int main() {
  // Day 5 学什么：三维重建。
  // 作用：把离散的点进一步组织成连续曲面。
  // 效果：输出一个 PolygonMesh 网格文件，让你看到“点云到模型”的最后一步。
  const std::string input_path = "../data/processed/bun000_filtered.pcd";
  CloudT::Ptr cloud = loadAndDownsample(input_path);
  if (!cloud) {
    return 1;
  }

  std::cout << "Day 5 输入点数: " << cloud->size() << std::endl;

  NormalCloudT::Ptr normals = computeNormals(cloud);
  PointNormalCloudT::Ptr cloud_with_normals = combinePointsAndNormals(cloud, normals);

  pcl::search::KdTree<PointNormalT>::Ptr tree(
      new pcl::search::KdTree<PointNormalT>);
  tree->setInputCloud(cloud_with_normals);

  // GreedyProjectionTriangulation 比泊松重建更轻量，
  // 更适合当前教学版和 Debug 模式先快速跑通。
  pcl::GreedyProjectionTriangulation<PointNormalT> gp3;
  pcl::PolygonMesh mesh;
  gp3.setSearchRadius(0.03);
  gp3.setMu(2.5);
  gp3.setMaximumNearestNeighbors(40);
  gp3.setMaximumSurfaceAngle(3.1415926f / 4.0f);
  gp3.setMinimumAngle(3.1415926f / 18.0f);
  gp3.setMaximumAngle(2.0f * 3.1415926f / 3.0f);
  gp3.setNormalConsistency(false);
  gp3.setInputCloud(cloud_with_normals);
  gp3.setSearchMethod(tree);
  gp3.reconstruct(mesh);

  std::filesystem::create_directories("../results/day5");
  pcl::io::savePLYFile("../results/day5/bun000_mesh.ply", mesh);

  std::cout << "PolygonMesh 重建完成。" << std::endl;
  std::cout << "网格面片数量: " << mesh.polygons.size() << std::endl;
  std::cout << "\nDay 5 完成。" << std::endl;
  std::cout << "这一天的作用：体验如何把点云变成更像模型的网格。" << std::endl;
  std::cout << "这一天的效果：得到一个可保存、可视化的重建结果文件。" << std::endl;
  return 0;
}
