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
  const std::string input_path =
      pathString(kDatasetRoot / "processed" / "bun000_filtered.pcd");
  CloudT::Ptr cloud = loadAndDownsample(input_path);
  if (!cloud) {
    return 1;
  }

  std::cout << "Day 5 input points: " << cloud->size() << std::endl;

  NormalCloudT::Ptr normals = computeNormals(cloud);
  PointNormalCloudT::Ptr cloud_with_normals = combinePointsAndNormals(cloud, normals);

  pcl::search::KdTree<PointNormalT>::Ptr tree(
      new pcl::search::KdTree<PointNormalT>);
  tree->setInputCloud(cloud_with_normals);

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

  std::filesystem::create_directories(kDatasetRoot / "results" / "day5");
  pcl::io::savePLYFile(pathString(kDatasetRoot / "results" / "day5" / "bun000_mesh.ply"), mesh);

  std::cout << "PolygonMesh reconstruction complete." << std::endl;
  std::cout << "Triangle count: " << mesh.polygons.size() << std::endl;
  std::cout << "\nDay 5 completed." << std::endl;
  std::cout << "Purpose: turn a point cloud into a more continuous surface." << std::endl;
  std::cout << "Effect: you get a mesh file that can be viewed or compared later." << std::endl;
  return 0;
}
