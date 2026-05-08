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

CloudT::Ptr loadBaseCloud(const std::string& path) {
  CloudT::Ptr cloud(new CloudT);
  if (pcl::io::loadPCDFile<PointT>(path, *cloud) != 0) {
    std::cerr << "读取点云失败: " << path << std::endl;
    return nullptr;
  }
  return cloud;
}

CloudT::Ptr buildSegmentationDemoCloud(const CloudT::Ptr& bunny_cloud) {
  CloudT::Ptr demo_cloud(new CloudT(*bunny_cloud));

  // Bunny 数据本身没有明显的大平面，不利于演示 RANSAC 平面分割。
  // 所以这里额外拼一块简单的合成平面，让 Day 4 的效果更直观，也更稳定。
  for (float x = -0.08f; x <= 0.08f; x += 0.01f) {
    for (float y = -0.08f; y <= 0.08f; y += 0.01f) {
      demo_cloud->push_back(PointT(x, y, -0.03f));
    }
  }

  // 再补一小簇偏移点，帮助欧式聚类更容易分出多个区域。
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
  // Day 4 学什么：RANSAC 分割 + 欧式聚类。
  // 作用：把混合在一起的点拆开，理解“模型拟合”和“区域分离”。
  // 效果：输出平面内点、非平面点和聚类数量。
  const std::string input_path = "../data/processed/bun000_filtered.pcd";
  CloudT::Ptr bunny_cloud = loadBaseCloud(input_path);
  if (!bunny_cloud) {
    return 1;
  }

  CloudT::Ptr demo_cloud = buildSegmentationDemoCloud(bunny_cloud);
  std::cout << "Day 4 输入点数: " << demo_cloud->size() << std::endl;

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
    std::cerr << "没有找到可用的平面模型。" << std::endl;
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

  std::cout << "RANSAC 平面分割完成。" << std::endl;
  std::cout << "平面内点数量: " << plane_cloud->size() << std::endl;
  std::cout << "剩余点数量: " << objects_cloud->size() << std::endl;

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

  std::cout << "EuclideanClusterExtraction 完成，聚类数量: "
            << cluster_indices.size() << std::endl;

  std::filesystem::create_directories("../results/day4");
  pcl::io::savePCDFileBinary("../results/day4/demo_plane_cloud.pcd", *plane_cloud);
  pcl::io::savePCDFileBinary("../results/day4/demo_objects_cloud.pcd", *objects_cloud);

  std::cout << "\nDay 4 完成。" << std::endl;
  std::cout << "这一天的作用：学会把点云拆成“模型”和“剩余对象”两部分。" << std::endl;
  std::cout << "这一天的效果：你能看到哪些点属于平面，哪些点被分到不同的空间簇。" << std::endl;
  return 0;
}
