# PCL 5 天实践重组 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将仓库重组为按 `day1` 到 `day5` 组织的 5 个 C++ 实践文件，并保留适合 Debug 模式的轻量参数、完整中文注释和基础测试。

**Architecture:** 保持单日单入口文件的结构，每个 `cpp` 负责一天的核心实践内容，并通过前一天的输出作为后一天输入形成学习流水线。优先复用现有 Bunny 数据，使用教学版轻量参数降低 Debug 模式运行成本，同时用结构测试约束文件布局与算法关键词。

**Tech Stack:** C++, PCL, Python `unittest`, Visual Studio / Debug 友好参数设计

---

### Task 1: 建立按天目录与结构测试

**Files:**
- Create: `src/day1/day1_registration.cpp`
- Create: `src/day2/day2_icp.cpp`
- Create: `src/day3/day3_features.cpp`
- Create: `src/day4/day4_segmentation.cpp`
- Create: `src/day5/day5_reconstruction.cpp`
- Create: `tests/test_pcl_5day_structure.py`

- [ ] **Step 1: 写结构测试并让它先失败**

```python
import pathlib
import unittest


ROOT = pathlib.Path("/workspace")


class PCLFiveDayStructureTest(unittest.TestCase):
    def test_day_cpp_files_exist(self):
        expected = [
            ROOT / "src" / "day1" / "day1_registration.cpp",
            ROOT / "src" / "day2" / "day2_icp.cpp",
            ROOT / "src" / "day3" / "day3_features.cpp",
            ROOT / "src" / "day4" / "day4_segmentation.cpp",
            ROOT / "src" / "day5" / "day5_reconstruction.cpp",
        ]
        missing = [str(path) for path in expected if not path.exists()]
        self.assertEqual([], missing, f"缺少按天组织的源码文件: {missing}")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试确认它失败**

Run: `python -m unittest tests/test_pcl_5day_structure.py -v`
Expected: FAIL，并提示缺少 `src/day1/day1_registration.cpp` 等文件。

- [ ] **Step 3: 创建最小源码骨架**

```cpp
#include <iostream>

int main() {
  std::cout << "占位骨架，后续补充完整实现。" << std::endl;
  return 0;
}
```

- [ ] **Step 4: 再次运行测试确认通过**

Run: `python -m unittest tests/test_pcl_5day_structure.py -v`
Expected: PASS

- [ ] **Step 5: 提交这一小步**

```bash
git add tests/test_pcl_5day_structure.py src/day1/day1_registration.cpp src/day2/day2_icp.cpp src/day3/day3_features.cpp src/day4/day4_segmentation.cpp src/day5/day5_reconstruction.cpp
git commit -m "test: add five-day cpp structure"
```

### Task 2: 实现 Day 1 预处理与 SAC-IA 合并文件

**Files:**
- Modify: `src/day1/day1_registration.cpp`
- Test: `tests/test_pcl_5day_structure.py`

- [ ] **Step 1: 先写失败测试，约束 Day 1 关键算法**

```python
    def test_day1_contains_preprocess_and_sac_ia(self):
        source = (ROOT / "src" / "day1" / "day1_registration.cpp").read_text(encoding="utf-8")
        self.assertIn("VoxelGrid", source)
        self.assertIn("StatisticalOutlierRemoval", source)
        self.assertIn("SampleConsensusInitialAlignment", source)
        self.assertIn("bun000.ply", source)
        self.assertIn("bun045.ply", source)
```

- [ ] **Step 2: 运行 Day 1 测试确认失败**

Run: `python -m unittest tests/test_pcl_5day_structure.py -v`
Expected: FAIL，提示缺少 `VoxelGrid` 或 `SampleConsensusInitialAlignment` 等关键词。

- [ ] **Step 3: 写最小实现并保留完整中文注释**

```cpp
// Day 1: 预处理 + SAC-IA 粗配准
// 先用体素滤波和统计滤波减少点数与噪声，再计算法向量和 FPFH 特征，
// 最后用 SAC-IA 求出一个“大致正确”的初始对齐结果。
```

实现内容必须包括：
- `preprocessCloud()`：读取 PLY、体素滤波、统计滤波、保存 PCD。
- `estimateNormals()`：计算法向量。
- `computeFPFH()`：计算 FPFH 特征。
- `runSacIaRegistration()`：执行粗配准并保存结果。
- `main()`：串联执行 Day 1 全部实践，参数使用轻量教学版。

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m unittest tests/test_pcl_5day_structure.py -v`
Expected: PASS

- [ ] **Step 5: 提交这一小步**

```bash
git add tests/test_pcl_5day_structure.py src/day1/day1_registration.cpp
git commit -m "feat: add day1 registration practice"
```

### Task 3: 实现 Day 2 精配准

**Files:**
- Modify: `src/day2/day2_icp.cpp`
- Test: `tests/test_pcl_5day_structure.py`

- [ ] **Step 1: 先写失败测试，约束 Day 2 关键算法**

```python
    def test_day2_contains_icp_or_gicp(self):
        source = (ROOT / "src" / "day2" / "day2_icp.cpp").read_text(encoding="utf-8")
        self.assertIn("IterativeClosestPoint", source)
        self.assertIn("GeneralizedIterativeClosestPoint", source)
        self.assertIn("setMaximumIterations", source)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m unittest tests/test_pcl_5day_structure.py -v`
Expected: FAIL，提示缺少 ICP / GICP 关键词。

- [ ] **Step 3: 写最小实现并说明作用与效果**

```cpp
// Day 2: ICP / GICP 精配准
// 这一阶段的目标不是“从零找姿态”，而是在 Day 1 粗配准结果基础上继续贴合，
// 让重叠区域对得更准。为了照顾 Debug 模式，默认迭代次数使用教学版较小值。
```

实现内容必须包括：
- 读取 Day 1 输出的粗配准结果和目标点云。
- 分别演示 ICP 与 GICP 的配置。
- 打印是否收敛、fitness score 和变换矩阵。
- 注释清楚点到点 ICP 与 GICP 的差别。

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m unittest tests/test_pcl_5day_structure.py -v`
Expected: PASS

- [ ] **Step 5: 提交这一小步**

```bash
git add tests/test_pcl_5day_structure.py src/day2/day2_icp.cpp
git commit -m "feat: add day2 icp practice"
```

### Task 4: 实现 Day 3 法向量与特征

**Files:**
- Modify: `src/day3/day3_features.cpp`
- Test: `tests/test_pcl_5day_structure.py`

- [ ] **Step 1: 先写失败测试，约束 Day 3 关键算法**

```python
    def test_day3_contains_normals_and_features(self):
        source = (ROOT / "src" / "day3" / "day3_features.cpp").read_text(encoding="utf-8")
        self.assertIn("NormalEstimation", source)
        self.assertIn("FPFHEstimation", source)
        self.assertIn("ISSKeypoint3D", source)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m unittest tests/test_pcl_5day_structure.py -v`
Expected: FAIL，提示缺少法向量、FPFH 或 ISS 关键词。

- [ ] **Step 3: 写最小实现并说明作用与效果**

```cpp
// Day 3: 法向量、FPFH 和关键点
// 这一天不是做最终配准，而是理解“点云如何描述局部几何形状”。
// 法向量帮助理解表面朝向，FPFH 描述局部结构，ISS 关键点帮助减少匹配开销。
```

实现内容必须包括：
- 读取处理后的 Bunny 点云。
- 计算法向量并输出点数。
- 计算 FPFH 特征并输出维度统计。
- 提取 ISS 关键点并输出关键点数量。

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m unittest tests/test_pcl_5day_structure.py -v`
Expected: PASS

- [ ] **Step 5: 提交这一小步**

```bash
git add tests/test_pcl_5day_structure.py src/day3/day3_features.cpp
git commit -m "feat: add day3 feature practice"
```

### Task 5: 实现 Day 4 分割与聚类

**Files:**
- Modify: `src/day4/day4_segmentation.cpp`
- Test: `tests/test_pcl_5day_structure.py`

- [ ] **Step 1: 先写失败测试，约束 Day 4 关键算法**

```python
    def test_day4_contains_segmentation_and_clustering(self):
        source = (ROOT / "src" / "day4" / "day4_segmentation.cpp").read_text(encoding="utf-8")
        self.assertIn("SACSegmentation", source)
        self.assertIn("EuclideanClusterExtraction", source)
        self.assertIn("ExtractIndices", source)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m unittest tests/test_pcl_5day_structure.py -v`
Expected: FAIL，提示缺少分割或聚类关键词。

- [ ] **Step 3: 写最小实现并说明作用与效果**

```cpp
// Day 4: 平面分割与聚类
// 先用 RANSAC 找最明显的平面，再把剩余点做欧式聚类，
// 这样可以直观看到“模型拟合”和“按空间邻近拆分对象”这两种思路。
```

实现内容必须包括：
- 读取处理后的 Bunny 或其变体点云。
- 用 `SACSegmentation` 提取主要平面或示例模型。
- 用 `ExtractIndices` 分离内点和外点。
- 用 `EuclideanClusterExtraction` 对剩余点聚类并输出聚类数量。

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m unittest tests/test_pcl_5day_structure.py -v`
Expected: PASS

- [ ] **Step 5: 提交这一小步**

```bash
git add tests/test_pcl_5day_structure.py src/day4/day4_segmentation.cpp
git commit -m "feat: add day4 segmentation practice"
```

### Task 6: 实现 Day 5 重建

**Files:**
- Modify: `src/day5/day5_reconstruction.cpp`
- Test: `tests/test_pcl_5day_structure.py`

- [ ] **Step 1: 先写失败测试，约束 Day 5 关键算法**

```python
    def test_day5_contains_reconstruction(self):
        source = (ROOT / "src" / "day5" / "day5_reconstruction.cpp").read_text(encoding="utf-8")
        self.assertIn("GreedyProjectionTriangulation", source)
        self.assertIn("NormalEstimation", source)
        self.assertIn("PolygonMesh", source)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m unittest tests/test_pcl_5day_structure.py -v`
Expected: FAIL，提示缺少重建关键词。

- [ ] **Step 3: 写最小实现并说明作用与效果**

```cpp
// Day 5: 三维重建
// 这一天把“离散点”进一步组织成“连续曲面”。
// 为了照顾 Debug 模式，优先演示参数较轻的贪婪三角化方案。
```

实现内容必须包括：
- 读取处理后的点云。
- 计算法向量。
- 组合点与法向量输入到 `GreedyProjectionTriangulation`。
- 保存 `PolygonMesh` 并打印网格顶点或面片统计信息。

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m unittest tests/test_pcl_5day_structure.py -v`
Expected: PASS

- [ ] **Step 5: 提交这一小步**

```bash
git add tests/test_pcl_5day_structure.py src/day5/day5_reconstruction.cpp
git commit -m "feat: add day5 reconstruction practice"
```

### Task 7: 删除被替代的旧入口并更新旧测试

**Files:**
- Delete: `src/day1_preprocess.cpp`
- Delete: `src/day1_sac_ia.cpp`
- Modify: `tests/test_day1_sources.py`

- [ ] **Step 1: 先写失败测试，改成检查新 Day 1 文件**

```python
class Day1SourceStructureTest(unittest.TestCase):
    def test_day1_registration_source_matches_plan(self):
        source = ROOT / "src" / "day1" / "day1_registration.cpp"
        self.assertTrue(source.exists(), "缺少 Day 1 合并源码文件")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m unittest tests/test_day1_sources.py -v`
Expected: FAIL，因为旧测试仍然引用 `src/day1_preprocess.cpp` 和 `src/day1_sac_ia.cpp`。

- [ ] **Step 3: 写最小实现，让旧测试转向新结构**

```python
        text = source.read_text(encoding="utf-8")
        self.assertIn("VoxelGrid", text)
        self.assertIn("StatisticalOutlierRemoval", text)
        self.assertIn("SampleConsensusInitialAlignment", text)
        self.assertIn("bun000.ply", text)
        self.assertIn("bun045.ply", text)
```

删除内容必须仅限：
- `src/day1_preprocess.cpp`
- `src/day1_sac_ia.cpp`

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m unittest tests/test_day1_sources.py tests/test_pcl_5day_structure.py -v`
Expected: PASS

- [ ] **Step 5: 提交这一小步**

```bash
git add tests/test_day1_sources.py tests/test_pcl_5day_structure.py src/day1/day1_registration.cpp src/day2/day2_icp.cpp src/day3/day3_features.cpp src/day4/day4_segmentation.cpp src/day5/day5_reconstruction.cpp
git rm src/day1_preprocess.cpp src/day1_sac_ia.cpp
git commit -m "refactor: reorganize pcl practice by day"
```

### Task 8: 最终检查

**Files:**
- Verify: `src/day1/day1_registration.cpp`
- Verify: `src/day2/day2_icp.cpp`
- Verify: `src/day3/day3_features.cpp`
- Verify: `src/day4/day4_segmentation.cpp`
- Verify: `src/day5/day5_reconstruction.cpp`
- Verify: `tests/test_day1_sources.py`
- Verify: `tests/test_pcl_5day_structure.py`

- [ ] **Step 1: 运行全部 Python 结构测试**

Run: `python -m unittest tests/test_day1_sources.py tests/test_pcl_5day_structure.py -v`
Expected: PASS

- [ ] **Step 2: 做源码级静态自检**

检查项：
- 5 个 `cpp` 都有“这一天做什么 / 作用 / 效果”的中文注释。
- 参数都偏轻量，适合 Debug 模式。
- 文件路径引用前后连贯，没有继续引用已删除的旧入口文件。

- [ ] **Step 3: 记录交付说明**

交付要点：
- 新目录结构
- 每天对应的学习内容
- 推荐运行顺序
- 已删除的旧文件
