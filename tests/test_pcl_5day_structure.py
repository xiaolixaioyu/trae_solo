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

    def test_day1_contains_preprocess_and_sac_ia(self):
        source = (ROOT / "src" / "day1" / "day1_registration.cpp").read_text(
            encoding="utf-8"
        )
        self.assertIn("VoxelGrid", source)
        self.assertIn("StatisticalOutlierRemoval", source)
        self.assertIn("SampleConsensusInitialAlignment", source)
        self.assertIn("bun000.ply", source)
        self.assertIn("bun045.ply", source)

    def test_day2_contains_icp_or_gicp(self):
        source = (ROOT / "src" / "day2" / "day2_icp.cpp").read_text(
            encoding="utf-8"
        )
        self.assertIn("IterativeClosestPoint", source)
        self.assertIn("GeneralizedIterativeClosestPoint", source)
        self.assertIn("setMaximumIterations", source)

    def test_day3_contains_normals_and_features(self):
        source = (ROOT / "src" / "day3" / "day3_features.cpp").read_text(
            encoding="utf-8"
        )
        self.assertIn("NormalEstimation", source)
        self.assertIn("FPFHEstimation", source)
        self.assertIn("ISSKeypoint3D", source)

    def test_day4_contains_segmentation_and_clustering(self):
        source = (ROOT / "src" / "day4" / "day4_segmentation.cpp").read_text(
            encoding="utf-8"
        )
        self.assertIn("SACSegmentation", source)
        self.assertIn("EuclideanClusterExtraction", source)
        self.assertIn("ExtractIndices", source)

    def test_day5_contains_reconstruction(self):
        source = (ROOT / "src" / "day5" / "day5_reconstruction.cpp").read_text(
            encoding="utf-8"
        )
        self.assertIn("GreedyProjectionTriangulation", source)
        self.assertIn("NormalEstimation", source)
        self.assertIn("PolygonMesh", source)


if __name__ == "__main__":
    unittest.main()
