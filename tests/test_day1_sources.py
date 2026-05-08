import pathlib
import unittest


ROOT = pathlib.Path("/workspace")


class Day1SourceStructureTest(unittest.TestCase):
    def test_day1_preprocess_source_matches_plan(self):
        source = ROOT / "src" / "day1_preprocess.cpp"
        self.assertTrue(source.exists(), "缺少 Day 1 预处理源码文件")

        text = source.read_text(encoding="utf-8")
        self.assertIn("bun000.ply", text)
        self.assertIn("bun045.ply", text)
        self.assertIn("VoxelGrid", text)
        self.assertIn("StatisticalOutlierRemoval", text)
        self.assertIn("savePCDFileBinary", text)

    def test_day1_sac_ia_source_matches_plan(self):
        source = ROOT / "src" / "day1_sac_ia.cpp"
        self.assertTrue(source.exists(), "缺少 Day 1 SAC-IA 源码文件")

        text = source.read_text(encoding="utf-8")
        self.assertIn("bun000_filtered.pcd", text)
        self.assertIn("bun045_filtered.pcd", text)
        self.assertIn("NormalEstimation", text)
        self.assertIn("FPFHEstimation", text)
        self.assertIn("SampleConsensusInitialAlignment", text)
        self.assertIn("registration_output", text)


if __name__ == "__main__":
    unittest.main()
