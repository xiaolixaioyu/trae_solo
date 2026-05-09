import pathlib
import unittest


ROOT = pathlib.Path("/workspace")


class Day1SourceStructureTest(unittest.TestCase):
    def test_day1_registration_source_matches_plan(self):
        source = ROOT / "src" / "day1" / "day1_registration.cpp"
        self.assertTrue(source.exists(), "缺少 Day 1 合并源码文件")

        text = source.read_text(encoding="utf-8")
        self.assertIn("bun000.ply", text)
        self.assertIn("bun045.ply", text)
        self.assertIn("VoxelGrid", text)
        self.assertIn("StatisticalOutlierRemoval", text)
        self.assertIn("savePCDFileBinary", text)
        self.assertIn("bun000_filtered.pcd", text)
        self.assertIn("bun045_filtered.pcd", text)
        self.assertIn("NormalEstimation", text)
        self.assertIn("FPFHEstimation", text)
        self.assertIn("SampleConsensusInitialAlignment", text)
        self.assertIn("registration_output", text)
        self.assertIn("removeNaNFromPointCloud", text)
        self.assertIn("Finite point count", text)

    def test_day1_uses_windows_dataset_root_and_ascii_safe_text(self):
        source = ROOT / "src" / "day1" / "day1_registration.cpp"
        text = source.read_text(encoding="utf-8")

        self.assertIn(
            r'R"(D:\PCL_Datasets\bunny1\bunny\data)"',
            text,
            "Day 1 代码应显式使用本地 Windows 数据集根路径",
        )

        try:
            text.encode("ascii")
        except UnicodeEncodeError as exc:
            self.fail(f"Day 1 代码包含非 ASCII 字符，VS 可能因代码页问题误解析: {exc}")


if __name__ == "__main__":
    unittest.main()
