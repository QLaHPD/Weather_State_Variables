import unittest

import pandas as pd
from torch.fx import GraphModule

from weather_state_variables.models import (
    FuXiShort,
    build_fuxi_time_embeddings,
    summarize_short_onnx_architecture,
)


class TestFuXiShort(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = FuXiShort()
        cls.summary = cls.model.summary()

    def test_time_embeddings_match_teacher_shape(self) -> None:
        tembs = build_fuxi_time_embeddings(pd.Timestamp("2024-01-01 00:00:00"), total_steps=3)
        self.assertEqual(tembs.shape, (3, 1, 12))

    def test_exact_graph_is_built_from_onnx(self) -> None:
        self.assertIsInstance(self.model.graph_module, GraphModule)
        self.assertEqual(self.summary["input_names"], ["input", "temb"])
        self.assertEqual(self.summary["forward_arg_names"], ["input", "temb"])
        self.assertEqual(self.summary["op_counts"]["call_module"], self.summary["onnx_node_count"])
        self.assertGreater(self.summary["meta_state_tensor_count"], 0)
        self.assertEqual(self.summary["graph_node_count"], 12768)

    def test_architecture_recipe_matches_expected_short_model(self) -> None:
        recipe = summarize_short_onnx_architecture()
        self.assertEqual(recipe["input_shape"], [1, 2, 70, 721, 1440])
        self.assertEqual(recipe["aux_chans"], 5)
        self.assertEqual(recipe["patch_size"], [4, 4])
        self.assertEqual(recipe["window_size"], [9, 9])
        self.assertEqual(recipe["num_heads"], 24)
        self.assertEqual(recipe["depths"], [12, 12, 12, 12])


if __name__ == "__main__":
    unittest.main()
