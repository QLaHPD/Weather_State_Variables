import unittest

from weather_state_variables.models import FuXiLowerResConfig
from weather_state_variables.scaling import (
    CHINCHILLA_TOKENS_PER_PARAMETER,
    chinchilla_target_tokens,
    classify_scaling_ratio,
    single_process_samples_per_epoch,
    tokens_per_sample_from_model_config,
)


class TestScalingHelpers(unittest.TestCase):
    def test_tokens_per_sample_uses_patch_grid(self) -> None:
        model_config = FuXiLowerResConfig(
            input_size=(181, 360),
            patch_size=(4, 4),
        )

        self.assertEqual(tokens_per_sample_from_model_config(model_config), 45 * 90)

    def test_chinchilla_target_tokens_uses_default_ratio(self) -> None:
        self.assertEqual(
            chinchilla_target_tokens(1_000_000),
            int(round(1_000_000 * CHINCHILLA_TOKENS_PER_PARAMETER)),
        )

    def test_classify_scaling_ratio(self) -> None:
        self.assertEqual(classify_scaling_ratio(0.25), "below Chinchilla heuristic")
        self.assertEqual(classify_scaling_ratio(1.00), "near Chinchilla heuristic")
        self.assertEqual(classify_scaling_ratio(3.00), "above Chinchilla heuristic")

    def test_single_process_samples_per_epoch(self) -> None:
        self.assertEqual(
            single_process_samples_per_epoch(100, batch_size=8, max_train_batches=None),
            (100, 13, True),
        )
        self.assertEqual(
            single_process_samples_per_epoch(100, batch_size=8, max_train_batches=20),
            (100, 13, True),
        )
        self.assertEqual(
            single_process_samples_per_epoch(100, batch_size=8, max_train_batches=10),
            (80, 13, False),
        )


if __name__ == "__main__":
    unittest.main()
