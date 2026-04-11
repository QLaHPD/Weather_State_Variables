import unittest

import numpy as np
import xarray as xr

from weather_state_variables.data import (
    ArcoEra5FuXiDataConfig,
    build_fuxi_channel_names,
    specific_humidity_to_relative_humidity,
)


class TestArcoEra5Helpers(unittest.TestCase):
    def test_config_loads_from_yaml(self) -> None:
        config = ArcoEra5FuXiDataConfig.from_yaml()

        self.assertEqual(config.input_time_offsets_hours, (-1, 0))
        self.assertEqual(config.lead_time_hours, 1)
        self.assertEqual(config.forecast_steps, 2)
        self.assertEqual(config.pressure_levels[0], 50)
        self.assertEqual(config.pressure_levels[-1], 1000)
        self.assertEqual(config.static_variables[0], "land_sea_mask")

    def test_build_fuxi_channel_names_matches_expected_layout(self) -> None:
        channel_names = build_fuxi_channel_names()

        self.assertEqual(len(channel_names), 70)
        self.assertEqual(channel_names[:5], ["Z50", "Z100", "Z150", "Z200", "Z250"])
        self.assertEqual(channel_names[13:16], ["T50", "T100", "T150"])
        self.assertEqual(channel_names[-5:], ["T2M", "U10", "V10", "MSL", "TP"])

    def test_specific_humidity_can_be_converted_to_relative_humidity(self) -> None:
        level = xr.DataArray(np.array([850, 1000], dtype=np.int64), dims=("level",))
        temperature = xr.DataArray(
            np.full((1, 2, 3, 4), 290.0, dtype=np.float32),
            dims=("time", "level", "latitude", "longitude"),
            coords={"level": level.values},
        )
        specific_humidity = xr.DataArray(
            np.full((1, 2, 3, 4), 0.008, dtype=np.float32),
            dims=("time", "level", "latitude", "longitude"),
            coords={"level": level.values},
        )

        relative_humidity = specific_humidity_to_relative_humidity(
            specific_humidity=specific_humidity,
            temperature=temperature,
            pressure_levels_hpa=level,
        )

        self.assertEqual(relative_humidity.shape, (1, 2, 3, 4))
        self.assertTrue(np.all(np.isfinite(relative_humidity.values)))
        self.assertTrue(np.all(relative_humidity.values >= 0.0))
        self.assertTrue(np.all(relative_humidity.values <= 100.0))


if __name__ == "__main__":
    unittest.main()
