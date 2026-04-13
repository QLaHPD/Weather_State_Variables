import json
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from weather_state_variables.data import (
    ArcoEra5DownloadPlan,
    ArcoEra5FuXiDataConfig,
    ArcoEra5FuXiDataset,
    ContiguousDistributedSampler,
    build_arco_era5_download_plan,
    build_fuxi_derived_static_maps,
    build_fuxi_channel_names,
    download_arco_era5_subset,
    inspect_local_zarr_time_axes,
    repair_local_zarr_time_consistency,
    resolve_arco_era5_download_window,
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
        self.assertFalse(config.include_sample_metadata)
        self.assertGreaterEqual(config.dynamic_ram_cache_time_steps, 0)

    def test_build_fuxi_channel_names_matches_expected_layout(self) -> None:
        channel_names = build_fuxi_channel_names()

        self.assertEqual(len(channel_names), 70)
        self.assertEqual(channel_names[:5], ["Z50", "Z100", "Z150", "Z200", "Z250"])
        self.assertEqual(channel_names[13:16], ["T50", "T100", "T150"])
        self.assertEqual(channel_names[-5:], ["T2M", "U10", "V10", "MSL", "TP"])

    def test_large_dynamic_cache_request_is_bounded_to_a_rolling_window(self) -> None:
        config = ArcoEra5FuXiDataConfig(dynamic_ram_cache_time_steps=1024)
        dataset = ArcoEra5FuXiDataset(config)
        dataset._dataset_step_hours = 1

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resolved_steps = dataset._resolved_dynamic_chunk_time_steps()

        self.assertEqual(resolved_steps, 64)
        self.assertTrue(any("caps the active RAM window" in str(item.message) for item in caught))

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

    def test_build_fuxi_derived_static_maps_returns_three_expected_fields(self) -> None:
        maps = build_fuxi_derived_static_maps(
            latitude=np.array([-90.0, 0.0, 90.0], dtype=np.float32),
            longitude=np.array([0.0, 90.0], dtype=np.float32),
        )

        self.assertEqual(set(maps.keys()), {"cos_latitude", "cos_longitude", "sin_longitude"})
        self.assertEqual(maps["cos_latitude"].shape, (3, 2))
        self.assertTrue(np.allclose(maps["cos_latitude"][:, 0], [0.0, 1.0, 0.0], atol=1e-6))
        self.assertTrue(np.allclose(maps["cos_longitude"][1], [1.0, 0.0], atol=1e-6))
        self.assertTrue(np.allclose(maps["sin_longitude"][1], [0.0, 1.0], atol=1e-6))

    def test_download_plan_derives_relative_humidity_from_specific_humidity(self) -> None:
        config = ArcoEra5FuXiDataConfig.from_yaml()
        available_variables = (
            "geopotential",
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind",
            "specific_humidity",
            "2m_temperature",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "mean_sea_level_pressure",
            "total_precipitation",
            "land_sea_mask",
            "geopotential_at_surface",
        )

        plan = build_arco_era5_download_plan(available_variables, config)

        self.assertIsInstance(plan, ArcoEra5DownloadPlan)
        self.assertTrue(plan.derive_relative_humidity)
        self.assertIn("specific_humidity", plan.source_pressure_variables)
        self.assertNotIn("specific_humidity", plan.output_dynamic_variables)
        self.assertIn("relative_humidity", plan.output_dynamic_variables)
        self.assertEqual(plan.source_static_variables, ("land_sea_mask", "geopotential_at_surface"))

    def test_download_window_matches_training_config(self) -> None:
        window = resolve_arco_era5_download_window()

        self.assertEqual(str(window.anchor_start), "2000-01-01 00:00:00")
        self.assertEqual(str(window.anchor_end), "2012-12-31 23:00:00")
        self.assertEqual(str(window.raw_start), "1999-12-31 23:00:00")
        self.assertEqual(str(window.raw_end), "2013-01-01 01:00:00")

    def test_contiguous_distributed_sampler_assigns_contiguous_ranges(self) -> None:
        dataset = list(range(10))

        rank0 = list(ContiguousDistributedSampler(dataset, num_replicas=3, rank=0))
        rank1 = list(ContiguousDistributedSampler(dataset, num_replicas=3, rank=1))
        rank2 = list(ContiguousDistributedSampler(dataset, num_replicas=3, rank=2))

        self.assertEqual(rank0, [0, 1, 2, 3])
        self.assertEqual(rank1, [4, 5, 6, 7])
        self.assertEqual(rank2, [8, 9, 9, 9])

    def test_dataset_targets_are_one_hour_and_two_hour_steps(self) -> None:
        times = pd.date_range("2018-01-01 00:00:00", periods=6, freq="h")
        levels = np.array([50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000], dtype=np.int64)
        latitude = np.array([-90.0, 90.0], dtype=np.float32)
        longitude = np.array([0.0, 1.0], dtype=np.float32)

        def pressure_field(offset: float) -> np.ndarray:
            data = np.arange(times.size * levels.size * latitude.size * longitude.size, dtype=np.float32)
            data = data.reshape(times.size, levels.size, latitude.size, longitude.size)
            return data + offset

        def surface_field(offset: float) -> np.ndarray:
            data = np.arange(times.size * latitude.size * longitude.size, dtype=np.float32)
            data = data.reshape(times.size, latitude.size, longitude.size)
            return data + offset

        source_ds = xr.Dataset(
            data_vars={
                "geopotential": (("time", "level", "latitude", "longitude"), pressure_field(0.0)),
                "temperature": (("time", "level", "latitude", "longitude"), pressure_field(1000.0) + 273.15),
                "u_component_of_wind": (("time", "level", "latitude", "longitude"), pressure_field(2000.0)),
                "v_component_of_wind": (("time", "level", "latitude", "longitude"), pressure_field(3000.0)),
                "relative_humidity": (
                    ("time", "level", "latitude", "longitude"),
                    np.full((times.size, levels.size, latitude.size, longitude.size), 50.0, dtype=np.float32),
                ),
                "2m_temperature": (("time", "latitude", "longitude"), surface_field(4000.0) + 273.15),
                "10m_u_component_of_wind": (("time", "latitude", "longitude"), surface_field(5000.0)),
                "10m_v_component_of_wind": (("time", "latitude", "longitude"), surface_field(6000.0)),
                "mean_sea_level_pressure": (("time", "latitude", "longitude"), surface_field(7000.0)),
                "total_precipitation": (("time", "latitude", "longitude"), surface_field(8000.0)),
                "land_sea_mask": (("latitude", "longitude"), np.ones((latitude.size, longitude.size), dtype=np.float32)),
                "geopotential_at_surface": (
                    ("latitude", "longitude"),
                    np.ones((latitude.size, longitude.size), dtype=np.float32) * 100.0,
                ),
            },
            coords={
                "time": times,
                "level": levels,
                "latitude": latitude,
                "longitude": longitude,
            },
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "source.zarr"
            source_ds.to_zarr(source_path, mode="w")

            config = ArcoEra5FuXiDataConfig(
                dataset_url=str(source_path),
                include_sample_metadata=True,
                config_path=Path("configs/model_config.yaml"),
            )
            dataset = ArcoEra5FuXiDataset(config)
            sample = dataset[0]

        self.assertEqual(sample["anchor_time"], "2018-01-01 01:00:00")
        self.assertEqual(sample["input_times"], ["2018-01-01 00:00:00", "2018-01-01 01:00:00"])
        self.assertEqual(sample["target_times"], ["2018-01-01 02:00:00", "2018-01-01 03:00:00"])

    def test_download_subset_can_resume_local_zarr(self) -> None:
        times = pd.date_range("2018-01-01 00:00:00", periods=4, freq="h")
        levels = np.array([50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000], dtype=np.int64)
        latitude = np.array([-90.0, 90.0], dtype=np.float32)
        longitude = np.array([0.0, 1.0, 2.0], dtype=np.float32)

        def pressure_field(offset: float) -> np.ndarray:
            data = np.arange(times.size * levels.size * latitude.size * longitude.size, dtype=np.float32)
            data = data.reshape(times.size, levels.size, latitude.size, longitude.size)
            return data + offset

        def surface_field(offset: float) -> np.ndarray:
            data = np.arange(times.size * latitude.size * longitude.size, dtype=np.float32)
            data = data.reshape(times.size, latitude.size, longitude.size)
            return data + offset

        def static_field(offset: float) -> np.ndarray:
            data = np.arange(latitude.size * longitude.size, dtype=np.float32)
            return data.reshape(latitude.size, longitude.size) + offset

        source_ds = xr.Dataset(
            data_vars={
                "geopotential": (("time", "level", "latitude", "longitude"), pressure_field(0.0)),
                "temperature": (("time", "level", "latitude", "longitude"), pressure_field(1000.0) + 273.15),
                "u_component_of_wind": (("time", "level", "latitude", "longitude"), pressure_field(2000.0)),
                "v_component_of_wind": (("time", "level", "latitude", "longitude"), pressure_field(3000.0)),
                "specific_humidity": (
                    ("time", "level", "latitude", "longitude"),
                    np.full((times.size, levels.size, latitude.size, longitude.size), 0.005, dtype=np.float32),
                ),
                "2m_temperature": (("time", "latitude", "longitude"), surface_field(4000.0) + 273.15),
                "10m_u_component_of_wind": (("time", "latitude", "longitude"), surface_field(5000.0)),
                "10m_v_component_of_wind": (("time", "latitude", "longitude"), surface_field(6000.0)),
                "mean_sea_level_pressure": (("time", "latitude", "longitude"), surface_field(7000.0)),
                "total_precipitation": (("time", "latitude", "longitude"), surface_field(8000.0)),
                "land_sea_mask": (("latitude", "longitude"), static_field(0.0)),
                "geopotential_at_surface": (("latitude", "longitude"), static_field(100.0)),
            },
            coords={
                "time": times,
                "level": levels,
                "latitude": latitude,
                "longitude": longitude,
            },
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source_path = tmp_path / "source.zarr"
            output_path = tmp_path / "download.zarr"
            source_ds.to_zarr(source_path, mode="w")

            first_summary = download_arco_era5_subset(
                output_path,
                config_path="configs/model_config.yaml",
                dataset_url=source_path,
                start_time="2018-01-01 00:00:00",
                end_time="2018-01-01 01:00:00",
                overwrite=True,
                chunk_size=1,
                verbose=False,
                show_progress=False,
            )
            second_summary = download_arco_era5_subset(
                output_path,
                config_path="configs/model_config.yaml",
                dataset_url=source_path,
                start_time="2018-01-01 00:00:00",
                end_time="2018-01-01 03:00:00",
                resume=True,
                chunk_size=1,
                verbose=False,
                show_progress=False,
            )

            downloaded = xr.open_zarr(output_path, consolidated=None)

            self.assertEqual(first_summary["time_steps"], 2)
            self.assertEqual(second_summary["time_steps"], 4)
            self.assertEqual(second_summary["resumed_from_time_steps"], 2)
            self.assertEqual(int(downloaded.sizes["time"]), 4)
            self.assertIn("relative_humidity", downloaded.data_vars)
            self.assertNotIn("specific_humidity", downloaded.data_vars)
            self.assertIn("total_precipitation", downloaded.data_vars)
            self.assertIn("land_sea_mask", downloaded.data_vars)
            self.assertEqual(str(pd.Timestamp(downloaded["time"].values[0])), "2018-01-01 00:00:00")
            self.assertEqual(str(pd.Timestamp(downloaded["time"].values[-1])), "2018-01-01 03:00:00")
            self.assertGreater(float(downloaded["latitude"].values[0]), float(downloaded["latitude"].values[-1]))

    def test_repair_local_zarr_time_consistency_trims_longer_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "broken.zarr"
            ds = xr.Dataset(
                data_vars={
                    "a": (("time", "x"), np.ones((4, 2), dtype=np.float32)),
                    "b": (("time", "x"), np.ones((4, 2), dtype=np.float32) * 2),
                },
                coords={
                    "time": pd.date_range("2000-01-01", periods=4, freq="h"),
                    "x": np.array([0, 1], dtype=np.int64),
                },
            )
            ds.to_zarr(store_path, mode="w")

            a_meta_path = store_path / "a" / ".zarray"
            b_meta_path = store_path / "b" / ".zarray"
            time_meta_path = store_path / "time" / ".zarray"

            for meta_path in (b_meta_path, time_meta_path):
                meta = json.loads(meta_path.read_text())
                meta["shape"][0] = 6
                meta_path.write_text(json.dumps(meta))

            before = inspect_local_zarr_time_axes(store_path)
            self.assertEqual(sorted({entry["time_size"] for entry in before}), [4, 6])

            summary = repair_local_zarr_time_consistency(store_path, verbose=False)
            repaired = xr.open_zarr(store_path, consolidated=None)

            self.assertEqual(summary["target_time_size"], 4)
            self.assertEqual(sorted(summary["touched_arrays"]), ["b", "time"])
            self.assertEqual(int(repaired.sizes["time"]), 4)
            self.assertEqual(tuple(repaired["b"].shape), (4, 2))


if __name__ == "__main__":
    unittest.main()
