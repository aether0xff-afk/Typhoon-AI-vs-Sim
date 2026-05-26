import unittest

from typhoon_ai_vs_sim.cli import build_parser
from typhoon_ai_vs_sim.config import ExperimentConfig
from typhoon_ai_vs_sim.data import prepare_data
from typhoon_ai_vs_sim.simulation import generate_storm_tracks


class CliDefaultTests(unittest.TestCase):
    def test_cli_default_data_source_is_ibtracs(self):
        args = build_parser().parse_args([])

        self.assertEqual(args.data_source, "ibtracs")

    def test_config_default_data_source_is_ibtracs(self):
        self.assertEqual(ExperimentConfig().data_source, "ibtracs")

    def test_repeat_seeds_default_is_empty(self):
        args = build_parser().parse_args([])

        self.assertEqual(args.repeat_seeds, "")


class DataPreparationTests(unittest.TestCase):
    def test_target_scalers_fit_train_only_and_storm_splits_do_not_overlap(self):
        config = ExperimentConfig(data_source="synthetic", n_storms=12, seed=42)
        storms = generate_storm_tracks(config)

        prepared = prepare_data(config, storms)

        train_ids = set(prepared.train.bundle.storm_ids.tolist())
        val_ids = set(prepared.val.bundle.storm_ids.tolist())
        test_ids = set(prepared.test.bundle.storm_ids.tolist())

        self.assertTrue(train_ids.isdisjoint(val_ids))
        self.assertTrue(train_ids.isdisjoint(test_ids))
        self.assertTrue(val_ids.isdisjoint(test_ids))
        self.assertAlmostEqual(
            prepared.scaler.residual_target_mean,
            float(prepared.train.bundle.residual_targets.mean()),
        )
        self.assertAlmostEqual(
            prepared.scaler.direct_target_mean,
            float(prepared.train.bundle.true_targets.mean()),
        )
        self.assertIn("residual_target", prepared.train.dataset[0])
        self.assertIn("direct_target", prepared.train.dataset[0])

    def test_repeat_seeds_are_parsed(self):
        args = build_parser().parse_args(["--repeat-seeds", "42,7,13"])

        self.assertEqual(args.repeat_seeds, "42,7,13")


if __name__ == "__main__":
    unittest.main()
