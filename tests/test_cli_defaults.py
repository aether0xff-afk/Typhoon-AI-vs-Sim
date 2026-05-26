import unittest

from typhoon_ai_vs_sim.cli import build_parser
from typhoon_ai_vs_sim.config import ExperimentConfig


class CliDefaultTests(unittest.TestCase):
    def test_cli_default_data_source_is_ibtracs(self):
        args = build_parser().parse_args([])

        self.assertEqual(args.data_source, "ibtracs")

    def test_config_default_data_source_is_ibtracs(self):
        self.assertEqual(ExperimentConfig().data_source, "ibtracs")


if __name__ == "__main__":
    unittest.main()
