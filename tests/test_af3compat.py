import csv
import sys
import tempfile
import unittest
from pathlib import Path

try:
    from protenixscore.score import ScoreResult
    from protenixscore.score import _strip_templates_from_json
    from protenixscore.score import _write_aggregate_csv
    from protenixscore.score import parse_seed_spec
except ModuleNotFoundError:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from score import ScoreResult  # type: ignore
    from score import _strip_templates_from_json  # type: ignore
    from score import _write_aggregate_csv  # type: ignore
    from score import parse_seed_spec  # type: ignore


class TestAF3CompatHelpers(unittest.TestCase):
    def test_parse_seed_spec_range(self):
        self.assertEqual(parse_seed_spec("0-3"), [0, 1, 2, 3])

    def test_parse_seed_spec_reverse_range(self):
        self.assertEqual(parse_seed_spec("3-1"), [3, 2, 1])

    def test_parse_seed_spec_mixed(self):
        self.assertEqual(parse_seed_spec("1,4-6,9"), [1, 4, 5, 6, 9])

    def test_strip_templates(self):
        payload = [
            {
                "name": "sample",
                "sequences": [
                    {"proteinChain": {"sequence": "AAAA", "templates": [{"id": "foo"}]}},
                    {"protein": {"sequence": "BBBB", "templates": [{"id": "bar"}]}},
                ],
            }
        ]
        _strip_templates_from_json(payload)
        self.assertEqual(payload[0]["sequences"][0]["proteinChain"]["templates"], [])
        self.assertEqual(payload[0]["sequences"][1]["protein"]["templates"], [])

    def test_aggregate_csv_header_contract(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "metrics.csv"
            result = ScoreResult(
                sample_name="x",
                summary={
                    "plddt": 1.0,
                    "ptm": 2.0,
                    "iptm": 3.0,
                    "ranking_score": 4.0,
                    "ipsae_interface_max": 5.0,
                    "ipsae_target_to_binder": 6.0,
                    "ipsae_binder_to_target": 7.0,
                },
                full_data=None,
                output_dir=Path(td),
            )
            _write_aggregate_csv([result], out)
            with out.open() as f:
                reader = csv.reader(f)
                header = next(reader)
            self.assertEqual(
                header,
                [
                    "sample",
                    "plddt",
                    "ptm",
                    "iptm",
                    "ranking_score",
                    "ipsae_interface_max",
                    "ipsae_target_to_binder",
                    "ipsae_binder_to_target",
                ],
            )

    def test_aggregate_csv_per_sample_rows(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "metrics.csv"
            result = ScoreResult(
                sample_name="x",
                summary={},
                full_data=None,
                output_dir=Path(td),
                per_sample_rows=[
                    {
                        "sample": "x__sample_000",
                        "plddt": 10.0,
                        "ptm": 1.0,
                        "iptm": 2.0,
                        "ranking_score": 3.0,
                        "ipsae_interface_max": 4.0,
                        "ipsae_target_to_binder": 5.0,
                        "ipsae_binder_to_target": 6.0,
                    },
                    {
                        "sample": "x__sample_001",
                        "plddt": 11.0,
                        "ptm": 1.1,
                        "iptm": 2.1,
                        "ranking_score": 3.1,
                        "ipsae_interface_max": 4.1,
                        "ipsae_target_to_binder": 5.1,
                        "ipsae_binder_to_target": 6.1,
                    },
                ],
            )
            _write_aggregate_csv([result], out)
            with out.open() as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["sample"], "x__sample_000")
            self.assertEqual(rows[1]["sample"], "x__sample_001")


if __name__ == "__main__":
    unittest.main()
