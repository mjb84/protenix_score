from __future__ import annotations

from argparse import Namespace

try:
    from protenixscore.score import run_predict
    from protenixscore.score import run_score
except ModuleNotFoundError:
    from score import run_predict  # type: ignore
    from score import run_score  # type: ignore


def run_score_only(args: Namespace) -> None:
    run_score(args)


def run_predict_refold(args: Namespace) -> None:
    run_predict(args)
