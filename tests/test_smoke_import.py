"""M0 smoke test: package imports + CLI parser builds + version string sane."""

from __future__ import annotations

import re


def test_package_imports():
    import bovin_demo

    assert hasattr(bovin_demo, "__version__")
    assert re.match(r"^\d+\.\d+\.\d+", bovin_demo.__version__)


def test_cli_parser_builds():
    from bovin_demo.cli import _build_parser

    parser = _build_parser()
    # The parser should expose the 4 planned subcommands.
    sub = next(a for a in parser._actions if a.dest == "command")
    assert set(sub.choices) == {"sanity", "train", "xai", "eval"}


def test_cli_sanity_subcommand_runs(capsys):
    from bovin_demo.cli import main

    code = main(["sanity"])
    assert code == 0
    out = capsys.readouterr().out
    assert "bovin_demo v" in out


def test_cli_eval_errors_on_missing_run_dir():
    """``eval`` lands in M6. With a non-existent run-dir it should raise —
    we just guard that it doesn't silently return 0."""
    import pytest as _pytest

    _pytest.importorskip("torch")
    _pytest.importorskip("pytorch_lightning")
    from bovin_demo.cli import main

    with _pytest.raises((FileNotFoundError, Exception)):
        main(["eval", "--run-dir", "nope/definitely_missing"])
