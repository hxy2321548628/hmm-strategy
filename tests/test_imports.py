"""Importing the installed package must not launch research or load market files."""

import importlib
import pkgutil

import pandas as pd

import benke


def test_import_has_no_io(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("Import triggered data I/O")

    monkeypatch.setattr(pd, "read_pickle", unexpected)
    monkeypatch.setattr(pd, "read_csv", unexpected)
    for info in pkgutil.walk_packages(benke.__path__, benke.__name__ + "."):
        if info.name != "benke.__main__":
            importlib.import_module(info.name)
