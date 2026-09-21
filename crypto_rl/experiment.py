# -*- coding: utf-8 -*-
"""Thin wrapper for experiment functionality.

The full implementation lives in :mod:`crypto_rl.experiment.runner` to keep
this module minimal and improve import times.
"""

from .runner import run_experiment  # noqa: F401

__all__ = ["run_experiment"]
