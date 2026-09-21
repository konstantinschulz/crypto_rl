# -*- coding: utf-8 -*-
"""Experiment package exposing the run_experiment function.

The heavy lifting is performed in :mod:`crypto_rl.experiment.runner`.
"""

from .runner import run_experiment

__all__ = ["run_experiment"]
