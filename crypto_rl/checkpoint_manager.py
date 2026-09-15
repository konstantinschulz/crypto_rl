"""
crypto_rl.checkpoint_manager
============================
Provides fold-level checkpointing and recovery for Optuna CV studies.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class CVCheckpointManager:
    """Manages fold-level checkpointing for cross-validation based on exact Trial ID."""

    def __init__(self, checkpoint_dir: Path | str = Path("logs/optuna_cv_checkpoints")):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def get_checkpoint_path(self, trial_number: int) -> Path:
        return self.checkpoint_dir / f"cv_ckpt_trial_{trial_number}.json"

    def load_checkpoint(self, trial_number: int) -> dict[str, Any] | None:
        """Load fold state if a checkpoint exists for this trial number."""
        ckpt_path = self.get_checkpoint_path(trial_number)
        if ckpt_path.exists():
            try:
                with open(ckpt_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load checkpoint {ckpt_path}: {e}")
        return None

    def save_fold_result(
        self,
        trial_number: int,
        fold_idx: int,
        score: float,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """Persist a completed fold score and metrics to disk."""
        ckpt_path = self.get_checkpoint_path(trial_number)
        data = self.load_checkpoint(trial_number) or {
            "trial_number": trial_number,
            "completed_folds": {},
        }
        data["completed_folds"][str(fold_idx)] = {
            "fold_idx": fold_idx,
            "score": float(score),
            "metrics": metrics or {},
        }
        with open(ckpt_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def clear_checkpoint(self, trial_number: int) -> None:
        """Remove checkpoint file once all folds for a trial complete or prune."""
        ckpt_path = self.get_checkpoint_path(trial_number)
        if ckpt_path.exists():
            try:
                ckpt_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete checkpoint {ckpt_path}: {e}")
