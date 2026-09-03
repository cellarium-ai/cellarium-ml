# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

from collections import deque

import numpy as np


class NoisyConvergenceTracker:
    def __init__(self, window_size: int = 20, patience: int = 50, min_delta: float = 1e-4):
        """
        Args:
            window_size: How many recent steps to average together to smooth the noise.
            patience: How many steps to wait without improvement before triggering convergence.
            min_delta: The minimum improvement required to reset the patience counter.
        """
        self.window = deque(maxlen=window_size)
        self.patience = patience
        self.min_delta = min_delta

        self.best_smoothed_loss = float("inf")
        self.wait_count = 0

    def check_convergence(self, current_loss: float) -> bool:
        """
        Returns True if the metric has converged, False otherwise.
        """
        self.window.append(current_loss)

        # Don't start checking until we have a full window
        if len(self.window) < self.window.maxlen:
            return False

        # Calculate the smoothed loss
        current_smoothed_loss = np.mean(self.window)

        # Check if it improved by at least min_delta
        if current_smoothed_loss < self.best_smoothed_loss - self.min_delta:
            # Significant improvement! Reset the tracker.
            self.best_smoothed_loss = current_smoothed_loss
            self.wait_count = 0
        else:
            # No significant improvement. Increment patience.
            self.wait_count += 1

        # If we have waited long enough without improvement, we are done
        return self.wait_count >= self.patience

    def converged(self) -> bool:
        """Convenience method to check if we have converged without needing to pass in a new loss."""
        return (len(self.window) >= self.window.maxlen) and (self.wait_count >= self.patience)

    def reset(self):
        """Resets the tracker to its initial state."""
        self.window.clear()
        self.best_smoothed_loss = float("inf")
        self.wait_count = 0
