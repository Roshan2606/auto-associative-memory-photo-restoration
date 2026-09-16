import numpy as np


class HopfieldNetwork:
    """A small Hopfield-style associative memory network for educational use."""

    def __init__(self, size, iterations=50, threshold=0.0):
        if size <= 0:
            raise ValueError("Network size must be greater than zero.")

        self.size = int(size)
        self.iterations = int(iterations)
        self.threshold = float(threshold)
        self.weights = np.zeros((self.size, self.size), dtype=float)
        self.patterns = []
        self.trained = False

    def _validate_pattern(self, pattern):
        vector = np.asarray(pattern, dtype=float).reshape(-1)
        if vector.size != self.size:
            raise ValueError(
                f"Expected pattern size {self.size}, but received {vector.size}."
            )
        return vector.astype(float)

    def store_pattern(self, pattern):
        """Train the network using Hebbian learning."""
        vector = self._validate_pattern(pattern)
        self.patterns.append(vector.copy())

        # Hebbian rule: W = (1/N) sum_i x_i x_i^T with diagonal set to zero.
        self.weights += np.outer(vector, vector)
        self.weights = self.weights / self.size
        np.fill_diagonal(self.weights, 0.0)
        self.trained = True
        return self.weights.copy()

    def recall(self, corrupted_pattern, iterations=None):
        """Recall a stored pattern from a damaged or incomplete version."""
        if not self.trained:
            raise ValueError("The network must be trained before recalling a pattern.")

        state = self._validate_pattern(corrupted_pattern).copy()
        steps = self.iterations if iterations is None else int(iterations)

        for _ in range(steps):
            activation = self.weights.dot(state)
            state = np.where(activation >= self.threshold, 1.0, -1.0)

        return state

    def summarize(self):
        return {
            "size": self.size,
            "iterations": self.iterations,
            "threshold": self.threshold,
            "trained": self.trained,
        }
