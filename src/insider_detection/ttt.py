"""Simple test-time adaptation utilities for the scikit-learn model.

This implements a tiny pseudo-labeling loop: for an unlabeled sample we take the
model's predicted probability, and if the max probability exceeds a threshold we
call `partial_fit` on that pseudo-label for a few steps.
"""
import numpy as np


def adapt_entropy_minimization(model, x_batch, steps: int = 5, threshold: float = 0.6):
    """Adapt `model` in-place using pseudo-labels from `x_batch`.

    - `model` should expose `predict_proba` and `model.partial_fit` (our wrapper
      uses `MLPClassifier.partial_fit` via `fit` loop in `model.fit`).
    - `x_batch` can be a numpy array or convertible to one with shape [B, D].
    """
    X = np.asarray(x_batch)
    # if the underlying sklearn estimator exposes predict_proba
    probs = model.predict_proba(X)
    for i in range(len(X)):
        p = probs[i]
        lab = int(np.argmax(p))
        if np.max(p) >= threshold:
            # perform a few small adaptation steps using partial_fit via fit(..., epochs=1)
            model.fit(X[i:i+1], np.array([lab]), epochs=steps)
    return model
