import numpy as np

def compute_ece(confidences: np.ndarray, accuracies: np.ndarray, num_bins: int = 10) -> float:
    """Expected Calibration Error."""
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    total = len(confidences)
    for i in range(num_bins):
        bin_mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        if i == num_bins - 1:
            bin_mask = (confidences >= bin_edges[i]) & (confidences <= bin_edges[i + 1])
        
        bin_count = np.sum(bin_mask)
        if bin_count > 0:
            bin_acc = np.mean(accuracies[bin_mask])
            bin_conf = np.mean(confidences[bin_mask])
            ece += (bin_count / total) * np.abs(bin_acc - bin_conf)
    return float(ece)

def compute_mce(confidences: np.ndarray, accuracies: np.ndarray, num_bins: int = 10) -> float:
    """Maximum Calibration Error."""
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    mce = 0.0
    for i in range(num_bins):
        bin_mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        if i == num_bins - 1:
            bin_mask = (confidences >= bin_edges[i]) & (confidences <= bin_edges[i + 1])
        
        bin_count = np.sum(bin_mask)
        if bin_count > 0:
            bin_acc = np.mean(accuracies[bin_mask])
            bin_conf = np.mean(confidences[bin_mask])
            mce = max(mce, float(np.abs(bin_acc - bin_conf)))
    return mce

def compute_brier_score(probs: np.ndarray, targets: np.ndarray) -> float:
    """Brier score."""
    one_hot_targets = np.zeros_like(probs)
    one_hot_targets[np.arange(len(targets)), targets] = 1.0
    return float(np.mean(np.sum((probs - one_hot_targets) ** 2, axis=1)))

def compute_nll(probs: np.ndarray, targets: np.ndarray) -> float:
    """Negative log likelihood."""
    eps = 1e-15
    probs = np.clip(probs, eps, 1 - eps)
    target_probs = probs[np.arange(len(targets)), targets]
    return float(-np.mean(np.log(target_probs)))

def compute_confidence(probs: np.ndarray) -> float:
    """Normalized entropy-based confidence: 1 - H(p)/log(K)."""
    eps = 1e-15
    probs = np.clip(probs, eps, 1.0)
    k = probs.shape[-1]
    if k <= 1:
        return 1.0
    entropy = -np.sum(probs * np.log(probs), axis=-1)
    max_entropy = np.log(k)
    return float(np.mean(1.0 - (entropy / max_entropy)))


def expected_calibration_error(probs: np.ndarray, target: np.ndarray, mask: np.ndarray, num_bins: int = 15) -> float:
    """Top-1 ECE for padded multi-option outputs; correct = prediction is one of the target's best options."""
    p = np.where(mask, probs, -1.0)
    pred = p.argmax(-1)
    conf = p.max(-1)
    best = np.where(mask, target, -1.0).max(-1)
    correct = (target[np.arange(len(pred)), pred] >= best - 1e-6).astype(np.float64)
    return float(compute_ece(conf, correct, num_bins))


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(c - h), float(c + h))
