import numpy as np
from lstm_inferer import (
    cluster_initial_window,
    track_from_window,
    seed_initial_labels,
    hungarian_acc,
)

# Synthetic dataset generator

def make_synth(n_obj=3, steps=200, noise=0.3):
    t = np.arange(steps, dtype=float)
    velocities = np.linspace(-0.2, 0.2, n_obj)
    offsets = np.linspace(10, 40, n_obj)
    T = np.repeat(t, n_obj)
    labels = np.tile(np.arange(n_obj), steps)
    A = np.zeros_like(T)
    for j in range(n_obj):
        idx = labels == j
        A[idx] = offsets[j] + velocities[j]*t + np.random.randn(steps)*noise
    return T, A, labels, velocities, offsets


def build_logits(T, A, velocities, offsets, noise=0.3):
    n = len(T)
    n_obj = len(velocities)
    probs = np.zeros((n, n_obj))
    for i in range(n):
        t = T[i]
        obs = A[i]
        dists = [(obs - (offsets[j] + velocities[j]*t))**2 for j in range(n_obj)]
        p = np.exp(-np.array(dists)/(2*noise**2))
        probs[i] = p / p.sum()
    return probs


def experiment():
    T, A, true_labels, vels, offs = make_synth()
    logits = build_logits(T, A, vels, offs)
    windows = [20, 40, 60, 80]
    results = []
    for w in windows:
        seeds = cluster_initial_window(T, A, window=float(w), eps=2.0)
        labels = track_from_window(T, A, logits, seeds)
        acc, _ = hungarian_acc(true_labels, labels)
        seed_mask = (seeds >= 0)
        seed_acc, _ = hungarian_acc(true_labels[seed_mask], seeds[seed_mask]) if seed_mask.any() else (0.0, seeds)
        results.append((w, acc, seed_acc, seed_mask.sum()))
    print("Window\tTrackAcc\tSeedAcc\tSeedCount")
    for w, acc, seed_acc, count in results:
        print(f"{w}\t{acc:.3f}\t{seed_acc:.3f}\t{count}")

    # Compare seeding methods on fixed window, including automatic selection
    w = 50.0
    for method in ["ransac", "kalman", "auto"]:
        seeds = seed_initial_labels(T, A, num_classes=3, method=method)
        mask = T <= w
        mask &= (seeds >= 0)
        acc = hungarian_acc(true_labels[mask], seeds[mask])[0] if mask.any() else 0.0
        print(f"seeding {method}: acc={acc:.3f}, labeled={mask.sum()}")

def experiment_real():
    import csv
    T, A, labels = [], [], []
    fn = "data real/generated_mode_1_ensemble_pred.csv"
    with open(fn) as f:
        reader = csv.DictReader(f)
        for row in reader:
            T.append(float(row["Time"]))
            A.append(float(row["Azimuth"]))
            labels.append(int(float(row["Label"])))
    T = np.array(T)
    A = np.array(A)
    labels = np.array(labels)
    seeds = cluster_initial_window(T, A, window=50.0, eps=0.6, min_samples=5)
    mask = (T <= T[0] + 50.0) & (seeds >= 0)
    acc, _ = hungarian_acc(labels[mask], seeds[mask]) if mask.any() else (0.0, None)
    print(f"real data: clusters={len(set(seeds[mask]))} acc={acc:.3f} labeled={mask.sum()}")

if __name__ == "__main__":
    experiment()
    experiment_real()
