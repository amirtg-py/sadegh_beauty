# -*- coding: utf-8 -*-
"""
Inference + plotting for trained LSTM/Transformer models (improved).
- Loads both models if available and does ENSEMBLE by default
- Post-process uniqueness uses a COMBINED cost (angle distance + -log prob)
- Confidence gating: do not force reassignment if model is not confident
- Saves CSV + plots; if Label exists -> reports Hungarian accuracy

Base file: lstm_inferer.py (updated)  # see: your previous file
"""

import os, math, warnings, json, glob
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.cluster import KMeans, DBSCAN

try:  # Kalman filter is optional
    from filterpy.kalman import KalmanFilter
    HAS_FILTERPY = True
except Exception:  # pragma: no cover - optional dependency
    KalmanFilter = None
    HAS_FILTERPY = False

# ---------------- FIXED PATHS ----------------
MODEL_DIR = r"D:\work_station\radar_co\radar\final\sadegh_beauty-codex-improve-data-generation-script-accuracy\generated_datalstm out"
OUT_DIR   = r"D:\work_station\radar_co\radar\final\sadegh_beauty-codex-improve-data-generation-script-accuracy\real_analyz"

# (1) پوشه یا فایل‌ها را اینجا معرفی کنید:
TEST_DIR = r"D:\work_station\data"
TEST_FILES = glob.glob(os.path.join(TEST_DIR, "*.csv"))

# ---------------- Inference Options ----------------
SEQ_LEN = 64
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

USE_ENSEMBLE = True       # اگر هر دو مدل باشند، میانگین نرم‌احتمال
PP_ALPHA = 1.0            # وزن فاصله زاویه‌ای در هزینه یگانگی
PP_BETA  = 2.0            # وزن -log(prob) در هزینه یگانگی
PP_CONF_THRESH = 0.40     # حداقل اعتماد برای جابجایی (prob کلاس فعلی < این مقدار)
SMOOTH_MIN_RUN = 6        # هموارسازی: حداقل طول سگمنت

# --- initial label seeding on a small time window ---
SEED_WINDOW = 50.0           # عرض پنجرهٔ اولیه (واحد زمان)
SEED_METHOD = "auto"         # 'ransac' | 'kalman' | 'auto'
SEED_MIN_SAMPLES = 5         # حداقل نقاط برای هر مدل
SEED_RANSAC_RESID = 1.0      # آستانهٔ خطای رانساک

# --- density-based clustering + tracking parameters ---
TRACK_ENABLE = True           # فعال‌سازی الگوریتم پنجره‌ای
TRACK_CLUSTER_EPS = 1.0       # حساسیت خوشه‌بندی DBSCAN
TRACK_CLUSTER_MIN = 3         # حداقل نمونه در خوشه ابتدایی
TRACK_GAP_THRESH = 20.0       # آستانه گپ زمانی برای پایان مسیر
TRACK_DIST_THRESH = 5.0       # آستانه فاصله زاویه‌ای برای نسبت دادن نقطه

# ---------------- Utils ----------------
def log(msg): print(msg, flush=True)
def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)
def ang_diff(a, b): return (a - b + 180.0) % 360.0 - 180.0

def robust_velocity(time: np.ndarray, az: np.ndarray) -> np.ndarray:
    n=len(az); v=np.zeros(n, dtype=float)
    for i in range(n):
        if i==0:
            dt=time[i+1]-time[i] if i+1<n else 1.0;  dt = 1e-6 if dt==0 else dt
            v[i]=ang_diff(az[i+1],az[i])/dt if i+1<n else 0.0
        elif i==n-1:
            dt=time[i]-time[i-1]; dt = 1e-6 if dt==0 else dt
            v[i]=ang_diff(az[i],az[i-1])/dt
        else:
            dt=time[i+1]-time[i-1]; dt = 1e-6 if dt==0 else dt
            v[i]=ang_diff(az[i+1],az[i-1])/dt
    return v

def robust_acceleration(v: np.ndarray) -> np.ndarray:
    n=len(v); a=np.zeros(n, dtype=float)
    for i in range(n):
        if i==0:    a[i]=v[i+1]-v[i] if n>1 else 0.0
        elif i==n-1:a[i]=v[i]-v[i-1]
        else:       a[i]=0.5*((v[i+1]-v[i])+(v[i]-v[i-1]))
    return a

def hungarian_acc(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, np.ndarray]:
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    _, inv = np.unique(y_pred, return_inverse=True)
    cm = confusion_matrix(y_true, inv)
    if cm.size == 0: return 0.0, y_pred
    cost = cm.max() - cm
    r, c = linear_sum_assignment(cost)
    acc = cm[r, c].sum() / cm.sum()
    mapping = {pc: tr for pc, tr in zip(c, r)}
    mapped = np.array([mapping[p] for p in inv])
    return acc, mapped

def plot_two(T, A, left_labels, right_labels, left_title, right_title, save_path: Path):
    fig, ax = plt.subplots(1,2, figsize=(14,5), sharey=True)
    ax[0].scatter(T, A, s=3, c=left_labels, cmap="tab20")
    ax[0].set_title(left_title); ax[0].set_xlabel("Time"); ax[0].set_ylabel("Azimuth (deg)")
    ax[1].scatter(T, A, s=3, c=right_labels, cmap="tab20")
    ax[1].set_title(right_title); ax[1].set_xlabel("Time")
    plt.tight_layout(); fig.savefig(save_path, dpi=160); plt.close(fig)
    log(f"[save] {save_path}")

def smooth_short_runs(labels: np.ndarray, min_len: int=SMOOTH_MIN_RUN) -> np.ndarray:
    lab=labels.copy(); n=len(lab); i=0
    while i<n:
        j=i
        while j+1<n and lab[j+1]==lab[i]: j+=1
        seglen=j-i+1
        if seglen<min_len:
            prev_lab = lab[i-1] if i>0 else None
            next_lab = lab[j+1] if j+1<n else None
            cand = prev_lab if prev_lab is not None else next_lab
            if cand is not None: lab[i:j+1]=cand
        i=j+1
    return lab

# ---------------- Initial Label Seeding ----------------
def seed_labels_ransac(T: np.ndarray, A: np.ndarray, num_classes: int,
                       window: float=SEED_WINDOW,
                       min_samples: int=SEED_MIN_SAMPLES,
                       resid: float=SEED_RANSAC_RESID) -> np.ndarray:
    mask = T <= (T[0] + window)
    Tw, Aw = T[mask], A[mask]
    out = np.full(len(T), -1, dtype=int)
    remain = np.arange(len(Tw))
    for lab in range(num_classes):
        if len(remain) < min_samples:
            break
        # scikit-learn >=1.1 uses 'estimator' instead of 'base_estimator'
        # for older versions this argument is still named 'base_estimator'.
        # Detect which one is available to keep compatibility.
        ransac_kwargs = dict(min_samples=min_samples,
                             residual_threshold=resid)
        if 'estimator' in RANSACRegressor.__init__.__code__.co_varnames:
            ransac_kwargs['estimator'] = LinearRegression()
        else:  # pragma: no cover - older scikit-learn
            ransac_kwargs['base_estimator'] = LinearRegression()
        ransac = RANSACRegressor(**ransac_kwargs)
        ransac.fit(Tw[remain].reshape(-1,1), Aw[remain])
        inliers = remain[ransac.inlier_mask_]
        if len(inliers) < min_samples:
            break
        out[np.where(mask)[0][inliers]] = lab
        remain = remain[~ransac.inlier_mask_]
    return out


def seed_labels_kalman(T: np.ndarray, A: np.ndarray, num_classes: int,
                       window: float=SEED_WINDOW) -> np.ndarray:
    mask = T <= (T[0] + window)
    out = np.full(len(T), -1, dtype=int)
    if not HAS_FILTERPY or mask.sum() < 2:
        return out
    k = min(num_classes, mask.sum())
    if k <= 0:
        return out
    km = KMeans(n_clusters=k, n_init=10).fit(A[mask].reshape(-1,1))
    labels = km.labels_
    out[np.where(mask)[0]] = labels
    # simple Kalman smoothing per cluster
    for lab in range(k):
        idx = np.where(out == lab)[0]
        if len(idx) < 2:
            continue
        kf = KalmanFilter(dim_x=2, dim_z=1)
        kf.H = np.array([[1.,0.]])
        kf.P *= 10
        kf.R *= 5
        kf.Q = np.eye(2) * 0.01
        kf.x = np.array([A[idx[0]], 0.0])
        prev_t = T[idx[0]]
        for j in idx[1:]:
            dt = T[j] - prev_t
            kf.F = np.array([[1., dt],[0.,1.]])
            kf.predict()
            kf.update(A[j])
            prev_t = T[j]
    return out


def seed_initial_labels(T: np.ndarray, A: np.ndarray, num_classes: int,
                        method: str=SEED_METHOD) -> np.ndarray:
    if method == "ransac":
        return seed_labels_ransac(T, A, num_classes)
    if method == "kalman":
        return seed_labels_kalman(T, A, num_classes)
    # auto: pick method with more labeled points
    r = seed_labels_ransac(T, A, num_classes)
    k = seed_labels_kalman(T, A, num_classes)
    if (k >= 0).sum() > (r >= 0).sum():
        return k
    return r


# ---------------- Window clustering and tracking ----------------
def cluster_initial_window(T: np.ndarray, A: np.ndarray,
                           window: float=SEED_WINDOW,
                           eps: float=TRACK_CLUSTER_EPS,
                           min_samples: int=TRACK_CLUSTER_MIN) -> np.ndarray:
    """Label dense groups in the first time window using DBSCAN."""
    mask = T <= (T[0] + window)
    out = np.full(len(T), -1, dtype=int)
    coords = np.column_stack((T[mask], A[mask]))
    if len(coords) == 0:
        return out
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labs = db.fit_predict(coords)
    valid = labs >= 0
    out[np.where(mask)[0][valid]] = labs[valid]
    return out


def track_from_window(T: np.ndarray, A: np.ndarray, logits: np.ndarray,
                      seeds: np.ndarray,
                      gap: float=TRACK_GAP_THRESH,
                      dist_thr: float=TRACK_DIST_THRESH) -> np.ndarray:
    """Propagate labels forward using constant-velocity prediction and model probs."""
    labels = seeds.copy()
    active = {}
    for lab in np.unique(seeds):
        if lab < 0:
            continue
        idxs = np.where(seeds == lab)[0]
        last = idxs[-1]
        if len(idxs) >= 2:
            i1, i2 = idxs[-2], idxs[-1]
            dt = T[i2] - T[i1]
            dt = 1e-6 if dt == 0 else dt
            vel = ang_diff(A[i2], A[i1]) / dt
        else:
            vel = 0.0
        active[lab] = {"last": last, "vel": vel}
    start = (np.where(seeds >= 0)[0].max() + 1) if np.any(seeds >= 0) else 0
    for i in range(start, len(T)):
        if labels[i] >= 0:
            continue
        best_lab = None
        best_cost = None
        for lab, st in list(active.items()):
            dt = T[i] - T[st["last"]]
            if dt > gap:
                continue
            pred = A[st["last"]] + st["vel"] * dt
            dist = abs(ang_diff(A[i], pred))
            prob = logits[i, lab] if lab < logits.shape[1] else 0.0
            cost = dist - prob
            if dist <= dist_thr and (best_cost is None or cost < best_cost):
                best_cost = cost
                best_lab = lab
        if best_lab is not None:
            prev = active[best_lab]["last"]
            dt = T[i] - T[prev]
            dt = 1e-6 if dt == 0 else dt
            active[best_lab]["vel"] = ang_diff(A[i], A[prev]) / dt
            active[best_lab]["last"] = i
            labels[i] = best_lab
    # fill remaining gaps by average model probability
    unlabeled = np.where(labels == -1)[0]
    start = 0
    while start < len(unlabeled):
        end = start
        while end + 1 < len(unlabeled) and unlabeled[end + 1] == unlabeled[end] + 1:
            end += 1
        block = unlabeled[start:end + 1]
        avg = logits[block].mean(axis=0)
        labels[block] = int(np.argmax(avg))
        start = end + 1
    return labels

# ---------------- Feature Engineering ----------------
def engineer_features(df: pd.DataFrame):
    T = df["Time"].to_numpy().astype(float)
    A = df["Azimuth"].to_numpy().astype(float)
    t_norm = 10 * (T - T.min()) / (T.max()-T.min() if T.max()>T.min() else 1.0)
    az = np.deg2rad(A)
    s = np.sin(az); c = np.cos(az)
    v = robust_velocity(T, A); a = robust_acceleration(v)
    vs = (v - v.mean()) /(v.std() if v.std()!=0 else 1.0)
    as_ = (a - a.mean())/(a.std() if a.std()!=0 else 1.0)
    X = np.vstack([t_norm, s, c, vs, as_]).T.astype(np.float32)
    return X, T, A

def build_sequences(X: np.ndarray, seq_len=SEQ_LEN):
    N, F = X.shape
    if N < seq_len:
        return np.empty((0,seq_len,F), np.float32), np.empty((0,), np.int64)
    M = N - seq_len + 1
    X_seq = np.zeros((M, seq_len, F), np.float32)
    idx_map = np.arange(seq_len-1, N, dtype=np.int64)
    for i in range(M):
        X_seq[i] = X[i:i+seq_len]
    return X_seq, idx_map

# ---------------- Models (same as training) ----------------
class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim=5, hidden=128, layers=2, num_classes=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.fc1  = nn.Linear(hidden*2, hidden)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2  = nn.Linear(hidden, num_classes)
    def forward(self, x):
        y,_ = self.lstm(x)
        h = y[:,-1,:]
        z = self.drop(self.relu(self.fc1(h)))
        return self.fc2(z)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(pos*div); pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:,:x.size(1),:]

class TransEncClassifier(nn.Module):
    def __init__(self, input_dim=5, d_model=64, nhead=4, num_layers=2,
                 dim_ff=128, num_classes=2, dropout=0.2, pos_max_len=2000):
        super().__init__()
        self.inp = nn.Linear(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pe  = PositionalEncoding(d_model, max_len=pos_max_len)
        self.fc  = nn.Linear(d_model, num_classes)

    def forward(self, x):
        z = self.inp(x)
        z = self.pe(z)
        y = self.enc(z)
        h = y[:, -1, :]
        return self.fc(h)


def detect_num_classes_from_state_dict(sd, head=("fc2.weight","fc.weight")):
    for k in head:
        if k in sd: return int(sd[k].shape[0])
    # fallback: کمترین out_features را پیدا می‌کنیم
    shapes = [v.shape for k,v in sd.items() if k.endswith(".weight")]
    shapes.sort(key=lambda s: s[0])
    return int(shapes[0][0])

def load_models(model_dir=MODEL_DIR):
    lstm_path  = Path(model_dir, "model_lstm.pt")
    trans_path = Path(model_dir, "model_trans.pt")
    lstm = trans = None; C=None

    # LSTM
    if lstm_path.exists():
        sd = torch.load(lstm_path, map_location="cpu")
        # تعداد کلاس‌ها از وزن لایهٔ آخر:
        C  = int(sd["fc2.weight"].shape[0]) if "fc2.weight" in sd else None
        lstm = BiLSTMClassifier(input_dim=5, hidden=128, layers=2,
                                num_classes=C if C is not None else 2, dropout=0.2)
        lstm.load_state_dict(sd); lstm.to(DEVICE).eval()

    # Transformer
    if trans_path.exists():
        sd = torch.load(trans_path, map_location="cpu")
        if C is None:
            # اگر هنوز C معلوم نیست، از هد ترنس بخوان
            if "fc.weight" in sd: C = int(sd["fc.weight"].shape[0])
            else:
                # fallback: کمترین out_features لایهٔ وزن
                shapes = [v.shape for k,v in sd.items() if k.endswith(".weight")]
                shapes.sort(key=lambda s: s[0])
                C = int(shapes[0][0])

        # d_model از وزن ورودی (inp.weight: [d_model, input_dim])
        d_model = sd["inp.weight"].shape[0] if "inp.weight" in sd else 64

        # طول positional از بافر pe.pe
        pos_max_len = 2000
        if "pe.pe" in sd:
            pos_max_len = int(sd["pe.pe"].shape[1])  # [1, max_len, d_model]

        trans = TransEncClassifier(input_dim=5, d_model=d_model, nhead=4, num_layers=2,
                                   dim_ff=128, num_classes=C, dropout=0.2,
                                   pos_max_len=pos_max_len)
        trans.load_state_dict(sd); trans.to(DEVICE).eval()

    if lstm is None and trans is None:
        raise FileNotFoundError("No trained model in MODEL_DIR")

    return lstm, trans, C


@torch.no_grad()
def infer_logits(model, X_seq, C):
    loader = DataLoader(torch.tensor(X_seq), batch_size=256, shuffle=False)
    logits_all = np.zeros((len(X_seq), C), np.float32)
    off=0
    for xb in loader:
        xb = xb.to(DEVICE)
        out = model(xb).softmax(1).cpu().numpy()
        logits_all[off:off+len(out)] = out
        off += len(out)
    return logits_all

# ---------------- Post-processing (improved uniqueness) ----------------
def enforce_uniqueness_prob_aware(T, A, labels, logits, alpha=PP_ALPHA, beta=PP_BETA, conf_thr=PP_CONF_THRESH):
    """
    If multiple points share same class at a timestamp, assign by cost:
       cost(i -> class L) = alpha * |ang - linear_pred_L(tt)| + beta * (-log prob[i,L])
    Only reassign if prob of current class < conf_thr and new cost is strictly lower.
    """
    out = labels.copy()
    from sklearn.linear_model import LinearRegression
    # build linear model for each class using its current assigned samples
    models={}
    for L in np.unique(labels):
        idx = np.where(labels==L)[0]
        if len(idx) < 3:
            models[L] = (0.0, float(np.median(A[idx]) if len(idx)>0 else 0.0))
        else:
            lr = LinearRegression().fit(T[idx].reshape(-1,1), A[idx])
            models[L] = (float(lr.coef_[0]), float(lr.intercept_))

    uniq_t = np.unique(T)
    for tt in uniq_t:
        inds = np.where(T==tt)[0]
        if len(inds)<=1: continue
        # group by predicted class
        groups={}
        for ii in inds:
            L = labels[ii]
            groups.setdefault(L, []).append(ii)
        for L, idxs in groups.items():
            if len(idxs)==1: continue
            # keep the one with best (lowest) cost for class L
            betaL,bL = models[L]
            def cost_for(i, Lcand):
                betaC, bC = models[Lcand] if Lcand in models else (0.0, np.median(A))
                angle_term = abs(ang_diff(A[i], betaC*tt + bC))
                prob_term  = -math.log(max(1e-6, float(logits[i, Lcand])))
                return alpha*angle_term + beta*prob_term

            # choose best for class L among idxs
            best_idx = min(idxs, key=lambda k: cost_for(k, L))
            for k in idxs:
                if k==best_idx: continue
                # consider reassigning k to an alternative class with smaller cost
                cur_prob = float(logits[k, L])
                if cur_prob >= conf_thr:  # confident enough -> do not force change
                    continue
                bestL = L; bestCost = cost_for(k, L)
                for L2 in models.keys():
                    if L2==L: continue
                    c2 = cost_for(k, L2)
                    if c2 < bestCost:
                        bestCost = c2; bestL = L2
                out[k] = bestL
    return out

# ---------------- Pipeline per file ----------------
def process_one(csv_path: str):
    df = pd.read_csv(csv_path).sort_values("Time").reset_index(drop=True)
    assert "Time" in df.columns and "Azimuth" in df.columns, "CSV must contain Time and Azimuth."
    X, T, A = engineer_features(df)
    X_seq, idx_map = build_sequences(X, SEQ_LEN)
    if len(X_seq)==0:
        log(f"[skip] too few rows for SEQ_LEN={SEQ_LEN}: {csv_path}")
        return

    # load models
    lstm, trans, C = load_models(MODEL_DIR)
    # logits
    if USE_ENSEMBLE and (lstm is not None) and (trans is not None):
        logits1 = infer_logits(lstm,  X_seq, C)
        logits2 = infer_logits(trans, X_seq, C)
        logits  = (logits1 + logits2)/2.0
        model_tag = "ensemble"
    else:
        if lstm is not None:
            logits = infer_logits(lstm, X_seq, C); model_tag = "lstm"
        else:
            logits = infer_logits(trans, X_seq, C); model_tag = "trans"

    y_pred = np.argmax(logits, axis=1)

    # expand to per-row
    y_row = np.zeros(len(df), dtype=int)
    y_row[:SEQ_LEN-1] = y_pred[0]
    y_row[SEQ_LEN-1:]  = y_pred

    # improved post-processing
    # 1) uniqueness with prob-aware cost
    # reshape logits to per-row as well
    logits_row = np.zeros((len(df), logits.shape[1]), np.float32)
    logits_row[:SEQ_LEN-1] = logits[0]
    logits_row[SEQ_LEN-1:] = logits
    y_pp = enforce_uniqueness_prob_aware(T, A, y_row, logits_row, alpha=PP_ALPHA, beta=PP_BETA, conf_thr=PP_CONF_THRESH)
    # 2) smoothing
    y_pp = smooth_short_runs(y_pp, min_len=SMOOTH_MIN_RUN)

    # 3) seed initial labels using classical methods
    seeds = seed_initial_labels(T, A, C, method=SEED_METHOD)
    m = seeds >= 0
    if m.any():
        y_row[m] = seeds[m]
        y_pp[m] = seeds[m]

    # 4) optional clustering + tracking to propagate labels
    if TRACK_ENABLE:
        seed_w = cluster_initial_window(T, A, window=SEED_WINDOW,
                                        eps=TRACK_CLUSTER_EPS,
                                        min_samples=TRACK_CLUSTER_MIN)
        comb = np.where(seed_w >= 0, seed_w, seeds)
        y_pp = track_from_window(T, A, logits_row, comb,
                                 gap=TRACK_GAP_THRESH,
                                 dist_thr=TRACK_DIST_THRESH)

    base  = Path(csv_path).stem
    out_csv = Path(OUT_DIR, "real", f"{base}_{model_tag}_inference.csv")
    dfo = df.copy(); dfo["PredRaw"]=y_row; dfo["PredPost"]=y_pp
    dfo.to_csv(out_csv, index=False); log(f"[save] {out_csv}")

    # plots
    if "Label" in df.columns:
        acc_raw, mapped_raw = hungarian_acc(df["Label"].to_numpy(), y_row)
        acc_pp,  mapped_pp  = hungarian_acc(df["Label"].to_numpy(), y_pp)
        plot_two(T, A, df["Label"], mapped_raw, "GT", f"Pred-raw ({model_tag}) acc={acc_raw:.3f}", Path(OUT_DIR, "real", f"{base}_gt_vs_pred_raw.png"))
        plot_two(T, A, df["Label"], mapped_pp,  "GT", f"Pred-pp  ({model_tag}) acc={acc_pp:.3f}",  Path(OUT_DIR, "real", f"{base}_gt_vs_pred_pp.png"))
    else:
        plot_two(T, A, y_row, y_pp, "Pred-raw", "Pred-postproc", Path(OUT_DIR, "real", f"{base}_pred_vs_post.png"))

def main():
    warnings.filterwarnings("ignore")
    ensure_dir(OUT_DIR)
    ensure_dir(Path(OUT_DIR, "real"))
    log("=== Inference (ensemble + prob-aware postproc) ===")
    log(f"[models] {MODEL_DIR}")
    log(f"[out   ] {OUT_DIR}")
    for p in TEST_FILES:
        if not Path(p).exists():
            log(f"[warn] not found: {p}"); continue
        process_one(p)
    log("=== Done ===")

if __name__ == "__main__":
    main()
