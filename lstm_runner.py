# -*- coding: utf-8 -*-
"""
Auto-train & infer on ALL CSVs in a folder to reduce generalization drop.
- Scans: generated_mode_*.csv + generated_link_*.csv in ROOT_DIR
- Feature engineering: t_norm, sin/cos(az), robust velocity/acceleration
- Models: BiLSTM + Transformer (sequence-to-label)
- Training: class-weighted CE, label-smoothing, augmentation, early-stopping,
            ReduceLROnPlateau, grad clipping
- Inference: best or ensemble; post-processing (uniqueness + short-run smoothing)
- Outputs: per-file CSVs + plots + summary JSON

Author: based on your lstm_runner.py, generalized and automated.
"""

import os, math, glob, json, warnings
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

# ---------------------- CONFIG (FIXED PATHS) ----------------------
ROOT_DIR = r"D:\work_station\radar_co\radar\final\sadegh_beauty-codex-improve-data-generation-script-accuracy\generated_data"
OUT_DIR  = r"D:\work_station\radar_co\radar\final\sadegh_beauty-codex-improve-data-generation-script-accuracy\generated_datalstm out"

PATTERNS = ["generated_mode_*.csv", "generated_link_*.csv"]  # auto-scan patterns
# protocol for validation selection:
VAL_MODE = "random"     # "random" | (می‌توانید "lofo_eval" بگذارید تا بعد از آموزش روی تک‌تک فایل‌ها ارزیابی شود)
USE_ENSEMBLE = True     # True: average softmax(LSTM, Transformer) at inference
SAVE_MODELS = True

# ---------------------- HYPERPARAMS ----------------------
SEQ_LEN = 64
BATCH   = 256
EPOCHS  = 20
LR      = 1e-3
VAL_SPLIT = 0.15
RANDOM_SEED = 42
LABEL_SMOOTH = 0.05

SMOOTH_MIN_RUN = 6   # حذف سوییچ‌های کوتاه
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------- UTILS ----------------------
def log(m): print(m, flush=True)
def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def ang_diff(a, b): return (a - b + 180.0) % 360.0 - 180.0

def robust_velocity(time: np.ndarray, az: np.ndarray) -> np.ndarray:
    n=len(az); v=np.zeros(n, dtype=float)
    for i in range(n):
        if i==0:
            dt = time[i+1]-time[i] if i+1<n else 1.0
            if dt==0: dt=1e-6
            v[i] = ang_diff(az[i+1],az[i])/dt if i+1<n else 0.0
        elif i==n-1:
            dt = time[i]-time[i-1]; 
            if dt==0: dt=1e-6
            v[i] = ang_diff(az[i],az[i-1])/dt
        else:
            dt = time[i+1]-time[i-1]
            if dt==0: dt=1e-6
            v[i] = ang_diff(az[i+1],az[i-1])/dt
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
    plt.tight_layout()
    fig.savefig(save_path, dpi=160); plt.close(fig)
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

def enforce_uniqueness_per_time(T: np.ndarray, A: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Resolve collisions per timestamp with local linear prediction."""
    out = labels.copy()
    from sklearn.linear_model import LinearRegression
    models={}
    for L in np.unique(labels):
        idx=np.where(labels==L)[0]
        if len(idx)<3:
            models[L]=(0.0, float(np.median(A[idx]) if len(idx)>0 else 0.0))
        else:
            lr=LinearRegression().fit(T[idx].reshape(-1,1), A[idx])
            models[L]=(float(lr.coef_[0]), float(lr.intercept_))
    for tt in np.unique(T):
        inds=np.where(T==tt)[0]
        if len(inds)<=1: continue
        groups={}
        for ii in inds:
            L=labels[ii]; groups.setdefault(L, []).append(ii)
        for L, idxs in groups.items():
            if len(idxs)==1: continue
            beta,b=models[L]
            jbest=min(idxs, key=lambda k: abs(ang_diff(A[k], beta*tt+b)))
            for k in idxs:
                if k==jbest: continue
                bestL=None; bestCost=float("inf")
                for L2,(bb,bb0) in models.items():
                    if L2==L: continue
                    cost=abs(ang_diff(A[k], bb*tt+bb0))
                    if cost<bestCost: bestCost, bestL=cost, L2
                if bestL is not None: out[k]=bestL
    return out

# ---------------------- DATA & FEATURES ----------------------
def list_csvs(root: str, patterns: List[str]) -> List[str]:
    files=[]
    for pat in patterns:
        files += glob.glob(os.path.join(root, pat))
    files = sorted(set(files))
    return files

def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).sort_values("Time").reset_index(drop=True)
    assert "Time" in df.columns and "Azimuth" in df.columns, f"{path} must have Time & Azimuth."
    return df

def engineer_features(df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
    T = df["Time"].to_numpy().astype(float)
    A = df["Azimuth"].to_numpy().astype(float)
    y = df["Label"].to_numpy().astype(int) if "Label" in df.columns else None

    t_norm = (T - T.min())/(T.max()-T.min() if T.max()>T.min() else 1.0)
    az = np.deg2rad(A)
    s = np.sin(az); c = np.cos(az)
    v = robust_velocity(T, A)
    a = robust_acceleration(v)
    vs = (v - v.mean())/(v.std() if v.std()!=0 else 1.0)
    as_ = (a - a.mean())/(a.std() if a.std()!=0 else 1.0)
    X = np.vstack([t_norm, s, c, vs, as_]).T.astype(np.float32)
    return X, y, T, A

def build_sequences(X: np.ndarray, y: Optional[np.ndarray], seq_len=SEQ_LEN):
    N, F = X.shape
    if N < seq_len:
        return np.empty((0, seq_len, F), np.float32), (np.empty((0,), np.int64) if y is not None else None), np.empty((0,), np.int64)
    M = N - seq_len + 1
    X_seq = np.zeros((M, seq_len, F), np.float32)
    y_seq = None if y is None else np.zeros((M,), np.int64)
    for i in range(M):
        X_seq[i] = X[i:i+seq_len]
        if y is not None:
            y_seq[i] = y[i+seq_len-1]
    idx_map = np.arange(seq_len-1, N)
    return X_seq, y_seq, idx_map

def remap_labels_global(train_y_list: List[np.ndarray], file_ids: List[int]) -> Tuple[List[np.ndarray], Dict[Tuple[int,int],int], int]:
    remap={}; next_id=0; out=[]
    for y, fid in zip(train_y_list, file_ids):
        if y is None: out.append(None); continue
        yy=np.zeros_like(y)
        for i,lab in enumerate(y):
            key=(fid, int(lab))
            if key not in remap:
                remap[key]=next_id; next_id+=1
            yy[i]=remap[key]
        out.append(yy)
    return out, remap, next_id

# ---------------------- MODELS ----------------------
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
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(pos*div); pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:,:x.size(1),:]

class TransEncClassifier(nn.Module):
    def __init__(self, input_dim=5, d_model=64, nhead=4, num_layers=2, dim_ff=128, num_classes=2, dropout=0.2):
        super().__init__()
        self.inp = nn.Linear(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pe  = PositionalEncoding(d_model)
        self.fc  = nn.Linear(d_model, num_classes)
    def forward(self, x):
        z = self.inp(x); z = self.pe(z); y = self.enc(z)
        h = y[:,-1,:]
        return self.fc(h)

# ---------------------- Datasets (with augmentation) ----------------------
# ---------------------- Datasets (with augmentation) ----------------------
class SeqDataset(Dataset):
    def __init__(self, X, y, aug=False):
        # X: (N, L, F)  y: (N,)
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
        self.aug = aug

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        # x: (L, F)
        x = self.X[i].copy()
        if self.aug:
            L, F = x.shape
            # ویژگی‌ها: [t_norm, sin(az), cos(az), v, a]
            # 1) بازسازی az از sin/cos و اعمال jitter کوچک
            az = np.arctan2(x[:, 1], x[:, 2])                # (L,)
            az += np.random.normal(0.0, 0.02, size=L)        # ~1.1 deg
            x[:, 1] = np.sin(az)
            x[:, 2] = np.cos(az)
            # 2) jitter روی t_norm, v, a
            x[:, 0] = np.clip(x[:, 0] + np.random.normal(0, 0.01, size=L), 0.0, 1.0)
            if F > 3:
                x[:, 3] += np.random.normal(0, 0.02, size=L)  # v
            if F > 4:
                x[:, 4] += np.random.normal(0, 0.02, size=L)  # a
        return torch.from_numpy(x), torch.tensor(self.y[i], dtype=torch.long)


# ---------------------- TRAIN / EVAL ----------------------
def split_random(X, y, ratio=VAL_SPLIT, seed=RANDOM_SEED):
    n=len(X); idx=np.arange(n); rs=np.random.RandomState(seed); rs.shuffle(idx)
    v=int(n*ratio); val_idx=idx[:v]; tr_idx=idx[v:]
    return tr_idx, val_idx

def class_weights_from(y, C):
    cnt = np.bincount(y, minlength=C).astype(np.float32)
    w = (cnt.mean() / (cnt + 1e-6))
    return torch.tensor(w, dtype=torch.float32, device=DEVICE)

def train_one(model, X_train, y_train, X_val, y_val, epochs=EPOCHS, lr=LR, class_weights=None):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2, verbose=True)
    if class_weights is not None:
        ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTH)
    else:
        ce = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

    best_acc = -1.0; best_state=None; patience = 5; no_improve=0
    train_loader = DataLoader(SeqDataset(X_train, y_train, aug=True), batch_size=BATCH, shuffle=True, drop_last=False)
    val_loader   = DataLoader(SeqDataset(X_val,   y_val,   aug=False), batch_size=BATCH, shuffle=False)

    def eval_acc():
        model.eval(); correct=tot=0
        with torch.no_grad():
            for xb,yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb)
                p = out.argmax(1)
                correct += (p==yb).sum().item(); tot += yb.size(0)
        return correct/max(1,tot)

    for ep in range(epochs):
        model.train(); loss_sum=0.0; steps=0
        for xb,yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            loss = ce(out, yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += loss.item(); steps += 1
        val_acc = eval_acc()
        sched.step(1.0 - val_acc)  # higher acc => lower 'loss' to scheduler
        log(f"   [ep {ep+1:02d}/{epochs}] loss={loss_sum/max(1,steps):.4f}  val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc; no_improve=0
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience: 
                log("   [early-stop] no improvement"); break

    if best_state is not None: model.load_state_dict(best_state)
    return model, best_acc

@torch.no_grad()
def infer_model(model, X_seq, num_classes):
    model=model.to(DEVICE); model.eval()
    logits_all = np.zeros((len(X_seq), num_classes), np.float32)
    loader = DataLoader(torch.tensor(X_seq), batch_size=BATCH, shuffle=False)
    off=0
    for xb in loader:
        xb = xb.to(DEVICE)
        out = model(xb).softmax(1).cpu().numpy()
        logits_all[off:off+len(out)] = out; off += len(out)
    y_pred = np.argmax(logits_all, axis=1)
    return y_pred, logits_all

# ---------------------- MAIN ----------------------
def main():
    warnings.filterwarnings("ignore")
    ensure_dir(OUT_DIR)
    np.random.seed(RANDOM_SEED); torch.manual_seed(RANDOM_SEED)

    log("=== Auto LSTM/Transformer (train on all labeled, infer on all) ===")
    files = list_csvs(ROOT_DIR, PATTERNS)
    if not files: 
        log(f"[error] no CSVs found in {ROOT_DIR} with {PATTERNS}"); return

    # Load all files + features
    file_data=[]
    for fid, path in enumerate(files):
        df = load_df(path)
        X, y, T, A = engineer_features(df)
        Xs, ys, idx_map = build_sequences(X, y, SEQ_LEN)
        file_data.append({
            "fid": fid, "path": path, "df": df,
            "X": X, "y": y, "T": T, "A": A,
            "X_seq": Xs, "y_seq": ys, "idx_map": idx_map
        })
        log(f"[load] {path} rows={len(df):5d} seqs={len(Xs):5d} has_GT={y is not None}")

    # Build training set from ALL labeled files
    labeled = [fd for fd in file_data if fd["y_seq"] is not None and len(fd["y_seq"])>0]
    if not labeled:
        log("[error] no labeled files to train on."); return

    # Global remap over LABELED sequences
    y_list = [fd["y_seq"] for fd in labeled]
    fids   = [fd["fid"]   for fd in labeled]
    remapped, remap, C = remap_labels_global(y_list, fids)
    for fd, yy in zip(labeled, remapped): fd["y_seq_global"]=yy
    X_all = np.concatenate([fd["X_seq"] for fd in labeled], axis=0)
    y_all = np.concatenate([fd["y_seq_global"] for fd in labeled], axis=0)
    log(f"[trainset] total seqs={len(X_all)}  classes(C)={C}")

    # Train/val split
    tr_idx, val_idx = split_random(X_all, y_all, ratio=VAL_SPLIT)
    X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
    X_va, y_va = X_all[val_idx], y_all[val_idx]
    class_w = class_weights_from(y_tr, C)

    # Train LSTM
    log("\n[train] BiLSTM …")
    lstm = BiLSTMClassifier(input_dim=X_all.shape[2], hidden=128, layers=2, num_classes=C, dropout=0.2)
    lstm, acc_lstm = train_one(lstm, X_tr, y_tr, X_va, y_va, epochs=EPOCHS, lr=LR, class_weights=class_w)
    log(f"[val] BiLSTM best_acc={acc_lstm:.4f}")

    # Train Transformer
    log("\n[train] Transformer …")
    trans = TransEncClassifier(input_dim=X_all.shape[2], d_model=64, nhead=4, num_layers=2, dim_ff=128, num_classes=C, dropout=0.2)
    trans, acc_trans = train_one(trans, X_tr, y_tr, X_va, y_va, epochs=EPOCHS, lr=LR, class_weights=class_w)
    log(f"[val] Transformer best_acc={acc_trans:.4f}")

    best_name = "lstm" if acc_lstm >= acc_trans else "trans"
    log(f"[select] best={best_name}")
    if SAVE_MODELS:
        torch.save(lstm.state_dict(), Path(OUT_DIR,"model_lstm.pt"))
        torch.save(trans.state_dict(), Path(OUT_DIR,"model_trans.pt"))
        log(f"[save] {Path(OUT_DIR,'model_lstm.pt')}")
        log(f"[save] {Path(OUT_DIR,'model_trans.pt')}")

    # --------- Inference on ALL files ----------
    summary=[]
    for fd in file_data:
        path=fd["path"]; df=fd["df"]; T=fd["T"]; A=fd["A"]
        Xs=fd["X_seq"]; idx_map=fd["idx_map"]
        if len(Xs)==0:
            log(f"[skip] too short for seq_len={SEQ_LEN}: {path}")
            continue

        # raw predictions
        if USE_ENSEMBLE:
            yp1, logits1 = infer_model(lstm, Xs, C)
            yp2, logits2 = infer_model(trans, Xs, C)
            logits = (logits1 + logits2)/2.0
            yp = np.argmax(logits, axis=1)
            model_used = "ensemble"
        else:
            if best_name=="lstm":
                yp, logits = infer_model(lstm, Xs, C); model_used="lstm"
            else:
                yp, logits = infer_model(trans, Xs, C); model_used="trans"

        # expand to per-row
        y_row = np.zeros(len(df), dtype=int)
        y_row[:SEQ_LEN-1] = yp[0]
        y_row[SEQ_LEN-1:] = yp

        # post-process
        y_pp = enforce_uniqueness_per_time(T, A, y_row)
        y_pp = smooth_short_runs(y_pp, min_len=SMOOTH_MIN_RUN)

        # save CSV + plots
        base = Path(path).stem
        out_csv = Path(OUT_DIR, f"{base}_{model_used}_pred.csv")
        dfo = df.copy(); dfo["PredRaw"]=y_row; dfo["PredPost"]=y_pp
        dfo.to_csv(out_csv, index=False); log(f"[save] {out_csv}")

        if "Label" in df.columns:
            acc_raw, mapped_raw = hungarian_acc(df["Label"].to_numpy(), y_row)
            acc_pp,  mapped_pp  = hungarian_acc(df["Label"].to_numpy(), y_pp)
            plot_two(T, A, df["Label"], mapped_raw, "GT", f"Pred-raw ({model_used}) acc={acc_raw:.3f}", Path(OUT_DIR, f"{base}_gt_vs_pred_raw.png"))
            plot_two(T, A, df["Label"], mapped_pp,  "GT", f"Pred-pp  ({model_used}) acc={acc_pp:.3f}",  Path(OUT_DIR, f"{base}_gt_vs_pred_pp.png"))
            summary.append({"file": path, "model": model_used, "acc_raw": float(acc_raw), "acc_post": float(acc_pp)})
        else:
            plot_two(T, A, y_row, y_pp, f"Pred-raw ({model_used})", "Pred-postproc", Path(OUT_DIR, f"{base}_pred_vs_post.png"))
            summary.append({"file": path, "model": model_used, "acc_raw": None, "acc_post": None})

    Path(OUT_DIR, "summary_auto.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"[save] {Path(OUT_DIR,'summary_auto.json')}")
    log("=== Done ===")

if __name__ == "__main__":
    main()
