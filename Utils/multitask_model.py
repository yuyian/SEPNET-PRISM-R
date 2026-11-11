import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import optuna


def _gpu_peak_memory_mb(device):
    """Peak GPU memory allocated in MB (CUDA only)."""
    if device.type != "cuda":
        return None
    try:
        return torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    except Exception:
        return None


def get_compute_cost(device):
    """
    Return a dict with current computational cost metrics (for reporting after a run).
    Keys: train_time_sec, eval_val_time_sec, eval_test_time_sec, gpu_peak_mb.
    """
    return {
        "train_time_sec": None,
        "eval_val_time_sec": None,
        "eval_test_time_sec": None,
        "gpu_peak_mb": _gpu_peak_memory_mb(device) if isinstance(device, torch.device) else None,
    }


class timer_context:
    """Wall-clock timer (seconds) for a code block."""

    def __init__(self):
        self.elapsed_sec = None
    def __enter__(self):
        self._t0 = time.perf_counter()
        return self
    def __exit__(self, *args):
        self.elapsed_sec = time.perf_counter() - self._t0
        return False


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.
    Args:
        alpha (float or list/tuple): Class weighting factor(s). 
        gamma (float): Focusing parameter.
        reduction (str): 'none', 'mean', or 'sum'.
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)

        if isinstance(self.alpha, (list, tuple)):
            alpha_tensor = torch.tensor(self.alpha, device=inputs.device, dtype=inputs.dtype)
            alpha_factor = alpha_tensor[targets.long()]
        else:
            alpha_factor = self.alpha

        focal_loss = alpha_factor * (1 - pt) ** self.gamma * bce

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


        
def evaluate(
    yreg_pred,
    ycls_pred,
    ycls_prob,
    yreg_true,
    ycls_true,
    reg_var_names=None,
    *,
    verbose: bool = True,
):
    """
    reg_var_names: optional list of names for each regression target (e.g. ['future_flare_num', 'future_CME_num']).
                   Used for per-variable metric labels.

    For **classification-only** baselines, pass ``yreg_pred`` as all-NaN (same shape as ``yreg_true``) so joint
    MSE/R² are omitted. Do **not** pass ``yreg_true`` as ``yreg_pred`` (that falsely yields MSE=0, R²=1).
    """
    yreg_true = np.asarray(yreg_true, dtype=np.float64)
    yreg_pred = np.asarray(yreg_pred, dtype=np.float64)
    ycls_prob = np.asarray(ycls_prob, dtype=np.float64).ravel()
    ycls_prob = np.nan_to_num(ycls_prob, nan=0.5, posinf=1.0, neginf=0.0)

    # Ensure 2D arrays (n_samples, n_targets)
    if yreg_true.ndim == 1:
        yreg_true = yreg_true.reshape(-1, 1)
    if yreg_pred.ndim == 1:
        yreg_pred = yreg_pred.reshape(-1, 1)

    n_targets = yreg_true.shape[1]
    if reg_var_names is None:
        reg_var_names = [f"var_{j}" for j in range(n_targets)]

    # Skip regression metrics: no targets, or explicit NaN preds (cls-only), or accidental y_true passed as y_pred
    skip_reg = (
        n_targets == 0
        or yreg_true.size == 0
        or yreg_pred.shape != yreg_true.shape
        or np.all(np.isnan(yreg_pred))
        or np.array_equal(yreg_pred, yreg_true)
    )

    rmse_per_var = []
    r2_per_var = []
    mae_per_var = []
    if skip_reg:
        mse = float("nan")
        r2 = float("nan")
    else:
        yreg_pred = np.nan_to_num(yreg_pred, nan=0.0, posinf=0.0, neginf=0.0)
        yreg_true = np.nan_to_num(yreg_true, nan=0.0, posinf=0.0, neginf=0.0)
        yreg_true = np.clip(yreg_true, -700, 700)
        yreg_pred = np.clip(yreg_pred, -700, 700)
        for j in range(n_targets):
            y_true_j = yreg_true[:, j]
            y_pred_j = yreg_pred[:, j]
            rmse_j = np.sqrt(mean_squared_error(y_true_j, y_pred_j))
            r2_j = r2_score(y_true_j, y_pred_j)
            mae_j = mean_absolute_error(y_true_j, y_pred_j)
            rmse_per_var.append(rmse_j)
            r2_per_var.append(r2_j)
            mae_per_var.append(mae_j)
        mse = mean_squared_error(yreg_true, yreg_pred)
        r2 = r2_score(yreg_true, yreg_pred)

    has_regression = (not skip_reg) and n_targets > 0 and len(rmse_per_var) > 0
    ycls_true_1d = np.asarray(ycls_true, dtype=np.float64).ravel()
    ycls_pred_1d = np.asarray(ycls_pred).ravel()
    accuracy = accuracy_score(ycls_true_1d, ycls_pred_1d)
    if len(np.unique(ycls_true_1d)) < 2:
        rocauc = float("nan")
    else:
        rocauc = roc_auc_score(ycls_true_1d, ycls_prob)

    # Force a stable 2x2 confusion matrix even if a batch/pool has one class.
    cm = confusion_matrix(ycls_true_1d, ycls_pred_1d, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel()
    precision = TP / (TP + FP) if (TP + FP) else 0
    fpr = FP / (FP + TN) if (FP + TN) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    pod = recall
    far = FP / (TP + FP) if (TP + FP) else 0
    tss = recall - fpr
    denom = ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN))
    hss = (2 * (TP * TN - FP * FN)) / denom if denom else 0.0

    if verbose:
        print(cm)
        if has_regression:
            print(f"MSE={mse:.4f}, R²={r2:.4f}, Acc={accuracy:.4f}, AUC={rocauc:.4f}, TSS={tss:.4f}, HSS={hss:.4f}")
        else:
            print(f"MSE=N/A, R²=N/A (cls-only / no regression preds), Acc={accuracy:.4f}, AUC={rocauc:.4f}, TSS={tss:.4f}, HSS={hss:.4f}")
        if has_regression:
            for j in range(n_targets):
                name = reg_var_names[j] if j < len(reg_var_names) else f"var_{j}"
                print(f"  [{name}] RMSE={rmse_per_var[j]:.4f}, R²={r2_per_var[j]:.4f}, MAE={mae_per_var[j]:.4f}")

    return dict(mse=mse, r2=r2, accuracy=accuracy, rocauc=rocauc,
                precision=precision, fpr=fpr, f1=f1, recall=recall,
                pod=pod, far=far, tss=tss, hss=hss,
                rmse_per_var=rmse_per_var, r2_per_var=r2_per_var, mae_per_var=mae_per_var,
                reg_var_names=reg_var_names,
                yreg_pred=yreg_pred, ycls_pred=ycls_pred, ycls_prob=ycls_prob,
                yreg_true=yreg_true, ycls_true=ycls_true)


def _mtm_time_nearest_lookup(time_series, history_ix_set):
    """Sort (time_ns, index) for rows in history with valid times."""
    import pandas as pd

    hist = [i for i in history_ix_set if i in time_series.index]
    if not hist:
        return np.array([], dtype=np.int64), np.array([], dtype=object)
    ts = pd.to_datetime(time_series.reindex(hist), utc=True, errors="coerce")
    ok = ts.notna()
    hist = [h for h, o in zip(hist, ok) if o]
    if not hist:
        return np.array([], dtype=np.int64), np.array([], dtype=object)
    tvals = np.asarray(ts.loc[hist].values, dtype="datetime64[ns]").astype(np.int64)
    order = np.argsort(tvals)
    return tvals[order], np.asarray(hist, dtype=object)[order]


def _mtm_nearest_ix(times_ns, ix_sorted, t_req_ns, tol_ns, *, t_target_ns=None):
    """
    Return index label whose time is nearest to t_req within tol_ns.
    If t_target_ns is set, only rows strictly before the target time are eligible.
    """
    if times_ns.size == 0:
        return None
    pos = int(np.searchsorted(times_ns, t_req_ns))
    best_ix = None
    best_d = None
    for c in (pos - 1, pos):
        if 0 <= c < times_ns.size:
            tn = int(times_ns[c])
            if t_target_ns is not None and tn >= int(t_target_ns):
                continue
            d = abs(tn - int(t_req_ns))
            if best_d is None or d < best_d:
                best_d = d
                best_ix = ix_sorted[c]
    if best_d is None or best_d > tol_ns:
        return None
    return best_ix


def build_mtm_sequences(
    order,
    X,
    y_reg,
    y_cls_op,
    target_ix,
    history_ix_set,
    seq_len,
    return_target_ix=False,
    stride=1,
    *,
    time_series=None,
    lag_hours=None,
    lag_tolerance_hours=None,
    pad_missing="nan",
):
    """
    Stack prior windows along time for LSTM/sequence MTM.

    **Index mode (default)** — time_series / lag_hours not set:
    consecutive rows in ``order`` (optionally spaced by ``stride``).

    **Time mode** — pass time_series (aligned to X.index, e.g. window_begin) and lag_hours:
    calendar lags e.g. seq_len=3, lag_hours=12 → [t−24h, t−12h, t_now]; missing lags → pad.
    Use this when rows are irregular after filtering (gaps in time).
    """
    import pandas as pd

    if pad_missing not in ("nan", "zero"):
        raise ValueError("pad_missing must be 'nan' or 'zero'")

    use_time = time_series is not None and lag_hours is not None
    if (time_series is None) ^ (lag_hours is None):
        raise ValueError("Provide both time_series and lag_hours for time mode, or neither for index mode.")

    if use_time:
        lag_td = pd.Timedelta(hours=float(lag_hours))
        tol_h = float(lag_tolerance_hours) if lag_tolerance_hours is not None else float(lag_hours) / 2.0
        tol_ns = int(pd.Timedelta(hours=tol_h).value)
        ts_all = pd.to_datetime(time_series.reindex(X.index), utc=True, errors="coerce")
        if ts_all.isna().any():
            raise ValueError("build_mtm_sequences (time mode): time_series has NaT for some X.index rows.")
        X_vals = X.to_numpy(dtype=np.float64)
        ix_pos = {ix: i for i, ix in enumerate(X.index)}
        n_feat = X_vals.shape[1]
        pad_row = np.full(n_feat, np.nan if pad_missing == "nan" else 0.0, dtype=np.float64)
        times_ns_hist, ix_hist_sorted = _mtm_time_nearest_lookup(ts_all, set(history_ix_set))

        yreg_m = y_reg.reindex(X.index).to_numpy(dtype=np.float64)
        ycls_m = y_cls_op.reindex(X.index).to_numpy(dtype=np.float64).ravel()

        X_list, yr_list, yc_list, kept_ix = [], [], [], []
        for ix in target_ix:
            if ix not in ix_pos or ix not in history_ix_set:
                continue
            t_target = ts_all.loc[ix]
            if pd.isna(t_target):
                continue
            t_target_ns = int(pd.Timestamp(t_target).value)
            p = ix_pos[ix]
            row_vecs = []
            for j in range(seq_len - 1):
                t_req = t_target - (seq_len - 1 - j) * lag_td
                t_req_ns = int(pd.Timestamp(t_req).value)
                lag_ix = _mtm_nearest_ix(
                    times_ns_hist, ix_hist_sorted, t_req_ns, tol_ns, t_target_ns=t_target_ns
                )
                if lag_ix is None or lag_ix not in ix_pos or lag_ix == ix:
                    row_vecs.append(pad_row.copy())
                else:
                    row_vecs.append(X_vals[ix_pos[lag_ix]])
            row_vecs.append(X_vals[p])
            X_list.append(np.stack(row_vecs, axis=0))
            yr_list.append(yreg_m[p])
            yc_list.append(ycls_m[p])
            kept_ix.append(ix)
        
        if not X_list:
            raise ValueError(
                "build_mtm_sequences (time mode): no valid samples (seq_len=%s, lag_hours=%s). "
                "Try smaller seq_len, larger lag_tolerance_hours, or relax history_ix_set."
                % (seq_len, lag_hours)
            )
        X_seq = np.stack(X_list, axis=0).astype(np.float32)
        y_reg_out = np.stack(yr_list, axis=0).astype(np.float32)
        y_cls_out = np.array(yc_list, dtype=np.float32).reshape(-1, 1)
        if return_target_ix:
            return X_seq, y_reg_out, y_cls_out, kept_ix
        return X_seq, y_reg_out, y_cls_out

    stride = int(stride)
    if stride < 1:
        raise ValueError("stride must be >= 1")

    order = pd.Index(order)
    pos_of = {ix: i for i, ix in enumerate(order)}
    Xo = X.reindex(order)
    if Xo.isna().any().any():
        raise ValueError("build_mtm_sequences: X.reindex(order) has NaNs — order must match X.index.")
    yreg_o = y_reg.reindex(order).to_numpy(dtype=np.float64)
    ycls_o = y_cls_op.reindex(order).to_numpy(dtype=np.float64).ravel()
    X_arr = Xo.to_numpy(dtype=np.float64)

    need_pos = (seq_len - 1) * stride

    X_list, yr_list, yc_list, kept_ix = [], [], [], []
    for ix in target_ix:
        if ix not in pos_of:
            continue
        p = pos_of[ix]
        if p < need_pos:
            continue
        labels = [order[p - need_pos + j * stride] for j in range(seq_len)]
        if not all(lbl in history_ix_set for lbl in labels):
            continue
        idx_rows = [p - need_pos + j * stride for j in range(seq_len)]
        X_list.append(X_arr[idx_rows])
        yr_list.append(yreg_o[p])
        yc_list.append(ycls_o[p])
        kept_ix.append(ix)

    if not X_list:
        raise ValueError(
            "build_mtm_sequences: no valid samples (seq_len=%s, stride=%s). "
            "Try smaller seq_len, stride=1, or relax history_ix_set." % (seq_len, stride)
        )
    X_seq = np.stack(X_list, axis=0).astype(np.float32)
    y_reg_out = np.stack(yr_list, axis=0).astype(np.float32)
    y_cls_out = np.array(yc_list, dtype=np.float32).reshape(-1, 1)
    if return_target_ix:
        return X_seq, y_reg_out, y_cls_out, kept_ix
    return X_seq, y_reg_out, y_cls_out


def _sanitize_feature_array(X_np: np.ndarray) -> np.ndarray:
    """Replace non-finite values and clip so f_classif SSE steps and torch do not overflow."""
    X = np.asarray(X_np, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=1e4, neginf=-1e4)
    return np.clip(X, -1e4, 1e4)


def _select_transform_features(X_np: np.ndarray, selector: SelectKBest) -> np.ndarray:
    """Apply sklearn selector to 2D (n, f) or 3D (n, t, f) features."""
    X_np = _sanitize_feature_array(X_np)
    if X_np.ndim == 2:
        return selector.transform(X_np)
    if X_np.ndim == 3:
        n, t, f = X_np.shape
        flat = X_np.reshape(-1, f)
        out = selector.transform(flat)
        return out.reshape(n, t, out.shape[1])
    raise ValueError("Expected X_np 2D or 3D, got shape %s" % (X_np.shape,))


def _fit_select_features(X_np: np.ndarray, ycls_np: np.ndarray, selector: SelectKBest) -> np.ndarray:
    """Fit selector; if 3D, fit on last timestep only (labels match target row)."""
    X_np = _sanitize_feature_array(X_np)
    if X_np.ndim == 2:
        return selector.fit_transform(X_np, ycls_np)
    if X_np.ndim == 3:
        return selector.fit_transform(X_np[:, -1, :], ycls_np)
    raise ValueError("Expected X_np 2D or 3D, got shape %s" % (X_np.shape,))


class _IdentitySelector:
    """Sklearn-like selector that does nothing (for disabling SelectKBest)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        return X


    
def prepare_data_loaders(
    X_tensor,
    yreg_tensor,
    ycls_tensor,
    feature_K,
    batch_size,
    device,
    sampler_mode="shuffle",
    ycls_teacher_prob_tensor=None,
    ycls_sample_weight_tensor=None,
):
    """
    sampler_mode:
      'shuffle' — standard random shuffle (default).
      'balanced' — WeightedRandomSampler so each minibatch is ~balanced for rare positives
                   (helps gradient signal for the minority SEP class).

    X_tensor may be (n, f) or (n, seq_len, f). If 3D, SelectKBest is fit on the last
    timestep (prediction row); the same transform is applied to every timestep.
    Returns (loader, selector, input_dim_per_step, seq_len).
    """
    X_np = X_tensor.cpu().numpy()
    ycls_np = ycls_tensor.cpu().numpy().ravel()
    seq_len = 1 if X_np.ndim == 2 else int(X_np.shape[1])
    n_feat = int(X_np.shape[-1])
    # feature_K can be None / 0 / >=n_feat → disable feature selection (use all features)
    disable_fs = feature_K is None
    if not disable_fs:
        try:
            k = int(feature_K)
        except Exception:
            disable_fs = True
            k = n_feat
    if disable_fs or int(feature_K) <= 0 or int(feature_K) >= n_feat:
        selector = _IdentitySelector()
        X_sel = _sanitize_feature_array(X_np)
    else:
        k = int(feature_K)
        ybin = (ycls_np >= 0.5).astype(np.int64)
        if np.unique(ybin).size < 2:
            raise ValueError(
                "SelectKBest (f_classif) needs both classes in the training fold; got only one class in y_cls. "
                "Widen the fit window or relax the validation split."
            )
        selector = SelectKBest(score_func=f_classif, k=k)
        _fit_select_features(X_np, ycls_np, selector)
        X_sel = _select_transform_features(X_np, selector)
    X_sel_tensor = torch.tensor(X_sel, dtype=torch.float32).to(device)
    yreg_tensor = yreg_tensor.to(device)
    ycls_tensor = ycls_tensor.to(device)
    input_dim_per_step = int(X_sel.shape[-1])

    ysw = None
    if ycls_sample_weight_tensor is not None:
        ysw = ycls_sample_weight_tensor.to(device).view(-1).float()
        if ysw.numel() != ycls_tensor.shape[0]:
            raise ValueError(
                "ycls_sample_weight_tensor length must match training sequences."
            )
    if ycls_teacher_prob_tensor is not None:
        yteach = ycls_teacher_prob_tensor.to(device)
        if ysw is not None:
            dataset = TensorDataset(X_sel_tensor, yreg_tensor, ycls_tensor, yteach, ysw)
        else:
            dataset = TensorDataset(X_sel_tensor, yreg_tensor, ycls_tensor, yteach)
    elif ysw is not None:
        dataset = TensorDataset(X_sel_tensor, yreg_tensor, ycls_tensor, ysw)
    else:
        dataset = TensorDataset(X_sel_tensor, yreg_tensor, ycls_tensor)
    if sampler_mode == "balanced":
        y_int = (ycls_np >= 0.5).astype(np.int64)
        n0 = int(np.sum(y_int == 0))
        n1 = int(np.sum(y_int == 1))
        w0 = 1.0 / max(n0, 1)
        w1 = 1.0 / max(n1, 1)
        sw = np.where(y_int == 0, w0, w1).astype(np.float64)
        sampler = WeightedRandomSampler(
            torch.from_numpy(sw), num_samples=len(sw), replacement=True
        )
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader, selector, input_dim_per_step, seq_len


def _unpack_mtm_train_batch(batch):
    """(X, yreg, ycls) | +yteach | +sample_w | +both (yteach, sample_w)."""
    n = len(batch)
    if n == 3:
        return batch[0], batch[1], batch[2], None, None
    if n == 4:
        xb, yreg, ycls, fourth = batch
        if fourth.ndim >= 2 and fourth.shape[-1] == 1:
            return xb, yreg, ycls, fourth, None
        return xb, yreg, ycls, None, fourth
    if n == 5:
        return batch[0], batch[1], batch[2], batch[3], batch[4]
    raise ValueError(f"Unexpected MTM train batch length {n}")


def _mtm_cls_loss(
    z,
    y,
    y_bce,
    crit_cls,
    crit_cls2_mean,
    crit_cls2_none,
    focal_scale,
    sample_w=None,
):
    """BCE (+ global pos_weight) + focal; optional per-sample weights (e.g. OSEP onset rows)."""
    if sample_w is None:
        return crit_cls(z, y_bce) + focal_scale * crit_cls2_mean(z, y)
    bce = F.binary_cross_entropy_with_logits(z, y_bce, reduction="none")
    pw = getattr(crit_cls, "pos_weight", None)
    if pw is not None:
        pwi = float(pw.view(-1)[0].item())
        bce = torch.where(y_bce > 0.5, bce * pwi, bce)
    focal = crit_cls2_none(z, y)
    w = sample_w.view(-1).to(z.device)
    w = w / w.sum().clamp(min=1e-6) * w.numel()
    return (bce + focal_scale * focal).mul(w).sum() / w.sum().clamp(min=1e-6)


def prepare_val_loader(
    X_tensor,
    yreg_tensor,
    ycls_tensor,
    selector,
    batch_size,
    device,
    ycls_teacher_prob_tensor=None,
):
    """Same feature selection as training; no shuffle."""
    X_np = X_tensor.cpu().numpy()
    if selector is None:
        selector = _IdentitySelector()
    X_sel = _select_transform_features(X_np, selector)
    X_sel_tensor = torch.tensor(X_sel, dtype=torch.float32).to(device)
    if ycls_teacher_prob_tensor is not None:
        ds = TensorDataset(
            X_sel_tensor,
            yreg_tensor.to(device),
            ycls_tensor.to(device),
            ycls_teacher_prob_tensor.to(device),
        )
    else:
        ds = TensorDataset(X_sel_tensor, yreg_tensor.to(device), ycls_tensor.to(device))
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def _tss_hss_from_counts(tn, fp, fn, tp):
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    tss = recall - fpr
    denom = ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    hss = (2.0 * (tp * tn - fp * fn)) / denom if denom else 0.0
    return tss, hss


def _f_beta(tp, fp, fn, beta=2.0):
    """F-beta; beta>1 emphasizes recall (minority detection)."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    b2 = beta * beta
    if precision + recall <= 0:
        return 0.0
    return (1.0 + b2) * precision * recall / (b2 * precision + recall + 1e-12)


def threshold_grid(n_uniform=51, n_high=120):
    """
    Finer steps in (0.85, 1.0] so argmax isn't stuck at 0.99 when probs are overconfident.
    """
    lo = np.linspace(0.0, 0.85, n_uniform, endpoint=True)
    hi = np.linspace(0.85, 1.0, n_high, endpoint=True)
    return np.unique(np.concatenate([lo, hi]))


def best_threshold_for_skill(ycls_true, ycls_prob, metric="min_tss_hss", n_grid=101, f_beta=2.0,
                           dense_high_end=True, t_min=0.0, t_max=1.0):
    """
    Pick probability threshold to maximize a skill score on labels.
    Default metric is 'min_tss_hss' (conservative balance of TSS and HSS).
    metric:
      'hss' | 'tss' | 'tss_plus_hss' | 'min_tss_hss'
      'hmean_tss_hss' — harmonic mean of TSS and HSS (penalizes one skill near zero vs min alone)
      'f2' — F-beta with beta=2 (recall-focused; good for rare SEP detection)
      'tss_plus_f2' — combine ordinal skill with recall emphasis
      'balanced_acc' — (TPR + TNR) / 2; can pick very low t on val — use t_min so test FP does not explode
    dense_high_end: extra grid points toward 1.0 (typical when sigmoid logits are extreme on train).
    t_min / t_max: search only thresholds in [t_min, t_max]. Use t_min≈0.65–0.75 if balanced_acc or
           low-t metrics pick t~0.5 on val but test scores shift (many FP at low t).
    Returns (best_threshold, best_score).
    """
    ycls_true = np.asarray(ycls_true, dtype=np.int64).ravel()
    ycls_prob = np.nan_to_num(
        np.asarray(ycls_prob, dtype=np.float64).ravel(),
        nan=0.5, posinf=1.0, neginf=0.0,
    )
    t_lo = float(t_min)
    t_hi = float(t_max)
    if t_lo > t_hi:
        t_lo, t_hi = t_hi, t_lo
    best_t = 0.5 * (t_lo + t_hi)
    best_score = -1e18
    grid = threshold_grid() if dense_high_end else np.linspace(0.0, 1.0, n_grid)
    grid = grid[(grid >= t_lo) & (grid <= t_hi)]
    if grid.size == 0:
        grid = np.array([best_t], dtype=np.float64)
    for t in grid:
        ypred = (ycls_prob > t).astype(np.int64)
        cm = confusion_matrix(ycls_true, ypred, labels=[0, 1])
        if cm.size != 4:
            continue
        tn, fp, fn, tp = cm.ravel()
        tss, hss = _tss_hss_from_counts(tn, fp, fn, tp)
        f2 = _f_beta(tp, fp, fn, beta=f_beta)
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        tnr = tn / (tn + fp) if (tn + fp) else 0.0
        balanced_acc = 0.5 * (tpr + tnr)
        if metric == "hss":
            sc = hss
        elif metric == "tss":
            sc = tss
        elif metric == "tss_plus_hss":
            sc = tss + hss
        elif metric == "min_tss_hss":
            sc = min(tss, hss)
        elif metric == "hmean_tss_hss":
            if tss > 0 and hss > 0:
                sc = 2.0 * tss * hss / (tss + hss)
            else:
                sc = min(tss, hss)
        elif metric == "f2":
            sc = f2
        elif metric == "tss_plus_f2":
            sc = tss + f2
        elif metric == "balanced_acc":
            sc = balanced_acc
        else:
            sc = tss + hss
        if sc > best_score:
            best_score = sc
            best_t = float(t)
    return best_t, best_score


def best_threshold_for_skill_under_max_fpr(
    ycls_true,
    ycls_prob,
    metric="min_tss_hss",
    max_fpr=0.05,
    f_beta=2.0,
    dense_high_end=True,
    t_min=0.0,
    t_max=1.0,
):
    """
    Maximize the same skill objective as ``best_threshold_for_skill``, but only among thresholds
    with **FPR <= max_fpr** on the given label set (e.g. validation).

    Helps when unconstrained ``min_tss_hss`` picks a low threshold that looks good on val but yields
    huge false-alarm rates under temporal / covariate shift on test.

    Returns
    -------
    best_t : float
    best_score : float
    cap_feasible : bool
        True if at least one grid point met the FPR cap; if False, ``best_t`` is the unconstrained
        skill optimum (same as ``best_threshold_for_skill``).
    """
    ycls_true = np.asarray(ycls_true, dtype=np.int64).ravel()
    ycls_prob = np.nan_to_num(
        np.asarray(ycls_prob, dtype=np.float64).ravel(),
        nan=0.5, posinf=1.0, neginf=0.0,
    )
    t_lo = float(t_min)
    t_hi = float(t_max)
    if t_lo > t_hi:
        t_lo, t_hi = t_hi, t_lo
    grid = threshold_grid() if dense_high_end else np.linspace(0.0, 1.0, 101)
    grid = grid[(grid >= t_lo) & (grid <= t_hi)]
    if grid.size == 0:
        bt, bs = best_threshold_for_skill(
            ycls_true, ycls_prob, metric=metric, t_min=t_min, t_max=t_max, dense_high_end=dense_high_end
        )
        return bt, bs, False

    mf = float(max_fpr)
    best_t = float(grid[0])
    best_score = -1e18
    n_ok = 0
    tie_eps = 1e-9
    for t in grid:
        ypred = (ycls_prob > t).astype(np.int64)
        cm = confusion_matrix(ycls_true, ypred, labels=[0, 1])
        if cm.size != 4:
            continue
        tn, fp, fn, tp = cm.ravel()
        denom = tn + fp
        fpr = fp / denom if denom > 0 else 0.0
        if fpr > mf:
            continue
        n_ok += 1
        tss, hss = _tss_hss_from_counts(tn, fp, fn, tp)
        f2 = _f_beta(tp, fp, fn, beta=f_beta)
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        tnr = tn / (tn + fp) if (tn + fp) else 0.0
        balanced_acc = 0.5 * (tpr + tnr)
        if metric == "hss":
            sc = hss
        elif metric == "tss":
            sc = tss
        elif metric == "tss_plus_hss":
            sc = tss + hss
        elif metric == "min_tss_hss":
            sc = min(tss, hss)
        elif metric == "hmean_tss_hss":
            if tss > 0 and hss > 0:
                sc = 2.0 * tss * hss / (tss + hss)
            else:
                sc = min(tss, hss)
        elif metric == "f2":
            sc = f2
        elif metric == "tss_plus_f2":
            sc = tss + f2
        elif metric == "balanced_acc":
            sc = balanced_acc
        else:
            sc = tss + hss
        # Prefer a stricter (higher) threshold when skill ties — fewer false alarms under covariate shift.
        if sc > best_score + tie_eps or (
            abs(sc - best_score) <= tie_eps and float(t) > best_t
        ):
            best_score = sc
            best_t = float(t)

    if n_ok == 0:
        bt, bs = best_threshold_for_skill(
            ycls_true, ycls_prob, metric=metric, t_min=t_min, t_max=t_max, dense_high_end=dense_high_end
        )
        return bt, bs, False
    return best_t, best_score, True


def threshold_for_max_fpr(ycls_true, ycls_prob, max_fpr=0.05, t_min=0.0, t_max=1.0, dense_high_end=True):
    """
    Choose the **smallest** threshold t in [t_min, t_max] such that **FPR <= max_fpr** on this label set.

    FPR = FP / (TN + FP) (fraction of **true negatives** incorrectly predicted positive).
    Raising t lowers FPR; this t maximizes SEP recall among thresholds meeting the FPR cap.

    Use on **validation** to target fewer false alarms; **test FPR may differ** under covariate shift.
    Returns (best_t, fpr_at_t).
    """
    ycls_true = np.asarray(ycls_true, dtype=np.int64).ravel()
    ycls_prob = np.nan_to_num(
        np.asarray(ycls_prob, dtype=np.float64).ravel(),
        nan=0.5, posinf=1.0, neginf=0.0,
    )
    t_lo = float(t_min)
    t_hi = float(t_max)
    if t_lo > t_hi:
        t_lo, t_hi = t_hi, t_lo
    grid = threshold_grid() if dense_high_end else np.linspace(0.0, 1.0, 101)
    grid = grid[(grid >= t_lo) & (grid <= t_hi)]
    if grid.size == 0:
        mid = 0.5 * (t_lo + t_hi)
        return float(mid), 1.0
    last_fpr = 1.0
    for t in sorted(grid):
        ypred = (ycls_prob > t).astype(np.int64)
        cm = confusion_matrix(ycls_true, ypred, labels=[0, 1])
        if cm.size != 4:
            continue
        tn, fp, fn, tp = cm.ravel()
        denom = tn + fp
        fpr = fp / denom if denom > 0 else 0.0
        last_fpr = fpr
        if fpr <= float(max_fpr):
            return float(t), fpr
    return float(t_hi), last_fpr


def _tpr_fpr_at_threshold(ycls_true, ycls_prob, t):
    """TPR (recall on positives) and FPR on negatives at threshold t."""
    ycls_true = np.asarray(ycls_true, dtype=np.int64).ravel()
    ycls_prob = np.nan_to_num(
        np.asarray(ycls_prob, dtype=np.float64).ravel(),
        nan=0.5, posinf=1.0, neginf=0.0,
    )
    ypred = (ycls_prob > float(t)).astype(np.int64)
    cm = confusion_matrix(ycls_true, ypred, labels=[0, 1])
    if cm.size != 4:
        return 0.0, 1.0
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (tn + fp) if (tn + fp) else 0.0
    return float(tpr), float(fpr)


def threshold_for_max_fpr_min_tpr(
    ycls_true,
    ycls_prob,
    max_fpr=0.05,
    min_tpr=0.0,
    t_min=0.0,
    t_max=1.0,
    dense_high_end=True,
):
    """
    Pick the **highest** (most conservative) threshold t in [t_min, t_max] such that **both**:
      FPR(t) <= max_fpr   and   TPR(t) >= min_tpr
    on this labeled pool (e.g. validation).

    This pushes t **up** vs `threshold_for_max_fpr` alone (which uses the **lowest** t meeting the FPR cap,
    maximizing val TPR). Here you cap false alarms **and** forbid val recall below `min_tpr`, so the chosen
    t is as strict as possible without breaking the hit-rate floor on the tuning set.
    If **no** t satisfies both (common when max_fpr is very tight), falls back to
    threshold_for_max_fpr (FPR cap only) and prints a warning; TPR may then fall below min_tpr.

    With **fixed** scores, lowering FPR by raising t always tends to lower TPR; min_tpr limits how
    much recall you sacrifice on the tuning set. Better separation from the model is what improves
    both simultaneously on held-out eras.

    Returns (best_t, fpr_at_t, tpr_at_t).
    """
    ycls_true = np.asarray(ycls_true, dtype=np.int64).ravel()
    ycls_prob = np.nan_to_num(
        np.asarray(ycls_prob, dtype=np.float64).ravel(),
        nan=0.5, posinf=1.0, neginf=0.0,
    )
    t_lo = float(t_min)
    t_hi = float(t_max)
    if t_lo > t_hi:
        t_lo, t_hi = t_hi, t_lo
    grid = threshold_grid() if dense_high_end else np.linspace(0.0, 1.0, 101)
    grid = grid[(grid >= t_lo) & (grid <= t_hi)]
    if grid.size == 0:
        mid = 0.5 * (t_lo + t_hi)
        tpr_m, fpr_m = _tpr_fpr_at_threshold(ycls_true, ycls_prob, mid)
        return float(mid), fpr_m, tpr_m

    min_tpr = float(min_tpr)
    max_fpr = float(max_fpr)
    best_t = None
    best_fpr = 1.0
    best_tpr = 0.0
    for t in grid:
        ypred = (ycls_prob > t).astype(np.int64)
        cm = confusion_matrix(ycls_true, ypred, labels=[0, 1])
        if cm.size != 4:
            continue
        tn, fp, fn, tp = cm.ravel()
        denom_neg = tn + fp
        denom_pos = tp + fn
        fpr = fp / denom_neg if denom_neg > 0 else 0.0
        tpr = tp / denom_pos if denom_pos > 0 else 0.0
        if fpr <= max_fpr and tpr >= min_tpr:
            if best_t is None or float(t) > best_t:
                best_t = float(t)
                best_fpr = float(fpr)
                best_tpr = float(tpr)
    if best_t is not None:
        return best_t, best_fpr, best_tpr

    t_fb, fpr_fb = threshold_for_max_fpr(
        ycls_true, ycls_prob, max_fpr=max_fpr, t_min=t_lo, t_max=t_hi, dense_high_end=dense_high_end
    )
    tpr_fb, _ = _tpr_fpr_at_threshold(ycls_true, ycls_prob, t_fb)
    print(
        "  warning: no threshold meets FPR<=", max_fpr, "and TPR>=", min_tpr,
        "— falling back to FPR cap only; val_TPR=", round(tpr_fb, 4),
    )
    return float(t_fb), float(fpr_fb), float(tpr_fb)


def fit_sigmoid_temperature(logits, y_true, t_hi=3.0, min_prob_std=0.10):
    """
    Scalar temperature T>0 for p = sigmoid(logits / T), fit on a labeled pool (e.g. val) by grid NLL.

    NLL alone can prefer **large T** that **compresses** probs (low std(p)); pos/neg quantiles then overlap
    and thresholding degrades. We use a **modest t_hi** (default 3) and reject if std(p) < min_prob_std
    (default 0.1) or if T sits on the grid cap with still-low spread.

    Returns (T, nll). On failure / flat probs / single class, returns (1.0, nan) and prints why.
    """
    logits = np.nan_to_num(
        np.asarray(logits, dtype=np.float64).ravel(), nan=0.0, posinf=0.0, neginf=0.0
    )
    y = np.asarray(y_true, dtype=np.float64).ravel()
    if logits.size != y.size:
        raise ValueError("logits and y_true must have the same length")
    n_pos = float(np.sum(y >= 0.5))
    n_neg = float(y.size - n_pos)
    if n_pos < 1.0 or n_neg < 1.0:
        print("cls_temperature: skipped (need both classes on cal pool), T=1.0")
        return 1.0, float("nan")

    best_T, best_nll = 1.0, np.inf
    t_hi = float(max(t_hi, 0.5))
    for T in np.geomspace(0.25, t_hi, 56):
        z = logits / T
        z = np.clip(z, -80.0, 80.0)
        p = 1.0 / (1.0 + np.exp(-z))
        p = np.clip(p, 1e-9, 1.0 - 1e-9)
        nll = -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        if nll < best_nll:
            best_nll, best_T = nll, T

    z = logits / max(best_T, 1e-9)
    z = np.clip(z, -80.0, 80.0)
    p = 1.0 / (1.0 + np.exp(-z))
    p_std = float(np.std(p))
    if p_std < float(min_prob_std):
        print(
            "cls_temperature: std(p)=%.4f < %.2f — NLL-opt T over-compresses; using T=1.0 for ranking. "
            "Try calibrate_cls_temperature=False or raise calibrate_cls_min_prob_std."
            % (p_std, min_prob_std)
        )
        return 1.0, float("nan")

    # NLL often sits on t_hi when logits are sharp; that flattens p and hurts threshold search unless spread remains wide.
    cap_spread = max(float(min_prob_std) + 0.06, 0.165)
    if best_T >= t_hi * 0.95 and p_std < cap_spread:
        print(
            "cls_temperature: T=%.3f at/near grid cap %.1f but std(p)=%.4f < %.3f — using T=1.0 (ambiguous cap optimum)."
            % (best_T, t_hi, p_std, cap_spread)
        )
        return 1.0, float("nan")

    if best_T >= t_hi * 0.95:
        print(
            "cls_temperature: note — T=%.3f is near grid cap %.1f; if metrics look off, set calibrate_cls_temperature=False."
            % (best_T, t_hi)
        )
    print(
        "cls_temperature: T=%.4f (val NLL grid), nll=%.4f, std(p)=%.4f"
        % (best_T, best_nll, p_std)
    )
    return float(best_T), float(best_nll)


def collect_cls_logits_np(model, selector, X_tensor, device, batch_size=2048):
    """Batched forward pass; returns 1-D numpy logits (before temperature scaling)."""
    model.eval()
    X_np = X_tensor.cpu().numpy()
    n = X_np.shape[0]
    chunks = []
    for i in range(0, n, batch_size):
        chunk = X_np[i : i + batch_size]
        X_sel = _select_transform_features(chunk, selector)
        X_sel_tensor = torch.tensor(X_sel, dtype=torch.float32).to(device)
        with torch.no_grad():
            _, logits, _ = model(X_sel_tensor)
        chunks.append(logits.squeeze(-1).detach().cpu().numpy().ravel())
    out = np.concatenate(chunks, axis=0)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def cls_pos_weight_tensor(ycls_tensor, device, multiplier=1.0, max_pos_weight=10.0):
    """
    BCEWithLogitsLoss pos_weight = multiplier * n_negative / n_positive (positive class).
    Capped to max_pos_weight: uncapped values + balanced batch sampling + focal loss often
    collapse logits to ~constant high sigmoid (~0.91+) with no usable ranking.
    """
    y = ycls_tensor.detach().cpu().numpy().ravel()
    n_pos = float(np.sum(y >= 0.5))
    n = float(len(y))
    n_neg = n - n_pos
    if n_pos < 1.0:
        w = 1.0
    else:
        w = (n_neg / max(n_pos, 1e-6)) * float(multiplier)
        w = min(float(w), float(max_pos_weight))
    return torch.tensor([w], dtype=torch.float32, device=device)


def init_classifier_bias_from_prevalence(model, y_cls_tensor):
    """
    Set the final classification layer bias to logit(p) where p is the positive rate on the **fit** set.
    Avoids starting from 0 when positives are rare (balanced batches still see ~50% pos, but this anchors
    scale to dataset prevalence and often improves val ranking under shift).
    """
    if y_cls_tensor is None:
        return
    branch = getattr(model, "cls_branch", None)
    if branch is None:
        return
    last = None
    for m in reversed(list(branch.modules())):
        if isinstance(m, nn.Linear) and m.out_features == 1:
            last = m
            break
    if last is None or last.bias is None:
        return
    y = y_cls_tensor.detach().cpu().numpy().ravel()
    pos = float(np.mean(y >= 0.5))
    pos = float(np.clip(pos, 5e-4, 1.0 - 5e-4))
    with torch.no_grad():
        last.bias.fill_(float(np.log(pos / (1.0 - pos))))


def _build_cls_mlp(transformer_dim, cls_hidden_dim, dropout_prob, num_hidden_layers):
    """num_hidden_layers: number of hidden blocks (Linear+GELU+Dropout) before the logit layer."""
    layers = []
    d_in = transformer_dim
    n_hid = max(1, int(num_hidden_layers))
    for _ in range(n_hid):
        layers += [
            nn.Linear(d_in, cls_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
        ]
        d_in = cls_hidden_dim
    layers.append(nn.Linear(d_in, 1))
    return nn.Sequential(*layers)


def _make_shared_trunk(in_dim, out_dim, dropout_prob):
    """LN → GELU → Dropout → Linear → LN → GELU bottleneck (used per-head when decoupled)."""
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.GELU(),
        nn.Dropout(dropout_prob),
        nn.Linear(in_dim, out_dim),
        nn.LayerNorm(out_dim),
        nn.GELU(),
    )


class RegressionHead(nn.Module):
    """
    Per-task regression head that separates each task's *location* (mean) from the
    feature-driven *deviation*. This removes the systematic per-task mean shift seen when a
    single shared trunk + plain ``nn.Linear`` underfits each target's offset:

        y_hat[:, j] = proj_j(feat) + task_bias_j

    All statistics come from the TRAINING split only, so there is no val/test leakage:
      * ``proj`` learns a (roughly) zero-centered deviation; its own bias starts at 0.
      * ``task_bias`` is a per-task location term initialized from the training per-task mean
        (see ``set_target_location``). It is excluded from weight decay (see
        ``_mtm_adam_param_groups``) so optimization does not shrink each task's mean toward 0.
      * ``apply_bias_correction`` folds the leftover per-task mean residual measured on the
        TRAINING predictions back into ``task_bias`` after training. This cancels the
        median-vs-mean gap that L1 / SmoothL1 losses introduce on skewed count targets
        (e.g. flare / CME counts), a common source of per-head mean shift.

    The head is a drop-in replacement for ``nn.Linear(in_dim, output_dim)`` and operates in
    whatever target space it is trained on (raw units, or the standardized space used when
    ``standardize_yreg=True``); ``set_target_location``/``apply_bias_correction`` are simply
    no-ops/zero in standardized space at init and still correct residual bias after training.
    """

    def __init__(self, in_dim, output_dim):
        super().__init__()
        self.output_dim = int(output_dim)
        self.proj = nn.Linear(in_dim, self.output_dim)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
        self.task_bias = nn.Parameter(torch.zeros(self.output_dim))
        # Frozen record of the training-set location estimate (kept for inspection / reset).
        self.register_buffer("init_task_mean", torch.zeros(self.output_dim))

    @torch.no_grad()
    def set_target_location(self, mean):
        """Initialize the per-task location from a training-set per-task mean (no leakage)."""
        if self.output_dim == 0 or mean is None:
            return
        mean = torch.as_tensor(mean, dtype=self.task_bias.dtype, device=self.task_bias.device).reshape(-1)
        if mean.numel() != self.output_dim:
            return
        self.task_bias.copy_(mean)
        self.init_task_mean.copy_(mean)

    @torch.no_grad()
    def apply_bias_correction(self, delta):
        """Add a per-task offset (e.g. the mean training residual) into the location term."""
        if self.output_dim == 0 or delta is None:
            return
        delta = torch.as_tensor(delta, dtype=self.task_bias.dtype, device=self.task_bias.device).reshape(-1)
        if delta.numel() != self.output_dim:
            return
        self.task_bias.add_(delta)

    def forward(self, feat):
        return self.proj(feat) + self.task_bias


class HurdleGate(nn.Module):
    """
    Occurrence gate for **zero-inflated** regression targets (a two-part / "hurdle" model).

    Some targets are not smooth continuous variables but a spike-at-zero mixed with a
    continuous part — e.g. ``Future_log_DONKICME_speed_km_s_max`` is ~34% exactly 0 ("no CME")
    and otherwise clusters near log10(speed) ≈ 2.7. A single regression head trained on such a
    target collapses toward the empty middle, giving large RMSE/MAE and low R².

    The hurdle model splits the prediction into:
        * an **occurrence** probability  p_j = sigmoid(gate_j(feat))  ≈ P(y_j > 0 | x), and
        * a **conditional value**       v_j = reg_head_j(feat)        ≈ E[y_j | y_j > 0, x].
    The point estimate is the expectation  E[y_j] = p_j · v_j.

    This module only produces the gate logits for the hurdle target columns (``target_idx``);
    the conditional value comes from the existing ``RegressionHead``. ``target_idx`` indexes
    columns of the regression output, and the returned logits are aligned to that list.
    """

    def __init__(self, in_dim, target_idx, output_dim):
        super().__init__()
        self.target_idx = [int(i) for i in (target_idx or [])]
        self.output_dim = int(output_dim)
        self.proj = nn.Linear(in_dim, len(self.target_idx)) if self.target_idx else None

    def forward(self, feat):
        if self.proj is None:
            return None
        return self.proj(feat)  # (n, n_hurdle), aligned to self.target_idx


def _model_hurdle_idx(model):
    """Hurdle target column indices configured on a model (empty list if none / legacy model)."""
    return list(getattr(model, "hurdle_target_idx", []) or [])


def _mtm_train_reg_residual_mean(model, train_loader, device, output_dim, hurdle_idx=None, hurdle_eps=1e-6):
    """
    Mean per-task regression residual (y_true - y_pred) over the TRAINING loader.

    Used as a leakage-free de-bias signal: adding this back into the head's per-task location
    makes training predictions mean-unbiased, which cancels the systematic per-head mean shift
    that median-seeking losses (L1 / SmoothL1) cause on skewed targets. Returns a 1D tensor of
    length ``output_dim`` or ``None`` if there is nothing to correct.

    For hurdle targets (``hurdle_idx``), the conditional value is only defined on occurrence
    (y > eps) rows, so the residual mean is taken over those rows only.
    """
    if output_dim <= 0:
        return None
    hset = set(int(i) for i in (hurdle_idx or []))
    model.eval()
    sum_res = torch.zeros(output_dim, device=device)
    cnt = torch.zeros(output_dim, device=device)
    with torch.no_grad():
        for batch in train_loader:
            xb, ybreg, _ybcls, _ybteach, _ybsw = _unpack_mtm_train_batch(batch)
            xb = xb.to(device)
            ybreg = ybreg.to(device)
            out = model(xb)
            reg_pred = out[0]
            res = ybreg - reg_pred
            if hset:
                mask = torch.ones_like(ybreg)
                for j in hset:
                    if 0 <= j < output_dim:
                        mask[:, j] = (ybreg[:, j] > hurdle_eps).float()
                res = res * mask
                cnt += mask.sum(dim=0)
            else:
                cnt += float(xb.size(0))
            sum_res += res.sum(dim=0)
    cnt = cnt.clamp(min=1.0)
    return (sum_res / cnt).detach()


def _mtm_column_reg_loss(
    reg_pred_j,
    ybreg_j,
    crit_reg,
    *,
    reg_loss_type="smooth_l1",
    smooth_l1_beta=1.0,
    zero_loss_floor=False,
    floor_std=0.0,
    floor_alpha=0.35,
    zero_eps=1e-6,
):
    """Per-column regression loss with optional floor on no-target rows (y <= eps)."""
    if not zero_loss_floor or floor_std <= 0:
        return crit_reg(reg_pred_j, ybreg_j)
    if reg_loss_type == "mse":
        per = F.mse_loss(reg_pred_j, ybreg_j, reduction="none")
    else:
        per = F.smooth_l1_loss(
            reg_pred_j, ybreg_j, beta=float(smooth_l1_beta), reduction="none"
        )
    floor_val = float(floor_alpha) * float(floor_std)
    no_target = ybreg_j <= zero_eps
    if bool(no_target.any()):
        per = torch.where(
            no_target,
            torch.clamp(per, min=reg_pred_j.new_tensor(floor_val)),
            per,
        )
    return per.mean()


def _mtm_hurdle_losses(
    reg_pred,
    ybreg,
    gate_logits,
    hurdle_idx,
    crit_reg,
    output_dim,
    hurdle_eps=1e-6,
    *,
    reg_loss_type="smooth_l1",
    smooth_l1_beta=1.0,
    zero_loss_floor_cols=None,
    zero_loss_floor_alpha=0.35,
    reg_col_std=None,
    reg_col_loss_weights=None,
):
    """
    Regression + occurrence-gate losses for a (possibly) hurdle multitask model.

    Returns ``(loss_reg, loss_gate)`` where:
      * ``loss_reg`` matches the legacy scaling ``crit_reg(reg_pred, ybreg) / output_dim`` exactly
        when there are no hurdle targets; for hurdle columns the conditional regression loss is
        computed **only on occurrence rows** (y > eps) so the head learns E[y | y>0] instead of
        being dragged toward the middle by the structural zeros.
      * ``loss_gate`` is the mean BCE of the occurrence logits against ``1[y > eps]`` over the
        hurdle columns (0 when there is no gate).
    """
    device = reg_pred.device
    if output_dim <= 0:
        z = torch.tensor(0.0, device=device)
        return z, z
    hset = set(int(i) for i in (hurdle_idx or []))
    zset = set(int(i) for i in (zero_loss_floor_cols or []))
    if reg_col_std is not None:
        std_t = reg_col_std.to(reg_pred.device).reshape(-1)
    else:
        std_t = None
    if reg_col_loss_weights is not None:
        w_t = torch.as_tensor(reg_col_loss_weights, dtype=reg_pred.dtype, device=reg_pred.device).reshape(-1)
    else:
        w_t = None
    # Per-column regression loss (legacy-equivalent normalization: sum_j mean_j / output_dim^2).
    per_col = []
    for j in range(output_dim):
        if j in hset:
            m = ybreg[:, j] > hurdle_eps
            if bool(m.any()):
                col_loss = crit_reg(reg_pred[m, j], ybreg[m, j])
            else:
                col_loss = reg_pred.new_zeros(())
        else:
            floor_std = float(std_t[j].item()) if std_t is not None and j < std_t.numel() else 0.0
            col_loss = _mtm_column_reg_loss(
                reg_pred[:, j], ybreg[:, j], crit_reg,
                reg_loss_type=reg_loss_type,
                smooth_l1_beta=smooth_l1_beta,
                zero_loss_floor=(j in zset),
                floor_std=floor_std,
                floor_alpha=zero_loss_floor_alpha,
                zero_eps=hurdle_eps,
            )
        if w_t is not None and j < w_t.numel():
            col_loss = col_loss * w_t[j]
        per_col.append(col_loss)
    loss_reg = torch.stack(per_col).sum() / float(output_dim * output_dim)

    loss_gate = torch.tensor(0.0, device=device)
    if gate_logits is not None and hset:
        idx_list = [int(i) for i in hurdle_idx if 0 <= int(i) < output_dim]
        acc = reg_pred.new_zeros(())
        for col, j in enumerate(idx_list):
            occ = (ybreg[:, j] > hurdle_eps).float()
            acc = acc + F.binary_cross_entropy_with_logits(gate_logits[:, col], occ)
        loss_gate = acc / max(1, len(idx_list))
    return loss_reg, loss_gate


def _mtm_hurdle_fill_means(yreg_train, hurdle_idx, mode="conditional", hurdle_eps=1e-6):
    """Per-column fill values for hurdle targets when occurrence is absent.

    Used at inference to avoid collapsing predictions to ~0 when the gate is low:
        pred = p * v + (1 - p) * fill_mean[j]

    mode:
      * ``'conditional'`` — mean of training rows with y > eps (typical CME speed when one occurs).
      * ``'unconditional'`` — mean over all training rows (includes structural zeros).
    """
    import torch

    y = torch.as_tensor(yreg_train, dtype=torch.float64)
    if y.ndim == 1:
        y = y.unsqueeze(1)
    n_out = int(y.shape[1])
    fill = y.mean(dim=0)
    for j in (hurdle_idx or []):
        j = int(j)
        if not (0 <= j < n_out):
            continue
        if mode == "unconditional":
            continue
        pos = y[:, j] > hurdle_eps
        if bool(pos.any()):
            fill[j] = y[pos, j].mean()
    return fill.detach().cpu()


def _mtm_hurdle_combine_predictions(
    yreg_pred,
    gate_prob,
    hurdle_idx,
    fill_mean,
    hurdle_missing_fill="conditional_mean",
):
    """Combine conditional regression values with gate probabilities for hurdle columns.

    Legacy (``hurdle_missing_fill='zero'``):  E[y] = p * v  → 0 when p→0.
    Mean-fill (default):  E[y] = p * v + (1-p) * μ  → μ when p→0 (no cliff to zero).
    """
    out = np.asarray(yreg_pred, dtype=np.float64).copy()
    if out.ndim == 1:
        out = out.reshape(-1, 1)
    mode = str(hurdle_missing_fill or "conditional_mean").lower()
    if mode in ("zero", "product", "none"):
        blend_fill = False
    else:
        blend_fill = True
        if fill_mean is None:
            blend_fill = False
        else:
            mu = np.asarray(fill_mean, dtype=np.float64).reshape(-1)
    gp = np.asarray(gate_prob, dtype=np.float64)
    if gp.ndim == 1:
        gp = gp.reshape(-1, 1)
    for col, j in enumerate(hurdle_idx or []):
        j = int(j)
        if not (0 <= j < out.shape[1] and col < gp.shape[1]):
            continue
        p = gp[:, col]
        v = out[:, j]
        if blend_fill and j < len(mu):
            out[:, j] = p * v + (1.0 - p) * float(mu[j])
        else:
            out[:, j] = p * v
    return out


def _is_mtm_regression_branch_param(name: str) -> bool:
    """
    Parameters that sit on the regression-only path (after decoupling), so a higher LR
    can strengthen the reg head without changing classification loss definitions.
    """
    if "reg_head" in name:
        return True
    if name.startswith("fc_reg.") or ".fc_reg." in name:
        return True
    if name.startswith("fuse_reg.") or ".fuse_reg." in name:
        return True
    return False


def _is_reg_location_bias_param(name: str) -> bool:
    """Per-task regression location bias (``RegressionHead.task_bias``).

    Kept out of weight decay so each task's estimated mean is not shrunk toward 0 — a direct
    cause of per-head mean shift when the head bias is regularized.
    """
    return name.endswith("reg_head.task_bias")


def _mtm_adam_param_groups(model: nn.Module, lr: float, weight_decay: float, reg_branch_lr_mult: float,
                           reg_weight_decay_mult: float = 1.0):
    """Adam param groups:
      * the per-task regression location bias is always placed in a ``weight_decay=0`` group;
      * optionally a higher LR (``reg_branch_lr_mult``) and/or lower weight decay
        (``reg_weight_decay_mult`` < 1, reduces shrinkage / amplitude collapse) on the rest of
        the regression branch (decoupled LSTM / hybrid).
    """
    m = float(reg_branch_lr_mult)
    use_reg_mult = np.isfinite(m) and m > 0.0 and abs(m - 1.0) >= 1e-8
    wd_m = float(reg_weight_decay_mult)
    use_wd_mult = np.isfinite(wd_m) and wd_m >= 0.0 and abs(wd_m - 1.0) >= 1e-8
    separate_reg = use_reg_mult or use_wd_mult
    no_wd_p, reg_p, other_p = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if _is_reg_location_bias_param(n):
            no_wd_p.append(p)
        elif separate_reg and _is_mtm_regression_branch_param(n):
            reg_p.append(p)
        else:
            other_p.append(p)
    groups = [{"params": other_p, "lr": lr, "weight_decay": weight_decay}]
    if reg_p:
        groups.append({
            "params": reg_p,
            "lr": lr * m if use_reg_mult else lr,
            "weight_decay": weight_decay * wd_m if use_wd_mult else weight_decay,
        })
    if no_wd_p:
        groups.append({"params": no_wd_p, "lr": lr, "weight_decay": 0.0})
    return groups


def train_loop_MTM(train_loader, input_dim_per_step, sequence_len, output_dim,
                   lr, weight_decay, dropout_prob,
                   max_epochs, patience, max_norm, device, LSTMTransformer,
                   Hybrid=False,
                   val_loader=None,
                   early_stop_on="val_skill",
                   val_skill_metric="min_tss_hss",
                   val_skill_metric_t_min=0.0,
                   val_skill_metric_t_max=1.0,
                   cls_pos_weight=None,
                   focal_gamma=2.0,
                   focal_scale=2.0,
                   reg_loss_scale=1.5,
                   cls_loss_scale=1.35,
                   focal_alpha_neg=0.25,
                   focal_alpha_pos=0.75,
                   reg_loss_type="smooth_l1",
                   smooth_l1_beta=1.0,
                   lstm_hidden_dim=64,
                   transformer_dim=16,
                   nhead=4,
                   num_transformer_layers=1,
                   cls_hidden_dim=32,
                   lstm_bidirectional=False,
                   sequence_pooling="last",
                   use_input_layernorm=True,
                   lstm_residual=True,
                   input_dropout_p=0.0,
                   cls_num_hidden_layers=1,
                   cls_label_smoothing=0.0,
                   ycls_fit_for_bias=None,
                   init_cls_bias_from_prevalence=True,
                   decouple_heads=False,
                   cls_transformer_dim=None,
                   # Hybrid dims
                   hybrid_seq_trunk_dim=32,
                   hybrid_tab_hidden_dim=96,
                   hybrid_tab_num_layers=2,
                   hybrid_fusion_dim=64,
                   hybrid_cls_fusion_dim=None,
                   # Distillation (teacher probs carried by loaders)
                   distill_weight=0.0,
                   distill_temperature=2.0,
                   freeze_reg_epochs=0,
                   reg_ramp_epochs=0,
                   log_head_losses=False,
                   reg_branch_lr_mult=1.0,
                   reg_weight_decay_mult=1.0,
                   reg_target_mean=None,
                   reg_debias=True,
                   hurdle_target_idx=None,
                   hurdle_gate_loss_scale=1.0,
                   zero_loss_floor_cols=None,
                   zero_loss_floor_alpha=0.35,
                   reg_col_std=None,
                   reg_col_loss_weights=None):

    hurdle_idx = [int(i) for i in (hurdle_target_idx or []) if 0 <= int(i) < output_dim]
    zero_floor_cols = [int(i) for i in (zero_loss_floor_cols or []) if 0 <= int(i) < output_dim]
    _reg_loss_kw = dict(
        reg_loss_type=reg_loss_type,
        smooth_l1_beta=smooth_l1_beta,
        zero_loss_floor_cols=zero_floor_cols,
        zero_loss_floor_alpha=float(zero_loss_floor_alpha),
        reg_col_std=reg_col_std,
        reg_col_loss_weights=reg_col_loss_weights,
    )
    if Hybrid:
        model = MultiTaskModel_HybridLSTMTabular(
            input_dim_per_step,
            output_dim,
            dropout_prob=dropout_prob,
            lstm_hidden_dim=lstm_hidden_dim,
            nhead=nhead,
            num_transformer_layers=num_transformer_layers,
            seq_trunk_dim=hybrid_seq_trunk_dim,
            tab_hidden_dim=hybrid_tab_hidden_dim,
            tab_num_layers=hybrid_tab_num_layers,
            fusion_dim=hybrid_fusion_dim,
            cls_hidden_dim=cls_hidden_dim,
            lstm_bidirectional=lstm_bidirectional,
            sequence_pooling=sequence_pooling,
            use_input_layernorm=use_input_layernorm,
            lstm_residual=lstm_residual,
            input_dropout_p=input_dropout_p,
            cls_num_hidden_layers=cls_num_hidden_layers,
            decouple_heads=True,
            cls_fusion_dim=hybrid_cls_fusion_dim,
            hurdle_target_idx=hurdle_idx,
        ).to(device)
    elif LSTMTransformer:
        model = MultiTaskModel_LSTMTransformer(
            input_dim_per_step, output_dim, dropout_prob=dropout_prob,
            lstm_hidden_dim=lstm_hidden_dim, transformer_dim=transformer_dim,
            nhead=nhead, num_transformer_layers=num_transformer_layers,
            cls_hidden_dim=cls_hidden_dim,
            lstm_bidirectional=lstm_bidirectional,
            sequence_pooling=sequence_pooling,
            use_input_layernorm=use_input_layernorm,
            lstm_residual=lstm_residual,
            input_dropout_p=input_dropout_p,
            cls_num_hidden_layers=cls_num_hidden_layers,
            decouple_heads=decouple_heads,
            cls_transformer_dim=cls_transformer_dim,
            hurdle_target_idx=hurdle_idx,
        ).to(device)
    else:
        mlp_in = int(input_dim_per_step * max(sequence_len, 1))
        model = MultiTaskModel(
            mlp_in, output_dim, dropout_prob=dropout_prob, cls_hidden_dim=cls_hidden_dim,
            cls_num_hidden_layers=cls_num_hidden_layers,
            hurdle_target_idx=hurdle_idx,
        ).to(device)

    if init_cls_bias_from_prevalence:
        init_classifier_bias_from_prevalence(model, ycls_fit_for_bias)

    # Warm-start each regression task's location from the TRAINING per-task mean (no leakage).
    if (
        output_dim > 0
        and reg_target_mean is not None
        and hasattr(model, "reg_head")
        and hasattr(model.reg_head, "set_target_location")
    ):
        model.reg_head.set_target_location(reg_target_mean)

    _pgs = _mtm_adam_param_groups(model, lr, weight_decay, float(reg_branch_lr_mult), float(reg_weight_decay_mult))
    optimizer = optim.Adam(_pgs)
    if reg_loss_type == "mse":
        crit_reg = nn.MSELoss()  
    else:
        crit_reg = nn.SmoothL1Loss(beta=smooth_l1_beta)
    pw = cls_pos_weight if cls_pos_weight is not None else torch.tensor([1.0], device=device)
    crit_cls = nn.BCEWithLogitsLoss(pos_weight=pw)
    crit_cls2_mean = FocalLoss(
        gamma=focal_gamma, alpha=(focal_alpha_neg, focal_alpha_pos), reduction="mean"
    )
    crit_cls2_none = FocalLoss(
        gamma=focal_gamma, alpha=(focal_alpha_neg, focal_alpha_pos), reduction="none"
    )
    use_val_stop = val_loader is not None and early_stop_on in (
        "val_skill",
        "val_auc",
        "val_multitask_loss",
    )
    if use_val_stop:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=4, threshold=1e-4
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=5
        )

    best_train_loss = float("inf")
    best_val_score = -1e18
    best_state = None
    patience_counter = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()

    freeze_reg = int(max(0, freeze_reg_epochs))
    ramp_n = int(max(0, reg_ramp_epochs))
    reg_full = float(reg_loss_scale)

    for epoch in range(max_epochs):

        if epoch < freeze_reg:
            reg_sc = 0.0
        elif ramp_n > 0 and epoch < freeze_reg + ramp_n:
            t = epoch - freeze_reg + 1
            reg_sc = reg_full * min(1.0, float(t) / float(ramp_n))
        else:
            reg_sc = reg_full

        model.train()
        for batch in train_loader:
            xb, ybreg, ybcls, ybteach, ybsw = _unpack_mtm_train_batch(batch)
            if ybteach is not None:
                ybteach = ybteach.to(device)
            if ybsw is not None:
                ybsw = ybsw.to(device)
            xb, ybreg, ybcls = xb.to(device), ybreg.to(device), ybcls.to(device)
            optimizer.zero_grad()
            reg_pred, cls_logits, gate_logits = model(xb)
            if output_dim > 0:
                loss_reg, loss_gate = _mtm_hurdle_losses(
                    reg_pred, ybreg, gate_logits, hurdle_idx, crit_reg, output_dim,
                    **_reg_loss_kw,
                )
            else:
                loss_reg = torch.tensor(0.0, device=xb.device)
                loss_gate = torch.tensor(0.0, device=xb.device)
            #print("loss_reg=",loss_reg)
            z = cls_logits.view(-1)
            y = ybcls.view(-1)
            ls = float(cls_label_smoothing)
            if ls > 0.0:
                y_bce = y * (1.0 - ls) + 0.5 * ls
            else:
                y_bce = y
            loss_cls = _mtm_cls_loss(
                z, y, y_bce, crit_cls, crit_cls2_mean, crit_cls2_none,
                focal_scale, sample_w=ybsw,
            )
            loss_distill = torch.tensor(0.0, device=xb.device)
            dw = float(distill_weight)
            if ybteach is not None and dw > 0.0:
                Td = float(max(distill_temperature, 1e-6))
                tprob = torch.clamp(ybteach.view(-1), 1e-5, 1.0 - 1e-5)
                loss_distill = F.binary_cross_entropy_with_logits(z / Td, tprob, reduction="mean") * (Td * Td)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            (
                reg_sc * loss_reg
                + cls_loss_scale * loss_cls
                + dw * loss_distill
                + reg_sc * float(hurdle_gate_loss_scale) * loss_gate
            ).backward()
            optimizer.step()

        model.eval()
        train_loss = 0.0
        sum_reg = sum_bce = sum_focal = 0.0
        n_loss = 0
        with torch.no_grad():
            for batch in train_loader:
                xb, ybreg, ybcls, _ybteach, ybsw = _unpack_mtm_train_batch(batch)
                bs = xb.size(0)
                xb, ybreg, ybcls = xb.to(device), ybreg.to(device), ybcls.to(device)
                if ybsw is not None:
                    ybsw = ybsw.to(device)
                reg_pred, cls_logits, _gate_logits = model(xb)
                if output_dim > 0:
                    loss_reg, _ = _mtm_hurdle_losses(
                        reg_pred, ybreg, None, hurdle_idx, crit_reg, output_dim,
                        **_reg_loss_kw,
                    )
                else:
                    loss_reg = torch.tensor(0.0, device=xb.device)
                z = cls_logits.view(-1)
                y = ybcls.view(-1)
                ls = float(cls_label_smoothing)
                y_bce = y * (1.0 - ls) + 0.5 * ls if ls > 0.0 else y
                loss_cls = _mtm_cls_loss(
                    z, y, y_bce, crit_cls, crit_cls2_mean, crit_cls2_none,
                    focal_scale, sample_w=ybsw,
                )
                loss_bce = F.binary_cross_entropy_with_logits(z, y_bce, reduction="mean")
                loss_focal = crit_cls2_mean(z, y)
                train_loss += (
                    reg_sc * loss_reg + cls_loss_scale * loss_cls
                ).item() * bs
                sum_reg += float(loss_reg.item()) * bs
                sum_bce += float(loss_bce.item()) * bs
                sum_focal += float(loss_focal.item()) * bs
                n_loss += bs

        train_loss /= len(train_loader.dataset)
        n_loss = max(n_loss, 1)
        m_reg = sum_reg / n_loss
        m_bce = sum_bce / n_loss
        m_focal = sum_focal / n_loss
        w_reg = reg_sc * m_reg
        w_bce = cls_loss_scale * m_bce
        w_foc = cls_loss_scale * focal_scale * m_focal
        w_total = w_reg + w_bce + w_foc
        if log_head_losses:
            if w_total > 1e-12:
                sr, sb, sf = 100.0 * w_reg / w_total, 100.0 * w_bce / w_total, 100.0 * w_foc / w_total
            else:
                sr = sb = sf = 0.0
            if reg_sc <= 0.0:
                reg_tag = "0*L_reg(cls_warmup)"
            elif reg_sc < reg_full - 1e-8:
                reg_tag = f"{reg_sc:.4g}*L_reg(ramp)"
            else:
                reg_tag = f"{reg_loss_scale}*L_reg"
            print(
                f"[epoch {epoch + 1}] train mean loss | "
                f"L_reg={m_reg:.6g} | L_bce={m_bce:.6g} | L_focal={m_focal:.6g} || "
                f"weighted: {reg_tag}={w_reg:.6g} | "
                f"{cls_loss_scale}*L_bce={w_bce:.6g} | "
                f"{cls_loss_scale}*{focal_scale}*L_focal={w_foc:.6g} | "
                f"sum={w_total:.6g} | "
                f"share% reg/bce/focal: {sr:.1f}/{sb:.1f}/{sf:.1f}"
            )

        if use_val_stop:
            if early_stop_on == "val_multitask_loss":
                # Joint val loss with full reg weight (comparable across epochs despite reg ramp on train).
                model.eval()
                sum_j, n_j = 0.0, 0
                with torch.no_grad():
                    for batch in val_loader:
                        if len(batch) == 4:
                            xb, ybreg, ybcls, _ = batch
                        else:
                            xb, ybreg, ybcls = batch
                        bs = xb.size(0)
                        xb = xb.to(device)
                        ybreg = ybreg.to(device)
                        ybcls = ybcls.to(device)
                        reg_pred, cls_logits, _gate_logits = model(xb)
                        if output_dim > 0:
                            lr_b, _ = _mtm_hurdle_losses(
                                reg_pred, ybreg, None, hurdle_idx, crit_reg, output_dim,
                                **_reg_loss_kw,
                            )
                        else:
                            lr_b = torch.tensor(0.0, device=device)
                        z = cls_logits.view(-1)
                        y = ybcls.view(-1)
                        ls = float(cls_label_smoothing)
                        y_bce = y * (1.0 - ls) + 0.5 * ls if ls > 0.0 else y
                        lc_b = crit_cls(z, y_bce) + focal_scale * crit_cls2_mean(z, y)
                        joint = float(reg_full) * float(lr_b.item()) + float(
                            cls_loss_scale
                        ) * float(lc_b.item())
                        sum_j += joint * bs
                        n_j += bs
                val_score = -(sum_j / max(n_j, 1))
            else:
                probs, labels = [], []
                model.eval()
                with torch.no_grad():
                    for batch in val_loader:
                        if len(batch) == 4:
                            xb, _ybreg, ybcls, _ybteach = batch
                        else:
                            xb, _ybreg, ybcls = batch
                        xb = xb.to(device)
                        ybcls = ybcls.to(device)
                        _, logits, _gate = model(xb)
                        p = torch.sigmoid(logits).squeeze(-1).detach().cpu().numpy()
                        if p.ndim == 0:
                            p = np.array([float(p)])
                        probs.append(p)
                        labels.append(ybcls.detach().cpu().numpy().ravel())
                y_prob = np.concatenate(probs).ravel()
                y_true = np.concatenate(labels).ravel()
                y_prob = np.nan_to_num(y_prob, nan=0.5, posinf=1.0, neginf=0.0)
                if early_stop_on == "val_auc":
                    y_int = y_true.astype(np.int64)
                    if len(np.unique(y_int)) < 2:
                        val_score = 0.5
                    elif float(np.std(y_prob)) < 0.02:
                        # Degenerate near-constant scores → AUC not meaningful; do not select checkpoint.
                        val_score = 0.5
                    else:
                        val_score = float(roc_auc_score(y_int, y_prob))
                else:
                    _, val_score = best_threshold_for_skill(
                        y_true, y_prob, metric=val_skill_metric,
                        t_min=val_skill_metric_t_min, t_max=val_skill_metric_t_max,
                    )
            scheduler.step(val_score)
            if val_score > best_val_score:
                best_val_score = val_score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        else:
            scheduler.step(train_loss)
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

    if use_val_stop and best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        best_train_loss = best_val_score

    # Leakage-free per-task mean de-bias: fold the mean TRAINING residual into the head's
    # location term so regression predictions are mean-unbiased (cancels the median-vs-mean
    # gap from SmoothL1/L1 on skewed targets). Only the selected/restored model is corrected.
    if (
        reg_debias
        and output_dim > 0
        and hasattr(model, "reg_head")
        and hasattr(model.reg_head, "apply_bias_correction")
    ):
        delta = _mtm_train_reg_residual_mean(model, train_loader, device, output_dim, hurdle_idx=hurdle_idx)
        if delta is not None:
            model.reg_head.apply_bias_correction(delta)

    train_time_sec = time.perf_counter() - t0
    return model, best_train_loss, train_time_sec

    
class MultiTaskModel(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob=0.2, cls_hidden_dim=32, cls_num_hidden_layers=1,
                 hurdle_target_idx=None):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 320), nn.LayerNorm(320), nn.ReLU(), nn.Dropout(dropout_prob),
            nn.Linear(320, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(dropout_prob),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(dropout_prob),
            nn.Linear(64, 16), nn.LayerNorm(16), nn.ReLU()
        )
        self.reg_head = RegressionHead(16, output_dim)
        self.hurdle_target_idx = [int(i) for i in (hurdle_target_idx or [])]
        self.gate = HurdleGate(16, self.hurdle_target_idx, output_dim) if self.hurdle_target_idx else None
        hid = max(1, int(cls_num_hidden_layers))
        if hid == 1:
            self.cls_branch = nn.Sequential(
                nn.Linear(16, cls_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_prob),
                nn.Linear(cls_hidden_dim, 1),
            )
        else:
            layers = []
            d = 16
            for _ in range(hid):
                layers += [
                    nn.Linear(d, cls_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_prob),
                ]
                d = cls_hidden_dim
            layers.append(nn.Linear(d, 1))
            self.cls_branch = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 3:
            x = x.reshape(x.size(0), -1)
        feat = self.shared(x)
        gate = self.gate(feat) if getattr(self, "gate", None) is not None else None
        return self.reg_head(feat), self.cls_branch(feat), gate





class MultiTaskModel_LSTMTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        dropout_prob=0.3,
        lstm_hidden_dim=64,
        transformer_dim=16,
        nhead=4,
        num_transformer_layers=1,
        cls_hidden_dim=32,
        lstm_bidirectional=False,
        sequence_pooling="last",
        use_input_layernorm=True,
        lstm_residual=True,
        input_dropout_p=0.0,
        cls_num_hidden_layers=1,
        decouple_heads=False,
        cls_transformer_dim=None,
        hurdle_target_idx=None,
    ):
        super().__init__()
        if sequence_pooling not in ("last", "attn"):
            raise ValueError('sequence_pooling must be "last" or "attn"')

        self.decouple_heads = bool(decouple_heads)
        self.lstm_bidirectional = lstm_bidirectional
        self.sequence_pooling = sequence_pooling
        self.lstm_residual = lstm_residual
        self.input_ln = nn.LayerNorm(input_dim) if use_input_layernorm else None
        p_drop = float(input_dropout_p)
        self.input_drop = nn.Dropout(p_drop) if p_drop > 0.0 else None
        num_directions = 2 if lstm_bidirectional else 1
        lstm_out_dim = lstm_hidden_dim * num_directions

        self.lstm = nn.LSTM(
            input_dim, lstm_hidden_dim, batch_first=True, bidirectional=lstm_bidirectional
        )
        self.pos_encoder = nn.Sequential(
            nn.LayerNorm(lstm_out_dim), nn.Dropout(dropout_prob)
        )

        n_tf = int(num_transformer_layers)
        if n_tf > 0:
            if lstm_out_dim % nhead != 0:
                raise ValueError(
                    f"LSTM output dim {lstm_out_dim} must be divisible by nhead={nhead}"
                )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=lstm_out_dim,
                nhead=nhead,
                dropout=dropout_prob,
                batch_first=True,
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=n_tf
            )
        else:
            self.transformer_encoder = None

        self.post_encoder_ln = nn.LayerNorm(lstm_out_dim)

        if sequence_pooling == "attn":
            self.attn_pool = nn.Linear(lstm_out_dim, 1)
        else:
            self.attn_pool = None

        td_cls = int(cls_transformer_dim) if cls_transformer_dim is not None else int(transformer_dim)

        if self.decouple_heads:
            self.fc = None
            self.fc_reg = _make_shared_trunk(lstm_out_dim, transformer_dim, dropout_prob)
            self.fc_cls = _make_shared_trunk(lstm_out_dim, td_cls, dropout_prob)
            self.reg_head = RegressionHead(transformer_dim, output_dim)
            self.cls_branch = _build_cls_mlp(
                td_cls, cls_hidden_dim, dropout_prob, cls_num_hidden_layers
            )
        else:
            self.fc = nn.Sequential(
                nn.LayerNorm(lstm_out_dim),
                nn.GELU(),
                nn.Dropout(dropout_prob),
                nn.Linear(lstm_out_dim, transformer_dim),
                nn.LayerNorm(transformer_dim),
                nn.GELU(),
            )
            self.fc_reg = None
            self.fc_cls = None
            self.reg_head = RegressionHead(transformer_dim, output_dim)
            self.cls_branch = _build_cls_mlp(
                transformer_dim, cls_hidden_dim, dropout_prob, cls_num_hidden_layers
            )

        # Occurrence gate (hurdle model) on the regression feature; None when no hurdle targets.
        self.hurdle_target_idx = [int(i) for i in (hurdle_target_idx or [])]
        self.gate = HurdleGate(int(transformer_dim), self.hurdle_target_idx, output_dim) if self.hurdle_target_idx else None

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, input_dim)
        If your input currently is (batch_size, input_dim), unsqueeze seq_len=1.
        """

        if x.dim() == 2:
            x = x.unsqueeze(1)

        if self.input_ln is not None:
            x = self.input_ln(x)
        if self.input_drop is not None:
            x = self.input_drop(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.pos_encoder(lstm_out)
        # batch_first=True: (N, L, E). Do not permute to (L, N, E) — that swapped batch and time.
        if self.transformer_encoder is not None:
            transformer_out = self.transformer_encoder(lstm_out)
            if self.lstm_residual:
                transformer_out = self.post_encoder_ln(transformer_out + lstm_out)
            else:
                transformer_out = self.post_encoder_ln(transformer_out)
        else:
            # LSTM-only temporal encoder (no Transformer); skip residual add (would duplicate lstm_out).
            transformer_out = self.post_encoder_ln(lstm_out)
        if self.sequence_pooling == "attn":
            scores = self.attn_pool(transformer_out).squeeze(-1)
            w = torch.softmax(scores, dim=1).unsqueeze(-1)
            feat_pool = (transformer_out * w).sum(dim=1)
        else:
            feat_pool = transformer_out[:, -1, :]

        if self.decouple_heads:
            h_reg = self.fc_reg(feat_pool)
            h_cls = self.fc_cls(feat_pool)
            gate = self.gate(h_reg) if getattr(self, "gate", None) is not None else None
            return self.reg_head(h_reg), self.cls_branch(h_cls), gate
        feat = self.fc(feat_pool)
        gate = self.gate(feat) if getattr(self, "gate", None) is not None else None
        return self.reg_head(feat), self.cls_branch(feat), gate


class MultiTaskModel_HybridLSTMTabular(nn.Module):
    """
    Hybrid model to borrow tree-like tabular strength:
      - Sequence branch: LSTM (+ optional TransformerEncoder) on (B, L, F)
      - Tabular branch: MLP on last-step features (B, F)
      - Fusion: concat(seq_feat, tab_feat) -> per-head trunks -> reg/cls heads

    This usually improves robustness vs pure sequence encoders on short SEQ_LEN tabular windows.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        dropout_prob=0.3,
        lstm_hidden_dim=64,
        nhead=4,
        num_transformer_layers=1,
        seq_trunk_dim=32,
        tab_hidden_dim=96,
        tab_num_layers=2,
        fusion_dim=64,
        cls_hidden_dim=32,
        lstm_bidirectional=False,
        sequence_pooling="last",
        use_input_layernorm=True,
        lstm_residual=True,
        input_dropout_p=0.0,
        cls_num_hidden_layers=1,
        decouple_heads=True,
        cls_fusion_dim=None,
        hurdle_target_idx=None,
    ):
        super().__init__()
        if sequence_pooling not in ("last", "attn"):
            raise ValueError('sequence_pooling must be "last" or "attn"')

        self.decouple_heads = bool(decouple_heads)
        self.sequence_pooling = sequence_pooling
        self.lstm_residual = bool(lstm_residual)
        self.input_ln = nn.LayerNorm(input_dim) if use_input_layernorm else None
        p_drop = float(input_dropout_p)
        self.input_drop = nn.Dropout(p_drop) if p_drop > 0.0 else None

        num_directions = 2 if lstm_bidirectional else 1
        lstm_out_dim = int(lstm_hidden_dim) * num_directions
        self.lstm = nn.LSTM(
            input_dim, int(lstm_hidden_dim), batch_first=True, bidirectional=lstm_bidirectional
        )
        self.pos_encoder = nn.Sequential(nn.LayerNorm(lstm_out_dim), nn.Dropout(dropout_prob))

        self.use_transformer = int(num_transformer_layers) > 0
        if self.use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=lstm_out_dim,
                nhead=int(nhead),
                dropout=float(dropout_prob),
                batch_first=True,
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=int(num_transformer_layers)
            )
        else:
            self.transformer_encoder = None
        self.post_encoder_ln = nn.LayerNorm(lstm_out_dim)

        if sequence_pooling == "attn":
            self.attn_pool = nn.Linear(lstm_out_dim, 1)
        else:
            self.attn_pool = None

        # Seq trunk: map pooled seq feature -> seq_trunk_dim
        self.seq_trunk = nn.Sequential(
            nn.LayerNorm(lstm_out_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(lstm_out_dim, int(seq_trunk_dim)),
            nn.LayerNorm(int(seq_trunk_dim)),
            nn.GELU(),
        )

        # Tab trunk on last-step features (tree-like interactions via MLP depth)
        tab_layers = []
        d = int(input_dim)
        tab_hidden_dim = int(tab_hidden_dim)
        for _ in range(max(1, int(tab_num_layers))):
            tab_layers += [
                nn.Linear(d, tab_hidden_dim),
                nn.LayerNorm(tab_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_prob),
            ]
            d = tab_hidden_dim
        self.tab_trunk = nn.Sequential(*tab_layers)

        # Fusion
        fusion_in = int(seq_trunk_dim) + tab_hidden_dim
        cls_fd = int(cls_fusion_dim) if cls_fusion_dim is not None else int(fusion_dim)

        if self.decouple_heads:
            self.fuse_reg = _make_shared_trunk(fusion_in, int(fusion_dim), dropout_prob)
            self.fuse_cls = _make_shared_trunk(fusion_in, cls_fd, dropout_prob)
            self.reg_head = RegressionHead(int(fusion_dim), output_dim)
            self.cls_branch = _build_cls_mlp(cls_fd, int(cls_hidden_dim), dropout_prob, cls_num_hidden_layers)
        else:
            self.fuse = _make_shared_trunk(fusion_in, int(fusion_dim), dropout_prob)
            self.reg_head = RegressionHead(int(fusion_dim), output_dim)
            self.cls_branch = _build_cls_mlp(int(fusion_dim), int(cls_hidden_dim), dropout_prob, cls_num_hidden_layers)

        # Occurrence gate (hurdle model) on the regression fusion feature; None when no hurdle targets.
        self.hurdle_target_idx = [int(i) for i in (hurdle_target_idx or [])]
        self.gate = HurdleGate(int(fusion_dim), self.hurdle_target_idx, output_dim) if self.hurdle_target_idx else None

    def forward(self, x):
        # x: (B, L, F) or (B, F)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        if self.input_ln is not None:
            x = self.input_ln(x)
        if self.input_drop is not None:
            x = self.input_drop(x)

        last_step = x[:, -1, :]  # (B, F)
        tab_feat = self.tab_trunk(last_step)

        lstm_out, _ = self.lstm(x)
        lstm_out = self.pos_encoder(lstm_out)
        if self.use_transformer:
            enc = self.transformer_encoder(lstm_out)
        else:
            enc = lstm_out
        if self.lstm_residual:
            enc = self.post_encoder_ln(enc + lstm_out)
        else:
            enc = self.post_encoder_ln(enc)

        if self.sequence_pooling == "attn":
            scores = self.attn_pool(enc).squeeze(-1)
            w = torch.softmax(scores, dim=1).unsqueeze(-1)
            pooled = (enc * w).sum(dim=1)
        else:
            pooled = enc[:, -1, :]
        seq_feat = self.seq_trunk(pooled)

        fused_in = torch.cat([seq_feat, tab_feat], dim=1)
        if self.decouple_heads:
            h_reg = self.fuse_reg(fused_in)
            h_cls = self.fuse_cls(fused_in)
            gate = self.gate(h_reg) if getattr(self, "gate", None) is not None else None
            return self.reg_head(h_reg), self.cls_branch(h_cls), gate
        h = self.fuse(fused_in)
        gate = self.gate(h) if getattr(self, "gate", None) is not None else None
        return self.reg_head(h), self.cls_branch(h), gate

    
def optuna_objective_MTM(trial,
                     Xtrain_tensor, yregtrain_tensor, yclstrain_tensor, n_repeats=1,
                     device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                     LSTMTransformer=True,
                     max_epochs=1000, patiences=25, max_norm=1):

    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-7, 1e-3, log=True)
    dropout_prob = trial.suggest_float('dropout_prob', 0.1, 0.5)
    # Allow disabling feature selection by passing feature_K=None
    feature_K = feature_K if feature_K is not None else None
    batch_size = trial.suggest_categorical('batch_size', [32,48, 64,80,96,112,128])

    output_dim = yregtrain_tensor.shape[1]

    train_losses = []
    for _ in range(n_repeats):
        train_loader, selector, input_dim_per_step, sequence_len = prepare_data_loaders(
            Xtrain_tensor, yregtrain_tensor, yclstrain_tensor,
            feature_K, batch_size, device)

        model, train_loss, _ = train_loop_MTM(
            train_loader, input_dim_per_step, sequence_len, output_dim,
            lr, weight_decay, dropout_prob,
            max_epochs, patiences, max_norm, device, LSTMTransformer,
            val_loader=None, early_stop_on="train_loss",
            reg_loss_type="mse")
        train_losses.append(train_loss)

    return sum(train_losses) / len(train_losses)


def threshold_chosen(model, selector, X_tensor, ycls_tensor, device, metric="min_tss_hss",
                       print_prob_summary=True, t_min=0.0, t_max=1.0, cls_temperature=1.0,
                       skill_max_fpr=None):
    X_np = X_tensor.cpu().numpy()
    X_sel = _select_transform_features(X_np, selector)
    X_sel_tensor = torch.tensor(X_sel, dtype=torch.float32).to(device)
    ycls_tensor = ycls_tensor.to(device)
    T = float(max(cls_temperature, 1e-6))

    model.eval()
    with torch.no_grad():
        _, logits, _ = model(X_sel_tensor)
        ycls_prob = torch.sigmoid(logits.squeeze(-1) / T).cpu().numpy()
        ycls_prob = np.nan_to_num(ycls_prob, nan=0.5, posinf=1.0, neginf=0.0)
        ycls_true = ycls_tensor.cpu().numpy().ravel()

    if print_prob_summary:
        pos = ycls_prob[ycls_true >= 0.5]
        neg = ycls_prob[ycls_true < 0.5]
        if len(pos) and len(neg):
            print(
                "threshold_pool prob quantiles | pos: p10,p50,p90=",
                np.round(np.quantile(pos, [0.1, 0.5, 0.9]), 4),
                "| neg: p50,p90,p99=",
                np.round(np.quantile(neg, [0.5, 0.9, 0.99]), 4),
                "| (high train scores → high optimal threshold is normal)",
            )

    if skill_max_fpr is not None and float(skill_max_fpr) > 0:
        best_threshold, _, cap_ok = best_threshold_for_skill_under_max_fpr(
            ycls_true,
            ycls_prob,
            metric=metric,
            max_fpr=float(skill_max_fpr),
            t_min=t_min,
            t_max=t_max,
        )
        tpr_v, fpr_v = _tpr_fpr_at_threshold(ycls_true, ycls_prob, best_threshold)
        print(
            "best_threshold=", best_threshold,
            "metric=", metric,
            "val_FPR_cap=", float(skill_max_fpr),
            "at_t: val_FPR=", round(fpr_v, 5), "val_TPR=", round(tpr_v, 5),
            "t_min=", t_min, "t_max=", t_max,
        )
        if not cap_ok:
            print(
                "  note: no grid threshold met val FPR cap — fell back to unconstrained skill optimum; "
                "raise cap or widen [t_min,t_max]."
            )
    else:
        best_threshold, _ = best_threshold_for_skill(
            ycls_true, ycls_prob, metric=metric, t_min=t_min, t_max=t_max
        )
        print("best_threshold=", best_threshold, "metric=", metric, "t_min=", t_min, "t_max=", t_max)
    eps = 1e-3
    if float(t_max) < 1.0 - eps and best_threshold >= float(t_max) - eps:
        print(
            "  note: threshold is on upper search bound — raise threshold_t_max (e.g. 0.97–0.99) "
            "if you want the grid to explore higher t; re-check val/test CM after."
        )
    if float(t_min) > eps and best_threshold <= float(t_min) + eps:
        print(
            "  note: threshold is on lower search bound — lower threshold_t_min if metric wants a lower t."
        )
    return best_threshold


def threshold_chosen_max_fpr(
    model, selector, X_tensor, ycls_tensor, device,
    max_fpr=0.05, t_min=0.0, t_max=1.0, print_prob_summary=True, cls_temperature=1.0,
):
    """Same pool as threshold_chosen; pick t by FPR cap (see threshold_for_max_fpr)."""
    X_np = X_tensor.cpu().numpy()
    X_sel = _select_transform_features(X_np, selector)
    X_sel_tensor = torch.tensor(X_sel, dtype=torch.float32).to(device)
    ycls_tensor = ycls_tensor.to(device)
    T = float(max(cls_temperature, 1e-6))

    model.eval()
    with torch.no_grad():
        _, logits, _ = model(X_sel_tensor)
        ycls_prob = torch.sigmoid(logits.squeeze(-1) / T).cpu().numpy()
        ycls_prob = np.nan_to_num(ycls_prob, nan=0.5, posinf=1.0, neginf=0.0)
        ycls_true = ycls_tensor.cpu().numpy().ravel()

    if print_prob_summary:
        pos = ycls_prob[ycls_true >= 0.5]
        neg = ycls_prob[ycls_true < 0.5]
        if len(pos) and len(neg):
            print(
                "threshold_pool prob quantiles | pos: p10,p50,p90=",
                np.round(np.quantile(pos, [0.1, 0.5, 0.9]), 4),
                "| neg: p50,p90,p99=",
                np.round(np.quantile(neg, [0.5, 0.9, 0.99]), 4),
                "| (FPR cap applied below)",
            )

    best_t, fpr_v = threshold_for_max_fpr(
        ycls_true, ycls_prob, max_fpr=max_fpr, t_min=t_min, t_max=t_max
    )
    print(
        "best_threshold=", best_t,
        "rule= max_fpr<=%s" % max_fpr,
        "val_FPR=", round(fpr_v, 5),
        "t_min=", t_min, "t_max=", t_max,
    )
    return best_t


def threshold_chosen_max_fpr_min_tpr(
    model,
    selector,
    X_tensor,
    ycls_tensor,
    device,
    max_fpr=0.05,
    min_tpr=0.0,
    t_min=0.0,
    t_max=1.0,
    print_prob_summary=True,
    cls_temperature=1.0,
):
    """Pick highest t with val FPR<=max_fpr and TPR>=min_tpr (see threshold_for_max_fpr_min_tpr)."""
    X_np = X_tensor.cpu().numpy()
    X_sel = _select_transform_features(X_np, selector)
    X_sel_tensor = torch.tensor(X_sel, dtype=torch.float32).to(device)
    ycls_tensor = ycls_tensor.to(device)
    T = float(max(cls_temperature, 1e-6))

    model.eval()
    with torch.no_grad():
        _, logits, _ = model(X_sel_tensor)
        ycls_prob = torch.sigmoid(logits.squeeze(-1) / T).cpu().numpy()
        ycls_prob = np.nan_to_num(ycls_prob, nan=0.5, posinf=1.0, neginf=0.0)
        ycls_true = ycls_tensor.cpu().numpy().ravel()

    if print_prob_summary:
        pos = ycls_prob[ycls_true >= 0.5]
        neg = ycls_prob[ycls_true < 0.5]
        if len(pos) and len(neg):
            print(
                "threshold_pool prob quantiles | pos: p10,p50,p90=",
                np.round(np.quantile(pos, [0.1, 0.5, 0.9]), 4),
                "| neg: p50,p90,p99=",
                np.round(np.quantile(neg, [0.5, 0.9, 0.99]), 4),
                "| (FPR cap + min TPR below)",
            )

    best_t, fpr_v, tpr_v = threshold_for_max_fpr_min_tpr(
        ycls_true, ycls_prob, max_fpr=max_fpr, min_tpr=min_tpr, t_min=t_min, t_max=t_max
    )
    print(
        "best_threshold=", best_t,
        "rule= max_fpr<=%s" % max_fpr,
        "min_tpr>=%s" % min_tpr,
        "val_FPR=", round(fpr_v, 5),
        "val_TPR=", round(tpr_v, 5),
        "t_min=", t_min,
        "t_max=", t_max,
    )
    return best_t


def threshold_chosen_GM(ycls_proba, ycls_true, metric="min_tss_hss", t_min=0.0, t_max=1.0):
    best_threshold, _ = best_threshold_for_skill(
        np.asarray(ycls_true).ravel(), np.asarray(ycls_proba).ravel(),
        metric=metric, t_min=t_min, t_max=t_max,
    )
    print("best_threshold=", best_threshold, "metric=", metric, "t_min=", t_min, "t_max=", t_max)
    return best_threshold

def _inverse_standardize_yreg(yreg_arr, yreg_mean, yreg_std):
    """Map standardized regression back to original units for metrics."""
    if yreg_mean is None or yreg_std is None:
        return yreg_arr
    m = yreg_mean.detach().cpu().numpy()
    s = yreg_std.detach().cpu().numpy()
    if m.ndim == 2 and m.shape[0] == 1:
        m = m.reshape(1, -1)
        s = s.reshape(1, -1)
    return yreg_arr * s + m


def rolling_online_reg_bias_correction(
    yreg_pred,
    yreg_true,
    obs_time,
    verify_time,
    window_days=30.0,
    min_samples=3,
):
    """
    Horizon-aware, leakage-free *online* bias correction for the regression heads.

    A static (training-only) de-bias cannot remove a per-task mean shift that appears **after**
    the training window (deployment distribution shift). Operationally, however, the truth for
    each forecast window becomes known once that window closes, so we may correct future
    forecasts from already-verified past ones without any future leakage.

    For each forecast ``i`` issued at ``obs_time[i]`` (= ``window_begin``), the per-task additive
    correction is the mean residual ``(true - pred)`` over verified past forecasts ``j`` whose
    verification time ``verify_time[j]`` (= ``window_end``) satisfies::

        obs_time[i] - window_days  <  verify_time[j]  <=  obs_time[i]

    i.e. only windows that fully closed *before* forecast ``i`` was issued (causal → no leakage)
    and lie within the trailing ``window_days``. Rows with fewer than ``min_samples`` verified
    neighbours (e.g. the start of the deployment period) or with missing times/truth are left
    uncorrected.

    Args:
        yreg_pred, yreg_true: (n, k) arrays in ORIGINAL target units (after de-standardizing).
        obs_time:    per-row forecast issue time (window_begin), length n.
        verify_time: per-row verification time (window_end), length n.
        window_days: trailing window length (days) for the residual mean.
        min_samples: minimum verified residuals required before a correction is applied.

    Returns:
        (corrected_pred, correction) — both (n, k) float64 arrays.
    """
    import pandas as pd

    pred = np.asarray(yreg_pred, dtype=np.float64)
    single_col = pred.ndim == 1
    if single_col:
        pred = pred.reshape(-1, 1)
    true = np.asarray(yreg_true, dtype=np.float64)
    if true.ndim == 1:
        true = true.reshape(-1, 1)
    n, k = pred.shape
    corr = np.zeros((n, k), dtype=np.float64)
    if obs_time is None or verify_time is None or n == 0 or k == 0:
        return (pred.ravel() if single_col else pred), corr

    t_obs = pd.to_datetime(pd.Series(list(obs_time)), utc=True, errors="coerce").to_numpy(dtype="datetime64[ns]")
    t_ver = pd.to_datetime(pd.Series(list(verify_time)), utc=True, errors="coerce").to_numpy(dtype="datetime64[ns]")
    if t_obs.shape[0] != n or t_ver.shape[0] != n:
        return (pred.ravel() if single_col else pred), corr
    win = np.timedelta64(int(round(float(window_days) * 24 * 3600)), "s")

    resid = true - pred  # (n, k); NaN where truth missing

    # Verified residual pool: rows with a known verification time, sorted by it for a causal scan.
    pool_idx = np.where(~np.isnat(t_ver))[0]
    if pool_idx.size == 0:
        return (pred.ravel() if single_col else pred), corr
    order = pool_idx[np.argsort(t_ver[pool_idx], kind="mergesort")]
    tv_sorted = t_ver[order]
    resid_sorted = resid[order, :]
    finite_sorted = np.isfinite(resid_sorted)

    # Per-task prefix sums over the time-sorted pool → O(n log n) trailing-window means.
    csum = np.vstack([np.zeros(k), np.cumsum(np.where(finite_sorted, resid_sorted, 0.0), axis=0)])
    ccnt = np.vstack([np.zeros(k), np.cumsum(finite_sorted.astype(np.float64), axis=0)])

    for i in range(n):
        ti = t_obs[i]
        if np.isnat(ti):
            continue
        hi = int(np.searchsorted(tv_sorted, ti, side="right"))          # tv <= ti
        lo = int(np.searchsorted(tv_sorted, ti - win, side="right"))    # tv <= ti - win (excluded)
        if hi <= lo:
            continue
        s = csum[hi] - csum[lo]
        c = ccnt[hi] - ccnt[lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            corr[i, :] = np.where(c >= float(min_samples), s / np.maximum(c, 1.0), 0.0)

    out = pred + corr
    return (out.ravel() if single_col else out), corr


def evaluate_model(
    model,
    selector,
    X_tensor,
    yreg_tensor,
    ycls_tensor,
    device,
    threshold=0.5,
    resultsOutput=False,
    reg_var_names=None,
    yreg_mean=None,
    yreg_std=None,
    cls_temperature=None,
    reg_online_debias=False,
    reg_obs_time=None,
    reg_verify_time=None,
    reg_online_window_days=30.0,
    reg_online_min_samples=3,
    hurdle_missing_fill=None,
    *,
    verbose: bool = True,
):
    if yreg_mean is None:
        yreg_mean = getattr(model, "_mtm_yreg_mean", None)
    if yreg_std is None:
        yreg_std = getattr(model, "_mtm_yreg_std", None)
    if cls_temperature is None:
        cls_temperature = float(getattr(model, "_cls_temperature", 1.0))
    if hurdle_missing_fill is None:
        hurdle_missing_fill = getattr(model, "_mtm_hurdle_missing_fill", "conditional_mean")
    T = float(max(cls_temperature, 1e-6))

    X_np = X_tensor.cpu().numpy()
    X_sel = _select_transform_features(X_np, selector)
    X_sel_tensor = torch.tensor(X_sel, dtype=torch.float32).to(device)

    yreg_tensor = yreg_tensor.to(device)
    ycls_tensor = ycls_tensor.to(device)

    model.eval()
    with torch.no_grad():
        out = model(X_sel_tensor)
        yreg_pred, logits = out[0], out[1]
        gate_logits = out[2] if len(out) > 2 else None
        ycls_prob = torch.sigmoid(logits.squeeze(-1) / T).cpu().numpy()
        ycls_prob = np.nan_to_num(ycls_prob, nan=0.5, posinf=1.0, neginf=0.0)
        yreg_pred = yreg_pred.cpu().numpy()
        gate_prob = torch.sigmoid(gate_logits).cpu().numpy() if gate_logits is not None else None
        yreg_true = yreg_tensor.cpu().numpy()
        ycls_pred = (ycls_prob > threshold).astype(int)
        ycls_true = ycls_tensor.cpu().numpy()

    # Targets are always in original units from the caller; preds are in train-scale if standardized.
    yreg_pred = _inverse_standardize_yreg(yreg_pred, yreg_mean, yreg_std)
    yreg_pred = np.nan_to_num(np.asarray(yreg_pred, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    # Hurdle (two-part) combination for zero-inflated targets.
    # Default blends toward the training conditional mean when occurrence is unlikely, so missing
    # CME speed does not collapse the prediction to ~0 (use hurdle_missing_fill='zero' for legacy).
    hidx = _model_hurdle_idx(model)
    if gate_prob is not None and hidx:
        fill_mean = getattr(model, "_mtm_hurdle_fill_mean", None)
        if fill_mean is None:
            # Legacy checkpoints: warm-start location on RegressionHead is the conditional mean.
            init_mu = getattr(getattr(model, "reg_head", None), "init_task_mean", None)
            if init_mu is not None:
                fill_mean = init_mu.detach().cpu().numpy()
        elif hasattr(fill_mean, "detach"):
            fill_mean = fill_mean.detach().cpu().numpy()
        yreg_pred = _mtm_hurdle_combine_predictions(
            yreg_pred, gate_prob, hidx, fill_mean, hurdle_missing_fill=hurdle_missing_fill,
        )

    # Optional horizon-aware online de-bias: correct deployment (post-training) mean shift from
    # already-verified past windows only. Requires per-row obs/verify times (leakage-free).
    if reg_online_debias and reg_obs_time is not None and reg_verify_time is not None and yreg_pred.size:
        yreg_pred, _ = rolling_online_reg_bias_correction(
            yreg_pred, yreg_true, reg_obs_time, reg_verify_time,
            window_days=reg_online_window_days, min_samples=reg_online_min_samples,
        )
        yreg_pred = np.nan_to_num(np.asarray(yreg_pred, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    if resultsOutput:
        return dict(yreg_pred=yreg_pred, ycls_pred=ycls_pred, ycls_prob=ycls_prob,
                    yreg_true=yreg_true, ycls_true=ycls_true)
    else:
        return evaluate(
            yreg_pred,
            ycls_pred,
            ycls_prob,
            yreg_true,
            ycls_true,
            reg_var_names=reg_var_names,
            verbose=verbose,
        )


def predictions_to_dataframe(
    raw,
    row_index,
    reg_var_names=None,
    *,
    model_prefix="mtm",
    threshold_used=None,
):
    """Export dict from evaluate_model(..., resultsOutput=True) to a DataFrame."""
    import pandas as pd

    ycls_prob = np.asarray(raw["ycls_prob"], dtype=np.float64).ravel()
    ycls_pred = np.asarray(raw["ycls_pred"]).ravel()
    yreg_pred = np.asarray(raw["yreg_pred"], dtype=np.float64)
    ycls_true = np.asarray(raw["ycls_true"], dtype=np.float64).ravel()
    yreg_true = np.asarray(raw["yreg_true"], dtype=np.float64)

    n = int(len(ycls_prob))
    idx = list(row_index)[:n]
    if len(idx) < n:
        idx = idx + [None] * (n - len(idx))

    if yreg_pred.ndim == 1:
        yreg_pred = yreg_pred.reshape(-1, 1)
    if yreg_true.ndim == 1:
        yreg_true = yreg_true.reshape(-1, 1)

    n_reg = yreg_pred.shape[1]
    if reg_var_names is None:
        reg_var_names = [f"reg_{j}" for j in range(n_reg)]

    def _safe(name):
        return str(name).replace(" ", "_").replace("/", "_")

    data = {
        f"{model_prefix}_cls_prob": ycls_prob,
        f"{model_prefix}_cls_pred": ycls_pred.astype(int),
        f"{model_prefix}_cls_true": ycls_true,
    }
    if threshold_used is not None:
        data[f"{model_prefix}_threshold_used"] = np.full(n, float(threshold_used), dtype=np.float64)
    for j, name in enumerate(reg_var_names):
        if j >= n_reg:
            break
        stem = _safe(name)
        data[f"{model_prefix}_pred_{stem}"] = yreg_pred[:, j]
        data[f"{model_prefix}_true_{stem}"] = yreg_true[:, j]

    return pd.DataFrame(data, index=pd.Index(idx, name="row_index"))


def tabular_classifier_probs_to_dataframe(index, proba, model_prefix, threshold):
    """Sklearn predict_proba[:,1] aligned to index."""
    import pandas as pd

    proba = np.asarray(proba, dtype=np.float64).ravel()
    ix = list(index)[: len(proba)]
    return pd.DataFrame(
        {
            f"{model_prefix}_cls_prob": proba,
            f"{model_prefix}_cls_pred": (proba >= float(threshold)).astype(int),
            f"{model_prefix}_threshold_used": np.full(len(proba), float(threshold), dtype=np.float64),
        },
        index=pd.Index(ix, name="row_index"),
    )


def attach_filtered_columns(df, filtered_df, row_index, columns):
    """Copy metadata columns (e.g. window_begin) onto a prediction frame."""
    import pandas as pd

    ix = pd.Index(list(row_index)[: len(df)])
    out = df.copy()
    for c in columns:
        if c in filtered_df.columns:
            out[c] = filtered_df.loc[ix, c].values
    return out


def _val_skill_scalar_from_eval_dict(res, metric):
    """Scalar for ranking repeats from evaluate_model-style dict (tss, hss, etc.)."""
    t = float(res.get("tss", np.nan))
    h = float(res.get("hss", np.nan))
    if metric == "min_tss_hss":
        return min(t, h) if np.isfinite(t) and np.isfinite(h) else np.nan
    if metric == "hmean_tss_hss":
        if np.isfinite(t) and np.isfinite(h) and t > 0 and h > 0:
            return 2.0 * t * h / (t + h)
        return min(t, h) if np.isfinite(t) and np.isfinite(h) else np.nan
    if metric == "tss_plus_hss":
        return t + h if np.isfinite(t) and np.isfinite(h) else np.nan
    if metric == "tss":
        return t if np.isfinite(t) else np.nan
    if metric == "hss":
        return h if np.isfinite(h) else np.nan
    return min(t, h) if np.isfinite(t) and np.isfinite(h) else np.nan


def _val_reg_min_pred_std_from_eval_dict(res) -> float:
    """Minimum across regression dims of nanstd(yreg_pred) on val (original units after evaluate)."""
    yp = np.asarray(res.get("yreg_pred"), dtype=np.float64)
    if yp.size == 0:
        return float("nan")
    if yp.ndim == 1:
        yp = yp.reshape(-1, 1)
    stds = [float(np.nanstd(yp[:, j])) for j in range(yp.shape[1])]
    return float(np.nanmin(stds)) if stds else float("nan")


def _val_min_r2_per_var_from_eval_dict(res) -> float:
    r2pv = res.get("r2_per_var") or []
    if not r2pv:
        return float("nan")
    return float(np.nanmin(np.asarray(r2pv, dtype=np.float64)))


def pick_median_repeat_index_from_val_results(
    results_val_list,
    metric="min_tss_hss",
    *,
    min_val_reg_pred_std=None,
    min_val_min_r2_per_var=None,
):
    """
    Index of the repeat whose validation scalar (e.g. min(TSS,HSS)) is the median
    among repeats. Not the median of each metric across repeats — one full repeat.

    If ``min_val_reg_pred_std`` is set, only repeats whose validation regression
    predictions have ``min_j std(yreg_pred[:, j]) >= min_val_reg_pred_std`` are
    eligible. If none qualify, falls back to all repeats with finite skill.

    If ``min_val_min_r2_per_var`` is set, repeats must also satisfy
    ``min(r2_per_var) >= min_val_min_r2_per_var``.
    """
    scores = np.array(
        [_val_skill_scalar_from_eval_dict(r, metric) for r in results_val_list],
        dtype=float,
    )
    n = len(results_val_list)
    base_ok = np.isfinite(scores)
    if not np.any(base_ok):
        return n - 1, {
            "repeat_selection_used_fallback": False,
            "repeat_selection_n_eligible": 0,
            "repeat_selection_n_total": int(n),
        }

    gate = np.ones(n, dtype=bool)
    if min_val_reg_pred_std is not None:
        thr = float(min_val_reg_pred_std)
        for i, r in enumerate(results_val_list):
            gate[i] = gate[i] and (_val_reg_min_pred_std_from_eval_dict(r) >= thr)
    if min_val_min_r2_per_var is not None:
        thr_r2 = float(min_val_min_r2_per_var)
        for i, r in enumerate(results_val_list):
            mnr = _val_min_r2_per_var_from_eval_dict(r)
            gate[i] = gate[i] and np.isfinite(mnr) and (mnr >= thr_r2)

    eligible = np.where(base_ok & gate)[0]
    used_fallback = False
    if len(eligible) == 0:
        eligible = np.where(base_ok)[0]
        used_fallback = True

    order = eligible[np.argsort(scores[eligible])]
    sel = int(order[len(order) // 2])
    return sel, {
        "repeat_selection_used_fallback": used_fallback,
        "repeat_selection_n_eligible": int(len(eligible)),
        "repeat_selection_n_total": int(n),
    }


def train_evaluate_MTM(
        Xtrain_tensor, yregtrain_tensor, yclstrain_tensor,
        Xval_tensor, yregval_tensor, yclsval_tensor,
        Xtest_tensor, yregtest_tensor, yclstest_tensor,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        LSTMTransformer=True, threshold_select=True,
        Hybrid=False,
        feature_K=10, dropout_prob=0.3, n_repeats=10,
        batch_size=64, lr=1e-4, weight_decay=1e-5, patiences=25, max_epochs=2000, max_norm=1, resultsOutput=False,
        track_cost=False, reg_var_names=None,
        repeat_selection_metric=None,
        early_stop_on="val_skill",
        early_stop_skill_metric="min_tss_hss",
        threshold_metric="min_tss_hss",
        threshold_t_min=0.0,
        threshold_t_max=1.0,
        threshold_max_fpr=None,
        threshold_min_tpr=None,
        threshold_skill_max_fpr=None,
        calibrate_cls_temperature=False,
        calibrate_cls_t_hi=3.0,
        calibrate_cls_min_prob_std=0.10,
        use_cls_pos_weight=True,
        pos_weight_multiplier=1.0,
        max_pos_weight=10.0,
        focal_scale=2.0,
        focal_gamma=2.0,
        focal_alpha_neg=0.25,
        focal_alpha_pos=0.75,
        reg_loss_scale=1.5,
        cls_loss_scale=1.35,
        train_sampler_mode="shuffle",
        lstm_hidden_dim=64,
        transformer_dim=16,
        nhead=4,
        num_transformer_layers=1,
        cls_hidden_dim=32,
        lstm_bidirectional=False,
        sequence_pooling="last",
        use_input_layernorm=True,
        lstm_residual=True,
        input_dropout_p=0.0,
        cls_num_hidden_layers=1,
        cls_label_smoothing=0.0,
        init_cls_bias_from_prevalence=True,
        decouple_heads=False,
        cls_transformer_dim=None,
        # Hybrid dims
        hybrid_seq_trunk_dim=32,
        hybrid_tab_hidden_dim=96,
        hybrid_tab_num_layers=2,
        hybrid_fusion_dim=64,
        hybrid_cls_fusion_dim=None,
        freeze_reg_epochs=0,
        reg_ramp_epochs=0,
        standardize_yreg=True,
        reg_loss_type="smooth_l1",
        smooth_l1_beta=1.0,
        X_threshold_tensor=None,
        yreg_threshold_tensor=None,
        ycls_threshold_tensor=None,
        # Distillation: teacher prob tensors aligned to sequence tensors
        ycls_teacher_prob_train_tensor=None,
        ycls_sample_weight_train_tensor=None,
        ycls_teacher_prob_val_tensor=None,
        ycls_teacher_prob_test_tensor=None,
        distill_weight=0.0,
        distill_temperature=2.0,
        log_head_losses=False,
        collect_test_raw_outputs=False,
        collect_val_raw_outputs=False,
        collect_train_raw_outputs=False,
        supervised_test_eval=True,
        eval_verbose=True,
        ycls_train_metrics_tensor=None,
        ycls_train_gsep_metrics_tensor=None,
        repeat_min_val_reg_pred_std=None,
        repeat_min_val_min_r2_per_var=None,
        reg_branch_lr_mult=1.0,
        reg_weight_decay_mult=1.0,
        reg_debias=True,
        hurdle_target_idx=None,
        hurdle_gate_loss_scale=1.0,
        hurdle_missing_fill="conditional_mean",
        reg_zero_loss_floor_cols=None,
        reg_zero_loss_floor_alpha=0.35,
        reg_col_loss_weights=None):
    """
    X_threshold_tensor / ycls_threshold_tensor: optional pool for probability threshold tuning.
    If provided, threshold is chosen on this set (e.g. full pre-2020 train) while Xval is only
    for early stopping metrics. If None, threshold uses the validation set (Xval_tensor).
    yreg_threshold_tensor is unused for thresholding but kept for API symmetry; may be None.

    **Optional label protocol (e.g. GSEP train / OSEP val):** callers may pass different binary
    labels for ``yclstrain_tensor`` (fit) vs ``yclsval_tensor`` / ``yclstest_tensor`` /
    ``ycls_threshold_tensor`` (operational). Training uses the fit labels for cls loss, sampler,
    ``SelectKBest``, and ``pos_weight``; validation metrics use the val labels. Do **not** use
    ``early_stop_on="val_multitask_loss"`` when train and val cls labels disagree, because the
    joint val cls loss would be misaligned with the training objective.

    early_stop_on:
      'val_auc' — checkpoint when validation ROC-AUC is best (good vs tree boosting on AUC).
      'val_skill' — checkpoint when val min(TSS,HSS) (or early_stop_skill_metric) at scanned thresholds is best.
      'val_multitask_loss' — checkpoint when weighted val (reg + cls) loss is lowest; aligns with joint MTM training.
      'train_loss' — minimize train loss (no val loader required).
    threshold_metric applies only to threshold_chosen on X_threshold_tensor / Xval.
    threshold_t_min / threshold_t_max: search band for threshold (see best_threshold_for_skill).
            If balanced_acc picks t~0.5 on val, set threshold_t_min≈0.65–0.75 to avoid huge FP on test.
    threshold_max_fpr: if set (e.g. 0.02), ignore threshold_metric and choose the smallest t such that
            val FPR <= threshold_max_fpr (fewer false alarms; test FPR may still differ).
    threshold_min_tpr: if set with threshold_max_fpr, choose the **largest** t with FPR<=cap **and**
            TPR>=threshold_min_tpr on the threshold pool (protects recall on val while tightening alarms).
            If infeasible, falls back to FPR-only rule (see threshold_for_max_fpr_min_tpr).
    threshold_skill_max_fpr: if set (and threshold_max_fpr is None), still optimize ``threshold_metric``
            on the threshold pool, but **only** among thresholds with val FPR <= this cap. Reduces
            unconstrained min(TSS,HSS) picking a very low t that explodes false alarms on a shifted test era.
    calibrate_cls_temperature: if True, fit scalar T on val (NLL grid on logits) and use p=sigmoid(z/T) for
            threshold search and evaluate_model. If std(p) is too low or T hits cap with low spread, falls back to T=1.
    calibrate_cls_t_hi: upper bound for T search (default 3; avoid 4+ which often over-flattens probs).
    calibrate_cls_min_prob_std: reject calibration and use T=1 if std(sigmoid(z/T)) below this after NLL-opt T.
    log_head_losses: if True, each epoch prints mean L_reg, L_bce, L_focal (raw and weighted) on the train loader.
    lstm_bidirectional: if True, LSTM is bidirectional; TransformerEncoder d_model becomes 2*lstm_hidden_dim (nhead must divide it).
    sequence_pooling: 'last' uses final timestep; 'attn' uses learned softmax weights over timesteps (better when signal is spread across lags).
    use_input_layernorm: LayerNorm on raw per-timestep features before LSTM (stabilizes scale across the 111 inputs).
    lstm_residual: add LSTM outputs back to Transformer outputs (post LayerNorm) before pooling.
    input_dropout_p: feature dropout on (B,L,F) inputs after input LayerNorm (training only); reduces co-adaptation.
    cls_num_hidden_layers: number of CLS MLP blocks (Linear+GELU+Dropout) before the logit; 2 often helps ranking.
    cls_label_smoothing: epsilon for BCE targets only (y*(1-eps)+0.5*eps); focal still uses hard labels.
    init_cls_bias_from_prevalence: set final logit bias to log(p/(1-p)) from **fit** y_cls (recommended).
    decouple_heads: if True (LSTM path), separate LN/Linear bottlenecks for regression vs classification so
            reg gradients do not squeeze the single shared fc used for SEP ranking.
    cls_transformer_dim: bottleneck width for the **cls** trunk when decoupled (default = transformer_dim).
    freeze_reg_epochs: for the first N epochs, regression loss weight is set to 0 (cls-only backbone warm-up).
    reg_ramp_epochs: after freeze, linearly scale reg loss from 0 to full over this many epochs (reduces cls shock).
    repeat_selection_metric: which val metric defines the median repeat (default: early_stop_skill_metric).
        Final returned model/selector are restored from that repeat, not the last repeat.
    collect_test_raw_outputs / collect_val_raw_outputs / collect_train_raw_outputs:
        store per-repeat raw preds in cost dict for CSV export.
    supervised_test_eval: if False (e.g. NRT rows with NaN future labels), skip ``evaluate()`` on the
        test set each repeat; still run ``evaluate_model(..., resultsOutput=True)`` so predictions are
        stored in ``cost['test_raw_by_repeat']`` (the list is created whenever this flag is False or
        ``collect_test_raw_outputs`` is True). ``results_test`` entries are minimal placeholders.
    eval_verbose: if False, suppress per-split prints inside ``evaluate`` during each repeat
        (useful when the caller prints a single train/val/test summary).
    ycls_train_metrics_tensor: optional cls labels aligned with ``Xtrain_tensor`` (e.g. OSEP on the
        train pool while ``yclstrain_tensor`` is GSEP). When set, each repeat appends a train
        ``evaluate`` dict to ``cost['results_train_metrics']`` (same threshold as val/test).
    ycls_train_gsep_metrics_tensor: optional second train-only label tensor (e.g. GSEP when the head
        was trained on GSEP). Appends per-repeat dicts to ``cost['results_train_gsep_metrics']``.
    repeat_min_val_reg_pred_std: if set (e.g. 0.05), median skill only among repeats with non-collapsed
        val regression preds (min_j std(yreg_pred[:,j]) >= threshold); see NRT util_multitask_model.
    repeat_min_val_min_r2_per_var: optional extra gate on min(r2_per_var) on val before median pick.
    reg_branch_lr_mult: if > 1, Adam uses higher lr on regression-only modules (reg_head, fc_reg, fuse_reg).
    reg_weight_decay_mult: weight-decay multiplier for the regression-only modules (reg_head, fc_reg,
        fuse_reg). Use < 1 (e.g. 0.1) to reduce shrinkage so regression predictions are not collapsed
        toward a near-constant mean (helps amplitude on targets like CME speed). With decoupled heads
        this affects regression only and leaves the classification branch unchanged.
    reg_debias: if True (default), after training fold the mean per-task TRAINING residual
        (y_true - y_pred) back into each regression head's per-task location term. Combined with
        the per-task location warm-start (set from the training mean) and excluding that bias
        from weight decay, this removes the systematic per-head mean shift that median-seeking
        losses (SmoothL1/L1) cause on skewed count targets. Uses training data only (no leakage).
    """

    output_dim = yregtrain_tensor.shape[1]
    thresholds, results_val, results_test = [], [], []
    results_train_metrics_list: list = []
    results_train_gsep_metrics_list: list = []
    cost = {"train_time_sec": 0.0, "eval_val_time_sec": 0.0, "eval_test_time_sec": 0.0, "gpu_peak_mb": None}
    if collect_test_raw_outputs or (not supervised_test_eval):
        cost["test_raw_by_repeat"] = []
    if collect_val_raw_outputs:
        cost["val_raw_by_repeat"] = []
    if collect_train_raw_outputs:
        cost["train_raw_by_repeat"] = []
    _repeat_snapshots = []
    if device.type == "cuda" and track_cost:
        torch.cuda.reset_peak_memory_stats(device)

    # Hurdle (zero-inflated) target columns: handled by a two-part model (occurrence gate +
    # conditional value). They are intentionally left UN-standardized so (a) occurrence
    # ``y > eps`` is trivially readable inside the train loop and (b) the gate combination in
    # evaluate_model sees the raw conditional value (inverse-standardize is identity for them).
    hurdle_idx_eff = [int(i) for i in (hurdle_target_idx or []) if 0 <= int(i) < output_dim]
    zero_floor_cols_eff = [
        int(i) for i in (reg_zero_loss_floor_cols or []) if 0 <= int(i) < output_dim
    ]

    yreg_mean = yreg_std = None
    if standardize_yreg and output_dim > 0:
        yreg_mean = yregtrain_tensor.mean(dim=0, keepdim=True)
        yreg_std = yregtrain_tensor.std(dim=0, unbiased=False, keepdim=True).clamp(min=1e-6)
        # Do not standardize hurdle columns (mean=0, std=1 → identity transform there).
        for j in hurdle_idx_eff:
            yreg_mean[0, j] = 0.0
            yreg_std[0, j] = 1.0
        yreg_train_u = (yregtrain_tensor - yreg_mean) / yreg_std
        yreg_val_u = (yregval_tensor - yreg_mean) / yreg_std
        yreg_test_u = (yregtest_tensor - yreg_mean) / yreg_std
    else:
        yreg_train_u = yregtrain_tensor
        yreg_val_u = yregval_tensor
        yreg_test_u = yregtest_tensor

    reg_col_std_tensor = None
    if output_dim > 0 and zero_floor_cols_eff:
        reg_col_std_tensor = yreg_train_u.float().std(dim=0, unbiased=False).clamp(min=1e-6)

    # Per-task location estimated on the TRAINING split only (no leakage). In standardized
    # space this is ~0; in raw space it is each task's mean. For hurdle columns we warm-start
    # with the CONDITIONAL (occurrence-row) mean, since the head models E[y | y>0].
    if output_dim > 0:
        reg_target_mean_tensor = yreg_train_u.mean(dim=0).detach().clone()
        for j in hurdle_idx_eff:
            pos = yregtrain_tensor[:, j] > 1e-6
            if bool(pos.any()):
                reg_target_mean_tensor[j] = yregtrain_tensor[pos, j].mean()
    else:
        reg_target_mean_tensor = None

    # Training-set means used to fill hurdle predictions when occurrence is unlikely (no cliff to 0).
    _fill_mode = "unconditional" if str(hurdle_missing_fill).lower() == "unconditional_mean" else "conditional"
    hurdle_fill_mean_tensor = (
        _mtm_hurdle_fill_means(yregtrain_tensor, hurdle_idx_eff, mode=_fill_mode)
        if hurdle_idx_eff else None
    )

    for _ in range(n_repeats):
        train_loader, selector, input_dim_per_step, sequence_len = prepare_data_loaders(
            Xtrain_tensor, yreg_train_u, yclstrain_tensor,
            feature_K, batch_size, device,
            sampler_mode=train_sampler_mode,
            ycls_teacher_prob_tensor=ycls_teacher_prob_train_tensor,
            ycls_sample_weight_tensor=ycls_sample_weight_train_tensor,
        )

        val_loader = prepare_val_loader(
            Xval_tensor, yreg_val_u, yclsval_tensor, selector, batch_size, device,
            ycls_teacher_prob_tensor=ycls_teacher_prob_val_tensor,
        )
        if use_cls_pos_weight:
            pos_w = cls_pos_weight_tensor(
                yclstrain_tensor, device, multiplier=pos_weight_multiplier,
                max_pos_weight=max_pos_weight,
            )
        else:
            pos_w = None

        val_for_train = val_loader if early_stop_on in (
            "val_skill",
            "val_auc",
            "val_multitask_loss",
        ) else None
        stop_mode = early_stop_on if val_for_train is not None else "train_loss"

        model, _, train_time = train_loop_MTM(
            train_loader, input_dim_per_step, sequence_len, output_dim,
            lr, weight_decay, dropout_prob=dropout_prob,
            max_epochs=max_epochs, patience=patiences,
            max_norm=max_norm, device=device, LSTMTransformer=LSTMTransformer,
            Hybrid=Hybrid,
            val_loader=val_for_train,
            early_stop_on=stop_mode,
            val_skill_metric=early_stop_skill_metric,
            val_skill_metric_t_min=threshold_t_min,
            val_skill_metric_t_max=threshold_t_max,
            cls_pos_weight=pos_w,
            focal_scale=focal_scale,
            focal_gamma=focal_gamma,
            reg_loss_scale=reg_loss_scale,
            cls_loss_scale=cls_loss_scale,
            focal_alpha_neg=focal_alpha_neg,
            focal_alpha_pos=focal_alpha_pos,
            reg_loss_type=reg_loss_type,
            smooth_l1_beta=smooth_l1_beta,
            lstm_hidden_dim=lstm_hidden_dim,
            transformer_dim=transformer_dim,
            nhead=nhead,
            num_transformer_layers=num_transformer_layers,
            cls_hidden_dim=cls_hidden_dim,
            lstm_bidirectional=lstm_bidirectional,
            sequence_pooling=sequence_pooling,
            use_input_layernorm=use_input_layernorm,
            lstm_residual=lstm_residual,
            input_dropout_p=input_dropout_p,
            cls_num_hidden_layers=cls_num_hidden_layers,
            cls_label_smoothing=cls_label_smoothing,
            ycls_fit_for_bias=yclstrain_tensor,
            init_cls_bias_from_prevalence=init_cls_bias_from_prevalence,
            decouple_heads=decouple_heads,
            cls_transformer_dim=cls_transformer_dim,
            hybrid_seq_trunk_dim=hybrid_seq_trunk_dim,
            hybrid_tab_hidden_dim=hybrid_tab_hidden_dim,
            hybrid_tab_num_layers=hybrid_tab_num_layers,
            hybrid_fusion_dim=hybrid_fusion_dim,
            hybrid_cls_fusion_dim=hybrid_cls_fusion_dim,
            distill_weight=distill_weight,
            distill_temperature=distill_temperature,
            freeze_reg_epochs=freeze_reg_epochs,
            reg_ramp_epochs=reg_ramp_epochs,
            log_head_losses=log_head_losses,
            reg_branch_lr_mult=float(reg_branch_lr_mult),
            reg_weight_decay_mult=float(reg_weight_decay_mult),
            reg_target_mean=reg_target_mean_tensor,
            reg_debias=reg_debias,
            hurdle_target_idx=hurdle_idx_eff,
            hurdle_gate_loss_scale=float(hurdle_gate_loss_scale),
            zero_loss_floor_cols=zero_floor_cols_eff,
            zero_loss_floor_alpha=float(reg_zero_loss_floor_alpha),
            reg_col_std=reg_col_std_tensor,
            reg_col_loss_weights=reg_col_loss_weights,
        )
        if track_cost:
            cost["train_time_sec"] += train_time

        T_cls = 1.0
        if calibrate_cls_temperature:
            X_cal = X_threshold_tensor if X_threshold_tensor is not None else Xval_tensor
            y_cal = ycls_threshold_tensor if ycls_threshold_tensor is not None else yclsval_tensor
            logits_np = collect_cls_logits_np(model, selector, X_cal, device)
            y_np = y_cal.detach().cpu().numpy().ravel()
            T_cls, nll_cal = fit_sigmoid_temperature(
                logits_np, y_np,
                t_hi=float(calibrate_cls_t_hi),
                min_prob_std=float(calibrate_cls_min_prob_std),
            )
            if np.isnan(nll_cal):
                T_cls = 1.0
        model._cls_temperature = float(T_cls)

        if threshold_select:
            if threshold_max_fpr is not None and float(threshold_max_fpr) > 0:
                use_min_tpr = (
                    threshold_min_tpr is not None and float(threshold_min_tpr) > 0
                )
                if X_threshold_tensor is not None and ycls_threshold_tensor is not None:
                    X_th, y_th = X_threshold_tensor, ycls_threshold_tensor
                else:
                    X_th, y_th = Xval_tensor, yclsval_tensor
                if use_min_tpr:
                    threshold = threshold_chosen_max_fpr_min_tpr(
                        model, selector, X_th, y_th, device,
                        max_fpr=float(threshold_max_fpr),
                        min_tpr=float(threshold_min_tpr),
                        t_min=threshold_t_min,
                        t_max=threshold_t_max,
                        cls_temperature=T_cls,
                    )
                else:
                    threshold = threshold_chosen_max_fpr(
                        model, selector, X_th, y_th, device,
                        max_fpr=float(threshold_max_fpr),
                        t_min=threshold_t_min,
                        t_max=threshold_t_max,
                        cls_temperature=T_cls,
                    )
            elif X_threshold_tensor is not None and ycls_threshold_tensor is not None:
                threshold = threshold_chosen(
                    model, selector, X_threshold_tensor, ycls_threshold_tensor, device,
                    metric=threshold_metric,
                    t_min=threshold_t_min,
                    t_max=threshold_t_max,
                    cls_temperature=T_cls,
                    skill_max_fpr=threshold_skill_max_fpr,
                )
            else:
                threshold = threshold_chosen(
                    model, selector, Xval_tensor, yclsval_tensor, device,
                    metric=threshold_metric,
                    t_min=threshold_t_min,
                    t_max=threshold_t_max,
                    cls_temperature=T_cls,
                    skill_max_fpr=threshold_skill_max_fpr,
                )
        else:
            threshold = 0.5

        thresholds.append(threshold)
        ev_kwargs = {}
        if yreg_mean is not None:
            ev_kwargs["yreg_mean"] = yreg_mean
            ev_kwargs["yreg_std"] = yreg_std
        ev_kwargs["cls_temperature"] = T_cls
        if track_cost:
            t0 = time.perf_counter()
        if collect_val_raw_outputs:
            res_val_raw = evaluate_model(
                model, selector, Xval_tensor, yregval_tensor, yclsval_tensor, device, threshold,
                True, reg_var_names=reg_var_names, verbose=eval_verbose, **ev_kwargs)
            cost["val_raw_by_repeat"].append(res_val_raw)
            results_val.append(
                evaluate(
                    res_val_raw["yreg_pred"],
                    res_val_raw["ycls_pred"],
                    res_val_raw["ycls_prob"],
                    res_val_raw["yreg_true"],
                    res_val_raw["ycls_true"],
                    reg_var_names=reg_var_names,
                    verbose=eval_verbose,
                )
            )
        else:
            res_val = evaluate_model(
                model, selector, Xval_tensor, yregval_tensor, yclsval_tensor, device, threshold,
                resultsOutput, reg_var_names=reg_var_names, verbose=eval_verbose, **ev_kwargs)
            results_val.append(res_val)
        if track_cost:
            cost["eval_val_time_sec"] += time.perf_counter() - t0

        if track_cost:
            t0 = time.perf_counter()
        if not supervised_test_eval:
            res_test_raw = evaluate_model(
                model, selector, Xtest_tensor, yregtest_tensor, yclstest_tensor, device, threshold,
                True, reg_var_names=reg_var_names, verbose=eval_verbose, **ev_kwargs)
            cost["test_raw_by_repeat"].append(res_test_raw)
            results_test.append(
                {
                    "supervised_test_eval": False,
                    "note": "test set has no usable labels (e.g. NRT); metrics undefined",
                }
            )
        elif collect_test_raw_outputs:
            res_test_raw = evaluate_model(
                model, selector, Xtest_tensor, yregtest_tensor, yclstest_tensor, device, threshold,
                True, reg_var_names=reg_var_names, verbose=eval_verbose, **ev_kwargs)
            cost["test_raw_by_repeat"].append(res_test_raw)
            results_test.append(
                evaluate(
                    res_test_raw["yreg_pred"],
                    res_test_raw["ycls_pred"],
                    res_test_raw["ycls_prob"],
                    res_test_raw["yreg_true"],
                    res_test_raw["ycls_true"],
                    reg_var_names=reg_var_names,
                    verbose=eval_verbose,
                )
            )
        else:
            res_test = evaluate_model(
                model, selector, Xtest_tensor, yregtest_tensor, yclstest_tensor, device, threshold,
                resultsOutput, reg_var_names=reg_var_names, verbose=eval_verbose, **ev_kwargs)
            results_test.append(res_test)
        if track_cost:
            cost["eval_test_time_sec"] += time.perf_counter() - t0

        if ycls_train_metrics_tensor is not None:
            res_train_m = evaluate_model(
                model,
                selector,
                Xtrain_tensor,
                yregtrain_tensor,
                ycls_train_metrics_tensor,
                device,
                threshold,
                resultsOutput,
                reg_var_names=reg_var_names,
                verbose=eval_verbose,
                **ev_kwargs,
            )
            results_train_metrics_list.append(res_train_m)

        if ycls_train_gsep_metrics_tensor is not None:
            res_train_gsep = evaluate_model(
                model,
                selector,
                Xtrain_tensor,
                yregtrain_tensor,
                ycls_train_gsep_metrics_tensor,
                device,
                threshold,
                resultsOutput,
                reg_var_names=reg_var_names,
                verbose=eval_verbose,
                **ev_kwargs,
            )
            results_train_gsep_metrics_list.append(res_train_gsep)

        if collect_train_raw_outputs:
            res_train_raw = evaluate_model(
                model, selector, Xtrain_tensor, yregtrain_tensor, yclstrain_tensor, device, threshold,
                True, reg_var_names=reg_var_names, verbose=eval_verbose, **ev_kwargs)
            cost["train_raw_by_repeat"].append(res_train_raw)

        _repeat_snapshots.append(
            {
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "selector": selector,
                "cls_temperature": float(getattr(model, "_cls_temperature", 1.0)),
            }
        )

    pick_metric = (
        repeat_selection_metric
        if repeat_selection_metric is not None
        else early_stop_skill_metric
    )
    sel_idx, _rpick = pick_median_repeat_index_from_val_results(
        results_val,
        metric=pick_metric,
        min_val_reg_pred_std=repeat_min_val_reg_pred_std,
        min_val_min_r2_per_var=repeat_min_val_min_r2_per_var,
    )
    cost["repeat_selection_used_fallback"] = bool(_rpick["repeat_selection_used_fallback"])
    cost["repeat_selection_n_eligible"] = int(_rpick["repeat_selection_n_eligible"])
    cost["repeat_selection_n_total"] = int(_rpick["repeat_selection_n_total"])
    cost["repeat_min_val_reg_pred_std"] = repeat_min_val_reg_pred_std
    cost["repeat_min_val_min_r2_per_var"] = repeat_min_val_min_r2_per_var
    _snap = _repeat_snapshots[sel_idx]
    model.load_state_dict({k: v.to(device) for k, v in _snap["state"].items()})
    selector = _snap["selector"]
    model._cls_temperature = _snap["cls_temperature"]
    cost["selected_repeat_idx"] = int(sel_idx)
    cost["repeat_selection_metric"] = str(pick_metric)
    if ycls_train_metrics_tensor is not None:
        cost["results_train_metrics"] = results_train_metrics_list
    if ycls_train_gsep_metrics_tensor is not None:
        cost["results_train_gsep_metrics"] = results_train_gsep_metrics_list

    if track_cost and device.type == "cuda":
        cost["gpu_peak_mb"] = _gpu_peak_memory_mb(device)

    if yreg_mean is not None:
        model._mtm_yreg_mean = yreg_mean.detach().cpu()
        model._mtm_yreg_std = yreg_std.detach().cpu()
    if hurdle_fill_mean_tensor is not None:
        model._mtm_hurdle_fill_mean = hurdle_fill_mean_tensor
        model._mtm_hurdle_missing_fill = str(hurdle_missing_fill)

    print("train_evaluate_MTM finished")
    print(
        f"  Restored model/selector: repeat index {sel_idx} "
        f"(median val scalar among {n_repeats} runs, metric={pick_metric!r}; "
        f"eligible_for_median={cost['repeat_selection_n_eligible']}/{cost['repeat_selection_n_total']}"
        + (
            ", regression_gate_fallback=True"
            if cost["repeat_selection_used_fallback"]
            else ""
        )
        + ")"
    )
    if track_cost:
        print(f"Compute cost: train={cost['train_time_sec']:.2f}s, eval_val={cost['eval_val_time_sec']:.2f}s, eval_test={cost['eval_test_time_sec']:.2f}s", end="")
        if cost["gpu_peak_mb"] is not None:
            print(f", GPU_peak={cost['gpu_peak_mb']:.2f} MB")
        else:
            print()
    if collect_test_raw_outputs:
        print(f"  Stored test raw outputs for {n_repeats} repeats in cost['test_raw_by_repeat'].")
    if collect_val_raw_outputs:
        print(f"  Stored val raw outputs for {n_repeats} repeats in cost['val_raw_by_repeat'].")
    if collect_train_raw_outputs:
        print(f"  Stored train raw outputs for {n_repeats} repeats in cost['train_raw_by_repeat'].")
    if ycls_train_metrics_tensor is not None:
        print(
            f"  Stored per-repeat train metrics ({n_repeats} dicts) in cost['results_train_metrics']."
        )
    if ycls_train_gsep_metrics_tensor is not None:
        print(
            f"  Stored per-repeat train GSEP metrics ({n_repeats} dicts) in cost['results_train_gsep_metrics']."
        )
    _return_cost = cost if (
        track_cost
        or collect_test_raw_outputs
        or collect_val_raw_outputs
        or collect_train_raw_outputs
        or ycls_train_metrics_tensor is not None
        or ycls_train_gsep_metrics_tensor is not None
    ) else None
    return model, selector, thresholds, results_val, results_test, _return_cost


