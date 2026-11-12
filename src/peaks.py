import numpy as np
import pandas as pd

MARGINS = [0.0, 0.01, 0.02, 0.03, 0.05]  # safety margins below the threshold at which a warning is issued


def episode_ids(flag, max_gap=6):
    """Label runs of flagged hours as episodes; flagged hours at most `max_gap` hours apart share one episode.
    Returns an int array (-1 = not in an episode). Assumes consecutive hourly rows."""
    flag = np.asarray(flag, bool)
    ids = np.full(len(flag), -1)
    pos = np.flatnonzero(flag)
    if len(pos) == 0:
        return ids
    eid = np.cumsum(np.r_[True, np.diff(pos) > max_gap]) - 1
    for e in range(eid.max() + 1):
        p = pos[eid == e]
        ids[p.min():p.max() + 1] = e
    return ids


def warning_metrics(true_high, warn):
    """Hour-level confusion metrics plus episode-level detection rate and precision."""
    true_high, warn = np.asarray(true_high, bool), np.asarray(warn, bool)
    tp, fp = int((true_high & warn).sum()), int((~true_high & warn).sum())
    fn, tn = int((true_high & ~warn).sum()), int((~true_high & ~warn).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    t_ids, w_ids = episode_ids(true_high), episode_ids(warn)
    n_t, n_w = t_ids.max() + 1, w_ids.max() + 1
    ep_rec = np.mean([warn[t_ids == e].any() for e in range(n_t)]) if n_t else np.nan
    ep_prec = np.mean([true_high[w_ids == e].any() for e in range(n_w)]) if n_w else np.nan
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "precision": prec, "recall": rec, "F1": f1,
            "false_alarm_rate": fp / (fp + tn) if fp + tn else 0.0,
            "true_episodes": int(n_t), "episode_recall": ep_rec,
            "warning_episodes": int(n_w), "episode_precision": ep_prec}


def sweep(actual, pred, thr, margins=MARGINS):
    """Warning performance for each safety margin: warn when pred > thr * (1 - margin)."""
    true_high = np.asarray(actual) > np.asarray(thr)
    return pd.DataFrame([{"margin": m, **warning_metrics(true_high, np.asarray(pred) > np.asarray(thr) * (1 - m))}
                         for m in margins])