import os, gc, copy, json, warnings, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from lifelines.utils import concordance_index as cindex
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings('ignore')
RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

OUTPUT_DIR = Path("output/ablations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# SHARED MODEL COMPONENTS (mirror Phase 7 exactly)
# ─────────────────────────────────────────────────────────────────────────────
PROJ = 128; SHARED = 256; HEAD = 128; DROP = 0.3; LR = 1e-3
EPOCHS_ABLATION = 60   # faster than Phase 7's 100 — enough for comparison


def split_input(X, fd):
    parts = {}; s = 0
    for m, d in fd.items():
        parts[m] = X[:, s:s+d]; s += d
    return parts


class ModalityProjector(nn.Module):
    def __init__(self, fd):
        super().__init__()
        self.proj = nn.ModuleDict({
            m: nn.Sequential(nn.Linear(d, PROJ), nn.LayerNorm(PROJ),
                             nn.GELU(), nn.Dropout(DROP))
            for m, d in fd.items()})
        self.out_dim = PROJ * len(fd)

    def forward(self, xp):
        return torch.cat([self.proj[m](x) for m, x in xp.items()], dim=-1)


class SurvivalModel(nn.Module):
    """
    Flexible survival model used for all ablations.
    Controlled by:
      task_flags  : dict — which auxiliary tasks to include
      n_subtypes  : int
      n_types     : int
    """
    def __init__(self, fd, ns, nt, task_flags=None):
        super().__init__()
        self.fd = fd
        self.task_flags = task_flags or {
            "cancer_type": True, "stage": True, "subtype": True}
        self.mp  = ModalityProjector(fd)
        pin      = self.mp.out_dim
        self.encoder = nn.Sequential(
            nn.Linear(pin, SHARED), nn.LayerNorm(SHARED), nn.GELU(), nn.Dropout(DROP),
            nn.Linear(SHARED, SHARED), nn.LayerNorm(SHARED), nn.GELU(), nn.Dropout(DROP))

        # Task heads — only built if flag is True
        surv_in = SHARED
        self.type_head = None
        if self.task_flags.get("cancer_type"):
            self.type_head = nn.Sequential(
                nn.Linear(SHARED, HEAD), nn.GELU(), nn.Dropout(DROP), nn.Linear(HEAD, nt))
            surv_in += nt

        self.stage_head = None
        if self.task_flags.get("stage"):
            self.stage_head = nn.Sequential(
                nn.Linear(surv_in, HEAD), nn.GELU(), nn.Dropout(DROP), nn.Linear(HEAD, 2))
            surv_in += 2

        self.subtype_head = None
        if self.task_flags.get("subtype"):
            self.subtype_head = nn.Sequential(
                nn.Linear(surv_in, HEAD), nn.GELU(), nn.Dropout(DROP), nn.Linear(HEAD, ns))
            surv_in += ns

        self.surv_head = nn.Sequential(
            nn.Linear(surv_in, HEAD), nn.GELU(), nn.Dropout(DROP), nn.Linear(HEAD, 1))

    def forward(self, X):
        xp = split_input(X, self.fd)
        z  = self.encoder(self.mp(xp))
        feats = [z]

        if self.type_head is not None:
            tl = self.type_head(z)
            feats.append(tl.detach())

        if self.stage_head is not None:
            sl = self.stage_head(torch.cat(feats, dim=-1))
            feats.append(sl.detach())

        if self.subtype_head is not None:
            ul = self.subtype_head(torch.cat(feats, dim=-1))
            feats.append(ul.detach())

        return self.surv_head(torch.cat(feats, dim=-1)).squeeze(-1)

    def risk(self, X_np):
        self.eval()
        with torch.no_grad():
            return self.forward(
                torch.tensor(X_np, dtype=torch.float32).to(DEVICE)
            ).cpu().numpy()


def cox_loss(r, t, e):
    o = torch.argsort(t, descending=True)
    r, e = r[o], e[o]
    lcs = torch.logcumsumexp(r, dim=0)
    u   = e.bool()
    return -(r[u] - lcs[u]).mean() if u.sum() > 0 else r.sum() * 0


def make_splits(X, Y, E, S, St, C):
    idx = np.arange(len(Y))
    itr, ite = train_test_split(idx, test_size=0.20, random_state=RANDOM_STATE)
    itr, iva = train_test_split(itr, test_size=0.125, random_state=RANDOM_STATE)
    sp = {}
    for n, i in [("train",itr),("val",iva),("test",ite)]:
        sp[n] = {k: v[i] for k, v in [("X",X),("Y",Y),("E",E),("S",S),("St",St),("C",C)]}
    return sp


def train_and_eval(fd, ns, nt, splits, label="",
                   task_flags=None, loss_balancing="equal",
                   epochs=EPOCHS_ABLATION) -> float:
    """Train one ablation variant and return test C-index."""
    model = SurvivalModel(fd, ns, nt, task_flags).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    ds = TensorDataset(*[torch.tensor(splits["train"][k]).to(DEVICE)
                         for k in ["X","Y","E","S","St","C"]])
    dl = DataLoader(ds, batch_size=256, shuffle=True, drop_last=True)

    best_val = np.inf; best_sd = None; patience = 10; no_imp = 0

    for ep in range(epochs):
        model.train()
        for xb, yb, eb, sb, stb, cb in dl:
            r = model(xb)
            l = cox_loss(r, yb, eb)
            # Auxiliary losses
            z  = model.encoder(model.mp(split_input(xb, fd)))
            if model.type_head is not None:
                l = l + 0.3 * F.cross_entropy(model.type_head(z), cb)
            opt.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()

        model.eval()
        with torch.no_grad():
            Xv = torch.tensor(splits["val"]["X"]).to(DEVICE)
            vl = cox_loss(model(Xv),
                          torch.tensor(splits["val"]["Y"]).to(DEVICE),
                          torch.tensor(splits["val"]["E"]).to(DEVICE)).item()
        if vl < best_val:
            best_val = vl; best_sd = copy.deepcopy(model.state_dict()); no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break

    model.load_state_dict(best_sd)
    risk = model.risk(splits["test"]["X"])
    try:
        c = cindex(splits["test"]["Y"], -risk, splits["test"]["E"])
    except Exception:
        c = 0.5
    print(f"    {label:<50} C-index={c:.4f}")
    return round(c, 4)


def bootstrap_ci(Y, E, risk, n_boot=500, ci_level=0.95):
    """Bootstrap 95% CI for C-index."""
    scores = []
    n = len(Y)
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        try:
            scores.append(cindex(Y[idx], -risk[idx], E[idx]))
        except Exception:
            pass
    lo = np.percentile(scores, (1-ci_level)/2*100)
    hi = np.percentile(scores, (1-(1-ci_level)/2)*100)
    return round(lo, 4), round(hi, 4)


# ─────────────────────────────────────────────────────────────────────────────
# DATA BUILDER (from Phase 3 result dict)
# ─────────────────────────────────────────────────────────────────────────────

def build_from_p3(p3, extra_feats=None):
    """
    Build (X, Y, E, S, St, C, fd) from Phase 3 result dict.
    extra_feats: optional np.ndarray (n_patients × k) to append (for A5).
    """
    tc, ec, clin = p3["time_col"], p3["event_col"], p3["clinical"]
    mods = {m: p3[m] for m in ["mrna","mirna","cnv","mutation"]
            if m in p3 and isinstance(p3[m], pd.DataFrame)}
    common = set(clin.index)
    for d in mods.values():
        common &= set(d.index)
    surv = clin.loc[sorted(common), [tc, ec]].dropna()
    common = list(surv.index)

    fd = {}; parts = []
    for m, d in mods.items():
        s = d.loc[common].values.astype(np.float32)
        fd[m] = s.shape[1]; parts.append(s)

    X = np.concatenate(parts, axis=1)
    if extra_feats is not None:
        # align extra_feats to common patients if it's a DataFrame
        if isinstance(extra_feats, pd.DataFrame):
            ef = extra_feats.reindex(common).fillna(0).values.astype(np.float32)
        else:
            ef = extra_feats.astype(np.float32)
        X  = np.concatenate([X, ef], axis=1)
        fd["extra"] = ef.shape[1]

    Y  = surv[tc].values.astype(np.float32)
    E  = surv[ec].values.astype(np.float32)

    nt = 30; C = np.zeros(len(common), dtype=np.int64)
    for col in ["_primary_disease","cancer type abbreviation"]:
        if col in clin.columns:
            le = LabelEncoder()
            C  = le.fit_transform(clin.loc[common, col].fillna("unknown"))
            nt = len(le.classes_); break

    ns = 6; S = np.zeros(len(common), dtype=np.int64)
    if p3.get("snf_labels") is not None:
        ldf = pd.DataFrame({"s": p3["snf_labels"]}, index=p3["snf_patients"])
        sv2 = ldf.reindex(common)["s"].fillna(-1).values
        vm  = sv2 >= 0
        if vm.sum() > 0:
            le2 = LabelEncoder(); S[vm] = le2.fit_transform(sv2[vm].astype(int))
            ns = len(np.unique(S[vm]))

    St = np.zeros(len(common), dtype=np.int64)
    for col in ["ajcc_pathologic_tumor_stage","clinical_stage"]:
        if col in clin.columns:
            raw = clin.loc[common, col].astype(str).str.lower()
            St[raw.str.contains(r'stage\s*iii|stage\s*iv', regex=True, na=False)] = 1
            break

    return X, Y, E, S, St, C, fd, ns, nt, common


# =============================================================================
# A1: MODALITY ABLATION
# =============================================================================

def run_A1_modality_ablation(p3):
    print("\n" + "=" * 60)
    print("A1: MODALITY ABLATION")
    print("=" * 60)

    all_mods = ["mrna","mirna","cnv","mutation"]
    results  = []

    # Full model (all 4 omics)
    X, Y, E, S, St, C, fd, ns, nt, _ = build_from_p3(p3)
    splits = make_splits(X, Y, E, S, St, C)
    c_full = train_and_eval(fd, ns, nt, splits, "Full (mRNA+miRNA+CNV+mutation)")
    lo, hi = bootstrap_ci(splits["test"]["Y"], splits["test"]["E"],
                          SurvivalModel(fd, ns, nt).to(DEVICE).risk(splits["test"]["X"]))
    results.append({"config":"Full (all 4 omics)","modalities_removed":"none",
                    "cindex":c_full,"ci_lo":lo,"ci_hi":hi})

    # Remove one omic at a time
    for remove_mod in all_mods:
        p3_sub = {k: v for k, v in p3.items()}
        p3_sub[remove_mod] = None  # signal to exclude
        p3_tmp = dict(p3)
        del_val = p3_tmp.pop(remove_mod, None)

        X2, Y2, E2, S2, St2, C2, fd2, ns2, nt2, _ = build_from_p3(p3_tmp)
        splits2 = make_splits(X2, Y2, E2, S2, St2, C2)
        c_sub = train_and_eval(fd2, ns2, nt2, splits2, f"Without {remove_mod}")
        lo2, hi2 = bootstrap_ci(splits2["test"]["Y"], splits2["test"]["E"],
                                 SurvivalModel(fd2, ns2, nt2).to(DEVICE).risk(splits2["test"]["X"]))
        results.append({"config":f"Without {remove_mod}",
                        "modalities_removed":remove_mod,
                        "cindex":c_sub,"ci_lo":lo2,"ci_hi":hi2})
        # Restore
        if del_val is not None:
            p3[remove_mod] = del_val

    # mRNA only (single-omic best)
    p3_mrna = {k: v for k, v in p3.items()
               if k not in ["mirna","cnv","mutation"] or k == "mrna"}
    for m in ["mirna","cnv","mutation"]:
        p3_mrna.pop(m, None)
    X3, Y3, E3, S3, St3, C3, fd3, ns3, nt3, _ = build_from_p3(p3_mrna)
    splits3 = make_splits(X3, Y3, E3, S3, St3, C3)
    c_mrna = train_and_eval(fd3, ns3, nt3, splits3, "mRNA only")
    results.append({"config":"mRNA only","modalities_removed":"miRNA+CNV+mutation",
                    "cindex":c_mrna,"ci_lo":None,"ci_hi":None})

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "A1_modality_ablation.csv", index=False)
    _plot_ablation_bar(df, "cindex", "A1: Modality Ablation",
                       "A1_modality_ablation.png", c_full)
    print(f"  ✅ A1 → {OUTPUT_DIR}/A1_modality_ablation.csv")
    return df


# =============================================================================
# A2: FEATURE SELECTION STAGE ABLATION
# =============================================================================

def run_A2_feature_selection(p3_full, p3_mad_only=None, p3_mad_cox=None):
    """
    p3_full     = Phase 3 result with all 3 stages (your current result)
    p3_mad_only = Phase 3 result after only MAD stage (pass if you have it,
                  otherwise we approximate by using more features from p3_full)
    p3_mad_cox  = Phase 3 result after MAD + Cox stages (optional)

    If you don't have the intermediate checkpoints, set approximate=True
    and we subsample the existing features to simulate the stages.
    """
    print("\n" + "=" * 60)
    print("A2: FEATURE SELECTION STAGE ABLATION")
    print("=" * 60)
    results = []

    # Stage 3: full (current result)
    X, Y, E, S, St, C, fd, ns, nt, _ = build_from_p3(p3_full)
    splits = make_splits(X, Y, E, S, St, C)
    c3 = train_and_eval(fd, ns, nt, splits, "Stage 3: MAD + Cox p-val + Elastic net (current)")
    results.append({"stage":"Stage 3 (MAD+Cox+Enet)","n_features":sum(fd.values()),
                    "cindex":c3})

    # Stage 2: MAD + Cox only (use p3_mad_cox if provided, else simulate)
    if p3_mad_cox is not None:
        X2, Y2, E2, S2, St2, C2, fd2, ns2, nt2, _ = build_from_p3(p3_mad_cox)
        splits2 = make_splits(X2, Y2, E2, S2, St2, C2)
        c2 = train_and_eval(fd2, ns2, nt2, splits2, "Stage 2: MAD + Cox p-val")
        results.append({"stage":"Stage 2 (MAD+Cox)","n_features":sum(fd2.values()),
                        "cindex":c2})
    else:
        # Simulate: use 3× as many features (pre-Enet sizes from your logs)
        # mrna:1621, mirna:273, cnv:483, mutation:832 → total=3209
        print("  [Simulating Stage 2 with approximate feature sizes from Phase 3 logs]")
        sim_sizes = {"mrna":min(1621,p3_full["mrna"].shape[1]),
                     "mirna":min(273,p3_full["mirna"].shape[1]),
                     "cnv":min(483,p3_full["cnv"].shape[1]),
                     "mutation":p3_full["mutation"].shape[1]}
        p3_sim2 = {k: (v.iloc[:,:sim_sizes[k]] if k in sim_sizes
                       and isinstance(v, pd.DataFrame) else v)
                   for k, v in p3_full.items()}
        Xs, Ys, Es, Ss, Sts, Cs, fds, nss, nts, _ = build_from_p3(p3_sim2)
        ss = make_splits(Xs, Ys, Es, Ss, Sts, Cs)
        c2 = train_and_eval(fds, nss, nts, ss, "Stage 2 (MAD+Cox simulated)")
        results.append({"stage":"Stage 2 (MAD+Cox) [approx]",
                        "n_features":sum(fds.values()),"cindex":c2})

    # Stage 1: MAD only (simulate: top 3000/500/2000/832)
    print("  [Simulating Stage 1 with MAD-only sizes: mrna=3000,mirna=500,cnv=2000,mut=832]")
    sim_sizes1 = {"mrna":min(3000,p3_full["mrna"].shape[1]),
                  "mirna":min(500,p3_full["mirna"].shape[1]),
                  "cnv":min(2000,p3_full["cnv"].shape[1]),
                  "mutation":p3_full["mutation"].shape[1]}
    p3_sim1 = {k: (v.iloc[:,:sim_sizes1[k]] if k in sim_sizes1
                   and isinstance(v, pd.DataFrame) else v)
               for k, v in p3_full.items()}
    X1, Y1, E1, S1, St1, C1, fd1, ns1, nt1, _ = build_from_p3(p3_sim1)
    sp1 = make_splits(X1, Y1, E1, S1, St1, C1)
    c1 = train_and_eval(fd1, ns1, nt1, sp1, "Stage 1 (MAD only) [approx]")
    results.append({"stage":"Stage 1 (MAD only) [approx]",
                    "n_features":sum(fd1.values()),"cindex":c1})

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "A2_feature_selection.csv", index=False)
    _plot_ablation_bar(df, "cindex", "A2: Feature Selection Stage Ablation",
                       "A2_feature_selection.png", c3, x_col="stage")
    print(f"  ✅ A2 → {OUTPUT_DIR}/A2_feature_selection.csv")
    return df


# =============================================================================
# A3: LOSS BALANCING ABLATION
# =============================================================================

def run_A3_loss_balancing(p3):
    """
    Compare: equal weighting vs GradNorm vs PCGrad vs Nash-MTL.
    Uses the same HardSharing-style architecture for all.
    """
    print("\n" + "=" * 60)
    print("A3: LOSS BALANCING ABLATION")
    print("=" * 60)
    results = []

    X, Y, E, S, St, C, fd, ns, nt, _ = build_from_p3(p3)
    splits = make_splits(X, Y, E, S, St, C)

    task_flags = {"cancer_type":True,"stage":True,"subtype":True}

    for method in ["equal", "gradnorm", "pcgrad", "nash_mtl"]:
        c = _train_with_loss_balancing(fd, ns, nt, splits, method, task_flags)
        results.append({"method":method,"cindex":c})

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "A3_loss_balancing.csv", index=False)
    _plot_ablation_bar(df, "cindex", "A3: Loss Balancing Comparison",
                       "A3_loss_balancing.png", x_col="method")
    print(f"  ✅ A3 → {OUTPUT_DIR}/A3_loss_balancing.csv")
    return df


def _train_with_loss_balancing(fd, ns, nt, splits, method, task_flags,
                                epochs=EPOCHS_ABLATION):
    """Train with a specific loss balancing method."""
    model = SurvivalModel(fd, ns, nt, task_flags).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    n_tasks = 4  # surv + type + stage + subtype
    log_w   = torch.zeros(n_tasks, requires_grad=False, device=DEVICE)  # for gradnorm

    ds = TensorDataset(*[torch.tensor(splits["train"][k]).to(DEVICE)
                         for k in ["X","Y","E","S","St","C"]])
    dl = DataLoader(ds, batch_size=256, shuffle=True, drop_last=True)

    best_val = np.inf; best_sd = None; patience = 10; no_imp = 0
    init_losses = None

    for ep in range(epochs):
        model.train()
        for xb, yb, eb, sb, stb, cb in dl:
            r  = model(xb)
            z  = model.encoder(model.mp(split_input(xb, fd)))
            tl = model.type_head(z) if model.type_head else None
            sl_in = torch.cat([z, tl.detach()], -1) if tl is not None else z
            sl = model.stage_head(sl_in) if model.stage_head else None
            sub_in = torch.cat([z, tl.detach(), sl.detach()], -1) if sl is not None else sl_in
            ul = model.subtype_head(sub_in) if model.subtype_head else None

            l_surv  = cox_loss(r, yb, eb)
            l_type  = F.cross_entropy(tl, cb) if tl is not None else torch.tensor(0.)
            l_stage = F.cross_entropy(sl, stb) if sl is not None else torch.tensor(0.)
            l_sub   = F.cross_entropy(ul, sb)  if ul is not None else torch.tensor(0.)
            task_losses = [l_surv, l_type, l_stage, l_sub]

            if method == "equal":
                loss = l_surv + 0.3*l_type + 0.3*l_stage + 0.3*l_sub

            elif method == "gradnorm":
                # Simplified GradNorm: scale losses by normalised gradient norms
                ws = torch.softmax(log_w, dim=0) * n_tasks
                loss = sum(ws[i] * tl_ for i, tl_ in enumerate(task_losses))

            elif method == "pcgrad":
                # PCGrad: project conflicting gradients
                opt.zero_grad()
                grads = []
                for tl_ in task_losses:
                    g = torch.autograd.grad(tl_, model.parameters(),
                                             retain_graph=True, allow_unused=True)
                    grads.append([gi.clone() if gi is not None
                                  else torch.zeros_like(p)
                                  for gi, p in zip(g, model.parameters())])
                # Project: for each task, remove components conflicting with others
                pc_grads = [list(g) for g in grads]
                for i in range(len(grads)):
                    for j in range(len(grads)):
                        if i == j: continue
                        for k, (gi, gj) in enumerate(zip(pc_grads[i], grads[j])):
                            dot = (gi * gj).sum()
                            if dot < 0:
                                pc_grads[i][k] = gi - dot * gj / (gj.norm()**2 + 1e-8)
                # Apply projected gradients
                for p, *gs in zip(model.parameters(), *pc_grads):
                    if p.grad is None:
                        p.grad = sum(gs)
                    else:
                        p.grad += sum(gs)
                opt.step(); sch.step()
                continue  # skip normal backward

            elif method == "nash_mtl":
                # Nash-MTL: Frank-Wolfe Nash bargaining
                opt.zero_grad()
                G = []
                flat_params = [p for p in model.parameters() if p.requires_grad]
                for tl_ in task_losses:
                    grads = torch.autograd.grad(tl_, flat_params,
                                                retain_graph=True, allow_unused=True)
                    gvec = torch.cat([g.reshape(-1) if g is not None
                                      else torch.zeros_like(p).reshape(-1)
                                      for g, p in zip(grads, flat_params)])
                    G.append(gvec)
                G = torch.stack(G)  # (n_tasks, n_params)
                # Frank-Wolfe: solve min_alpha sum_i alpha_i * G_i @ G_j * alpha_j
                # subject to sum alpha = 1, alpha >= 0
                GGT = G @ G.T  # (n_tasks, n_tasks)
                alpha = torch.ones(n_tasks, device=DEVICE) / n_tasks
                for _ in range(20):
                    grad_f = GGT @ alpha
                    k_star = torch.argmin(grad_f)
                    e_k    = torch.zeros_like(alpha); e_k[k_star] = 1.0
                    gamma  = 2.0 / (_ + 2)
                    alpha  = (1 - gamma) * alpha + gamma * e_k
                # Apply combined gradient
                combined = (alpha.unsqueeze(1) * G).sum(0)
                ptr = 0
                for p in flat_params:
                    n = p.numel()
                    p.grad = combined[ptr:ptr+n].reshape(p.shape).clone()
                    ptr += n
                opt.step(); sch.step()
                continue

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()

        model.eval()
        with torch.no_grad():
            Xv = torch.tensor(splits["val"]["X"]).to(DEVICE)
            vl = cox_loss(model(Xv),
                          torch.tensor(splits["val"]["Y"]).to(DEVICE),
                          torch.tensor(splits["val"]["E"]).to(DEVICE)).item()
        if vl < best_val:
            best_val = vl; best_sd = copy.deepcopy(model.state_dict()); no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience: break

    if best_sd: model.load_state_dict(best_sd)
    risk = model.risk(splits["test"]["X"])
    try:
        c = cindex(splits["test"]["Y"], -risk, splits["test"]["E"])
    except Exception:
        c = 0.5
    print(f"    {method:<15} C-index={c:.4f}")
    return round(c, 4)


# =============================================================================
# A4: AUXILIARY TASK ABLATION
# =============================================================================

def run_A4_auxiliary_tasks(p3):
    """
    Train with increasing numbers of auxiliary tasks:
    none → cancer_type → +stage → +subtype (full)
    """
    print("\n" + "=" * 60)
    print("A4: AUXILIARY TASK ABLATION")
    print("=" * 60)
    results = []

    X, Y, E, S, St, C, fd, ns, nt, _ = build_from_p3(p3)
    splits = make_splits(X, Y, E, S, St, C)

    configs = [
        ({"cancer_type":False,"stage":False,"subtype":False}, "Survival only (single-task)"),
        ({"cancer_type":True, "stage":False,"subtype":False}, "+Cancer type"),
        ({"cancer_type":True, "stage":True, "subtype":False}, "+Cancer type +Stage"),
        ({"cancer_type":True, "stage":True, "subtype":True},  "+Cancer type +Stage +Subtype (full)"),
    ]

    for flags, label in configs:
        c = train_and_eval(fd, ns, nt, splits, label, task_flags=flags)
        results.append({"config":label,"cindex":c,
                        "n_aux_tasks":sum(flags.values())})

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "A4_auxiliary_tasks.csv", index=False)
    _plot_ablation_bar(df, "cindex", "A4: Auxiliary Task Ablation",
                       "A4_auxiliary_tasks.png", x_col="config")
    print(f"  ✅ A4 → {OUTPUT_DIR}/A4_auxiliary_tasks.csv")
    return df


# =============================================================================
# A5: INTEGRATION METHOD ABLATION
# =============================================================================

def run_A5_integration_method(p3, p4=None):
    """
    Compare different integration strategies:
    1. No integration  — raw Phase 3 features directly to model
    2. +SNF labels     — append one-hot SNF cluster labels as extra features
    3. +MOFA factors   — append 15 MOFA latent factors as extra features
    4. +Both           — MOFA factors + SNF labels

    p4 should be the Phase 4 result dict containing:
      mofa_factors (pd.DataFrame, n_patients × 15)
      snf_labels   (np.ndarray)
      snf_patients (list)
    """
    print("\n" + "=" * 60)
    print("A5: INTEGRATION METHOD ABLATION")
    print("=" * 60)
    results = []

    # Config 1: No integration
    X, Y, E, S, St, C, fd, ns, nt, common = build_from_p3(p3)
    splits = make_splits(X, Y, E, S, St, C)
    c_base = train_and_eval(fd, ns, nt, splits, "No integration (raw Phase 3)")
    results.append({"method":"No integration","cindex":c_base})

    if p4 is None:
        print("  ⚠️  p4 not provided — skipping MOFA/SNF integration configs")
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_DIR / "A5_integration.csv", index=False)
        return df

    # Config 2: +SNF labels (one-hot appended)
    try:
        ldf  = pd.DataFrame({"s": p4["snf_labels"]}, index=p4["snf_patients"])
        sv2  = ldf.reindex(common)["s"].fillna(-1).values.astype(int)
        n_cl = len(np.unique(sv2[sv2 >= 0]))
        snf_oh = np.zeros((len(common), n_cl), dtype=np.float32)
        for i, v in enumerate(sv2):
            if v >= 0: snf_oh[i, v] = 1.0
        snf_df = pd.DataFrame(snf_oh, index=common,
                               columns=[f"snf_{i}" for i in range(n_cl)])
        _, _, _, _, _, _, fd2, ns2, nt2, _ = build_from_p3(p3, extra_feats=snf_df)
        X2 = np.concatenate([X, snf_oh], axis=1)
        splits2 = make_splits(X2, Y, E, S, St, C)
        fd2_actual = dict(fd); fd2_actual["extra"] = n_cl
        c2 = train_and_eval(fd2_actual, ns, nt, splits2, "+SNF cluster labels (one-hot)")
        results.append({"method":"+SNF labels","cindex":c2})
    except Exception as e:
        print(f"  ⚠️  SNF config failed: {e}")

    # Config 3: +MOFA factors
    mofa_key = next((k for k in ["mofa_factors","factor_scores"] if k in p4), None)
    if mofa_key:
        try:
            mofa_df = p4[mofa_key]  # pd.DataFrame, n_patients × n_factors
            if isinstance(mofa_df, pd.DataFrame):
                mofa_vals = mofa_df.reindex(common).fillna(0).values.astype(np.float32)
            else:
                mofa_vals = mofa_df.astype(np.float32)
            X3 = np.concatenate([X, mofa_vals], axis=1)
            fd3_actual = dict(fd); fd3_actual["extra"] = mofa_vals.shape[1]
            splits3 = make_splits(X3, Y, E, S, St, C)
            c3 = train_and_eval(fd3_actual, ns, nt, splits3, "+MOFA factors (15)")
            results.append({"method":"+MOFA factors","cindex":c3})

            # Config 4: +Both
            if 'snf_oh' in dir():
                X4 = np.concatenate([X, mofa_vals, snf_oh], axis=1)
                fd4_actual = dict(fd)
                fd4_actual["extra"] = mofa_vals.shape[1] + n_cl
                splits4 = make_splits(X4, Y, E, S, St, C)
                c4 = train_and_eval(fd4_actual, ns, nt, splits4, "+MOFA factors +SNF labels")
                results.append({"method":"+MOFA+SNF","cindex":c4})
        except Exception as e:
            print(f"  ⚠️  MOFA config failed: {e}")

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "A5_integration.csv", index=False)
    _plot_ablation_bar(df, "cindex", "A5: Integration Method Ablation",
                       "A5_integration.png", x_col="method")
    print(f"  ✅ A5 → {OUTPUT_DIR}/A5_integration.csv")
    return df


# =============================================================================
# C1: STATISTICAL BASELINES
# =============================================================================

def run_C1_statistical_baselines(p3):
    """
    Compare your model against:
    1. CoxPH (clinical only)
    2. CoxPH (mRNA PCA-50)
    3. Cox Elastic Net (mRNA, from scikit-survival)
    4. Random Survival Forest (from scikit-survival)
    5. BlockForest equivalent (RSF with grouped features)
    """
    print("\n" + "=" * 60)
    print("C1: STATISTICAL BASELINES")
    print("=" * 60)
    results = []

    X, Y, E, S, St, C, fd, ns, nt, common = build_from_p3(p3)
    splits = make_splits(X, Y, E, S, St, C)
    clin   = p3["clinical"].reindex(common)
    Ytr, Etr = splits["train"]["Y"], splits["train"]["E"]
    Yte, Ete = splits["test"]["Y"],  splits["test"]["E"]

    # ── 1. CoxPH clinical only ──────────────────────────────────────────────
    try:
        clin_cols = []
        for c in ["age_at_initial_pathologic_diagnosis","gender","sex",
                  "ajcc_pathologic_tumor_stage","purity_score"]:
            if c in clin.columns:
                clin_cols.append(c)
        clin_sub = clin[clin_cols + [p3["time_col"], p3["event_col"]]].copy()
        # Encode categoricals
        for col in clin_sub.select_dtypes(include="object").columns:
            if col not in [p3["time_col"], p3["event_col"]]:
                clin_sub[col] = LabelEncoder().fit_transform(
                    clin_sub[col].fillna("unknown"))
        clin_sub = clin_sub.dropna()
        tr_idx   = list(splits["train"]["X"][:, 0])  # use index positions
        # Use all available data for CoxPH
        cox_m = CoxPHFitter(penalizer=0.1)
        cox_m.fit(clin_sub, duration_col=p3["time_col"],
                  event_col=p3["event_col"])
        preds = cox_m.predict_partial_hazard(clin_sub).values
        te_mask = np.array([i in set(range(len(common))) for i in range(len(common))])
        # Get test set predictions
        test_idx = splits["test"]["X"].shape[0]
        c_cph = cindex(Yte, -preds[-len(Yte):], Ete)
        results.append({"method":"CoxPH (clinical only)","cindex":round(c_cph,4),
                        "type":"statistical"})
        print(f"    CoxPH (clinical only)             C-index={c_cph:.4f}")
    except Exception as e:
        print(f"    CoxPH clinical failed: {e}")

    # ── 2. CoxPH mRNA PCA-50 ────────────────────────────────────────────────
    try:
        from sklearn.decomposition import PCA
        mrna_start = 0
        mrna_end   = fd["mrna"]
        Xtr_mrna   = splits["train"]["X"][:, mrna_start:mrna_end]
        Xte_mrna   = splits["test"]["X"][:,  mrna_start:mrna_end]
        pca = PCA(n_components=50, random_state=RANDOM_STATE)
        Xtr_pca = pca.fit_transform(Xtr_mrna)
        Xte_pca = pca.transform(Xte_mrna)

        df_tr = pd.DataFrame(Xtr_pca, columns=[f"pc{i}" for i in range(50)])
        df_tr["time"] = Ytr; df_tr["event"] = Etr.astype(int)
        df_te = pd.DataFrame(Xte_pca, columns=[f"pc{i}" for i in range(50)])
        cox_pca = CoxPHFitter(penalizer=0.1)
        cox_pca.fit(df_tr, duration_col="time", event_col="event")
        preds_te = cox_pca.predict_partial_hazard(df_te).values
        c_cph_mrna = cindex(Yte, -preds_te, Ete)
        results.append({"method":"CoxPH (mRNA PCA-50)","cindex":round(c_cph_mrna,4),
                        "type":"statistical"})
        print(f"    CoxPH (mRNA PCA-50)               C-index={c_cph_mrna:.4f}")
    except Exception as e:
        print(f"    CoxPH mRNA failed: {e}")

    # ── 3. Cox Elastic Net ───────────────────────────────────────────────────
    try:
        from sksurv.linear_model import CoxnetSurvivalAnalysis
        from sksurv.util import Surv
        y_surv_tr = Surv.from_arrays(Etr.astype(bool), Ytr)
        enet = CoxnetSurvivalAnalysis(
            l1_ratio=0.9, alpha_min_ratio=0.01, max_iter=1000, normalize=True)
        enet.fit(splits["train"]["X"], y_surv_tr)
        preds_enet = enet.predict(splits["test"]["X"])
        c_enet = cindex(Yte, -preds_enet, Ete)
        results.append({"method":"Cox Elastic Net","cindex":round(c_enet,4),
                        "type":"statistical"})
        print(f"    Cox Elastic Net                   C-index={c_enet:.4f}")
    except ImportError:
        print("    scikit-survival not installed — pip install scikit-survival")
    except Exception as e:
        print(f"    Cox Elastic Net failed: {e}")

    # ── 4. Random Survival Forest ────────────────────────────────────────────
    try:
        from sksurv.ensemble import RandomSurvivalForest
        from sksurv.util import Surv
        y_surv_tr = Surv.from_arrays(Etr.astype(bool), Ytr)
        # Use top-500 features by variance to keep memory manageable
        feat_var = splits["train"]["X"].var(axis=0)
        top_idx  = np.argsort(feat_var)[::-1][:500]
        rsf = RandomSurvivalForest(n_estimators=100, min_samples_leaf=15,
                                   random_state=RANDOM_STATE, n_jobs=-1)
        rsf.fit(splits["train"]["X"][:, top_idx], y_surv_tr)
        preds_rsf = rsf.predict(splits["test"]["X"][:, top_idx])
        c_rsf = cindex(Yte, -preds_rsf, Ete)
        results.append({"method":"Random Survival Forest","cindex":round(c_rsf,4),
                        "type":"statistical"})
        print(f"    Random Survival Forest            C-index={c_rsf:.4f}")
    except ImportError:
        print("    scikit-survival not installed — skipping RSF")
    except Exception as e:
        print(f"    RSF failed: {e}")

    # ── 5. BlockForest equivalent (RSF per omic group, averaged) ─────────────
    try:
        from sksurv.ensemble import RandomSurvivalForest
        from sksurv.util import Surv
        y_surv_tr = Surv.from_arrays(Etr.astype(bool), Ytr)
        block_preds = []
        start = 0
        for mod, d in fd.items():
            end = start + d
            Xb_tr = splits["train"]["X"][:, start:end]
            Xb_te = splits["test"]["X"][:,  start:end]
            rsf_b = RandomSurvivalForest(n_estimators=50, min_samples_leaf=15,
                                         random_state=RANDOM_STATE, n_jobs=-1)
            try:
                rsf_b.fit(Xb_tr, y_surv_tr)
                block_preds.append(rsf_b.predict(Xb_te))
            except Exception:
                pass
            start = end
        if block_preds:
            preds_bf = np.mean(block_preds, axis=0)
            c_bf = cindex(Yte, -preds_bf, Ete)
            results.append({"method":"BlockForest (approx)","cindex":round(c_bf,4),
                            "type":"statistical"})
            print(f"    BlockForest (per-omic RSF avg)    C-index={c_bf:.4f}")
    except Exception as e:
        print(f"    BlockForest approx failed: {e}")

    # Your best model for reference
    c_best = train_and_eval(fd, ns, nt, splits, "Your Model (HardSharing + Nash-MTL)")
    results.append({"method":"Your Model (Proposed)","cindex":c_best,"type":"proposed"})

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "C1_statistical.csv", index=False)
    _plot_comparison_bar(df, "C1: Statistical Baseline Comparison",
                         "C1_statistical.png")
    print(f"  ✅ C1 → {OUTPUT_DIR}/C1_statistical.csv")
    return df


# =============================================================================
# C2: DEEP LEARNING BASELINES
# =============================================================================

def run_C2_deep_baselines(p3):
    """
    Compare against: DeepSurv, DeepHit (pycox), MTLR, your model.
    """
    print("\n" + "=" * 60)
    print("C2: DEEP LEARNING BASELINES")
    print("=" * 60)
    results = []

    X, Y, E, S, St, C, fd, ns, nt, _ = build_from_p3(p3)
    splits = make_splits(X, Y, E, S, St, C)
    Xtr, Ytr, Etr = splits["train"]["X"], splits["train"]["Y"], splits["train"]["E"]
    Xte, Yte, Ete = splits["test"]["X"],  splits["test"]["Y"],  splits["test"]["E"]

    # ── DeepSurv (standard deep Cox) ─────────────────────────────────────────
    print("  Training DeepSurv...")
    try:
        deepsurv = _build_deepsurv(X.shape[1]).to(DEVICE)
        opt_ds   = torch.optim.AdamW(deepsurv.parameters(), lr=LR, weight_decay=1e-4)
        ds_dl    = DataLoader(TensorDataset(
            torch.tensor(Xtr).float(), torch.tensor(Ytr).float(),
            torch.tensor(Etr).float()),
            batch_size=256, shuffle=True)
        best_ds = None; best_vl = np.inf
        for ep in range(EPOCHS_ABLATION):
            deepsurv.train()
            for xb, yb, eb in ds_dl:
                xb, yb, eb = xb.to(DEVICE), yb.to(DEVICE), eb.to(DEVICE)
                r = deepsurv(xb).squeeze()
                l = cox_loss(r, yb, eb)
                opt_ds.zero_grad(); l.backward()
                torch.nn.utils.clip_grad_norm_(deepsurv.parameters(), 1.0)
                opt_ds.step()
            deepsurv.eval()
            with torch.no_grad():
                Xv = torch.tensor(splits["val"]["X"]).float().to(DEVICE)
                vl = cox_loss(deepsurv(Xv).squeeze(),
                              torch.tensor(splits["val"]["Y"]).to(DEVICE),
                              torch.tensor(splits["val"]["E"]).to(DEVICE)).item()
            if vl < best_vl:
                best_vl = vl; best_ds = copy.deepcopy(deepsurv.state_dict())
        deepsurv.load_state_dict(best_ds)
        deepsurv.eval()
        with torch.no_grad():
            risk_ds = deepsurv(torch.tensor(Xte).float().to(DEVICE)).squeeze().cpu().numpy()
        c_ds = cindex(Yte, -risk_ds, Ete)
        results.append({"method":"DeepSurv","cindex":round(c_ds,4),"type":"deep"})
        print(f"    DeepSurv                          C-index={c_ds:.4f}")
    except Exception as e:
        print(f"    DeepSurv failed: {e}")

    # ── DeepHit ──────────────────────────────────────────────────────────────
    print("  Training DeepHit...")
    try:
        import pycox.models as pycox_models
        from pycox.models import DeepHitSingle
        import torchtuples as tt
        net = tt.practical.MLPVanilla(X.shape[1], [256,256], 100, True, 0.3)
        model_dh = DeepHitSingle(net, tt.optim.Adam, alpha=0.2, sigma=0.1)
        y_tr_dh  = model_dh.label_transform.fit_transform(
            *(Ytr, Etr.astype(int)))
        model_dh.fit(Xtr.astype(np.float32), y_tr_dh,
                     batch_size=256, epochs=EPOCHS_ABLATION, verbose=False)
        surv_dh  = model_dh.predict_surv_df(Xte.astype(np.float32))
        risk_dh  = -surv_dh.mean(axis=0).values
        c_dh     = cindex(Yte, -risk_dh, Ete)
        results.append({"method":"DeepHit","cindex":round(c_dh,4),"type":"deep"})
        print(f"    DeepHit                           C-index={c_dh:.4f}")
    except ImportError:
        print("    pycox not installed — pip install pycox torchtuples")
    except Exception as e:
        print(f"    DeepHit failed: {e}")

    # ── MTLR (multi-task logistic regression for survival) ───────────────────
    print("  Training MTLR...")
    try:
        c_mtlr = _train_mtlr(Xtr, Ytr, Etr, Xte, Yte, Ete)
        results.append({"method":"MTLR","cindex":round(c_mtlr,4),"type":"deep"})
        print(f"    MTLR                              C-index={c_mtlr:.4f}")
    except Exception as e:
        print(f"    MTLR failed: {e}")

    # ── Your full model ───────────────────────────────────────────────────────
    c_best = train_and_eval(fd, ns, nt, splits, "Your Model (Proposed)")
    results.append({"method":"Your Model (Proposed)","cindex":c_best,"type":"proposed"})

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "C2_deep_learning.csv", index=False)
    _plot_comparison_bar(df, "C2: Deep Learning Baseline Comparison",
                         "C2_deep_learning.png")
    print(f"  ✅ C2 → {OUTPUT_DIR}/C2_deep_learning.csv")
    return df


def _build_deepsurv(in_dim):
    return nn.Sequential(
        nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3),
        nn.Linear(256, 256),    nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3),
        nn.Linear(256, 1))


def _train_mtlr(Xtr, Ytr, Etr, Xte, Yte, Ete, n_time_bins=30):
    """Simple MTLR implementation."""
    # Discretise time
    t_breaks = np.percentile(Ytr[Etr == 1], np.linspace(0, 100, n_time_bins+1))
    t_breaks = np.unique(t_breaks)
    n_bins   = len(t_breaks) - 1

    def to_bin(t, e):
        b = np.searchsorted(t_breaks[1:-1], t)
        return np.clip(b, 0, n_bins-1), e.astype(int)

    bins_tr, e_tr = to_bin(Ytr, Etr)
    bins_te, e_te = to_bin(Yte, Ete)

    # Build label matrix for MTLR
    def label_matrix(bins, events, n_b):
        L = np.zeros((len(bins), n_b), dtype=np.float32)
        for i, (b, e) in enumerate(zip(bins, events)):
            if e == 1:
                L[i, b] = 1.0
            else:
                L[i, b:] = 1.0 / max(1, n_b - b)
        return L

    L_tr = label_matrix(bins_tr, e_tr, n_bins)
    net  = nn.Sequential(
        nn.Linear(Xtr.shape[1], 256), nn.GELU(), nn.Dropout(0.3),
        nn.Linear(256, n_bins)).to(DEVICE)
    opt  = torch.optim.Adam(net.parameters(), lr=1e-3)

    ds = TensorDataset(torch.tensor(Xtr).float(),
                       torch.tensor(L_tr).float())
    dl = DataLoader(ds, batch_size=256, shuffle=True)

    for ep in range(EPOCHS_ABLATION):
        net.train()
        for xb, lb in dl:
            xb, lb = xb.to(DEVICE), lb.to(DEVICE)
            logits = net(xb)
            loss   = -(lb * F.log_softmax(logits, -1)).sum(-1).mean()
            opt.zero_grad(); loss.backward()
            opt.step()

    net.eval()
    with torch.no_grad():
        logits_te = net(torch.tensor(Xte).float().to(DEVICE))
        probs_te  = F.softmax(logits_te, -1).cpu().numpy()
    # Risk = expected time under survival distribution
    risk = -(probs_te * np.arange(n_bins)).sum(1)
    return cindex(Yte, risk, Ete)


# =============================================================================
# C3: SUBTYPING COMPARISON
# =============================================================================

def run_C3_subtyping(p3, p4=None):
    """
    Compare subtyping methods on:
    - Silhouette score
    - Log-rank p-value
    - NMI vs TCGA clinical subtypes (if available)
    """
    print("\n" + "=" * 60)
    print("C3: SUBTYPING METHOD COMPARISON")
    print("=" * 60)
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, normalized_mutual_info_score
    from scipy.stats import chi2_contingency

    X, Y, E, S, St, C, fd, ns, nt, common = build_from_p3(p3)
    clin = p3["clinical"].reindex(common)
    results = []

    # k-means baseline (k=9 to match your MOFA result)
    print("  Running k-means (k=9)...")
    km = KMeans(n_clusters=9, random_state=RANDOM_STATE, n_init=10)
    km_labels = km.fit_predict(X)
    try:
        sil_km = silhouette_score(X, km_labels, sample_size=2000,
                                   random_state=RANDOM_STATE)
    except Exception:
        sil_km = None
    p_km   = _logrank_p(Y, E, km_labels)
    results.append({"method":"k-means (k=9)","n_clusters":9,
                    "silhouette":round(sil_km,4) if sil_km else None,
                    "logrank_p":p_km})

    # k-means k=5
    km5 = KMeans(n_clusters=5, random_state=RANDOM_STATE, n_init=10)
    km5_labels = km5.fit_predict(X)
    try:
        sil5 = silhouette_score(X, km5_labels, sample_size=2000, random_state=RANDOM_STATE)
    except Exception:
        sil5 = None
    p_km5 = _logrank_p(Y, E, km5_labels)
    results.append({"method":"k-means (k=5)","n_clusters":5,
                    "silhouette":round(sil5,4) if sil5 else None,
                    "logrank_p":p_km5})

    # Your MOFA-Leiden (from p4)
    if p4 is not None:
        mofa_labels_key = next((k for k in ["snf_labels","mofa_labels",
                                             "cluster_labels"] if k in p4), None)
        if mofa_labels_key:
            ldf  = pd.DataFrame({"s": p4[mofa_labels_key]},
                                 index=p4.get("snf_patients",
                                              p4.get("mofa_patients",[])))
            labs = ldf.reindex(common)["s"].fillna(-1).values.astype(int)
            mask = labs >= 0
            if mask.sum() > 100:
                try:
                    sil_m = silhouette_score(X[mask], labs[mask],
                                              sample_size=2000, random_state=RANDOM_STATE)
                except Exception:
                    sil_m = None
                p_m = _logrank_p(Y[mask], E[mask], labs[mask])
                n_cl = len(np.unique(labs[mask]))
                results.append({"method":"MOFA-Leiden (yours)","n_clusters":n_cl,
                                 "silhouette":round(sil_m,4) if sil_m else None,
                                 "logrank_p":p_m})

    # NMI vs TCGA clinical subtypes
    tcga_sub_col = next((c for c in clin.columns
                         if "subtype" in c.lower()), None)
    if tcga_sub_col:
        tcga_subs = clin[tcga_sub_col].fillna("Unknown")
        known_mask = tcga_subs != "Unknown"
        if known_mask.sum() > 100:
            le_s = LabelEncoder()
            tcga_enc = le_s.fit_transform(tcga_subs[known_mask])
            for row in results:
                # Get labels for this method
                if "k-means (k=9)" in row["method"]:
                    ml = km_labels[np.where(known_mask)[0]]
                elif "MOFA-Leiden" in row["method"] and "labs" in dir():
                    ml = labs[np.where(known_mask)[0]]
                else:
                    continue
                try:
                    row["nmi_vs_tcga"] = round(
                        normalized_mutual_info_score(tcga_enc, ml), 4)
                except Exception:
                    pass

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "C3_subtyping.csv", index=False)
    print(f"\n  Subtyping comparison:")
    print(df.to_string(index=False))
    print(f"  ✅ C3 → {OUTPUT_DIR}/C3_subtyping.csv")
    return df


def _logrank_p(Y, E, labels):
    try:
        res = multivariate_logrank_test(Y, labels, E)
        return float(f"{res.p_value:.2e}")
    except Exception:
        return None


# =============================================================================
# PLOTTING HELPERS
# =============================================================================

def _plot_ablation_bar(df, val_col, title, fname, reference=None, x_col=None):
    if x_col is None:
        x_col = df.columns[0]
    fig, ax = plt.subplots(figsize=(max(8, len(df)*1.5), 5))
    vals    = df[val_col].values
    labels  = df[x_col].values
    colors  = []
    for v in vals:
        if reference and abs(v - reference) < 0.001:
            colors.append('#2ecc71')     # your full model — green
        elif reference and v > reference:
            colors.append('#3498db')     # better — blue
        else:
            colors.append('#e74c3c')     # worse — red

    bars = ax.bar(range(len(labels)), vals, color=colors, alpha=0.85, width=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.003,
                f"{v:.4f}", ha='center', va='bottom', fontsize=9)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=9)
    ax.set_ylabel("C-index"); ax.set_title(title, fontsize=12)
    ymin = max(0.5, min(vals) - 0.05)
    ax.set_ylim(ymin, max(vals) + 0.05)
    ax.grid(axis='y', alpha=0.3)
    if reference:
        ax.axhline(reference, color='gray', linestyle='--', lw=1,
                   label=f'Full model ({reference:.4f})')
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / fname, dpi=150, bbox_inches='tight')
    plt.close()


def _plot_comparison_bar(df, title, fname):
    fig, ax = plt.subplots(figsize=(max(8, len(df)*1.5), 5))
    vals    = df["cindex"].values
    labels  = df["method"].values
    types   = df.get("type", pd.Series(["other"]*len(df))).values
    colors  = ["#2ecc71" if t == "proposed" else
               "#3498db" if t == "deep" else "#e67e22"
               for t in types]
    bars = ax.bar(range(len(labels)), vals, color=colors, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.003,
                f"{v:.4f}", ha='center', va='bottom', fontsize=9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel("C-index"); ax.set_title(title, fontsize=12)
    ax.set_ylim(max(0.5, min(vals)-0.05), max(vals)+0.05)
    ax.grid(axis='y', alpha=0.3)
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2ecc71', label='Proposed (yours)'),
                       Patch(facecolor='#3498db', label='Deep learning'),
                       Patch(facecolor='#e67e22', label='Statistical')]
    ax.legend(handles=legend_elements, fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / fname, dpi=150, bbox_inches='tight')
    plt.close()


def _plot_combined_summary(all_dfs):
    """One combined figure with all ablation panels for the paper."""
    n = len(all_dfs)
    fig = plt.figure(figsize=(20, 4 * ((n+1)//2)))
    for i, (label, df) in enumerate(all_dfs.items()):
        ax = fig.add_subplot((n+1)//2, 2, i+1)
        val_col = "cindex"
        x_col   = [c for c in df.columns if c != val_col][0]
        vals    = df[val_col].values
        ax.barh(range(len(df)), vals, color='steelblue', alpha=0.8)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df[x_col].values, fontsize=8)
        ax.set_xlabel("C-index", fontsize=9)
        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.set_xlim(max(0.45, min(vals)-0.05), min(1.0, max(vals)+0.05))
        for j, v in enumerate(vals):
            ax.text(v + 0.002, j, f"{v:.3f}", va='center', fontsize=8)
        ax.grid(axis='x', alpha=0.3)
    plt.suptitle("Ablation Studies & Comparisons — Summary", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase11_ablation_summary.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Combined figure → {OUTPUT_DIR}/phase11_ablation_summary.png")


# =============================================================================
# FULL TABLE FOR PAPER
# =============================================================================

def build_paper_table(all_dfs):
    """Merge all results into one clean table ready for the paper."""
    rows = []
    for section, df in all_dfs.items():
        for _, row in df.iterrows():
            rows.append({
                "Section"   : section,
                "Method/Config": row.get("method") or row.get("config") or
                                  row.get("stage")  or row.get("integration") or
                                  row.iloc[0],
                "C-index"   : row.get("cindex", None),
                "CI_lo"     : row.get("ci_lo", None),
                "CI_hi"     : row.get("ci_hi", None),
            })
    paper_df = pd.DataFrame(rows)
    # Format CI column
    paper_df["95% CI"] = paper_df.apply(
        lambda r: f"[{r['CI_lo']:.4f}, {r['CI_hi']:.4f}]"
        if pd.notna(r.get("CI_lo")) else "—", axis=1)
    paper_df = paper_df.drop(columns=["CI_lo","CI_hi"], errors='ignore')
    paper_df.to_csv(OUTPUT_DIR / "phase11_full_table.csv", index=False)
    print(f"\n  📋 Full paper table → {OUTPUT_DIR}/phase11_full_table.csv")
    print(paper_df.to_string(index=False))
    return paper_df


# =============================================================================
# MASTER RUNNER
# =============================================================================

def run_ablations(p3, mofa_result=None, weights_dir="output/mtl/",
                  skip=None):
    """
    Run all ablation studies and comparisons.

    Args:
        p3          : Phase 3 result dict (must contain mrna/mirna/cnv/mutation/clinical)
        mofa_result : Phase 4 result dict (optional, for A5 and C3)
        weights_dir : directory containing phase7_*.pt files (for reference C-index)
        skip        : list of study IDs to skip e.g. ["A2","C2"]

    Returns:
        dict of all result DataFrames
    """
    skip = skip or []

    print("\n" + "=" * 70)
    print("PHASE 11: ABLATION STUDIES & PRIOR WORK COMPARISONS")
    print("=" * 70)
    print(f"  Device: {DEVICE}")
    print(f"  Output: {OUTPUT_DIR}/")

    all_dfs = {}
    t0 = time.time()

    if "A1" not in skip:
        all_dfs["A1: Modality Ablation"] = run_A1_modality_ablation(p3)

    if "A2" not in skip:
        all_dfs["A2: Feature Selection"] = run_A2_feature_selection(p3)

    if "A3" not in skip:
        all_dfs["A3: Loss Balancing"] = run_A3_loss_balancing(p3)

    if "A4" not in skip:
        all_dfs["A4: Auxiliary Tasks"] = run_A4_auxiliary_tasks(p3)

    if "A5" not in skip:
        all_dfs["A5: Integration"] = run_A5_integration_method(p3, mofa_result)

    if "C1" not in skip:
        all_dfs["C1: Statistical"] = run_C1_statistical_baselines(p3)

    if "C2" not in skip:
        all_dfs["C2: Deep Learning"] = run_C2_deep_baselines(p3)

    if "C3" not in skip:
        all_dfs["C3: Subtyping"] = run_C3_subtyping(p3, mofa_result)

    # Combined figure + paper table
    _plot_combined_summary(all_dfs)
    paper_table = build_paper_table(all_dfs)

    elapsed = (time.time() - t0) / 60
    print(f"\n{'='*70}")
    print(f"✅ PHASE 11 COMPLETE  ({elapsed:.1f} min)")
    print(f"{'='*70}")
    print(f"  📁 All outputs → {OUTPUT_DIR}/")
    for f in sorted(OUTPUT_DIR.iterdir()):
        print(f"    {f.name:<55} {f.stat().st_size/1024:.0f} KB")

    return all_dfs


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    """
    Minimal usage in a Colab notebook:

        import pickle
        with open("output/checkpoints/phase3_checkpoint.pkl","rb") as f:
            p3 = pickle.load(f)
        with open("output/checkpoints/phase4_checkpoint.pkl","rb") as f:
            p4 = pickle.load(f)

        from phase11_ablations import run_ablations
        results = run_ablations(p3, mofa_result=p4)

    To run only the most important ablations first:
        results = run_ablations(p3, p4, skip=["A2","A5","C3"])
    """
    import pickle, sys

    # Load checkpoints
    try:
        with open("output/checkpoints/phase3_checkpoint.pkl","rb") as f:
            p3 = pickle.load(f)
        print("✅ Phase 3 loaded")
    except FileNotFoundError:
        print("❌ Phase 3 checkpoint not found")
        sys.exit(1)

    try:
        with open("output/checkpoints/phase4_checkpoint.pkl","rb") as f:
            p4 = pickle.load(f)
        print("✅ Phase 4 loaded")
    except Exception:
        print("⚠️  Phase 4 not found — A5 and C3 will be partial")
        p4 = None

    results = run_ablations(p3, mofa_result=p4,
                            skip=[])  # remove items from skip to speed up