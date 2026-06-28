"""
Models 2 and 3 LOLO — using saved Model 1 predictions
=======================================================
Model 1 already run and saved at:
  extended_analysis/lolo_PV/lolo_PV_predictions.csv

This script:
  1. Loads Model 1 predictions from disk (no retraining)
  2. Runs LOLO for Model 2 (Lateral only — Left+Right beads)
  3. Runs LOLO for Model 3 (Size >= 0.10mm only)
  4. Compares all three models

Physical motivation:
  Model 2: Middle beads sit directly under the laser — no lateral
           pool asymmetry → not detectable → remove from training
  Model 3: Beads < 0.10mm cause sub-resolution perturbations
           → not detectable → remove from training

Key: evaluation always uses TRUE original labels so models are
     honestly penalized for missing Middle/Small beads.
"""

import random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from sklearn.metrics import roc_auc_score, f1_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR        = Path("extended_analysis/model_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL1_PREDS_PATH = Path("extended_analysis/lolo_PV/lolo_PV_predictions.csv")

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

ALL_LAYERS     = list(range(226, 232)) + list(range(245, 253))
SEQ_LEN        = 6
IMAGE_SIZE     = (128, 128)
BATCH_SIZE     = 8
MAX_EPOCHS     = 60
PATIENCE       = 10
LR             = 1e-3
WEIGHT_DECAY   = 1e-3
MAX_TRAIN_AUC  = 0.95
SIZE_THRESHOLD = 0.10


# ── MODEL ─────────────────────────────────────────────────────────

class DiffFrameCNN(nn.Module):
    def __init__(self, feature_dim=64, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,32,5,padding=2), nn.BatchNorm2d(32),
            nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64),
            nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(64,64,3,padding=1), nn.BatchNorm2d(64),
            nn.ReLU(True), nn.AdaptiveMaxPool2d((4,4)),
        )
        self.proj = nn.Sequential(
            nn.Flatten(), nn.Linear(64*4*4, feature_dim),
            nn.BatchNorm1d(feature_dim), nn.ReLU(True), nn.Dropout(dropout),
        )
    def forward(self, x): return self.proj(self.encoder(x))


class DiffImageDetectorPV(nn.Module):
    def __init__(self, feature_dim=64, lstm_hidden=64,
                 n_layers=2, pv_hidden=16, dropout=0.35):
        super().__init__()
        self.diff_cnn  = DiffFrameCNN(feature_dim, dropout)
        self.frame_cnn = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.BatchNorm2d(16),
            nn.ReLU(True), nn.MaxPool2d(4),
            nn.Conv2d(16,32,3,padding=1), nn.BatchNorm2d(32),
            nn.ReLU(True), nn.AdaptiveAvgPool2d((4,4)),
            nn.Flatten(), nn.Linear(32*4*4,32),
            nn.ReLU(True), nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(feature_dim, lstm_hidden, n_layers,
                            batch_first=True,
                            dropout=dropout if n_layers>1 else 0.0)
        self.pv_branch = nn.Sequential(
            nn.Linear(2, pv_hidden), nn.ReLU(), nn.Dropout(dropout),
        )
        fused = lstm_hidden*2 + 32 + pv_hidden
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(fused,64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64,1),
        )
    def forward(self, diffs, last_frame, pv):
        B,T,C,H,W = diffs.shape
        f = self.diff_cnn(diffs.view(B*T,C,H,W)).view(B,T,-1)
        out,(h,_) = self.lstm(f)
        fused = torch.cat([h[-1], out.mean(1),
                           self.frame_cnn(last_frame),
                           self.pv_branch(pv)], dim=1)
        return self.classifier(fused)


# ── LABEL DEFINITIONS ─────────────────────────────────────────────

def apply_model2_labels(df):
    df = df.copy()
    mask = (df["balling"]==1) & (df["bead_type"]=="Middle")
    df.loc[mask,"balling"] = 0
    return df, int(mask.sum())

def apply_model3_labels(df):
    df = df.copy()
    sm = ((df["balling"]==1) & df["bead_size_mm"].notna() &
          (df["bead_size_mm"].astype(float) < SIZE_THRESHOLD))
    uk = (df["balling"]==1) & df["bead_size_mm"].isna()
    df.loc[sm,"balling"] = 0
    df.loc[uk,"balling"] = 0
    return df, int(sm.sum())+int(uk.sum())


# ── DATASET ───────────────────────────────────────────────────────

class DiffDatasetPV(Dataset):
    def __init__(self, samples_df, source_df,
                 p_mean=0., p_std=1., v_mean=0., v_std=1., augment=False):
        self.df     = samples_df.reset_index(drop=True)
        self.p_mean = float(p_mean); self.p_std = float(p_std) if p_std!=0 else 1.
        self.v_mean = float(v_mean); self.v_std = float(v_std) if v_std!=0 else 1.
        self.augment = augment
        self.fmap = {}
        for layer,g in source_df.groupby("layer"):
            g = g.sort_values("frame")
            self.fmap[int(layer)] = {int(r.frame):str(r.image_path)
                                     for r in g.itertuples(index=False)}
    def _load(self, path):
        x = TF.pil_to_tensor(Image.open(path)).float()
        x = torch.clamp(x,0,65535)/65535.
        x = TF.resize(x, list(IMAGE_SIZE), antialias=True)
        if x.shape[0]>1: x = x.mean(0,keepdim=True)
        return x
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        layer = int(row["layer"]); start = int(row["start_frame"])
        fmap  = self.fmap[layer]; all_f = sorted(fmap.keys())
        mn,mx = all_f[0],all_f[-1]
        frames = [max(mn,min(mx,start+i)) for i in range(SEQ_LEN)]
        imgs   = [self._load(fmap[f]) for f in frames]
        diffs  = torch.stack([torch.abs(imgs[t]-imgs[t-1])
                               for t in range(1,SEQ_LEN)],0)
        dm = diffs.max()
        if dm>1e-8: diffs = diffs/dm
        last = (imgs[-1]-0.5)/0.5
        if self.augment and random.random()>0.5:
            diffs = torch.flip(diffs,[3]); last = torch.flip(last,[2])
        pv = torch.tensor([(float(row["P"])-self.p_mean)/self.p_std,
                            (float(row["V"])-self.v_mean)/self.v_std],
                           dtype=torch.float32)
        return diffs, last, pv, torch.tensor(float(row["label"]),
                                              dtype=torch.float32)


# ── SAMPLE BUILDERS ───────────────────────────────────────────────

def build_train_samples(layer_df, neg_ratio=1.0, seed=42):
    rng = random.Random(seed); rows = []
    for layer,g in layer_df.groupby("layer"):
        g = g.sort_values("frame").reset_index(drop=True)
        mf,xf = int(g["frame"].min()),int(g["frame"].max())
        p,v   = float(g["P"].iloc[0]),float(g["V"].iloc[0])
        ball  = set(g[g["balling"]==1]["frame"].astype(int))
        half  = SEQ_LEN//2; pos = set()
        for fn in ball:
            ws = max(mf,min(fn-half,xf-SEQ_LEN+1))
            if any(f in ball for f in range(ws,ws+SEQ_LEN)): pos.add(ws)
        for ws in pos:
            rows.append({"layer":int(layer),"start_frame":ws,"label":1,"P":p,"V":v})
        negs = [s for s in range(mf,xf-SEQ_LEN+2)
                if all(f not in ball for f in range(s,s+SEQ_LEN))]
        n_neg = max(int(round(len(pos)*neg_ratio)),1)
        if negs:
            ch = rng.sample(negs,k=min(n_neg,len(negs)))
            while len(ch)<n_neg: ch.append(rng.choice(negs))
            for ws in ch:
                rows.append({"layer":int(layer),"start_frame":ws,"label":0,"P":p,"V":v})
    if not rows: return pd.DataFrame(columns=["layer","start_frame","label","P","V"])
    return pd.DataFrame(rows).sample(frac=1,random_state=seed).reset_index(drop=True)


def build_full_test_samples(layer, enriched_df):
    g    = enriched_df[enriched_df["layer"]==layer].sort_values("frame").reset_index(drop=True)
    mf,xf = int(g["frame"].min()),int(g["frame"].max())
    p,v  = float(g["P"].iloc[0]),float(g["V"].iloc[0])
    ball = set(g[g["balling"]==1]["frame"].astype(int)); rows=[]
    for start in range(mf,xf-SEQ_LEN+2):
        label = int(any(f in ball for f in range(start,start+SEQ_LEN)))
        cf    = start+SEQ_LEN//2
        cr    = g[g["frame"]==cf]
        if cr.empty: cr = g.iloc[[(g["frame"]-cf).abs().argmin()]]
        r = cr.iloc[0]
        rows.append({"layer":int(layer),"start_frame":start,"center_frame":int(cf),
                     "label":label,"P":p,"V":v,"bead_type":r.get("bead_type"),
                     "bead_size":r.get("bead_size_vis"),
                     "bead_size_mm":float(r["bead_size_mm"]) if pd.notna(r.get("bead_size_mm")) else np.nan,
                     "n_beads":int(r.get("n_beads",0))})
    return pd.DataFrame(rows).reset_index(drop=True)


# ── TRAINING UTILITIES ────────────────────────────────────────────

def run_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    all_p,all_t,total_loss = [],[],0.
    for diffs,lf,pv,y in loader:
        diffs,lf,pv = diffs.to(device),lf.to(device),pv.to(device)
        y_dev = y.to(device).unsqueeze(1)
        if is_train: optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            logits = model(diffs,lf,pv)
            loss   = F.binary_cross_entropy_with_logits(logits,y_dev)
            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.0)
                optimizer.step()
        total_loss += loss.item()*diffs.size(0)
        all_p.extend(torch.sigmoid(logits).detach().cpu().numpy().ravel())
        all_t.extend((y.numpy()>=0.5).astype(int))
    pa,ta = np.array(all_p),np.array(all_t)
    try: auc = roc_auc_score(ta,pa)
    except: auc = float("nan")
    return {"probs":pa,"targets":ta,"auc":auc,
            "f1":f1_score(ta,(pa>=0.5).astype(int),zero_division=0),
            "loss":total_loss/max(len(loader.dataset),1)}

def best_threshold(probs,targets):
    best_t,best_f1 = 0.5,-1.
    for t in np.arange(0.05,0.96,0.05):
        f = f1_score(targets,(probs>=t).astype(int),zero_division=0)
        if f>best_f1: best_f1,best_t = f,float(t)
    return best_t,best_f1

def safe_auc(pos,neg):
    if len(pos)<2 or len(neg)<2: return np.nan
    try: return roc_auc_score([1]*len(pos)+[0]*len(neg),list(pos)+list(neg))
    except: return np.nan

def evaluate_strata(pred_df):
    clean = pred_df[pred_df["label"]==0]
    strata = {"Overall":pred_df["label"]==1,
               "Type=Middle":pred_df["bead_type"]=="Middle",
               "Type=Left":pred_df["bead_type"]=="Left",
               "Type=Right":pred_df["bead_type"]=="Right",
               "Size=L":pred_df["bead_size"]=="L",
               "Size=M":pred_df["bead_size"]=="M",
               "Size=S":pred_df["bead_size"]=="S"}
    results = {}
    for name,mask in strata.items():
        pos = pred_df[mask&(pred_df["label"]==1)]
        if len(pos)<2: results[name]={"n_pos":len(pos),"auc":np.nan}; continue
        ev  = pd.concat([pos,clean],ignore_index=True)
        try: auc = roc_auc_score((ev["label"]==1).astype(int).values,ev["pred_prob"].values)
        except: auc = np.nan
        results[name] = {"n_pos":len(pos),"auc":auc}
    return results


# ── ONE LOLO FOLD ─────────────────────────────────────────────────

def run_lolo_fold(held_out, enriched_df, modified_df,
                  p_mean, p_std, v_mean, v_std, ckpt_dir):
    train_layers = [l for l in ALL_LAYERS if l!=held_out]
    val_layer    = random.choice(train_layers)
    pure_tr      = [l for l in train_layers if l!=val_layer]
    ptr_df = modified_df[modified_df["layer"].isin(pure_tr)].reset_index(drop=True)
    pvl_df = modified_df[modified_df["layer"]==val_layer].reset_index(drop=True)
    tr_samp = build_train_samples(ptr_df, seed=42)
    vl_samp = build_train_samples(pvl_df, seed=43)
    n_tr_pos = len(tr_samp[tr_samp["label"]==1])
    n_vl_pos = len(vl_samp[vl_samp["label"]==1])
    skip = n_tr_pos<4 or n_vl_pos<2
    tr_ds = DiffDatasetPV(tr_samp,ptr_df,p_mean,p_std,v_mean,v_std,augment=True)
    vl_ds = DiffDatasetPV(vl_samp,pvl_df,p_mean,p_std,v_mean,v_std,augment=False)
    tr_loader = DataLoader(tr_ds,BATCH_SIZE,shuffle=True,num_workers=0)
    vl_loader = DataLoader(vl_ds,BATCH_SIZE,shuffle=False,num_workers=0)
    model     = DiffImageDetectorPV().to(device)
    ckpt_path = ckpt_dir/f"fold_{held_out}.pth"
    # Always save initial checkpoint — guarantees file exists
    torch.save({"state":model.state_dict(),"thresh":0.5,
                "p_mean":p_mean,"p_std":p_std,"v_mean":v_mean,"v_std":v_std},
               ckpt_path)
    if skip:
        print(f"      WARNING: too few positives (tr={n_tr_pos},vl={n_vl_pos}) — using untrained model")
    else:
        opt   = torch.optim.Adam(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,"max",factor=0.5,patience=5,min_lr=1e-5)
        best_val_auc,no_imp = -1.,0
        for epoch in range(1,MAX_EPOCHS+1):
            tr_m = run_epoch(model,tr_loader,opt)
            vl_m = run_epoch(model,vl_loader)
            t,_  = best_threshold(vl_m["probs"],vl_m["targets"])
            sched.step(vl_m["auc"])
            if not np.isnan(vl_m["auc"]) and vl_m["auc"]>best_val_auc:
                best_val_auc,no_imp = vl_m["auc"],0
                torch.save({"state":model.state_dict(),"thresh":t,
                            "p_mean":p_mean,"p_std":p_std,"v_mean":v_mean,"v_std":v_std},
                           ckpt_path)
            else:
                no_imp+=1
                if no_imp>=PATIENCE: break
            if not np.isnan(tr_m["auc"]) and tr_m["auc"]>=MAX_TRAIN_AUC: break
    ckpt = torch.load(ckpt_path,map_location=device,weights_only=False)
    model.load_state_dict(ckpt["state"])
    te_df   = enriched_df[enriched_df["layer"]==held_out].reset_index(drop=True)
    full    = build_full_test_samples(held_out,enriched_df)
    full_ds = DiffDatasetPV(full,te_df,p_mean,p_std,v_mean,v_std,augment=False)
    fl_load = DataLoader(full_ds,BATCH_SIZE,shuffle=False,num_workers=0)
    model.eval(); probs=[]
    with torch.no_grad():
        for di,lf,pv,_ in fl_load:
            logits = model(di.to(device),lf.to(device),pv.to(device))
            probs.extend(torch.sigmoid(logits).cpu().numpy().ravel())
    full["pred_prob"] = probs
    strata = evaluate_strata(full)
    p_val  = int(te_df["P"].iloc[0]); v_val = int(te_df["V"].iloc[0])
    print(f"    L{held_out} P={p_val}W V={v_val}: AUC={strata['Overall']['auc']:.3f}")
    return {"held_out":held_out,"P":p_val,"V":v_val,
            "overall_auc":strata["Overall"]["auc"],"strata":strata,"predictions":full}


# ── FULL LOLO ─────────────────────────────────────────────────────

def run_lolo(model_name, enriched_df, modified_df, p_mean, p_std, v_mean, v_std):
    print(f"\n{'='*65}\nLOLO: {model_name}\n{'='*65}")
    ckpt_dir = OUTPUT_DIR/f"ckpts_{model_name}"; ckpt_dir.mkdir(exist_ok=True)
    all_results = []
    for layer in sorted(ALL_LAYERS):
        n_true = int((enriched_df[enriched_df["layer"]==layer]["balling"]==1).sum())
        n_mod  = int((modified_df[modified_df["layer"]==layer]["balling"]==1).sum())
        p = int(enriched_df[enriched_df["layer"]==layer]["P"].iloc[0])
        v = int(enriched_df[enriched_df["layer"]==layer]["V"].iloc[0])
        print(f"\n  Fold L{layer} P={p}W V={v} — true_pos={n_true}  train_pos={n_mod}")
        result = run_lolo_fold(layer,enriched_df,modified_df,p_mean,p_std,v_mean,v_std,ckpt_dir)
        all_results.append(result)
        pd.to_pickle(all_results, OUTPUT_DIR/f"{model_name}_intermediate.pkl")
    all_preds = pd.concat([r["predictions"].assign(held_out=r["held_out"])
                           for r in all_results], ignore_index=True)
    all_preds.to_csv(OUTPUT_DIR/f"{model_name}_predictions.csv", index=False)
    return all_results, all_preds


# ── LOAD MODEL 1 FROM DISK ────────────────────────────────────────

def load_model1(enriched_df):
    print(f"\nLoading Model 1 from: {MODEL1_PREDS_PATH}")
    if not MODEL1_PREDS_PATH.exists():
        raise FileNotFoundError(f"Not found: {MODEL1_PREDS_PATH}\nRun lolo_PV.py first.")
    preds = pd.read_csv(MODEL1_PREDS_PATH)
    # Normalise column names
    if "bead_size" not in preds.columns and "bead_size_vis" in preds.columns:
        preds = preds.rename(columns={"bead_size_vis":"bead_size"})
    if "bead_size" not in preds.columns:
        preds["bead_size"] = None
    print(f"  Loaded {len(preds)} rows")
    results = []
    for layer in sorted(preds["held_out"].unique()):
        fold  = preds[preds["held_out"]==layer].copy()
        p_val = int(enriched_df[enriched_df["layer"]==layer]["P"].iloc[0])
        v_val = int(enriched_df[enriched_df["layer"]==layer]["V"].iloc[0])
        strata = evaluate_strata(fold)
        results.append({"held_out":int(layer),"P":p_val,"V":v_val,
                         "overall_auc":strata["Overall"]["auc"],
                         "strata":strata,"predictions":fold})
        print(f"  L{layer} P={p_val}W V={v_val}: AUC={strata['Overall']['auc']:.3f}")
    return results, preds


# ── COMPARISON ────────────────────────────────────────────────────

def compare_models(results_dict, preds_dict):
    print(f"\n{'='*65}\nTHREE-MODEL COMPARISON\n{'='*65}")
    model_names = list(results_dict.keys())
    strata_keys = ["Overall","Type=Middle","Type=Left","Type=Right",
                   "Size=L","Size=M","Size=S"]
    layer_aucs = {mn:{} for mn in model_names}
    for mn,results in results_dict.items():
        for r in results: layer_aucs[mn][r["held_out"]] = r["overall_auc"]

    print(f"\n  Per-layer overall AUC:")
    hdr = f"  {'Layer':>7} {'P':>5} {'V':>6}"
    for mn in model_names: hdr += f"  {mn[:16]:>16}"
    print(hdr); print("  "+"-"*(22+18*len(model_names)))
    for layer in sorted(ALL_LAYERS):
        r0    = results_dict[model_names[0]]
        p_val = next(r["P"] for r in r0 if r["held_out"]==layer)
        v_val = next(r["V"] for r in r0 if r["held_out"]==layer)
        row   = f"  {layer:>7} {p_val:>5} {v_val:>6}"
        for mn in model_names:
            auc = layer_aucs[mn].get(layer,np.nan)
            row += f"  {auc:>16.3f}"
        print(row)
    print(f"\n  {'Mean AUC':>20}", end="")
    for mn in model_names:
        vals = [v for v in layer_aucs[mn].values() if not np.isnan(v)]
        print(f"  {np.mean(vals):>16.3f}", end="")
    print()

    strata_aucs = {}
    for mn,preds in preds_dict.items():
        clean_all = preds[preds["label"]==0]
        strata_aucs[mn] = {}
        for sk in strata_keys:
            if sk=="Overall": pos=preds[preds["label"]==1]
            elif sk.startswith("Type="):
                btype=sk.split("=")[1]
                pos=preds[(preds["label"]==1)&(preds["bead_type"]==btype)]
            else:
                bsize=sk.split("=")[1]
                col = "bead_size" if "bead_size" in preds.columns else "bead_size_vis"
                pos=preds[(preds["label"]==1)&(preds[col]==bsize)]
            strata_aucs[mn][sk] = safe_auc(pos["pred_prob"].values,
                                             clean_all["pred_prob"].values)

    print(f"\n  Per-stratum AUC (combined across all folds):")
    hdr = f"  {'Stratum':<18}"
    for mn in model_names: hdr += f"  {mn[:16]:>16}"
    print(hdr); print("  "+"-"*(20+18*len(model_names)))
    for sk in strata_keys:
        row = f"  {sk:<18}"
        for mn in model_names:
            auc  = strata_aucs[mn].get(sk,np.nan)
            flag = " ✓" if not np.isnan(auc) and auc>0.65 else "  "
            row += f"  {auc:>14.3f}{flag}"
        print(row)

    print(f"\n  Best model per stratum:")
    for sk in strata_keys:
        aucs  = {mn:strata_aucs[mn].get(sk,np.nan) for mn in model_names}
        valid = {mn:a for mn,a in aucs.items() if not np.isnan(a)}
        if not valid: continue
        best_mn  = max(valid,key=valid.get); best_auc=valid[best_mn]
        m1_auc   = strata_aucs[model_names[0]].get(sk,np.nan)
        delta    = best_auc-m1_auc if not np.isnan(m1_auc) else np.nan
        d_str    = f" (+{delta:.3f})" if not np.isnan(delta) and delta>0 \
                   else f" ({delta:.3f})" if not np.isnan(delta) else ""
        print(f"    {sk:<18}: {best_mn} (AUC={best_auc:.3f}{d_str})")

    colors = {model_names[0]:"#1565C0",model_names[1]:"#2E7D32",model_names[2]:"#C62828"}
    fig,axes = plt.subplots(1,3,figsize=(18,6))
    fig.suptitle("Three-model comparison: physically motivated label definitions\n"
                 "M1=All | M2=Lateral only | M3=Size≥0.10mm\n"
                 "Identical architecture — difference is training labels only",
                 fontsize=10,fontweight="bold")
    ax=axes[0]; x=np.arange(len(ALL_LAYERS)); w=0.25
    for i,mn in enumerate(model_names):
        aucs=[layer_aucs[mn].get(l,np.nan) for l in sorted(ALL_LAYERS)]
        ax.bar(x+(i-1)*w,aucs,w,label=mn,color=colors[mn],alpha=0.8,
               edgecolor="black",linewidth=0.5)
    ax.axhline(0.5,color="black",linewidth=1.2,linestyle="--")
    ax.axhline(0.65,color="green",linewidth=1,linestyle=":")
    ax.set_xticks(x); ax.set_xticklabels([f"L{l}" for l in sorted(ALL_LAYERS)],
                                           fontsize=7,rotation=45,ha="right")
    ax.set_ylabel("Overall AUC"); ax.set_ylim(0.3,1.0)
    ax.set_title("Per-layer AUC by model",fontsize=10); ax.legend(fontsize=8)

    ax=axes[1]; x=np.arange(len(strata_keys))
    for i,mn in enumerate(model_names):
        aucs=[strata_aucs[mn].get(sk,np.nan) for sk in strata_keys]
        ax.bar(x+(i-1)*w,aucs,w,label=mn,color=colors[mn],alpha=0.8,
               edgecolor="black",linewidth=0.5)
    ax.axhline(0.5,color="black",linewidth=1.2,linestyle="--")
    ax.axhline(0.65,color="green",linewidth=1,linestyle=":")
    ax.set_xticks(x); ax.set_xticklabels(strata_keys,fontsize=8,rotation=30,ha="right")
    ax.set_ylabel("AUC"); ax.set_ylim(0.3,1.0)
    ax.set_title("Per-stratum AUC by model",fontsize=10); ax.legend(fontsize=8)

    ax=axes[2]; x=np.arange(len(strata_keys))
    for i,mn in enumerate(model_names[1:],start=1):
        deltas=[]
        for sk in strata_keys:
            a1=strata_aucs[model_names[0]].get(sk,np.nan)
            ai=strata_aucs[mn].get(sk,np.nan)
            deltas.append(ai-a1 if not(np.isnan(a1) or np.isnan(ai)) else 0.)
        bar_colors=["#2E7D32" if d>0.01 else "#C62828" if d<-0.01 else "#888888"
                    for d in deltas]
        ax.bar(x+(i-1)*0.35,deltas,0.32,label=f"{mn} vs M1",
               color=bar_colors,alpha=0.85,edgecolor="black",linewidth=0.5)
    ax.axhline(0,color="black",linewidth=1.5,linestyle="--")
    ax.set_xticks(x); ax.set_xticklabels(strata_keys,fontsize=8,rotation=30,ha="right")
    ax.set_ylabel("Delta AUC vs Model 1")
    ax.set_title("Improvement over baseline\nGreen=better, Red=worse",fontsize=10)
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = OUTPUT_DIR/"model_comparison.png"
    plt.savefig(path,dpi=130,bbox_inches="tight"); plt.close()
    print(f"\n  Saved: {path}")


# ── MAIN ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nLoading enriched_df...")
    enriched_df = pd.read_csv(
        r"C:\Users\erfan\Downloads\balling_dataset\enriched_df.csv",
        encoding="utf-8-sig", low_memory=False)
    for col in ["frame","balling","event_id","bead_id_rich","n_beads","layer","P","V"]:
        if col in enriched_df.columns:
            enriched_df[col] = pd.to_numeric(enriched_df[col],errors="coerce").astype("Int64")
    for col in ["bead_size_mm","start_pixel","end_pixel","size_actual"]:
        if col in enriched_df.columns:
            enriched_df[col] = pd.to_numeric(enriched_df[col],errors="coerce").astype(float)
    for col in ["bead_type","bead_size_vis","image_path"]:
        if col in enriched_df.columns:
            enriched_df[col] = enriched_df[col].where(enriched_df[col].astype(str)!="nan",other=None)
    enriched_df["is_multi_bead"] = enriched_df["is_multi_bead"].astype(bool)
    print(f"  {len(enriched_df)} frames  balling={int(enriched_df['balling'].sum())}")
    p_mean=float(enriched_df["P"].mean()); p_std=float(enriched_df["P"].std())
    v_mean=float(enriched_df["V"].mean()); v_std=float(enriched_df["V"].std())
    for s,n in [(p_std,"P"),(v_std,"V")]:
        if pd.isna(s) or s==0:
            if n=="P": p_std=1.
            else: v_std=1.

    m2_df,n2 = apply_model2_labels(enriched_df)
    m3_df,n3 = apply_model3_labels(enriched_df)
    print(f"\nModel 2 (Lateral)  : {int(m2_df['balling'].sum())} positives ({n2} Middle relabeled)")
    print(f"Model 3 (Size≥0.10): {int(m3_df['balling'].sum())} positives ({n3} Small relabeled)")

    results_dict,preds_dict = {},{}

    # Model 1 — load from disk
    m1_results,m1_preds = load_model1(enriched_df)
    results_dict["Model1_All"]     = m1_results
    preds_dict["Model1_All"]       = m1_preds

    # Model 2 — train
    m2_results,m2_preds = run_lolo("Model2_Lateral",enriched_df,m2_df,p_mean,p_std,v_mean,v_std)
    results_dict["Model2_Lateral"] = m2_results
    preds_dict["Model2_Lateral"]   = m2_preds

    # Model 3 — train
    m3_results,m3_preds = run_lolo("Model3_SizeGe10",enriched_df,m3_df,p_mean,p_std,v_mean,v_std)
    results_dict["Model3_SizeGe10"] = m3_results
    preds_dict["Model3_SizeGe10"]   = m3_preds

    compare_models(results_dict,preds_dict)

    print(f"\nOutputs: {OUTPUT_DIR}/")
    for f in sorted(OUTPUT_DIR.iterdir()):
        if f.is_file(): print(f"  {f.name}")
