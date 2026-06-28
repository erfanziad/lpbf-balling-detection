"""
Experiment A SEQ=8 + Augmentation + Hyperparameter Search
==========================================================
Best setup so far: Exp A SEQ=8 (mean AUC=0.720)

Additions:
1. AUGMENTATION (training only, all applied consistently across frames):
   - Horizontal flip (50%): reverses scan direction
   - Brightness/contrast jitter (50%): simulates camera variation
   - Gaussian noise (50%): simulates sensor noise
   - Temporal shift (50%): shifts window by +-1 frame

2. HYPERPARAMETER SEARCH (on L250 fold, 25 epochs each):
   Grid: dropout=[0.25,0.35,0.45], lstm_hidden=[64,128], lr=[5e-4,1e-3]
   Best params used for all 14 folds.

3. THRESHOLD TUNING:
   Per-fold optimal F1 threshold reported alongside AUC.

Output: extended_analysis/lolo_enhanced/
"""

import random, numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
from itertools import product

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from sklearn.metrics import roc_auc_score, f1_score
from tqdm import tqdm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR   = Path("extended_analysis")
LOLO_DIR     = OUTPUT_DIR / "lolo_enhanced"
LOLO_DIR.mkdir(parents=True, exist_ok=True)
BASELINE_CSV = OUTPUT_DIR / "lolo_PV" / "lolo_PV_predictions.csv"
EXPA_CSV     = OUTPUT_DIR / "lolo_experiment_len8" / "predictions.csv"
PROFIL_CSV   = Path(r"C:\Users\erfan\Downloads\qq_exp3_c6.csv")
TRACK_CSV_DIR= Path(r"C:\Users\erfan\Downloads\Erfan_balling_data_updated 2\Erfan_balling_data_updated")

def seed_everything(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_everything()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

ALL_LAYERS=list(range(226,232))+list(range(245,253))
SEQ_LEN=8; IMAGE_SIZE=(128,128); BATCH_SIZE=8
MAX_EPOCHS=60; PATIENCE=10; MAX_TRAIN_AUC=0.95
TRIM_START=0; TRIM_END=0
AUG_BRIGHTNESS=0.15; AUG_CONTRAST=0.15
AUG_NOISE_STD=0.02; AUG_TEMP_SHIFT=1
PX_X_MM=0.00789; PX_Y_MM=0.01250
PATCH_HALF_X=25; PATCH_HALF_Y=16; PATCH_OUT=16

PROFIL_LAYER_CONFIG={
    248:{"x_offset":25.5440,"row_center":62, "half_band":20},
    249:{"x_offset":25.5740,"row_center":143,"half_band":20},
    250:{"x_offset":25.4540,"row_center":221,"half_band":20},
    251:{"x_offset":25.4400,"row_center":301,"half_band":20},
    252:{"x_offset":25.5140,"row_center":379,"half_band":20},
}
PROFIL_LAYERS=set(PROFIL_LAYER_CONFIG.keys())

PARAM_GRID={"dropout":[0.25,0.35,0.45],"lstm_hidden":[64,128],
            "lr":[5e-4,1e-3],"weight_decay":[1e-3]}

# ── Profilometry ──────────────────────────────────────────────────
def build_height_lookup():
    if not PROFIL_CSV.exists():
        print("  WARNING: profil CSV not found"); return None,{},{}
    print("Loading profilometry CSV...")
    Z=pd.read_csv(PROFIL_CSV,header=None).values.astype(float)
    nr,nc=Z.shape; print(f"  {nr}x{nc}")
    lookup={}; frame_xy={}
    for layer,cfg in PROFIL_LAYER_CONFIG.items():
        rc=cfg["row_center"]; hb=cfg["half_band"]
        band=Z[max(0,rc-hb):min(nr-1,rc+hb)+1,:]
        hm=float(np.nanmean(band)); hs=float(np.nanstd(band))
        if hs<1e-8: hs=1.0
        lookup[layer]={"h_mean":hm,"h_std":hs,"n_rows":nr,"n_cols":nc}
        print(f"  L{layer}: rc={rc} h={hm*1000:.1f}um std={hs*1000:.1f}um")
        csv=TRACK_CSV_DIR/f"L0{layer}.csv"
        if not csv.exists(): continue
        df_t=pd.read_csv(csv); df_t.columns=[c.strip() for c in df_t.columns]
        for _,row in df_t.iterrows():
            fn=int(row["frame_number"]); xm=float(row["x(mm)"])
            xp=xm-cfg["x_offset"]; cc=int(round(xp/PX_X_MM))
            frame_xy[(layer,fn)]=(cc,rc)
    return Z,lookup,frame_xy

def extract_height_patch(Z,col_c,row_c,h_mean,h_std,n_rows,n_cols):
    zero=torch.zeros(1,PATCH_OUT,PATCH_OUT,dtype=torch.float32)
    r_lo=row_c-PATCH_HALF_Y; r_hi=row_c+PATCH_HALF_Y+1
    c_lo=col_c-PATCH_HALF_X; c_hi=col_c+PATCH_HALF_X+1
    if r_lo<0 or r_hi>n_rows or c_lo<0 or c_hi>n_cols: return zero
    patch=Z[r_lo:r_hi,c_lo:c_hi].copy()
    patch=np.where(np.isnan(patch),h_mean,patch)
    patch=(patch-h_mean)/h_std
    p_t=torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0)
    p_t=F.interpolate(p_t,size=(PATCH_OUT,PATCH_OUT),mode="bilinear",align_corners=False)
    return p_t.squeeze(0)

# ── Model ─────────────────────────────────────────────────────────
class HeightPatchCNN(nn.Module):
    def __init__(self,height_dim=16,dropout=0.3):
        super().__init__()
        self.enc=nn.Sequential(
            nn.Conv2d(1,16,3,padding=1),nn.BatchNorm2d(16),nn.ReLU(True),
            nn.Conv2d(16,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(True),
            nn.AdaptiveAvgPool2d((4,4)),nn.Flatten(),
            nn.Linear(32*16,height_dim),nn.ReLU(True),nn.Dropout(dropout))
    def forward(self,x): return self.enc(x)

class DiffFrameCNN(nn.Module):
    def __init__(self,feature_dim=64,dropout=0.3):
        super().__init__()
        self.enc=nn.Sequential(
            nn.Conv2d(1,32,5,padding=2),nn.BatchNorm2d(32),nn.ReLU(True),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(True),nn.MaxPool2d(2),
            nn.Conv2d(64,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(True),nn.AdaptiveMaxPool2d((4,4)))
        self.proj=nn.Sequential(
            nn.Flatten(),nn.Linear(64*16,feature_dim),
            nn.BatchNorm1d(feature_dim),nn.ReLU(True),nn.Dropout(dropout))
    def forward(self,x): return self.proj(self.enc(x))

def build_model(lstm_hidden=64,dropout=0.35,height_dim=16,pv_hidden=16,feature_dim=64,n_layers=2):
    class Det(nn.Module):
        def __init__(self):
            super().__init__()
            self.diff_cnn=DiffFrameCNN(feature_dim,dropout)
            self.frame_cnn=nn.Sequential(
                nn.Conv2d(1,16,3,padding=1),nn.BatchNorm2d(16),nn.ReLU(True),nn.MaxPool2d(4),
                nn.Conv2d(16,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(True),nn.AdaptiveAvgPool2d((4,4)),
                nn.Flatten(),nn.Linear(32*16,32),nn.ReLU(True),nn.Dropout(dropout))
            self.height_cnn=HeightPatchCNN(height_dim,dropout)
            self.lstm=nn.LSTM(feature_dim,lstm_hidden,n_layers,batch_first=True,
                              dropout=dropout if n_layers>1 else 0.)
            self.pv_branch=nn.Sequential(nn.Linear(2,pv_hidden),nn.ReLU(),nn.Dropout(dropout))
            fused=lstm_hidden*2+32+height_dim+pv_hidden+1
            self.clf=nn.Sequential(nn.Dropout(dropout),nn.Linear(fused,64),nn.ReLU(),
                                   nn.Dropout(dropout),nn.Linear(64,1))
        def forward(self,diffs,last_frame,pv,height_patch,has_height):
            B,T,C,H,W=diffs.shape
            f=self.diff_cnn(diffs.view(B*T,C,H,W)).view(B,T,-1)
            out,(h,_)=self.lstm(f)
            ff=self.frame_cnn(last_frame)
            hf=self.height_cnn(height_patch)*has_height
            pvf=self.pv_branch(pv)
            return self.clf(torch.cat([h[-1],out.mean(1),ff,hf,pvf,has_height],dim=1))
    return Det()

# ── Dataset with augmentation ─────────────────────────────────────
class AugDataset(Dataset):
    def __init__(self,samples_df,source_df,p_mean,p_std,v_mean,v_std,
                 Z,height_lookup,frame_xy,augment=False):
        self.df=samples_df.reset_index(drop=True)
        self.p_mean=float(p_mean); self.p_std=float(p_std) if p_std!=0 else 1.
        self.v_mean=float(v_mean); self.v_std=float(v_std) if v_std!=0 else 1.
        self.Z=Z; self.hl=height_lookup; self.fxy=frame_xy; self.aug=augment
        self.fmap={}
        for layer,g in source_df.groupby("layer"):
            g=g.sort_values("frame")
            self.fmap[int(layer)]={int(r.frame):str(r.image_path)
                                   for r in g.itertuples(index=False)}
    def _load(self,path):
        img=Image.open(path)
        x=TF.pil_to_tensor(img).float()
        x=torch.clamp(x,0,65535)/65535.
        x=TF.resize(x,list(IMAGE_SIZE),antialias=True)
        if x.shape[0]>1: x=x.mean(dim=0,keepdim=True)
        return x
    def _get_patch(self,layer,cf):
        zero=torch.zeros(1,PATCH_OUT,PATCH_OUT)
        if self.Z is None or layer not in self.hl: return zero
        key=(layer,cf)
        if key not in self.fxy: return zero
        cc,rc=self.fxy[key]; cfg=self.hl[layer]
        return extract_height_patch(self.Z,cc,rc,
            cfg["h_mean"],cfg["h_std"],cfg["n_rows"],cfg["n_cols"])
    def _augment(self,imgs):
        do_flip=self.aug and random.random()>0.5
        do_bc=self.aug and random.random()>0.5
        do_noise=self.aug and random.random()>0.5
        br=1.0+random.uniform(-AUG_BRIGHTNESS,AUG_BRIGHTNESS)
        co=1.0+random.uniform(-AUG_CONTRAST,AUG_CONTRAST)
        out=[]
        for x in imgs:
            if do_flip: x=torch.flip(x,dims=[2])
            if do_bc:
                x=torch.clamp(x*br,0,1)
                m=x.mean(); x=torch.clamp((x-m)*co+m,0,1)
            if do_noise: x=torch.clamp(x+torch.randn_like(x)*AUG_NOISE_STD,0,1)
            out.append(x)
        return out
    def __len__(self): return len(self.df)
    def __getitem__(self,idx):
        row=self.df.iloc[idx]
        layer=int(row["layer"]); start=int(row["start_frame"])
        fmap=self.fmap[layer]; all_f=sorted(fmap.keys()); mn,mx=all_f[0],all_f[-1]
        shift=random.choice([-AUG_TEMP_SHIFT,AUG_TEMP_SHIFT]) \
              if self.aug and random.random()>0.5 else 0
        frames=[max(mn,min(mx,start+i+shift)) for i in range(SEQ_LEN)]
        imgs=self._augment([self._load(fmap[f]) for f in frames])
        diffs=torch.stack([torch.abs(imgs[t]-imgs[t-1]) for t in range(1,SEQ_LEN)],dim=0)
        d_max=diffs.max()
        if d_max>1e-8: diffs=diffs/d_max
        last=(imgs[-1]-0.5)/0.5
        pv=torch.tensor([(float(row["P"])-self.p_mean)/self.p_std,
                          (float(row["V"])-self.v_mean)/self.v_std],dtype=torch.float32)
        cf=frames[SEQ_LEN//2]; hp=self._get_patch(layer,cf)
        has_h=torch.tensor([[1.0]] if layer in PROFIL_LAYERS else [[0.0]],
                            dtype=torch.float32).squeeze(0)
        y=torch.tensor(float(row["label"]),dtype=torch.float32)
        return diffs,last,pv,hp,has_h,y

# ── Sample builders ───────────────────────────────────────────────
def get_valid_frames(g):
    all_f=sorted(g["frame"].astype(int).tolist())
    if TRIM_START==0 and TRIM_END==0: return set(all_f)
    if len(all_f)<=TRIM_START+TRIM_END: return set(all_f)
    return set(all_f[TRIM_START:len(all_f)-TRIM_END if TRIM_END>0 else len(all_f)])

def build_samples(split_df,neg_ratio=1.0,seed=42):
    rng=random.Random(seed); rows=[]
    for layer,g in split_df.groupby("layer"):
        g=g.sort_values("frame").reset_index(drop=True)
        valid=get_valid_frames(g); all_f=sorted(valid)
        if len(all_f)<SEQ_LEN: continue
        mf=all_f[0]; xf=all_f[-1]
        p=float(g["P"].iloc[0]); v=float(g["V"].iloc[0])
        ball=set(g[g["balling"]==1]["frame"].astype(int))&valid
        half=SEQ_LEN//2; pos=set()
        for fn in ball:
            ws=max(mf,min(fn-half,xf-SEQ_LEN+1))
            seq=range(ws,ws+SEQ_LEN)
            if all(f in valid for f in seq) and any(f in ball for f in seq):
                pos.add(ws)
        for ws in pos:
            rows.append({"layer":int(layer),"start_frame":ws,"label":1,"P":p,"V":v})
        negs=[s for s in range(mf,xf-SEQ_LEN+2)
              if all(f in valid for f in range(s,s+SEQ_LEN)) and
                 all(f not in ball for f in range(s,s+SEQ_LEN))]
        n_neg=max(int(round(len(pos)*neg_ratio)),1)
        if negs:
            ch=rng.sample(negs,k=min(n_neg,len(negs)))
            while len(ch)<n_neg: ch.append(rng.choice(negs))
            for ws in ch:
                rows.append({"layer":int(layer),"start_frame":ws,"label":0,"P":p,"V":v})
    return pd.DataFrame(rows).sample(frac=1,random_state=seed).reset_index(drop=True)

def build_full_samples(layer_df):
    rows=[]
    for layer,g in layer_df.groupby("layer"):
        g=g.sort_values("frame").reset_index(drop=True)
        valid=get_valid_frames(g); all_f=sorted(valid)
        if len(all_f)<SEQ_LEN: continue
        mf=all_f[0]; xf=all_f[-1]
        p=float(g["P"].iloc[0]); v=float(g["V"].iloc[0])
        ball=set(g[g["balling"]==1]["frame"].astype(int))&valid
        for start in range(mf,xf-SEQ_LEN+2):
            if not all(f in valid for f in range(start,start+SEQ_LEN)): continue
            label=int(any(f in ball for f in range(start,start+SEQ_LEN)))
            cf=start+SEQ_LEN//2
            cr=g[g["frame"]==cf]
            if cr.empty: cr=g.iloc[[(g["frame"]-cf).abs().argmin()]]
            r=cr.iloc[0]
            rows.append({"layer":int(layer),"start_frame":start,"center_frame":int(cf),
                          "label":label,"P":p,"V":v,
                          "bead_type":r.get("bead_type"),"bead_size":r.get("bead_size_vis"),
                          "bead_size_mm":float(r["bead_size_mm"]) if pd.notna(r.get("bead_size_mm")) else np.nan})
    return pd.DataFrame(rows).reset_index(drop=True)

# ── Training utils ────────────────────────────────────────────────
def best_threshold(probs,targets):
    best_t,best_f1=0.5,-1.
    for t in np.arange(0.05,0.96,0.05):
        f=f1_score(targets,(probs>=t).astype(int),zero_division=0)
        if f>best_f1: best_f1,best_t=f,float(t)
    return best_t,best_f1

def run_epoch(model,loader,optimizer=None,desc=""):
    is_train=optimizer is not None
    model.train() if is_train else model.eval()
    all_p,all_t=[],[]; total_loss=0.
    pbar=tqdm(loader,desc=desc,leave=False,ncols=80,disable=not is_train)
    for diffs,lf,pv,hp,hg,y in pbar:
        diffs=diffs.to(device); lf=lf.to(device)
        pv=pv.to(device); hp=hp.to(device); hg=hg.to(device)
        y_dev=y.to(device).unsqueeze(1)
        if is_train: optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            logits=model(diffs,lf,pv,hp,hg)
            loss=F.binary_cross_entropy_with_logits(logits,y_dev)
            if is_train:
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step(); pbar.set_postfix(loss=f"{loss.item():.3f}")
        total_loss+=loss.item()*diffs.size(0)
        all_p.extend(torch.sigmoid(logits).detach().cpu().numpy().ravel())
        all_t.extend((y.numpy()>=0.5).astype(int))
    pa,ta=np.array(all_p),np.array(all_t)
    try: auc=roc_auc_score(ta,pa)
    except: auc=float("nan")
    return {"probs":pa,"targets":ta,"auc":auc,
            "f1":f1_score(ta,(pa>=0.5).astype(int),zero_division=0),
            "loss":total_loss/max(len(loader.dataset),1)}

def train_one(model,tr_loader,vl_loader,lr,wd,ckpt_path,verbose=True,max_ep=None):
    opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=wd)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,"max",factor=0.5,patience=5,min_lr=1e-5)
    best_v,no_imp,best_t=(-1.,0,0.5)
    torch.save({"state":model.state_dict(),"thresh":0.5},ckpt_path)
    ep_max=max_ep or MAX_EPOCHS
    for epoch in range(1,ep_max+1):
        tr_m=run_epoch(model,tr_loader,opt,desc=f"Ep{epoch:02d} train")
        vl_m=run_epoch(model,vl_loader,desc=f"Ep{epoch:02d} val  ")
        t,_=best_threshold(vl_m["probs"],vl_m["targets"])
        sched.step(vl_m["auc"])
        improved=not np.isnan(vl_m["auc"]) and vl_m["auc"]>best_v
        if verbose:
            print(f"    Ep{epoch:02d}: tr={tr_m['auc']:.3f} vl={vl_m['auc']:.3f} "
                  f"f1={vl_m['f1']:.3f} no_imp={no_imp}{' *' if improved else ''}",flush=True)
        if improved:
            best_v,no_imp,best_t=vl_m["auc"],0,t
            torch.save({"state":model.state_dict(),"thresh":t},ckpt_path)
        else:
            no_imp+=1
            if no_imp>=PATIENCE:
                if verbose: print(f"    Early stop ep{epoch}")
                break
        if not np.isnan(tr_m["auc"]) and tr_m["auc"]>=MAX_TRAIN_AUC:
            if verbose: print(f"    Train ceiling ep{epoch}")
            break
    return best_v,best_t

def evaluate_strata(pred_df):
    clean=pred_df[pred_df["label"]==0]
    strata={"Overall":pred_df["label"]==1,"Type=Middle":pred_df["bead_type"]=="Middle",
            "Type=Left":pred_df["bead_type"]=="Left","Type=Right":pred_df["bead_type"]=="Right",
            "Size=L":pred_df["bead_size"]=="L","Size=M":pred_df["bead_size"]=="M",
            "Size=S":pred_df["bead_size"]=="S"}
    results={}
    for name,mask in strata.items():
        pos=pred_df[mask&(pred_df["label"]==1)]
        if len(pos)<2: results[name]={"n_pos":len(pos),"auc":np.nan}; continue
        ev=pd.concat([pos,clean],ignore_index=True)
        try: auc=roc_auc_score((ev["label"]==1).astype(int).values,ev["pred_prob"].values)
        except: auc=np.nan
        results[name]={"n_pos":len(pos),"auc":auc}
    return results

# ── Hyperparameter search ─────────────────────────────────────────
def hyperparameter_search(enriched_df,Z_profil,height_lookup,frame_xy):
    print(f"\n{'='*60}\nHYPERPARAMETER SEARCH\n{'='*60}")
    keys=list(PARAM_GRID.keys())
    combos=list(product(*[PARAM_GRID[k] for k in keys]))
    print(f"Combinations: {len(combos)}")

    search_layer=250
    train_layers=[l for l in ALL_LAYERS if l!=search_layer]
    train_df=enriched_df[enriched_df["layer"].isin(train_layers)].reset_index(drop=True)
    p_mean=float(train_df["P"].mean()); p_std=float(train_df["P"].std()) or 1.
    v_mean=float(train_df["V"].mean()); v_std=float(train_df["V"].std()) or 1.

    def pv_dist(l,tl=search_layer):
        lp=float(enriched_df[enriched_df["layer"]==l]["P"].iloc[0])
        lv=float(enriched_df[enriched_df["layer"]==l]["V"].iloc[0])
        tp=float(enriched_df[enriched_df["layer"]==tl]["P"].iloc[0])
        tv=float(enriched_df[enriched_df["layer"]==tl]["V"].iloc[0])
        return ((lp-tp)/140)**2+((lv-tv)/200)**2

    val_layer=min(train_layers,key=pv_dist)
    pure_tr=[l for l in train_layers if l!=val_layer]
    tr_df_s=enriched_df[enriched_df["layer"].isin(pure_tr)].reset_index(drop=True)
    vl_df_s=enriched_df[enriched_df["layer"]==val_layer].reset_index(drop=True)
    tr_samp=build_samples(tr_df_s,seed=42); vl_samp=build_samples(vl_df_s,seed=43)
    kw=dict(p_mean=p_mean,p_std=p_std,v_mean=v_mean,v_std=v_std,
            Z=Z_profil,height_lookup=height_lookup,frame_xy=frame_xy)

    best_auc=-1.; best_params={k:PARAM_GRID[k][0] for k in keys}
    for combo in combos:
        params=dict(zip(keys,combo))
        print(f"  {params} ...",end=" ",flush=True)
        tr_ds=AugDataset(tr_samp,tr_df_s,**kw,augment=True)
        vl_ds=AugDataset(vl_samp,vl_df_s,**kw,augment=False)
        tr_l=DataLoader(tr_ds,BATCH_SIZE,shuffle=True,num_workers=0)
        vl_l=DataLoader(vl_ds,BATCH_SIZE,shuffle=False,num_workers=0)
        model=build_model(lstm_hidden=params["lstm_hidden"],dropout=params["dropout"]).to(device)
        ckpt=LOLO_DIR/f"search_tmp.pth"
        bv,_=train_one(model,tr_l,vl_l,params["lr"],params["weight_decay"],
                       ckpt,verbose=False,max_ep=25)
        print(f"val_auc={bv:.3f}")
        if bv>best_auc: best_auc=bv; best_params=params.copy()

    print(f"\n  Best: {best_params}  val_auc={best_auc:.3f}")
    return best_params

# ── LOLO fold ─────────────────────────────────────────────────────
def train_fold(held_out,enriched_df,Z_profil,height_lookup,frame_xy,best_params):
    train_layers=[l for l in ALL_LAYERS if l!=held_out]
    train_df=enriched_df[enriched_df["layer"].isin(train_layers)].reset_index(drop=True)
    p_mean=float(train_df["P"].mean()); p_std=float(train_df["P"].std()) or 1.
    v_mean=float(train_df["V"].mean()); v_std=float(train_df["V"].std()) or 1.

    def pv_dist(l):
        lp=float(enriched_df[enriched_df["layer"]==l]["P"].iloc[0])
        lv=float(enriched_df[enriched_df["layer"]==l]["V"].iloc[0])
        tp=float(enriched_df[enriched_df["layer"]==held_out]["P"].iloc[0])
        tv=float(enriched_df[enriched_df["layer"]==held_out]["V"].iloc[0])
        return ((lp-tp)/140)**2+((lv-tv)/200)**2

    val_layer=min(train_layers,key=pv_dist)
    pure_tr=[l for l in train_layers if l!=val_layer]
    print(f"    Val:L{val_layer} dropout={best_params['dropout']} "
          f"lstm={best_params['lstm_hidden']} lr={best_params['lr']}")

    tr_df=enriched_df[enriched_df["layer"].isin(pure_tr)].reset_index(drop=True)
    vl_df=enriched_df[enriched_df["layer"]==val_layer].reset_index(drop=True)
    te_df=enriched_df[enriched_df["layer"]==held_out].reset_index(drop=True)
    tr_samp=build_samples(tr_df,seed=42); vl_samp=build_samples(vl_df,seed=43)
    print(f"    Train:{len(tr_samp)} Val:{len(vl_samp)}")

    kw=dict(p_mean=p_mean,p_std=p_std,v_mean=v_mean,v_std=v_std,
            Z=Z_profil,height_lookup=height_lookup,frame_xy=frame_xy)
    tr_ds=AugDataset(tr_samp,tr_df,**kw,augment=True)
    vl_ds=AugDataset(vl_samp,vl_df,**kw,augment=False)
    tr_l=DataLoader(tr_ds,BATCH_SIZE,shuffle=True,num_workers=0)
    vl_l=DataLoader(vl_ds,BATCH_SIZE,shuffle=False,num_workers=0)

    model=build_model(lstm_hidden=best_params["lstm_hidden"],
                      dropout=best_params["dropout"]).to(device)
    ckpt=LOLO_DIR/f"fold_{held_out}.pth"
    train_one(model,tr_l,vl_l,best_params["lr"],
              best_params["weight_decay"],ckpt,verbose=True)

    ckpt_d=torch.load(ckpt,map_location=device,weights_only=False)
    model.load_state_dict(ckpt_d["state"]); thresh=ckpt_d.get("thresh",0.5)

    full=build_full_samples(te_df)
    full_ds=AugDataset(full,te_df,**kw,augment=False)
    fl_l=DataLoader(full_ds,BATCH_SIZE,shuffle=False,num_workers=0)

    model.eval(); probs=[]
    with torch.no_grad():
        for di,lf,pv,hp,hg,_ in fl_l:
            logits=model(di.to(device),lf.to(device),pv.to(device),hp.to(device),hg.to(device))
            probs.extend(torch.sigmoid(logits).cpu().numpy().ravel())

    full["pred_prob"]=probs
    try: overall_auc=roc_auc_score(full["label"],full["pred_prob"])
    except: overall_auc=float("nan")
    overall_f1=f1_score(full["label"],(np.array(probs)>=thresh).astype(int),zero_division=0)
    strata=evaluate_strata(full)
    p_val=int(te_df["P"].iloc[0]); v_val=int(te_df["V"].iloc[0])
    n_ball=int((te_df["balling"]==1).sum()); has_h=held_out in PROFIL_LAYERS

    print(f"  L{held_out} P={p_val}W V={v_val} n_ball={n_ball} "
          f"height={'YES' if has_h else 'no'}: "
          f"AUC={overall_auc:.3f} F1={overall_f1:.3f}(t={thresh:.2f})")
    for s,r in strata.items():
        if not np.isnan(r["auc"]) and r["n_pos"]>=2:
            print(f"    {s:<18} n={r['n_pos']:>3}  AUC={r['auc']:.3f}")

    return {"held_out":held_out,"P":p_val,"V":v_val,
            "overall_auc":overall_auc,"overall_f1":overall_f1,
            "threshold":thresh,"has_height":has_h,
            "strata":strata,"predictions":full,"best_params":best_params}

# ── Comparison ────────────────────────────────────────────────────
def compare_results(new_results):
    print(f"\n{'='*70}")
    print("COMPARISON: Enhanced (Aug+Tune) vs Baseline vs Exp-A-SEQ8")
    print(f"{'='*70}")
    old_auc={}
    if BASELINE_CSV.exists():
        old_df=pd.read_csv(BASELINE_CSV)
        for layer in ALL_LAYERS:
            sub=old_df[old_df["held_out"]==layer]
            if sub.empty: continue
            ev=pd.concat([sub[sub["label"]==0],sub[sub["label"]==1]],ignore_index=True)
            try: old_auc[layer]=roc_auc_score((ev["label"]==1).astype(int).values,ev["pred_prob"].values)
            except: pass
    expa_auc={}
    if EXPA_CSV.exists():
        expa_df=pd.read_csv(EXPA_CSV)
        if "held_out" in expa_df.columns:
            for layer in ALL_LAYERS:
                sub=expa_df[expa_df["held_out"]==layer]
                if sub.empty: continue
                ev=pd.concat([sub[sub["label"]==0],sub[sub["label"]==1]],ignore_index=True)
                try: expa_auc[layer]=roc_auc_score((ev["label"]==1).astype(int).values,ev["pred_prob"].values)
                except: pass

    print(f"\n  {'Layer':>7} {'Baseline':>10} {'ExpA-S8':>9} {'Enhanced':>10} {'Delta':>8} {'F1':>6} {'t':>5}")
    print("  "+"-"*65)
    deltas=[]; dh=[]; dnh=[]
    for r in sorted(new_results,key=lambda x:x["held_out"]):
        layer=r["held_out"]; new_auc=r["overall_auc"]
        old_a=old_auc.get(layer,np.nan); exp_a=expa_auc.get(layer,np.nan)
        delta=new_auc-old_a if not(np.isnan(new_auc) or np.isnan(old_a)) else np.nan
        marker=" [H]" if r["has_height"] else "    "
        print(f"  {layer:>7}{marker} {old_a:>8.3f}  {exp_a:>7.3f}  "
              f"{new_auc:>8.3f}  {delta:>+7.3f}  "
              f"{r.get('overall_f1',np.nan):>5.3f}  {r.get('threshold',0.5):>4.2f}")
        if not np.isnan(delta):
            deltas.append(delta)
            (dh if r["has_height"] else dnh).append(delta)

    aucs=[r["overall_auc"] for r in new_results if not np.isnan(r["overall_auc"])]
    print(f"\n  All:  mean={np.mean(deltas):+.3f}  ({sum(d>0 for d in deltas)}/{len(deltas)} improved)")
    if dh: print(f"  [H]:  mean={np.mean(dh):+.3f}  ({sum(d>0 for d in dh)}/{len(dh)} improved)")
    if dnh: print(f"  no-H: mean={np.mean(dnh):+.3f}  ({sum(d>0 for d in dnh)}/{len(dnh)} improved)")
    print(f"\n  Mean AUC enhanced: {np.mean(aucs):.3f}")
    print(f"  Exp A SEQ=8:       0.720")
    print(f"  Baseline:          0.673")

# ── Main ──────────────────────────────────────────────────────────
if __name__=="__main__":
    print("Loading enriched_df...")
    enriched_df=pd.read_csv(r"C:\Users\erfan\Downloads\balling_dataset\enriched_df.csv",
                             encoding="utf-8-sig",low_memory=False)
    for col in ["frame","balling","event_id","bead_id_rich","n_beads","layer","P","V"]:
        if col in enriched_df.columns:
            enriched_df[col]=pd.to_numeric(enriched_df[col],errors="coerce").astype("Int64")
    for col in ["bead_size_mm","start_pixel","end_pixel","size_actual"]:
        if col in enriched_df.columns:
            enriched_df[col]=pd.to_numeric(enriched_df[col],errors="coerce").astype(float)
    for col in ["bead_type","bead_size_vis","image_path"]:
        if col in enriched_df.columns:
            enriched_df[col]=enriched_df[col].where(enriched_df[col].astype(str)!="nan",other=None)
    enriched_df["is_multi_bead"]=enriched_df["is_multi_bead"].astype(bool)
    print(f"  {len(enriched_df)} frames")

    print("\nBuilding profilometry...")
    Z_profil,height_lookup,frame_xy=build_height_lookup()

    print(f"\nAugmentation: hflip | brightness+-{AUG_BRIGHTNESS*100:.0f}% | "
          f"noise_std={AUG_NOISE_STD} | temp_shift+-{AUG_TEMP_SHIFT}")

    best_params=hyperparameter_search(enriched_df,Z_profil,height_lookup,frame_xy)

    print(f"\n{'='*65}\nLOLO — 14 FOLDS  Best params:{best_params}\n{'='*65}")
    all_results=[]
    for held_out in sorted(ALL_LAYERS):
        p=int(enriched_df[enriched_df["layer"]==held_out]["P"].iloc[0])
        v=int(enriched_df[enriched_df["layer"]==held_out]["V"].iloc[0])
        n=int((enriched_df[enriched_df["layer"]==held_out]["balling"]==1).sum())
        print(f"\nFold: L{held_out} P={p}W V={v} n_ball={n}")
        result=train_fold(held_out,enriched_df,Z_profil,height_lookup,frame_xy,best_params)
        all_results.append(result)
        pd.to_pickle(all_results,LOLO_DIR/"intermediate.pkl")

    pd.concat([r["predictions"].assign(held_out=r["held_out"])
               for r in all_results],ignore_index=True
              ).to_csv(LOLO_DIR/"predictions.csv",index=False)

    compare_results(all_results)
    print(f"\nOutputs: {LOLO_DIR}/")
