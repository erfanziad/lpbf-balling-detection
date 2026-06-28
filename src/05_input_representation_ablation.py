"""
Process Parameter Ablation Study
==================================
Tests 4 variants to measure contribution of P and V:

  Variant 1: P+V   — both power and speed (current best)
  Variant 2: P     — power only, V zeroed out
  Variant 3: V     — speed only, P zeroed out
  Variant 4: none  — no process params (visual signal only)

All variants use the same architecture and hyperparams as
the best experiment (SEQ=8, gated height patch, smart val,
dropout=0.45, lstm_hidden=128, lr=5e-4).

Zeroing a param means setting its normalized value to 0.0
which equals the training mean — the model receives no
information about that parameter.

Research questions:
  1. How much does each process parameter contribute to AUC?
  2. Which layers benefit most from P vs V?
  3. Is the model primarily visual or process-condition-aware?
  4. Do V=1800 and V=2000 layers respond differently?

Output: extended_analysis/pv_ablation/
"""

import random, numpy as np, pandas as pd
from pathlib import Path
from PIL import Image

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from sklearn.metrics import roc_auc_score, f1_score
from tqdm import tqdm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR   = Path("extended_analysis")
ABLATION_DIR = OUTPUT_DIR / "pv_ablation"
ABLATION_DIR.mkdir(parents=True, exist_ok=True)

PROFIL_CSV    = Path(r"C:\Users\erfan\Downloads\qq_exp3_c6.csv")
TRACK_CSV_DIR = Path(r"C:\Users\erfan\Downloads\Erfan_balling_data_updated 2\Erfan_balling_data_updated")

def seed_everything(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_everything()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

ALL_LAYERS    = list(range(226,232))+list(range(245,253))
SEQ_LEN=8; IMAGE_SIZE=(128,128); BATCH_SIZE=8
MAX_EPOCHS=60; PATIENCE=10; MAX_TRAIN_AUC=0.95
LR=1e-3; WEIGHT_DECAY=1e-3; DROPOUT=0.35; LSTM_HIDDEN=64
PX_X_MM=0.00789; PATCH_HALF_X=25; PATCH_HALF_Y=16; PATCH_OUT=16

PROFIL_LAYER_CONFIG={
    248:{"x_offset":25.5440,"row_center":62, "half_band":20},
    249:{"x_offset":25.5740,"row_center":143,"half_band":20},
    250:{"x_offset":25.4540,"row_center":221,"half_band":20},
    251:{"x_offset":25.4400,"row_center":301,"half_band":20},
    252:{"x_offset":25.5140,"row_center":379,"half_band":20},
}
PROFIL_LAYERS=set(PROFIL_LAYER_CONFIG.keys())

VARIANTS={
    "PV"  :{"use_P":True, "use_V":True },
    "P"   :{"use_P":True, "use_V":False},
    "V"   :{"use_P":False,"use_V":True },
    "none":{"use_P":False,"use_V":False},
}

# ── Profilometry ──────────────────────────────────────────────────
def build_height_lookup():
    if not PROFIL_CSV.exists(): return None,{},{}
    print("Loading profilometry...")
    Z=pd.read_csv(PROFIL_CSV,header=None).values.astype(float)
    nr,nc=Z.shape; lookup={}; frame_xy={}
    for layer,cfg in PROFIL_LAYER_CONFIG.items():
        rc=cfg["row_center"]; hb=cfg["half_band"]
        band=Z[max(0,rc-hb):min(nr-1,rc+hb)+1,:]
        hm=float(np.nanmean(band)); hs=float(np.nanstd(band))
        if hs<1e-8: hs=1.0
        lookup[layer]={"h_mean":hm,"h_std":hs,"n_rows":nr,"n_cols":nc}
        csv=TRACK_CSV_DIR/f"L0{layer}.csv"
        if not csv.exists(): continue
        df_t=pd.read_csv(csv); df_t.columns=[c.strip() for c in df_t.columns]
        for _,row in df_t.iterrows():
            fn=int(row["frame_number"])
            xp=float(row["x(mm)"])-cfg["x_offset"]
            cc=int(round(xp/PX_X_MM))
            frame_xy[(layer,fn)]=(cc,rc)
    return Z,lookup,frame_xy

def get_patch(Z,col_c,row_c,h_mean,h_std,n_rows,n_cols):
    zero=torch.zeros(1,PATCH_OUT,PATCH_OUT,dtype=torch.float32)
    r_lo=row_c-PATCH_HALF_Y; r_hi=row_c+PATCH_HALF_Y+1
    c_lo=col_c-PATCH_HALF_X; c_hi=col_c+PATCH_HALF_X+1
    if r_lo<0 or r_hi>n_rows or c_lo<0 or c_hi>n_cols: return zero
    patch=Z[r_lo:r_hi,c_lo:c_hi].copy()
    patch=np.where(np.isnan(patch),h_mean,patch)
    patch=(patch-h_mean)/h_std
    p_t=torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0)
    return F.interpolate(p_t,size=(PATCH_OUT,PATCH_OUT),
                         mode="bilinear",align_corners=False).squeeze(0)

# ── Model ─────────────────────────────────────────────────────────
class HPatchCNN(nn.Module):
    def __init__(self,hd=16,dr=0.3):
        super().__init__()
        self.enc=nn.Sequential(
            nn.Conv2d(1,16,3,padding=1),nn.BatchNorm2d(16),nn.ReLU(True),
            nn.Conv2d(16,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(True),
            nn.AdaptiveAvgPool2d((4,4)),nn.Flatten(),
            nn.Linear(512,hd),nn.ReLU(True),nn.Dropout(dr))
    def forward(self,x): return self.enc(x)

class DiffCNN(nn.Module):
    def __init__(self,fd=64,dr=0.3):
        super().__init__()
        self.enc=nn.Sequential(
            nn.Conv2d(1,32,5,padding=2),nn.BatchNorm2d(32),nn.ReLU(True),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(True),nn.MaxPool2d(2),
            nn.Conv2d(64,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(True),nn.AdaptiveMaxPool2d((4,4)))
        self.proj=nn.Sequential(nn.Flatten(),nn.Linear(1024,fd),
                                nn.BatchNorm1d(fd),nn.ReLU(True),nn.Dropout(dr))
    def forward(self,x): return self.proj(self.enc(x))

class Detector(nn.Module):
    def __init__(self,lh=128,dr=0.45,hd=16,pvh=16,fd=64,nl=2):
        super().__init__()
        self.dc=DiffCNN(fd,dr)
        self.fc=nn.Sequential(
            nn.Conv2d(1,16,3,padding=1),nn.BatchNorm2d(16),nn.ReLU(True),nn.MaxPool2d(4),
            nn.Conv2d(16,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(True),nn.AdaptiveAvgPool2d((4,4)),
            nn.Flatten(),nn.Linear(512,32),nn.ReLU(True),nn.Dropout(dr))
        self.hc=HPatchCNN(hd,dr)
        self.lstm=nn.LSTM(fd,lh,nl,batch_first=True,dropout=dr if nl>1 else 0.)
        self.pv=nn.Sequential(nn.Linear(2,pvh),nn.ReLU(),nn.Dropout(dr))
        self.clf=nn.Sequential(nn.Dropout(dr),nn.Linear(lh*2+32+hd+pvh+1,64),
                               nn.ReLU(),nn.Dropout(dr),nn.Linear(64,1))
    def forward(self,diffs,lf,pv,hp,hg):
        B,T,C,H,W=diffs.shape
        f=self.dc(diffs.view(B*T,C,H,W)).view(B,T,-1)
        out,(h,_)=self.lstm(f)
        return self.clf(torch.cat([h[-1],out.mean(1),self.fc(lf),
                                   self.hc(hp)*hg,self.pv(pv),hg],dim=1))

# ── Dataset ───────────────────────────────────────────────────────
class AblDS(Dataset):
    def __init__(self,sdf,src,pm,ps,vm,vs,Z,hl,fxy,use_P,use_V,aug=False):
        self.df=sdf.reset_index(drop=True)
        self.pm=float(pm); self.ps=float(ps) if ps!=0 else 1.
        self.vm=float(vm); self.vs=float(vs) if vs!=0 else 1.
        self.Z=Z; self.hl=hl; self.fxy=fxy
        self.use_P=use_P; self.use_V=use_V; self.aug=aug
        self.fmap={}
        for layer,g in src.groupby("layer"):
            g=g.sort_values("frame")
            self.fmap[int(layer)]={int(r.frame):str(r.image_path)
                                   for r in g.itertuples(index=False)}
    def _load(self,p):
        img=Image.open(p); x=TF.pil_to_tensor(img).float()
        x=torch.clamp(x,0,65535)/65535.
        x=TF.resize(x,list(IMAGE_SIZE),antialias=True)
        if x.shape[0]>1: x=x.mean(dim=0,keepdim=True)
        return x
    def _hp(self,layer,cf):
        if self.Z is None or layer not in self.hl: return torch.zeros(1,PATCH_OUT,PATCH_OUT)
        k=(layer,cf)
        if k not in self.fxy: return torch.zeros(1,PATCH_OUT,PATCH_OUT)
        cc,rc=self.fxy[k]; c=self.hl[layer]
        return get_patch(self.Z,cc,rc,c["h_mean"],c["h_std"],c["n_rows"],c["n_cols"])
    def __len__(self): return len(self.df)
    def __getitem__(self,idx):
        row=self.df.iloc[idx]; layer=int(row["layer"]); start=int(row["start_frame"])
        fm=self.fmap[layer]; af=sorted(fm.keys()); mn,mx=af[0],af[-1]
        frames=[max(mn,min(mx,start+i)) for i in range(SEQ_LEN)]
        imgs=[self._load(fm[f]) for f in frames]
        if self.aug and random.random()>0.5:
            imgs=[torch.flip(x,dims=[2]) for x in imgs]
        diffs=torch.stack([torch.abs(imgs[t]-imgs[t-1]) for t in range(1,SEQ_LEN)],dim=0)
        dm=diffs.max()
        if dm>1e-8: diffs=diffs/dm
        last=(imgs[-1]-0.5)/0.5
        pn=(float(row["P"])-self.pm)/self.ps if self.use_P else 0.0
        vn=(float(row["V"])-self.vm)/self.vs if self.use_V else 0.0
        pv=torch.tensor([pn,vn],dtype=torch.float32)
        cf=frames[SEQ_LEN//2]; hp=self._hp(layer,cf)
        hg=torch.tensor([[1.0]] if layer in PROFIL_LAYERS else [[0.0]],
                         dtype=torch.float32).squeeze(0)
        y=torch.tensor(float(row["label"]),dtype=torch.float32)
        return diffs,last,pv,hp,hg,y

# ── Sample builders ───────────────────────────────────────────────
def build_samp(df,seed=42):
    rng=random.Random(seed); rows=[]
    for layer,g in df.groupby("layer"):
        g=g.sort_values("frame").reset_index(drop=True)
        valid=set(g["frame"].astype(int).tolist()); af=sorted(valid)
        if len(af)<SEQ_LEN: continue
        mf=af[0]; xf=af[-1]; p=float(g["P"].iloc[0]); v=float(g["V"].iloc[0])
        ball=set(g[g["balling"]==1]["frame"].astype(int))&valid
        half=SEQ_LEN//2; pos=set()
        for fn in ball:
            ws=max(mf,min(fn-half,xf-SEQ_LEN+1))
            if all(f in valid for f in range(ws,ws+SEQ_LEN)) and \
               any(f in ball for f in range(ws,ws+SEQ_LEN)): pos.add(ws)
        for ws in pos: rows.append({"layer":int(layer),"start_frame":ws,"label":1,"P":p,"V":v})
        negs=[s for s in range(mf,xf-SEQ_LEN+2)
              if all(f in valid for f in range(s,s+SEQ_LEN)) and
                 all(f not in ball for f in range(s,s+SEQ_LEN))]
        nn=max(int(round(len(pos)*1.0)),1)
        if negs:
            ch=rng.sample(negs,k=min(nn,len(negs)))
            while len(ch)<nn: ch.append(rng.choice(negs))
            for ws in ch: rows.append({"layer":int(layer),"start_frame":ws,"label":0,"P":p,"V":v})
    return pd.DataFrame(rows).sample(frac=1,random_state=seed).reset_index(drop=True)

def build_full(df):
    rows=[]
    for layer,g in df.groupby("layer"):
        g=g.sort_values("frame").reset_index(drop=True)
        valid=set(g["frame"].astype(int).tolist()); af=sorted(valid)
        if len(af)<SEQ_LEN: continue
        mf=af[0]; xf=af[-1]; p=float(g["P"].iloc[0]); v=float(g["V"].iloc[0])
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

# ── Training ──────────────────────────────────────────────────────
def bth(p,t):
    bt,bf=0.5,-1.
    for th in np.arange(0.05,0.96,0.05):
        f=f1_score(t,(p>=th).astype(int),zero_division=0)
        if f>bf: bf,bt=f,float(th)
    return bt,bf

def run_ep(model,loader,opt=None,desc=""):
    itr=opt is not None
    model.train() if itr else model.eval()
    ap,at=[],[]; tl=0.
    pb=tqdm(loader,desc=desc,leave=False,ncols=80,disable=True)  # disabled to avoid IOPub flood
    for diffs,lf,pv,hp,hg,y in pb:
        diffs=diffs.to(device); lf=lf.to(device)
        pv=pv.to(device); hp=hp.to(device); hg=hg.to(device)
        yd=y.to(device).unsqueeze(1)
        if itr: opt.zero_grad()
        with torch.set_grad_enabled(itr):
            lo=model(diffs,lf,pv,hp,hg)
            loss=F.binary_cross_entropy_with_logits(lo,yd)
            if itr:
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.)
                opt.step(); pb.set_postfix(loss=f"{loss.item():.3f}")
        tl+=loss.item()*diffs.size(0)
        ap.extend(torch.sigmoid(lo).detach().cpu().numpy().ravel())
        at.extend((y.numpy()>=0.5).astype(int))
    pa,ta=np.array(ap),np.array(at)
    try: auc=roc_auc_score(ta,pa)
    except: auc=float("nan")
    return {"probs":pa,"targets":ta,"auc":auc}

def run_fold(held_out,enriched_df,Z,hl,fxy,use_P,use_V,vname):
    tl=[l for l in ALL_LAYERS if l!=held_out]
    trd=enriched_df[enriched_df["layer"].isin(tl)].reset_index(drop=True)
    ted=enriched_df[enriched_df["layer"]==held_out].reset_index(drop=True)
    pm=float(trd["P"].mean()); ps=float(trd["P"].std()) or 1.
    vm=float(trd["V"].mean()); vs=float(trd["V"].std()) or 1.
    tp=float(ted["P"].iloc[0]); tv=float(ted["V"].iloc[0])
    def pd_(l):
        lp=float(enriched_df[enriched_df["layer"]==l]["P"].iloc[0])
        lv=float(enriched_df[enriched_df["layer"]==l]["V"].iloc[0])
        return ((lp-tp)/140)**2+((lv-tv)/200)**2
    vl=min(tl,key=pd_); ptr=[l for l in tl if l!=vl]
    trd2=enriched_df[enriched_df["layer"].isin(ptr)].reset_index(drop=True)
    vld=enriched_df[enriched_df["layer"]==vl].reset_index(drop=True)
    tr_samp=build_samp(trd2,42); vl_samp=build_samp(vld,43)
    kw=dict(pm=pm,ps=ps,vm=vm,vs=vs,Z=Z,hl=hl,fxy=fxy,use_P=use_P,use_V=use_V)
    tds=AblDS(tr_samp,trd2,**kw,aug=True); vds=AblDS(vl_samp,vld,**kw,aug=False)
    tl_=DataLoader(tds,BATCH_SIZE,shuffle=True,num_workers=0)
    vl_=DataLoader(vds,BATCH_SIZE,shuffle=False,num_workers=0)
    model=Detector().to(device)
    opt=torch.optim.Adam(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,"max",factor=0.5,patience=5,min_lr=1e-5)
    bva,ni=(-1.,0); ckpt=ABLATION_DIR/f"{vname}_{held_out}.pth"
    torch.save({"state":model.state_dict()},ckpt)
    for ep in range(1,MAX_EPOCHS+1):
        run_ep(model,tl_,opt,f"")
        vm_=run_ep(model,vl_,desc=f"")
        sch.step(vm_["auc"])
        if not np.isnan(vm_["auc"]) and vm_["auc"]>bva:
            bva,ni=vm_["auc"],0; torch.save({"state":model.state_dict()},ckpt)
        else:
            ni+=1
            if ni>=PATIENCE: break  # silent early stop
    model.load_state_dict(torch.load(ckpt,map_location=device,weights_only=False)["state"])
    fs=build_full(ted); fds=AblDS(fs,ted,**kw,aug=False)
    fl=DataLoader(fds,BATCH_SIZE,shuffle=False,num_workers=0)
    model.eval(); probs=[]
    with torch.no_grad():
        for di,lf,pv,hp,hg,_ in fl:
            lo=model(di.to(device),lf.to(device),pv.to(device),hp.to(device),hg.to(device))
            probs.extend(torch.sigmoid(lo).cpu().numpy().ravel())
    fs["pred_prob"]=probs
    try: auc=roc_auc_score(fs["label"],fs["pred_prob"])
    except: auc=float("nan")
    return auc,fs

# ── Main ──────────────────────────────────────────────────────────
if __name__=="__main__":
    import sys, io
    # Log all output to file AND console
    log_path = ABLATION_DIR / "ablation_log.txt"
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    class Tee:
        def __init__(self, *files): self.files = files
        def write(self, s):
            for f in self.files: f.write(s); f.flush()
        def flush(self):
            for f in self.files: f.flush()
    log_f = open(log_path, "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_f)
    print(f"Logging to: {log_path}")
    print("Loading enriched_df...")
    enriched_df=pd.read_csv(
        r"C:\Users\erfan\Downloads\balling_dataset\enriched_df.csv",
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

    Z,hl,fxy=build_height_lookup()

    results={v:{} for v in VARIANTS}
    for vname,vcfg in VARIANTS.items():
        print(f"\n{'='*60}\nVARIANT: {vname}  use_P={vcfg['use_P']} use_V={vcfg['use_V']}\n{'='*60}")
        for held_out in sorted(ALL_LAYERS):
            p=int(enriched_df[enriched_df["layer"]==held_out]["P"].iloc[0])
            v=int(enriched_df[enriched_df["layer"]==held_out]["V"].iloc[0])
            n=int((enriched_df[enriched_df["layer"]==held_out]["balling"]==1).sum())
            print(f"\n  Fold L{held_out} P={p}W V={v} n_ball={n}")
            auc,_=run_fold(held_out,enriched_df,Z,hl,fxy,
                           vcfg["use_P"],vcfg["use_V"],vname)
            results[vname][held_out]=auc
            print(f"  [{vname}] L{held_out}: AUC={auc:.3f}")
        pd.DataFrame({"layer":list(results[vname].keys()),
                      "auc":list(results[vname].values()),
                      "variant":vname}
                     ).to_csv(ABLATION_DIR/f"preds_{vname}.csv",index=False)

    # Print results table
    layer_info={}
    for l in ALL_LAYERS:
        lp=int(enriched_df[enriched_df["layer"]==l]["P"].iloc[0])
        lv=int(enriched_df[enriched_df["layer"]==l]["V"].iloc[0])
        layer_info[l]=(lp,lv)
    sep75 = "="*75
    print(f"\n{sep75}\nPROCESS PARAMETER ABLATION RESULTS\n{sep75}")
    print("  {:>7} {:>5} {:>5} | {:>7} {:>7} {:>7} {:>7} | {:>7} {:>7} {:>8}".format(
          "Layer","P","V","P+V","P","V","none","dP","dV","d_none"))
    print("  "+"-"*75)
    for layer in sorted(ALL_LAYERS):
        pv_=layer_info[layer]
        a={v:results[v].get(layer,np.nan) for v in ["PV","P","V","none"]}
        dp=a["PV"]-a["P"] if not np.isnan(a["P"]) else np.nan
        dv=a["PV"]-a["V"] if not np.isnan(a["V"]) else np.nan
        dn=a["PV"]-a["none"] if not np.isnan(a["none"]) else np.nan
        hf=" [H]" if layer in PROFIL_LAYERS else "    "
        print(f"  {layer:>7}{hf} {pv_[0]:>5} {pv_[1]:>5} | "
              f"{a['PV']:>7.3f} {a['P']:>7.3f} {a['V']:>7.3f} {a['none']:>7.3f} | "
              f"{dp:>+7.3f} {dv:>+7.3f} {dn:>+8.3f}")

    for vname in ["PV","P","V","none"]:
        aucs=[x for x in results[vname].values() if not np.isnan(x)]
        print(f"  {vname:>6}: mean={np.mean(aucs):.3f}")

    pc=np.nanmean([results["PV"].get(l,np.nan)-results["V"].get(l,np.nan) for l in ALL_LAYERS])
    vc=np.nanmean([results["PV"].get(l,np.nan)-results["P"].get(l,np.nan) for l in ALL_LAYERS])
    bc=np.nanmean([results["PV"].get(l,np.nan)-results["none"].get(l,np.nan) for l in ALL_LAYERS])
    print(f"\n  P adds over V-only:  {pc:+.3f}")
    print(f"  V adds over P-only:  {vc:+.3f}")
    print(f"  Both over none:      {bc:+.3f}")

    # Figure
    ls=sorted(ALL_LAYERS); x=np.arange(len(ls))
    xl=[f"L{l}\n{layer_info[l][0]}W"+("\n[H]" if l in PROFIL_LAYERS else "") for l in ls]
    clrs={"PV":"#1565C0","P":"#E65100","V":"#2E7D32","none":"#888888"}; w=0.2
    fig,axes=plt.subplots(1,2,figsize=(20,6))
    fig.suptitle("Process Parameter Ablation: P+V vs P-only vs V-only vs none\n"
                 "Same model (SEQ=8, gated height, smart val, dropout=0.45, lstm=128)",
                 fontsize=10,fontweight="bold")
    ax=axes[0]
    for i,(vn,co) in enumerate(clrs.items()):
        ax.bar(x+(i-1.5)*w,[results[vn].get(l,np.nan) for l in ls],
               w,label=vn,color=co,alpha=0.85,edgecolor="black",linewidth=0.7)
    ax.axhline(0.5,color="black",linewidth=1.2,linestyle="--")
    ax.set_xticks(x); ax.set_xticklabels(xl,fontsize=6)
    ax.set_ylabel("AUC"); ax.set_ylim(0.3,1.0)
    ax.set_title("AUC per layer"); ax.legend(fontsize=9)
    ax=axes[1]
    for i,(vn,co) in enumerate([("P","#E65100"),("V","#2E7D32"),("none","#888888")]):
        ax.bar(x+(i-1)*w,[results["PV"].get(l,np.nan)-results[vn].get(l,np.nan) for l in ls],
               w,label=f"PV-{vn}",color=co,alpha=0.85,edgecolor="black",linewidth=0.7)
    ax.axhline(0,color="black",linewidth=1.5,linestyle="--")
    ax.set_xticks(x); ax.set_xticklabels(xl,fontsize=6)
    ax.set_ylabel("Delta AUC (P+V minus variant)")
    ax.set_title("Contribution of each parameter\nPositive = P+V better")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(ABLATION_DIR/"pv_ablation.png",dpi=130,bbox_inches="tight")
    plt.close()
    pd.DataFrame([{"layer":l,**{f"auc_{v}":results[v].get(l,np.nan) for v in VARIANTS}}
                  for l in sorted(ALL_LAYERS)]
                 ).to_csv(ABLATION_DIR/"ablation_summary.csv",index=False)
    print(f"\nOutputs: {ABLATION_DIR}/")
