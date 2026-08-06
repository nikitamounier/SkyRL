import torch, os, shutil, glob
from safetensors.torch import load_file, save_file
BASE=os.environ["BASE_CKPT"]; OUT=os.environ["MERGED_OUT"]; FULL=os.environ["FULL_SD_OUT"]
os.makedirs(OUT, exist_ok=True)
d=torch.load(FULL, map_location="cpu", weights_only=False)
pairs={}
for k,v in d["lora"].items():
    bk=k.replace('base_model.model.model.','language_model.model.')
    if '.lora_A.' in k: pairs.setdefault(bk.replace('.lora_A.default.weight','.weight'),{})['A']=v
    elif '.lora_B.' in k: pairs.setdefault(bk.replace('.lora_B.default.weight','.weight'),{})['B']=v
shard=sorted(glob.glob(f"{BASE}/*.safetensors"))[0]
w=load_file(shard); merged=0; miss=[]
for bk,ab in pairs.items():
    if bk in w and {'A','B'}<=set(ab):
        w[bk]=(w[bk].float()+(ab['B'].float()@ab['A'].float())*1.0).to(w[bk].dtype); merged+=1
    else: miss.append(bk)
save_file(w, f"{OUT}/{os.path.basename(shard)}", metadata={"format":"pt"})
for f in os.listdir(BASE):
    if not f.endswith('.safetensors') and os.path.isfile(os.path.join(BASE,f)):
        shutil.copy2(os.path.join(BASE,f), os.path.join(OUT,f))
torch.save({k.replace('_skyrl_modality_projections.state_mod.',''):v for k,v in d["proj"].items()}, f"{OUT}/cell_projection.pt")
print(f"merged {merged}/{len(pairs)} missing={len(miss)} -> {OUT}")
