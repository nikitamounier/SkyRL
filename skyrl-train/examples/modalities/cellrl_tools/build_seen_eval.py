"""Seen-gene held-out eval: genes the SFT model knows, prompts it has never seen
(excludes the 12,500 SFT rows AND every row used in RL training)."""
import os, sys, collections, random
import pandas as pd, torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
P="/home/parsaidp/BioReasonCell/experiments/sft_gene_pathway_v7_5_50k/9b_full_answeronly/prompts"
sys.path.insert(0,P)
from template import QUESTION
from bioreason_cell.synth.utils import (build_interaction_index, build_reasoning_prompt,
                                        load_pathway_mapping)
RD="wanglab/BioReasonCell-ReasoningData"; CFG="reasoning_nadig_replogle_new"
CELL_BLOCK="Cell A: <|CELL_START|><|cell_pad|><|CELL_END|>"
MAXES=dict(max_pathways=-1,max_interactions=100,max_domains=-1,max_go_terms=-1,max_pathway_genes=500,
 max_cell_overall_genes=50,max_cell_specific_genes=200,max_cell_pathways=25,max_pathway_interactions=50,max_pathway_words=500)
tok=os.environ["HF_TOKEN"]
gl={r["gene"]:r for r in load_dataset(RD,"genes",split="genes")}
pl={r["pathway"]:r for r in load_dataset(RD,"pathways",split="pathways")}
cl={r["name"]:r for r in load_dataset(RD,"cells",split="cells")}
istr,iidx,imeta,dft=build_interaction_index(load_dataset(RD,"pathway_interactions",split="train"))
pm=load_pathway_mapping()
df=pd.read_parquet(hf_hub_download(RD,f"{CFG}/train-00000-of-00001.parquet",repo_type="dataset",token=tok))
v75=pd.read_parquet('/large_storage/goodarzilab/bioreason_cell/batches/genetic_v7.5_50k.parquet')
sft_ids=set(v75['unique_id']); sft_genes=set(v75[v75['unique_id'].str.startswith('nadig::')]['gene_target'])
used=set()
for f in ('cell_pathway_nadig_rl/train.parquet','cell_pathway_nadig_rl/validation.parquet','cell_pathway_nadig_1104/train.parquet'):
    try: used|={r['sample_id'] for r in pd.read_parquet(f'/large_storage/goodarzilab/bioreason_cell/rl_data/{f}')['extra_info']}
    except Exception: pass
print(f"excluding {len(sft_ids)} SFT ids + {len(used)} RL-train ids")
df=df[(~df['drop'])&(~df['is_offtarget'])&(~df['unique_id'].isin(sft_ids))&(~df['unique_id'].isin(used))
      &df['gene_target'].isin(sft_genes)&df['gene_target'].isin(gl.keys())&df['pathway_name'].isin(pl.keys())
      &df['pathway_change'].isin(("upregulated","downregulated","unchanged"))]
print(f"candidate pool (SEEN genes, unseen prompts): {len(df):,}")
ed='/large_storage/goodarzilab/bioreason_cell/embeddings/genetic_v7_5/cells'
vec={}
for s in {os.path.splitext(os.path.basename(c))[0] for c in df['cell_file'].unique() if c}:
    p=f'{ed}/{s}.pt'
    if os.path.isfile(p):
        e=torch.load(p,map_location='cpu',weights_only=False); vec[s]=e.reshape(-1).to(torch.float32).tolist()
df=df[df['cell_file'].map(lambda c: os.path.splitext(os.path.basename(c))[0] in vec)]
recs=df.to_dict('records'); random.Random(7).shuffle(recs)
want={'downregulated':400,'upregulated':400,'unchanged':400}; got=collections.Counter(); out=[]
for r in recs:
    g=r['pathway_change']
    if got[g]>=want[g]: continue
    q=build_reasoning_prompt(sample=r,gene_lookup=gl,pathway_lookup=pl,prompt_template=QUESTION,cell_lookup=cl,
        interaction_strings=istr,interaction_index=iidx,interaction_meta=imeta,desc_for_type=dft,pathway_mapping=pm,**MAXES)
    stem=os.path.splitext(os.path.basename(r['cell_file']))[0]
    out.append({"data_source":f"{RD}::{CFG}","prompt":[{"role":"user","content":f"{CELL_BLOCK}\n\n{q}"}],
      "env_class":"cell_pathway","reward_spec":{"method":"rule","ground_truth":g,"format_bonus":0.0},
      "extra_info":{"split":"validation","index":len(out),"sample_id":r['unique_id'],"dataset":"nadig",
        "gene_target":r['gene_target'],"cell_type":r['cell_type'],"pathway_name":r['pathway_name'],"cell_file":stem},
      "modalities":{"state_mod":[vec[stem]]}})
    got[g]+=1
    if len(out)>=1200: break
O='/large_storage/goodarzilab/bioreason_cell/rl_data/cell_pathway_nadig_seen1200'
os.makedirs(O,exist_ok=True)
pd.DataFrame(out).to_parquet(f'{O}/validation.parquet'); pd.DataFrame(out[:48]).to_parquet(f'{O}/train.parquet')
print(f"WROTE {len(out)} labels={dict(got)} genes={len({o['extra_info']['gene_target'] for o in out})} -> {O}")
