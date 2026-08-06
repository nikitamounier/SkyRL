import os, torch, torch.distributed as dist
from torch.distributed.tensor import DTensor
C=os.environ["CKPT_POLICY_DIR"]; OUT=os.environ["FULL_SD_OUT"]
dist.init_process_group("gloo"); rank=dist.get_rank()
sd=torch.load(f"{C}/model_world_size_3_rank_{rank}.pt", map_location="cpu", weights_only=False)
full={k:(v.full_tensor() if isinstance(v,DTensor) else v) for k,v in sd.items()}
if rank==0:
    lora={k:v for k,v in full.items() if 'lora_' in k}
    proj={k:v for k,v in full.items() if k.startswith('_skyrl_modality_projections')}
    print(f"[rank0] lora={len(lora)} proj={len(proj)}", flush=True)
    torch.save({"lora":lora,"proj":proj}, OUT)
    print("[rank0] saved", OUT, flush=True)
dist.barrier(); dist.destroy_process_group()
