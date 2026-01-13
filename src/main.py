import os
from src.ablate import run_ablation
from src.visualize import visualize_with_z
from src.utils import load_yaml

def load_z_tokens_from_cache(idx, cache_path):
    import torch
    cache = torch.load(cache_path, map_location="cpu")
    return cache[idx]["z_pred"]

if __name__ == "__main__":
    cfg = load_yaml("config.yaml")

    ROOT = cfg["paths"]["root"]
    results_dir = os.path.join(ROOT, "results")
    cache_path  = os.path.join(ROOT, "checkpoints", "decoder_cache.pt")

    #--Ablation grid--
    idx_list   = [0, 5, 10]
    strengths  = [0.02, 0.05, 0.10]
    steps_list = [10, 20, 30]
    cfg_scales = [1.0, 3.0, 5.0, 7.0]
  
    csv_path = run_ablation(
        idx_list=idx_list,
        strengths=strengths,
        steps_list=steps_list,
        cfg_scales=cfg_scales,
        split="train", 
        results_dir=results_dir,
        load_z_tokens_fn=lambda i: load_z_tokens_from_cache(i, cache_path),
        visualize_fn=visualize_with_z
    )

    print("Ablation complete. Metrics saved to:", csv_path)

