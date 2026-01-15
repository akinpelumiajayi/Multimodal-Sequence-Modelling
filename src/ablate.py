import os, csv
from PIL import Image
import torch

from .metrics import ClipScorer
from .visualize import visualize_with_z 

def run_ablation(
    # idx_list,
    strengths,
    steps_list,
    cfg_scales,
    split,
    results_dir,
    load_z_tokens_fn,   # function: idx -> z_tokens
    visualize_fn,       # function that returns dict with pred_text4, gt_text4 and save the visualization.
):
    os.makedirs(results_dir, exist_ok=True)
    panels_dir = os.path.join(results_dir, "ablations")
    os.makedirs(panels_dir, exist_ok=True)

    clip = ClipScorer(device="cuda" if torch.cuda.is_available() else "cpu")

    csv_path = os.path.join(results_dir, "ablation_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "idx","split","steps","strength","cfg_scale",
            "panel_path",
            "clip_predimg_predtxt","clip_predimg_gttxt"
        ])
        writer.writeheader()

        for idx in idx_list:
            z_tokens = load_z_tokens_fn(idx)

            for steps in steps_list:
                for strength in strengths:
                    for cfg_scale in cfg_scales:
                        panel_path = os.path.join(
                            panels_dir, f"idx{idx}_s{steps}_str{strength:.3f}_cfg{cfg_scale:.2f}.png"
                        )

                        out = visualize_fn(
                            idx=idx,
                            z_tokens=z_tokens,
                            split=split,
                            steps=steps,
                            strength=strength,
                            cfg_scale=cfg_scale,
                            save_path=panel_path,
                            show=False,
                        )
                      
                        pred_pil = out.get("pred_pil", None)
                        gt_txt   = out.get("gt_text4", "")
                        pred_txt = out.get("pred_text4", "")

                        clip_pp = clip.image_text_cosine(pred_pil, pred_txt) if pred_pil else float("nan")
                        clip_pg = clip.image_text_cosine(pred_pil, gt_txt)   if pred_pil else float("nan")

                        writer.writerow({
                            "idx": idx,
                            "split": split,
                            "steps": steps,
                            "strength": strength,
                            "cfg_scale": cfg_scale,
                            "panel_path": panel_path,
                            "clip_predimg_predtxt": clip_pp,
                            "clip_predimg_gttxt": clip_pg,
                        })

    return csv_path
pred_pil = Image.fromarray(to_u8(pred_img))
return {"pred_text4": pred_txt, "gt_text4": gt_txt, "pred_pil": pred_pil, "save_path": save_path}
