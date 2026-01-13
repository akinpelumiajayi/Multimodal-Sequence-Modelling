DEC_CACHE_PATH = os.path.join(CKPT_DIR, "decoder_cache.pt")  
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_z_tokens_from_cache(idx, cache_path=DEC_CACHE_PATH):
    """
    Load z_tokens from cache
    Expects cache to be saved as a list of dicts
      [{"z_pred": (T,512), "gt_latent":, "text": },]
    
    Returns: z_tokens (T,512) torch.Tensor
    """
    assert os.path.exists(cache_path), f"Missing decoder cache: {cache_path}"
    cache = torch.load(cache_path, map_location="cpu")

    if isinstance(cache, dict) and "data" in cache:
        cache = cache["data"]

    assert isinstance(cache, (list, tuple)), f"Cache must be list/tuple, got: {type(cache)}"
    assert 0 <= idx < len(cache), f"idx out of range: {idx} (len={len(cache)})"
    assert "z_pred" in cache[idx], f"cache[{idx}] missing 'z_pred' keys={list(cache[idx].keys())}"

    z = cache[idx]["z_pred"]
    if not isinstance(z, torch.Tensor):
        z = torch.tensor(z)
    return z

def visualize_with_z(idx, z_tokens, split="test", steps=20, strength=0.05, save_path=None, show=True):
    ds = StoryDataset(split=split, max_len=MAX_LEN)
    item = ds[idx]

    imgs224 = item["imgs224"]
    texts   = item["texts"]
    gt_img512 = item["gt_img512"]
    gt_txt = texts[3]

    unet, latent_mapper, dec_text, vae, ddpm, ddim = build_decoder_models_from_ckpt(DEC_CKPT_PATH)

    gt_lat = encode_latents(vae, gt_img512.unsqueeze(0))
    gt_img = decode_latents(vae, gt_lat)[0]

    pred_img = gt_like_reconstruct_unet(
        unet, latent_mapper, vae, ddim,
        z_tokens.to(torch.float32),
        gt_lat,
        steps=steps,
        strength=strength
    )[0]

    pred_txt = generate_text_from_z(
        dec_text,
        z_tokens.to(torch.float32),
        max_len=MAX_LEN,
        temperature=0.9,
        top_k=50
    )

    plt.figure(figsize=(18,7))

    titles = ["Image 1", "Image 2", "Image 3", "GT Image 4", "Pred (GT-like)"]
    imgs_top = [imgs224[0], imgs224[1], imgs224[2], gt_img, pred_img]

    for i in range(5):
        ax = plt.subplot(2,5,i+1)
        ax.imshow(Image.fromarray(to_u8(imgs_top[i])))
        ax.axis("off")
        ax.set_title(titles[i], fontsize=12)

    ax6  = plt.subplot(2,5,6);  draw_text(ax6,  "Text 1",     texts[0])
    ax7  = plt.subplot(2,5,7);  draw_text(ax7,  "Text 2",     texts[1])
    ax8  = plt.subplot(2,5,8);  draw_text(ax8,  "Text 3",     texts[2])
    ax9  = plt.subplot(2,5,9);  draw_text(ax9,  "GT Text 4",  gt_txt)
    ax10 = plt.subplot(2,5,10); draw_text(ax10, "Pred Text 4", pred_txt)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    print("\n--- TEXT DEBUG ---")
    print("GT text 4:\n", gt_txt)
    print("\nPred text 4:\n", pred_txt)

    return {"save_path": save_path, "pred_text4": pred_txt, "gt_text4": gt_txt}
  
  def visualize_(idx=0, split="test", steps=20, strength=0.05, save_name=None, show=True):
    z_tokens = load_z_tokens_from_cache(idx)
    if save_name is None:
        save_name = f"visualize_idx{idx}.png"
    save_path = os.path.join(RESULTS_DIR, save_name)

    return visualize_with_z(
        idx=idx,
        z_tokens=z_tokens,
        split=split,
        steps=steps,
        strength=strength,
        save_path=save_path,
        show=show
    )
if __name__ == "__main__":
    out = visualize_(
        idx=0,
        split="test",
        steps=20,
        strength=0.05,
        show=True
    )
    print("Saved visualize to:", out["save_path"])

