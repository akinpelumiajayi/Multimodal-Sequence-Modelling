import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from bs4 import BeautifulSoup
from torchvision import transforms
from transformers import AutoTokenizer
from transformers import AutoTokenizer as GPT2Tokenizer
from diffusers import AutoencoderKL, DDPMScheduler
from accelerate import Accelerator

from .utils import (
    load_yaml, make_paths, set_seed, get_device, get_dtype,
    save_checkpoint, maybe_drop_condition, to_minus1_1
)
from .model import (
    MultimodalSequenceModel, mem_collapse_loss,
    DecoderText, build_unet_decoder
)

def parse_gdi(text):
  """
    Extracts story segments stored inside <gdi> tags from an HTML-like story field.
    The StoryReasoning dataset stores the story as an HTML string where each
    step is wrapped in <gdi>...</gdi>. This function parses the HTML and returns
    the clean text per step.

    Args:
        html_text: HTML string containing <gdi> tags.
    Returns:
        A list of strings, one per <gdi> segment, stripped of whitespace.
    Instance:
        steps = parse_gdi(sample["story"])
        # steps might have length 4 (context1, context2, context3, target(context4))
    """
    soup = BeautifulSoup(text, "html.parser")
    return [x.get_text().strip() for x in soup.find_all("gdi")]

class StoryDataset(torch.utils.data.Dataset):
   """
    StoryReasoning dataset wrapper for multimodal sequential prediction.

    Each dataset item contains 4 ordered steps (image+text). The model is trained
    to predict the 4th step (image/text) given the first 3 steps.

    Output structure is designed for:
      - Sequence encoder training (Image1-3 + Text1-3 -> memory tokens)
      - Target text supervision (Text4 token ids)
      - Decoder cache building (Z tokens + GT latent + GT text)

    Attributes:
        ds: HuggingFace dataset split (train/test).
        max_len: Max token length for BERT tokenization.
        img_tf_224: Transform applied to context images (224x224).
        img_tf_512: Transform applied to GT image4 for VAE latent encoding (512x512).
        tokenizer: BERT tokenizer used for context/target text tokenization.

    __getitem__ returns dict containing:
        imgs224: Tensor (3,3,224,224) context images.
        ids: Tensor (3,max_len) context token ids.
        mask: Tensor (3,max_len) attention masks.
        texts: list[str] length (4) [text1,text2,text3,text4].
        gt_img512: Tensor (3,512,512) ground-truth image4 in [0,1] -> normalized.

    Notes:
        - imgs224 are used only for sequence encoding.
        - gt_img512 is used to build stable diffusion latents via VAE for decoder training.
    """
    def __init__(self, split="train", max_len=64):
        self.ds = load_dataset("daniel3303/StoryReasoning", split=split)
        self.max_len = max_len
        self.bert_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.img_tf_224 = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
        self.img_tf_512 = transforms.Compose([transforms.Resize((512,512)), transforms.ToTensor()])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        s = self.ds[i]
        texts = (parse_gdi(s["story"]) + [""] * 4)[:4]

        imgs224 = torch.stack([self.img_tf_224(s["images"][j]) for j in range(4)])
        gt_img512 = self.img_tf_512(s["images"][3])

        ctx = self.bert_tok(
            texts[:3],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "imgs224": imgs224,            # (4,3,224,224)
            "ids": ctx["input_ids"],       # (3,L)
            "mask": ctx["attention_mask"], # (3,L)
            "texts": texts,                # list[str] length 4
            "gt_img512": gt_img512         # (3,512,512)
        }
@torch.no_grad()
def encode_latents(vae, img_1x3_512, device, dtype):
  """
    Encodes an image into Stable Diffusion VAE latent space.
    Converts image from [0,1] to [-1,1], then runs VAE encoder.
    Multiplies by SD scaling factor 0.18215.
    Args:
        vae: diffusers AutoencoderKL.
        img_1x3_512: Tensor (1,3,512,512) in [0,1].
    Returns:
        Tensor (1,4,64,64) latents scaled by 0.18215.
    """
    x = img_1x3_512.to(device, dtype=dtype)
    if x.min() >= 0 and x.max() <= 1:
        x = to_minus1_1(x)
    lat = vae.encode(x).latent_dist.sample() * 0.18215
    return lat

def main(config_path="config.yaml"):
    cfg = load_yaml(config_path)
    root, ckpt_dir, res_dir = make_paths(cfg)

    device = get_device()
    dtype = get_dtype(device)
    set_seed(cfg["train"]["seed"])

    sd_model_id = cfg["models"]["sd_model_id"]
    latent_dim = cfg["models"]["latent_dim"]
    max_len = cfg["models"]["max_len"]

    seq_ckpt_path = os.path.join(ckpt_dir, "MultimodalSequenceModel.pt")
    cache_path    = os.path.join(ckpt_dir, "decoder_cache.pt")
    dec_ckpt_path = os.path.join(ckpt_dir, "decoder_ckpt.pt")

    grad_accum = cfg["train"]["grad_accum"]
    mp = "bf16" if (device=="cuda" and torch.cuda.is_bf16_supported()) else ("fp16" if device=="cuda" else "no")
    accelerator = Accelerator(mixed_precision=mp, gradient_accumulation_steps=grad_accum)

    seq_model = MultimodalSequenceModel(latent_dim)
    opt = torch.optim.AdamW(seq_model.parameters(), lr=cfg["train"]["seq_lr"])

    train_dl = DataLoader(
        StoryDataset("train", max_len),
        batch_size=cfg["train"]["seq_batch"],
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    seq_model, opt, train_dl = accelerator.prepare(seq_model, opt, train_dl)

    for epoch in range(cfg["train"]["seq_epochs"]):
        seq_model.train()
        tot = 0.0
        for batch in train_dl:
            imgs3 = batch["imgs224"][:, :3].to(accelerator.device)
            ids   = batch["ids"].to(accelerator.device)
            msk   = batch["mask"].to(accelerator.device)

            _, tokens = seq_model.encode_sequence(imgs3, ids, msk, return_tokens=True)
            loss = mem_collapse_loss(tokens)

            opt.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            opt.step()
            tot += float(loss.detach().cpu())
        accelerator.print(f"Seq Epoch {epoch+1}/{cfg['train']['seq_epochs']} | loss={tot/len(train_dl):.4f}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        torch.save(accelerator.unwrap_model(seq_model).state_dict(), seq_ckpt_path)
        print("Saved:", seq_ckpt_path)
      
    #Build Cache
    seq_model_cpu = MultimodalSequenceModel(latent_dim).to(device)
    seq_model_cpu.load_state_dict(torch.load(seq_ckpt_path, map_location="cpu"), strict=False)
    seq_model_cpu.eval().float()

    vae = AutoencoderKL.from_pretrained(sd_model_id, subfolder="vae").to(device, dtype=dtype)
    vae.eval().requires_grad_(False)

    dl = DataLoader(StoryDataset("train", max_len), batch_size=1, shuffle=False, num_workers=2)
    cache = []
    max_items = int(cfg["train"]["cache_items"])
    for i, batch in enumerate(dl):
        imgs3 = batch["imgs224"][:, :3].to(device).float()
        ids   = batch["ids"].to(device)
        msk   = batch["mask"].to(device)

        with torch.no_grad():
            _, z_tokens = seq_model_cpu.encode_sequence(imgs3, ids, msk, return_tokens=True)
            z_tokens = z_tokens[0].detach().float().cpu()  # (3,512)

            gt_lat = encode_latents(vae, batch["gt_img512"], device, dtype).squeeze(0).detach().float().cpu()
            gt_txt = batch["texts"][3][0]

        cache.append({"z_pred": z_tokens, "gt_latent": gt_lat, "text": str(gt_txt)})

        if (i + 1) % 200 == 0:
            print("cached:", i + 1)
        if len(cache) >= max_items:
            break

    torch.save(cache, cache_path)
    print("Saved cache:", cache_path, "| items:", len(cache))

    # Decoder dataset
    gpt2_tok = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2_tok.pad_token = gpt2_tok.eos_token
    vocab_size = gpt2_tok.vocab_size

    class CachedDecoderDataset(torch.utils.data.Dataset):
        def __init__(self, path):
            self.data = torch.load(path, map_location="cpu")
        def __len__(self): return len(self.data)
        def __getitem__(self, i):
            it = self.data[i]
            enc = gpt2_tok(str(it["text"]), max_length=max_len, truncation=True,
                           padding="max_length", return_tensors="pt")
            return {
                "z_pred": it["z_pred"],
                "gt_latent": it["gt_latent"],
                "text_ids": enc.input_ids.squeeze(0),
            }
          
    dec_ds = CachedDecoderDataset(cache_path)
    dec_loader = DataLoader(dec_ds, batch_size=cfg["train"]["dec_batch"], shuffle=True, num_workers=2, pin_memory=True)

    # build decoder models
    unet = build_unet_decoder(sd_model_id, device, torch.float16 if device=="cuda" else torch.float32)

    # train attention
    for p in unet.parameters():
        p.requires_grad = False
    for name, p in unet.named_parameters():
        if ("attn" in name) or ("to_q" in name) or ("to_k" in name) or ("to_v" in name) or ("to_out" in name) or ("conv_out" in name):
            p.requires_grad = True

    latent_mapper = torch.nn.Sequential(torch.nn.Linear(512, 768), torch.nn.LayerNorm(768))
    dec_text = DecoderText(vocab_size=vocab_size, max_len=max_len, n_layers=3)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    optimizer = torch.optim.AdamW(
        [p for p in unet.parameters() if p.requires_grad] + list(latent_mapper.parameters()) + list(dec_text.parameters()),
        lr=cfg["train"]["dec_lr"]
    )
    unet, latent_mapper, dec_text, optimizer, dec_loader = accelerator.prepare(
        unet, latent_mapper, dec_text, optimizer, dec_loader
    )
    cfg_drop = float(cfg["train"]["cfg_drop_prob"])
    autosave_every = int(cfg["train"]["autosave_every"])
    global_step = 0

    # Train decoder with autosave and training resuming
    for epoch in range(1, int(cfg["train"]["dec_epochs"]) + 1):
        unet.train(); latent_mapper.train(); dec_text.train()
        tot = 0.0

        for batch in dec_loader:
            global_step += 1
            z = batch["z_pred"].to(accelerator.device)           # (B,T,512)
            lat = batch["gt_latent"].to(accelerator.device)      # (B,4,64,64) or (4,64,64)
            if lat.dim() == 3:
                lat = lat.unsqueeze(0)
            txt_ids = batch["text_ids"].to(accelerator.device)

            noise = torch.randn_like(lat)
            t = torch.randint(0, 1000, (lat.size(0),), device=lat.device).long()
            noisy = noise_scheduler.add_noise(lat, noise, t)

            cond = latent_mapper(z).to(noisy.dtype)
            cond = maybe_drop_condition(cond, cfg_drop)

            with accelerator.accumulate(unet):
                optimizer.zero_grad(set_to_none=True)
                pred_noise = unet(noisy, t, encoder_hidden_states=cond).sample
                img_loss = F.mse_loss(pred_noise, noise)

                logits = dec_text(txt_ids[:, :-1], z)
                text_loss = F.cross_entropy(
                    logits.reshape(-1, vocab_size),
                    txt_ids[:, 1:].reshape(-1),
                    ignore_index=gpt2_tok.pad_token_id
                )
                loss = img_loss + text_loss
                accelerator.backward(loss)
                optimizer.step()

            tot += float(loss.detach().cpu())

            # autosave
            if global_step % autosave_every == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    save_checkpoint(dec_ckpt_path, {
                        "epoch": epoch,
                        "global_step": global_step,
                        "unet": accelerator.unwrap_model(unet).state_dict(),
                        "latent_mapper": accelerator.unwrap_model(latent_mapper).state_dict(),
                        "dec_text": accelerator.unwrap_model(dec_text).state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "config": cfg
                    })
                    accelerator.print(f"Autosaved at step {global_step} -> {dec_ckpt_path}")

        accelerator.print(f"UNet Decoder Epoch {epoch}/{cfg['train']['dec_epochs']} | loss={tot/len(dec_loader):.4f}")

        # end-of-epoch save
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            save_checkpoint(dec_ckpt_path, {
                "epoch": epoch,
                "global_step": global_step,
                "unet": accelerator.unwrap_model(unet).state_dict(),
                "latent_mapper": accelerator.unwrap_model(latent_mapper).state_dict(),
                "dec_text": accelerator.unwrap_model(dec_text).state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": cfg
            })
            accelerator.print(f"Saved epoch ckpt -> {dec_ckpt_path}")

if __name__ == "__main__":
    main("config.yaml")
