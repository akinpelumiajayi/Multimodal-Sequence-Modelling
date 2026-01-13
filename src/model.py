import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from transformers import AutoModel
from diffusers import UNet2DConditionModel

class ImageEncoder(nn.Module):
     """
    Encodes an image into a fixed-dimensional latent vector.
    Uses a frozen ResNet50 backbone (ImageNet pretrained) to produce a 2048-d
    global feature, then projects it into `latent_dim`.

    Args:
        latent_dim: Output embedding size. Common choices:
            - 512 for CLIP-alignment space
            - 256 for smaller models
    Inputs:
        x: Float tensor (B,3,H,W) in [0,1] (or normalized depending on transform).
    Outputs:
        Tensor (B, latent_dim) image embedding.
    Notes:
        - Backbone is frozen to reduce compute and stabilize training.
        - Only the final linear projection trains.
    """
    def __init__(self, d: int):
        super().__init__()
        bb = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(bb.children())[:-1])
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.fc = nn.Linear(2048, d)

    def forward(self, x):
        x = self.backbone(x).squeeze(-1).squeeze(-1)
        return self.fc(x)

class TextEncoder(nn.Module):
    """
    Encodes a text sequence into a fixed-dimensional latent vector.
    Uses a pretrained BERT encoder and takes the [CLS] token embedding
    (last_hidden_state[:, 0]) as the sentence representation, then projects
    to `latent_dim`.
    Args:
        latent_dim: Output embedding size.
    Inputs:
        ids: Long tensor (B, L) token ids.
        mask: Long tensor (B, L) attention mask.
    Outputs:
        Tensor (B, latent_dim) text embedding.
    """
    def __init__(self, d: int, name="bert-base-uncased"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(name)
        self.fc = nn.Linear(768, d)

    def forward(self, ids, mask):
        out = self.bert(ids, attention_mask=mask)
        return self.fc(out.last_hidden_state[:, 0])

class TemporalFusion(nn.Module):
    """
    Fuses image and text embeddings for a single time step without collapse.
    Instead of naive addition (zi + zt), which often collapses variance and
    makes the memory representation constant, this module:
      - Normalizes modalities separately
      - Concatenates and fuses with MLP
      - Uses a learnable gate to mix fused and residual information
    Args:
        d: Latent dimension for each modality.
    Inputs:
        zi: Tensor (B, d) image embedding.
        zt: Tensor (B, d) text embedding.
    Outputs:
        Tensor (B, d) fused embedding.
    """
    def __init__(self, d: int):
        super().__init__()
        self.norm_i = nn.LayerNorm(d)
        self.norm_t = nn.LayerNorm(d)
        self.fuse = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.GELU(),
            nn.LayerNorm(d),
            nn.Dropout(0.1),
        )
        self.gate = nn.Sequential(nn.Linear(2 * d, d), nn.Sigmoid())

    def forward(self, zi, zt):
        zi = self.norm_i(zi)
        zt = self.norm_t(zt)
        cat = torch.cat([zi, zt], dim=-1)
        fused = self.fuse(cat)
        g = self.gate(cat)
        return g * fused + (1 - g) * (0.5 * (zi + zt))

class MultimodalSequenceModel(nn.Module):
    """
    Multimodal sequence model that encodes (image,text) steps 1-3 and predicts step 4.

    Model Pipeline:
      -- Encode each of the 3 context images with ImageEncoder -> zi_t
      -- Encode each of the 3 context texts with TextEncoder -> zt_t
      -- Fuse per time-step using TemporalFusion -> z_t
      -- Temporal TransformerEncoder over steps (t=1-3) -> contextual tokens h_t
      -- Use last token h_3 as `mem` for text prediction and image-embedding prediction.

    Heads(Image & Text):
      - predict_text(mem, tokens): autoregressive text logits for Text4
      - predict_image_latent(mem): predicts an embedding (often aligned to CLIP image space)
    Args:
        latent_dim: Shared embedding dimension.

    Key methods:
        encode_sequence(..., return_tokens=False)
        predict_text(mem, tokens)
        predict_image_latent(mem)

    Notes:
        - During decoder-cache building, `return_tokens=True` is used to capture
          temporal tokens (e.g. h_1..h_3) as conditioning sequence `z_pred`.
    """
    def __init__(self, d=512):
        super().__init__()
        self.img_enc = ImageEncoder(d)
        self.txt_enc = TextEncoder(d)
        self.fusion = TemporalFusion(d)
        self.temporal = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, nhead=8, batch_first=True, norm_first=True),
            num_layers=3,
        )
    
  def encode_sequence(self, imgs, ids, mask, return_tokens=False):
        """
        Encodes the first (3) multimodal steps into a memory vector and dependently tokens.
        Args:
            imgs: Tensor (B, 3, 3, 224, 224) context images.
            ids: Tensor (B, 3, L) context token ids.
            mask: Tensor (B, 3, L) attention masks.
            return_tokens: If True, returns both (mem, tokens) where tokens is (B,3,D).
        Returns:
            If return_tokens is False:
                mem: Tensor (B, D) final memory vector.
            If return_tokens is True:
                (mem, tokens):
                    mem: Tensor (B, D)
                    tokens: Tensor (B, 3, D) temporal tokens (one per context step).
        """
        z = []
        for t in range(3):
            zi = self.img_enc(imgs[:, t])
            zt = self.txt_enc(ids[:, t], mask[:, t])
            z.append(self.fusion(zi, zt))
        z = torch.stack(z, dim=1)      # (B,3,512)
        h = self.temporal(z)           # (B,3,512)
        mem = h[:, -1]
        return (mem, h) if return_tokens else mem

def mem_collapse_loss(tokens: torch.Tensor):
    """
    Serves as anti-collapse regularizer for memory vectors.

    Allows per-dimension standard deviation to stay above a threshold
    across the batch, preventing the model from producing constant memory
    representations or same output generation.
    Args:
        mem: Tensor (B, D) memory vectors.

    Returns:
        Scalar tensor regularization loss.
    """
    x = tokens.reshape(tokens.size(0), -1)
    if x.size(0) < 2:
        return x.new_tensor(0.0)
    std = x.std(dim=0)
    return F.relu(0.05 - std).mean()

class DecoderText(nn.Module):
    """
    Autoregressive text decoder conditioned on memory tokens (z_pred).

    Uses a TransformerDecoder:
      - Input: GPT-2 token ids (teacher forcing during training)
      - Memory: conditioning tokens z_pred (B,T,512)
      - Output: logits over GPT-2 vocabulary for next-token prediction

    Args:
        vocab_size: GPT-2 vocab size.
        d_model: Hidden size (must match z_pred embedding dim, typically 512).
        n_layers: Number of TransformerDecoder layers.
        n_heads: Number of attention heads.

    Inputs:
        input_ids: LongTensor (B, L)
        memory_tokens: FloatTensor (B, T, d_model)

    Outputs:
        logits: FloatTensor (B, L, vocab_size)
    """
    def __init__(self, vocab_size, max_len, d_model=512, n_layers=3, n_heads=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, batch_first=True, activation="gelu"
        )
        self.dec = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, memory_tokens):
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x = self.embed(input_ids) + self.pos(pos)
        causal = nn.Transformer.generate_square_subsequent_mask(L, device=input_ids.device)
        h = self.dec(x, memory_tokens, tgt_mask=causal)
        return self.out(h)

def build_unet_decoder(sd_model_id: str, device: str, dtype):
    """
    Build and configure a Stable Diffusion UNet decoder for latent diffusion training/inference.
    This function enable and loads the pretrained UNet from a Stable Diffusion checkpoint, enables memory saving
    features the gradient checkpointing and optional xFormers attention, which moves the model to the
    requested device and dtype.
    
    Args:
        sd_model_id (str):
            HuggingFace model ID or local path for the Stable Diffusion checkpoint
            (e.g., "runwayml/stable-diffusion-v1-5").
        device (str):
            Device where the UNet should run, typically "cuda" or "cpu".
        dtype:
            Torch dtype to load the UNet weights with, e.g.:
            - torch.float16 (recommended for CUDA)
            - torch.float32 (recommended for CPU)
    Returns:
        UNet2DConditionModel:
            The configured UNet model moved to `device` and using the given `dtype`.
    Notes:
        - "enable_gradient_checkpointing()" reduces VRAM usage during training by trading compute for memory.
        - "enable_xformers_memory_efficient_attention()" further reduces VRAM if xFormers is installed.
          And if xFormers is not available, the function safely skips it.
        - This UNet expects latents shaped like (B, 4, 64, 64) and conditioning embeddings
          provided through "encoder_hidden_states".
    """
    unet = UNet2DConditionModel.from_pretrained(sd_model_id, subfolder="unet", torch_dtype=dtype)
    unet.enable_gradient_checkpointing()
    try:
        unet.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    return unet.to(device)

