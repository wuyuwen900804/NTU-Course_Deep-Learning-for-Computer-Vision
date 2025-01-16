import os
import timm
import json
import math
import torch
import random
import collections
import numpy as np
from torch import nn
from PIL import Image
from tqdm import tqdm
import loralib as lora
import torch.nn.functional as F
from torchvision import transforms
from tokenizer import BPETokenizer
from timm.data import resolve_data_config
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from scheduler import CosineAnnealingWarmupRestarts
from timm.data.transforms_factory import create_transform

def set_seed(seed):
  np.random.seed(seed)
  random.seed(seed)
  torch.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  if torch.cuda.is_available():
      torch.cuda.manual_seed_all(seed)
      torch.cuda.manual_seed(seed)
set_seed(0)

BOS_TOKEN, EOS_TOKEN, PAD_TOKEN = 50256, 50256, 50256

### ImgTextDataset ###
class ImgTextDataset(Dataset):
    def __init__(self, folder, tokenizer, transform=None, mode='train'):
        self.folder = folder
        self.image_folder = os.path.join(folder, f'images/{mode}')
        self.json_path = os.path.join(folder, f'{mode}.json')
        self.tokenizer = tokenizer
        self.transform = transform
        self.mode = mode
        img_annotation = json.load(open(self.json_path))['images']
        self.images = {img['id']: img['file_name'] for img in img_annotation}
        self.annotations = json.load(open(self.json_path))['annotations']

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        # image file
        image_id = annotation.get('image_id', '')
        image_path = os.path.join(self.image_folder, self.images[image_id])
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        # tokenize caption    
        caption = annotation.get('caption', '')
        caption_tokens = self.tokenizer.encode(caption, allowed_special=['<|endoftext|>'])
        caption_tokens = [BOS_TOKEN] + caption_tokens
        # pad captions to the same length
        max_caption_length = 64
        if len(caption_tokens) < max_caption_length:
            caption_tokens += [PAD_TOKEN] * (max_caption_length - len(caption_tokens))
        else:
            caption_tokens = caption_tokens[:max_caption_length]
        caption_tokens = torch.tensor(caption_tokens, dtype=torch.long)
          
        return image, caption_tokens, image_id
### ImgTextDataset ###

### Decoder ###
class Config:
    def __init__(self, checkpoint=None):
        self.n_layer = 12
        self.n_head = 12
        self.n_embd = 768
        self.vocab_size = 50257
        self.block_size = 1024
        self.checkpoint = checkpoint

class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.c_attn = lora.Linear(cfg.n_embd, 3 * cfg.n_embd, r=40)
        self.c_proj = lora.Linear(cfg.n_embd, cfg.n_embd, r=40)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        size = cfg.block_size
        self.register_buffer('bias', torch.tril(torch.ones(size, size)).view(1, 1, size, size))

    def forward(self, x):
        B, T, C = x.size() # batch, context, embedding
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        return self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, C)), att

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        # multi-layer perceptron
        self.mlp = nn.Sequential(collections.OrderedDict([
            ('c_fc', nn.Linear(cfg.n_embd, 4 * cfg.n_embd)),
            ('act', nn.GELU(approximate='tanh')),
            ('c_proj', nn.Linear(4 * cfg.n_embd, cfg.n_embd))
        ]))

    def forward(self, x):
        attn_out, attn_weights = self.attn(self.ln_1(x))
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, attn_weights

class Decoder(nn.Module):
    def __init__(self, cfg, visual_dim=1664):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.block_size

        hidden_dim = 896
        self.visual_projection = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, cfg.n_embd)
        )
        
        # learnable gates to adjust the relative importance of image and text features dynamically
        self.gate_image = nn.Parameter(torch.ones(cfg.n_embd))
        self.gate_text = nn.Parameter(torch.ones(cfg.n_embd))

        self.transformer = nn.ModuleDict(dict(
            wte=lora.Embedding(cfg.vocab_size, cfg.n_embd),  # (50257,768)
            wpe=lora.Embedding(cfg.block_size, cfg.n_embd),  # (1024,768)
            h=nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f=nn.LayerNorm(cfg.n_embd),
        ))
        self.lm_head = lora.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        
        # load checkpoint
        if self.cfg.checkpoint is not None:
            state_dict = torch.load(self.cfg.checkpoint)
            transposed = ['.c_attn.weight', '.c_fc.weight', '.c_proj.weight']
            for key, value in state_dict.items():
                if any(key.endswith(w) for w in transposed):
                    state_dict[key] = value.t()
            self.transformer.load_state_dict(state_dict, strict=False)

    def forward(self, image_features, text_tokens, attn_map=False):
        # remove the first token of vit features
        projected_features = self.visual_projection(image_features[:, 1:, :])
        # add positional embeddings to text tokens
        pos = torch.arange(text_tokens.size(1), dtype=torch.long, device=text_tokens.device).unsqueeze(0)
        text_embeddings = self.transformer.wte(text_tokens) + self.transformer.wpe(pos)
        # adjust the relative importance of image and text features
        projected_features = projected_features * self.gate_image
        text_embeddings = text_embeddings * self.gate_text
        
        attn_weights_list = []
        inputs = torch.cat([projected_features, text_embeddings], dim=1)
        x = inputs
        for block in self.transformer.h:
            x, attn_weights = block(x)
            attn_weights_list.append(attn_weights)   
        x = self.transformer.ln_f(x)
        # [layer_num, batch_size, head_num, max_len, encode_size**2]
        attn_weights_list = torch.stack(attn_weights_list)
        attn_weights_list = torch.mean(attn_weights_list, dim=2)
        
        # [batch_size, sequence_length, vocab_size]
        logits = self.lm_head(x)
        # print(logits.size())

        # [batch_size, text_sequence_length, vocab_size]
        logits_text = logits[:, 256:,]
        
        if attn_map:
            return logits_text, attn_weights_list
        else:
            return logits_text
### Decoder ###

### ImgTextModel ###
class ImgTextModel(nn.Module):
    def __init__(self):
        super(ImgTextModel, self).__init__()
        # Encoder
        self.encoder = timm.create_model("vit_gigantic_patch14_clip_224", pretrained=True, num_classes=0)
        for param in self.encoder.parameters():
            param.requires_grad = False
        # Decoder
        self.cfg = Config("hw3_data/p2_data/decoder_model.bin")
        self.decoder = Decoder(self.cfg)

    def forward(self, images, captions):
        features = self.encoder.forward_features(images)
        # features size [batch_size, patches, embedding]
        outputs = self.decoder(features, captions)
        return outputs
### ImgTextModel ###

def main():
    EPOCHS = 50

    tokenizer = BPETokenizer("encoder.json", "vocab.bpe")
    # transform = create_transform(**resolve_data_config({}, model="vit_gigantic_patch14_clip_224"))
    train_transform = transforms.Compose([
        transforms.Resize((224,224)),  
        transforms.AutoAugment(),
        transforms.ToTensor(),
    ])
    valid_transform = transforms.Compose([
        transforms.Resize((224,224)),  
        transforms.ToTensor(),
    ])
    train_dataset = ImgTextDataset('hw3_data/p2_data', tokenizer=tokenizer, transform=train_transform, mode = 'train')
    valid_dataset = ImgTextDataset('hw3_data/p2_data', tokenizer=tokenizer, transform=valid_transform, mode = 'val')
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = ImgTextModel().to(device)
    lora.mark_only_lora_as_trainable(model)

    for param in model.decoder.visual_projection.parameters():        
        param.requires_grad = True

    # for param in model.decoder.self_attention.parameters():
    #     param.requires_grad = True
    # model.decoder.gate_image.requires_grad = True
    # model.decoder.gate_text.requires_grad = True

    # for param in model.decoder.transformer.h.parameters():
    #     param.requires_grad = True

    # for block in model.decoder.transformer.h:
    #     for param in block.attn_cross.parameters():
    #         param.requires_grad = True

    # saved_weights_path = "best.pth"
    # if os.path.exists(saved_weights_path):
    #     state_dict = torch.load(saved_weights_path)
    #     model.load_state_dict(state_dict, strict=False)
    #     print(f"Loaded saved model weights from {saved_weights_path}")

    # AMP to accelerate training (merge precision)
    # ref : https://hackmd.io/@Hong-Jia/H1hmbNr1d
    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    total_steps = len(train_dataloader) * EPOCHS
    warmup_steps = 0.1 * total_steps  
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=total_steps,
        cycle_mult=1.0,
        max_lr=1e-4,
        min_lr=1e-5,
        warmup_steps=warmup_steps,
        gamma=1.0,
    )

    for epoch in range(EPOCHS):
        train_loss = 0
        model.train()
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for images, captions_token, image_ids in progress_bar:
            images = images.to(device)
            captions_token = captions_token.to(device)
            optimizer.zero_grad()
            with autocast():
                # output size [batch_size, caption_length, vocab_size]
                outputs = model(images, captions_token)
                # flatten caption_length and batch_size to get logits with shape [batch_size * caption_length, vocab_size]
                outputs = outputs[:, :-1,:].contiguous()
                logits_flat = outputs.reshape(-1, 50257)
                # flatten caption_length and batch_size to get targets with shape [batch_size * caption_length]
                captions_token = captions_token[:,1:].contiguous()
                for i in range(captions_token.size(0)):
                    # get the index of the [END] token
                    end_token_indices = (captions_token[i] == EOS_TOKEN).nonzero()
                    # if there is an [END] token, numel() would be greater than 0
                    # numel() returns the number of elements in a tensor
                    if end_token_indices.numel() > 0:
                        # get the column index of the first [END] token
                        col = end_token_indices[0]
                        if col < captions_token.size(1) - 1:
                            captions_token[i, col + 1:] = -100
                targets_flat = captions_token.reshape(-1)
                loss = criterion(logits_flat, targets_flat)
                if torch.isnan(loss).any():
                    print("NaN detected in loss!")
                    continue
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_dataloader)
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Learning rate: {scheduler.get_lr()[0]}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, caption_tokens, _ in tqdm(valid_dataloader):
                images, captions_token = images.to(device), caption_tokens.to(device)
                with autocast():
                    outputs = model(images, captions_token)
                    outputs = outputs[:, :-1,:].contiguous()
                    logits_flat = outputs.reshape(-1, 50257)
                    captions_token = captions_token[:,1:].contiguous()
                    for i in range(captions_token.size(0)):
                        end_token_indices = (captions_token[i] == 50256).nonzero()
                        if end_token_indices.numel() > 0:
                            col = end_token_indices[0]
                            if col < captions_token.size(1) - 1:
                                captions_token[i, col + 1:] = -100
                    targets_flat = captions_token.reshape(-1)

                    loss = criterion(logits_flat, targets_flat)
                val_loss += loss.item()
            avg_val_loss = val_loss / len(valid_dataloader)
            print(f"Validation Loss: {avg_val_loss:.4f}")

        trainable_weights = [name for name, param in model.named_parameters() if param.requires_grad == True]
        print(f"Total trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6}M")
        save_weights = {k: v for k, v in model.state_dict().items() if k in trainable_weights}
        checkpoint_path = os.path.join('./p2/model_checkpoints', f'epoch_{epoch}.pth')
        torch.save(save_weights, checkpoint_path)

if __name__ == "__main__":
    main()