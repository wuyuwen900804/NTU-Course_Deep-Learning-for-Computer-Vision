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

### ImgDataset ###
class ImgDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.images = os.listdir(image_folder)
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.images[idx])
        image = Image.open(image_path).convert("RGB")
        name = self.images[idx].split('.')[0]
        if self.transform:
            image = self.transform(image)
        return image, name
### ImgDataset ###

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
        for param in self.decoder.parameters():
            param.requires_grad = False

    def forward(self, images, captions):
        features = self.encoder.forward_features(images)
        # features size [batch_size, patches, embedding]
        outputs = self.decoder(features, captions)
        return outputs
### ImgTextModel ###

### BeamSearch ###
def beam_search(model, image_features, tokenizer, start_token, index, beam_width=3, max_len=15):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 初始候選序列，每個序列包含 start_token，並將分數初始化為 0
    sequences = [[torch.tensor([start_token], device=device)]]
    scores = torch.zeros(1, device=device)
    for step in range(max_len-1):
        all_candidates = []
        # 將當前的序列轉換為 [beam_width, seq_len]
        if step  == 1:
            image_features = image_features.repeat(beam_width, 1, 1) 
        text_tokens = [torch.cat(seq) for seq in sequences]
        text_tokens = torch.stack(text_tokens, dim=0) 
        # 通過 Decoder 獲取當前步的預測
        predictions, attn_weights_list = model.decoder(image_features, text_tokens, attn_map=True)  # [beam_width, seq_len, vocab_size]
        logits = predictions[:, -1, :]  # 取得最新生成的詞的 logits [beam_width, vocab_size]
        log_probs = torch.log_softmax(logits, dim=-1)  # 計算 log 概率
        # 選擇每個序列中得分最高的 beam_width 個詞
        topk_scores, topk_indices = torch.topk(log_probs, beam_width, dim=-1)  # [beam_width, beam_width]
        # 更新候選序列
        for i in range(len(sequences)):
            seq, score = sequences[i], scores[i]
            for j in range(beam_width):
                next_token = topk_indices[i, j].unsqueeze(0)  # 新生成的 token
                candidate_seq = seq + [next_token]  # 更新候選序列
                candidate_score = score + topk_scores[i, j].item()  # 更新分數
                all_candidates.append((candidate_seq, candidate_score))
        # 選擇分數最高的 beam_width 條序列
        ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
        sequences = [candidate[0] for candidate in ordered[:beam_width]]
        scores = torch.tensor([candidate[1] for candidate in ordered[:beam_width]], device=device)
        # 檢查是否生成了 end_token，若有則提前結束
        if all(seq[-1].item() == start_token for seq in sequences):  # 假設 end_token 與 start_token 相同
            break
    # 選取分數最高的序列並解碼
    best_sequence = sequences[0]
    best_sequence_ids = [token.item() for token in best_sequence[1:]]  # 去掉 start_token
    for idx, (token) in enumerate(best_sequence_ids):
        if token == start_token:
            best_sequence_ids = best_sequence_ids[:idx]
            break
    decoded_text = tokenizer.decode(best_sequence_ids)

    return decoded_text 
### BeamSearch ###

### Top-K ###
def top_k_sampling(model, image_features, tokenizer, start_token, image_id, k=10, max_len=30):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 初始序列包含 start_token，並初始化分數
    generate_kwargs = {
        "max_new_tokens": 30,
        "temperature": 0.3,
        "top_p": 0.9,
        "top_k": 30,
        "do_sample": True,
    }
    # 提取生成參數
    max_len = generate_kwargs.get("max_new_tokens", 30)
    temperature = generate_kwargs.get("temperature", 3.0)
    top_p = generate_kwargs.get("top_p", 0.9)
    k = generate_kwargs.get("top_k", 30)
    # 初始序列包含 start_token，並初始化分數
    generated_sequence = [torch.tensor([start_token], device=device)]
    for step in range(max_len-1):
        # 將目前的序列轉換為 Tensor 並加入維度
        text_tokens = torch.cat(generated_sequence, dim=0).unsqueeze(0)  # [1, seq_len]
        # 通過 Decoder 獲取當前步的預測
        predictions, attn_weights_list = model.decoder(image_features, text_tokens, attn_map=True)   # [1, seq_len, vocab_size]
        logits = predictions[:, -1, :]  # 取得最新生成的詞的 logits [1, vocab_size]
        # 溫度縮放
        logits /= temperature
        # 計算詞彙的機率並使用 Top-k 取樣
        probs = torch.softmax(logits, dim=-1)  # [1, vocab_size]
         # Top-p (核取樣) 過濾
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()  # 保證至少有一個保留
        sorted_indices_to_remove[..., 0] = 0  # 保留最高概率的詞彙
        sorted_probs[sorted_indices_to_remove] = 0
        # 從 top-k 概率中隨機選擇下一個 token
        topk_probs, topk_indices = torch.topk(probs, k, dim=-1)  # 取得 top-k 的概率和索引 [1, k]
        next_token = topk_indices[0, torch.multinomial(topk_probs[0], 1)]  # 選出下一個 token
        # 將選中的 token 加入序列
        generated_sequence.append(next_token)
        # 停止條件檢查，如果生成了 end_token 可以提前結束
        if next_token.item() == start_token:  # 假設 end_token 與 start_token 相同
            break
    # 將生成的序列解碼
    best_sequence_ids = [token.item() for token in generated_sequence[1:-1]]  # 去掉 start_token
    decoded_text = tokenizer.decode(best_sequence_ids)
    
    return decoded_text 
### Top-K ###

def main():
    tokenizer = BPETokenizer("encoder.json", "vocab.bpe")
    # transform = create_transform(**resolve_data_config({}, model="vit_gigantic_patch14_clip_224"))
    valid_transform = transforms.Compose([
        transforms.Resize((224,224)),  
        transforms.ToTensor(),
    ])

    valid_dataset = ImgDataset('hw3_data/p2_data/images/val', transform=valid_transform)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = ImgTextModel().to(device)   
    lora_checkpoint = "p2/model_checkpoints/epoch_14.pth"
    print(lora_checkpoint)
    lora_state_dict = torch.load(lora_checkpoint, map_location=device)
    model.load_state_dict(lora_state_dict, strict=False)

    model.eval()

    results = {}

    with torch.no_grad():
        for index, (images, image_id) in enumerate(valid_dataloader):
            images = images.to(device)
            start_token = tokenizer.encode('<|endoftext|>', allowed_special=['<|endoftext|>'])[0]
            with torch.no_grad():
                features = model.encoder.forward_features(images)
                image_id = image_id[0].split('.')[0]
                predicted_tokens = beam_search(model, features, tokenizer, start_token, index, beam_width=3, max_len=15)
                # predicted_tokens = top_k_sampling(model, features, tokenizer, start_token, image_id, k=20, max_len=15)
                results[image_id] = predicted_tokens  # 去掉 end_token

    output_path = "p2_result_e14.json"
    with open(output_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
