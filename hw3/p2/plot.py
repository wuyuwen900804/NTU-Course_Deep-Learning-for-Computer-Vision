import os
import timm
import json
import math
import torch
import argparse
import collections
from torch import nn
from PIL import Image
import loralib as lora
import torch.nn.functional as F
from tokenizer import BPETokenizer
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

if __name__ == '__main__':
    
    args = argparse.ArgumentParser()
    args.add_argument('--image_folder', type=str, default='hw3_data/p3_data/images')
    args.add_argument('--output_json', type=str, default='hw3/output_p3/pred.json')
    args.add_argument('--decoder_checkpoint', type=str, default='./hw3_data/p2_data/decoder_model.bin')
    
    # get arguments
    image_folder = args.parse_args().image_folder
    output_json = args.parse_args().output_json
    decoder_checkpoint = args.parse_args().decoder_checkpoint

    encoder_json_path = 'encoder.json'
    vocab_file_path = 'vocab.bpe'
    tokenizer = BPETokenizer(encoder_file=encoder_json_path, vocab_file=vocab_file_path)

    # prepare dataset
    val_transform = transforms.Compose([
        transforms.Resize((224,224)),  
        transforms.ToTensor(),
    ])
    val_dataset = ImgDataset(image_folder, transform=val_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    # prepare model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImgTextModel().to(device)

    checkpoint_path = 'inference_model/best.pth'
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict, strict=False)

    def top_k_sampling_decoder(model, image_features, tokenizer, start_token, image_id, k=10, max_len=30):
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
        repetition_penalty = generate_kwargs.get("repetition_penalty", 1.2)
        
        # 初始序列包含 start_token，並初始化分數
        generated_sequence = [torch.tensor([start_token], device=device)]
        
        fig, ax = plt.subplots(ncols=5, nrows=4, figsize=(16, 8))
        for i in range(4):
            for j in range(5):
                ax[i][j].axis('off')
        ax[0][0].imshow(images.squeeze(0).permute(1, 2, 0).cpu().numpy())
        ax[0][0].set_title('<|endoftext|>')
        image_list = []
        attn_list = []
        token_list = []
        for step in range(max_len-1):
            # 將目前的序列轉換為 Tensor 並加入維度
            # print(generated_sequence)
            text_tokens = torch.cat(generated_sequence, dim=0).unsqueeze(0)  # [1, seq_len]
            
            # 通過 Decoder 獲取當前步的預測
            predictions, attn_weights_list = model.decoder(image_features, text_tokens, attn_map=True)   # [1, seq_len, vocab_size]
            logits = predictions[:, -1, :]  # 取得最新生成的詞的 logits [1, vocab_size]
            
            # 溫度縮放
            logits /= temperature
            
            # 懲罰重複詞彙
            # for token_id in generated_sequence:
            #     logits[0, token_id] /= repetition_penalty
            
            
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

            if step < len(ax.flatten()) - 1:
                # attn_weights size : [head, seq_len, seq_len]
                attn_weights = attn_weights_list.mean(dim=1)  # Average over heads
                token_idx = 256 + step  # Current token's index
                patch_attention = attn_weights[0, token_idx, :256]  
                # print(patch_attention)
                patch_attention = patch_attention.view(16, 16)  # Reshape to grid (16x16)

                # Resize attention map to match image dimensions
                patch_attention_resized = F.interpolate(
                    patch_attention.unsqueeze(0).unsqueeze(0),  # Add batch and channel dimensions
                    size=(images.size(2), images.size(3)),  # Match image size (H, W)
                    mode="bilinear",
                    align_corners=False
                ).squeeze()
                
                # image_list.append(images.squeeze(0).permute(1, 2, 0).cpu().numpy())
                # attn_list.append(patch_attention_resized.cpu().numpy())
                # token_list.append(next_token)
                count = step + 1
                # Overlay attention map on the image
                ax[count // 5][count % 5].imshow(images.squeeze(0).permute(1, 2, 0).cpu().numpy(), alpha=0.7)
                ax[count // 5][count % 5].imshow(patch_attention_resized.cpu().numpy(), cmap="viridis", alpha=0.5)
                token_text = tokenizer.decode([next_token.item()])
                ax[count // 5][count % 5].set_title(token_text, fontsize=12)
            
            # 將選中的 token 加入序列
            generated_sequence.append(next_token)
            
            # 停止條件檢查，如果生成了 end_token 可以提前結束
            if next_token.item() == start_token:  # 假設 end_token 與 start_token 相同
                break
        # generate_text = tokenizer.decode(token_list)  # 去掉 start_token
        # 將生成的序列解碼
        best_sequence_ids = [token.item() for token in generated_sequence[1:-1]]  # 去掉 start_token
        decoded_text = tokenizer.decode(best_sequence_ids)
        word_list = decoded_text.split(" ")
        
        plt.tight_layout()
        attn_map_path = './att_map'
        os.makedirs(attn_map_path, exist_ok=True)
        plt.savefig(os.path.join(attn_map_path,f'attention_map_{image_id}.jpg'))
        plt.close()
        
        return decoded_text

    output = {}
    model.eval()
    with torch.no_grad():
        for index, (images, image_id) in enumerate(val_dataloader):
            images = images.to(device)
            start_token = tokenizer.encode('<|endoftext|>', allowed_special=['<|endoftext|>'])[0]
        
            with torch.no_grad():
                # 提取圖像特徵並投影到嵌入空間
                features = model.encoder.forward_features(images)
                image_id = image_id[0].split('.')[0]
                predicted_tokens = top_k_sampling_decoder(model, features, tokenizer, start_token, index, k=20, max_len=20)
                output[image_id] = predicted_tokens  # 去掉 end_token

    
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(output, f, indent=4) 