import os
import sys
import torch
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration

import matplotlib
matplotlib.use('Agg')

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

class ImgDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.image_list = os.listdir(image_folder)
        self.image_list.sort()
        self.image_path = [os.path.join(image_folder, img) for img in self.image_list]
        self.transform = transform

    def __len__(self):
        return len(self.image_list)
    def __getitem__(self, idx):
        return self.image_path[idx]

def visualize_individual_attention(image, attentions, inputs_len, output_tokens, output_dir, idx):
    os.makedirs(output_dir, exist_ok=True)

    # 動態計算行數和列數
    tokens_per_row = 5
    total_tokens = len(output_tokens) + 1  # 包括初始圖像
    nrows = (total_tokens + tokens_per_row - 1) // tokens_per_row
    ncols = min(tokens_per_row, total_tokens)

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 4 * nrows))

    # 確保 ax 是 2D 陣列
    if nrows == 1:
        ax = [ax]

    for i in range(nrows):
        for j in range(ncols):
            ax[i][j].axis('off')

    transform = transforms.ToTensor()
    image_tensor = transform(image)
    ax[0][0].imshow(image_tensor.permute(1, 2, 0).cpu().numpy())
    ax[0][0].set_title('<|start|>')

    attention_map = attentions.mean(dim=1)

    for i, token_text in enumerate(output_tokens):
        attention = attention_map[0, i + 576, 1:577]
        attention = attention.view(24, 24).cpu().detach()

        attention_resized = F.interpolate(
            attention.unsqueeze(0).unsqueeze(0),
            size=(image.size[1], image.size[0]),
            mode="bilinear",
            align_corners=False,
        ).squeeze()

        row, col = divmod(i + 1, tokens_per_row)
        ax[row][col].imshow(image_tensor.permute(1, 2, 0).cpu().numpy(), alpha=0.7)
        ax[row][col].imshow(attention_resized.cpu().numpy(), cmap="coolwarm", alpha=0.7)
        ax[row][col].set_title(token_text, fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"att_summary_{idx}.jpg"))
    plt.close()


def main():
  model_id = "llava-hf/llava-1.5-7b-hf"
  model = LlavaForConditionalGeneration.from_pretrained(
      model_id,
      torch_dtype=torch.float,
      output_attentions=True,
      low_cpu_mem_usage=True
  ).to(0)
  if not model.config.output_attentions:
    model.config.output_attentions = True
  processor = AutoProcessor.from_pretrained(model_id)
  prompt = "<image>\nGenerate a concise caption that describes the main subject, action, or context in the image. \
        Focus on essential elements, such as people, objects, and activities, while keeping the description \
        straightforward and informative.\nCaption:"

  image_folder = sys.argv[1]
  image_dataset = ImgDataset(image_folder, transform=None)

  for idx in range(5):
    image_path = image_dataset[idx]
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")

    generate_kwargs = {
      "max_new_tokens": 20,       
      "do_sample": True,
      "top_k": 25
    }
    outputs = model.generate(**inputs, **generate_kwargs)
  
    with torch.no_grad():
      outputs_with_attn = model(
          input_ids=inputs["input_ids"],
          pixel_values=inputs["pixel_values"],
          attention_mask=inputs["attention_mask"],
          output_attentions=True,
      )
    
    attentions = outputs_with_attn.attentions[-1] 

    input_ids = inputs["input_ids"][0]
    default_prompts_len = len(input_ids)+1

    outputs = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    outputs = outputs[0].split()[33:]

    output_path = "./hw3/output_p3"
    os.makedirs(output_path, exist_ok=True)

    visualize_individual_attention(
        image=image,
        attentions=attentions,
        inputs_len=default_prompts_len,
        output_tokens=outputs,
        output_dir=output_path,
        idx = idx
    )

if __name__ == "__main__":
  main()