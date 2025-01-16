import os
import sys
import json
import torch
import random
import numpy as np
from PIL import Image
from transformers import pipeline

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

def main():
  model_id = "llava-hf/llava-1.5-7b-hf"
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  pipe = pipeline("image-to-text", model=model_id, torch_dtype=torch.float16, device=device)
  prompt = "<image>\nGenerate a concise caption that describes the main subject, action, or context in the image. \
        Focus on essential elements, such as people, objects, and activities, while keeping the description \
        straightforward and informative.\nCaption:"

  image_folder = sys.argv[1]

  images = []
  image_names = []
  for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)
    image = Image.open(image_path).convert("RGB")
    images.append(image)
    image_names.append(os.path.splitext(image_name)[0])

  generate_kwargs = {
    "max_new_tokens": 20,   
    "do_sample": True,
    "num_beams": 8,
    "top_k": 25
  }
  # CIDEr: 1.1638129928988272 | CLIPScore: 0.7813665771484375
  # generate_kwargs = {
  #   "max_new_tokens": 15,   
  #   "do_sample": True,
  #   "num_beams": 8,
  #   "top_k": 25
  # }
  # CIDEr: 1.1800517657900251 | CLIPScore: 0.771376953125

  outputs = pipe(images, prompt=prompt, generate_kwargs=generate_kwargs)

  results = {image_name: output[0]["generated_text"].split("Caption:")[-1].strip() 
            for image_name, output in zip(image_names, outputs)}

  output_file = sys.argv[2]
  output_dir = os.path.dirname(output_file)
  if output_dir and not os.path.exists(output_dir):
      os.makedirs(output_dir)
  with open(output_file, "w") as f:
      json.dump(results, f, ensure_ascii=False, indent=2)

  print(f"Results saved to {output_file}")

if __name__ == "__main__":
  main()