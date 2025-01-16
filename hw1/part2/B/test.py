import os
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt
from torchsummary import summary
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from dataset import ImageDataset
from mean_iou_evaluate import read_masks, mean_iou_score
from MobileNet import Deeplabv3_MobilenetV3_Model

def mask_to_color(mask):
    color_map = {
        0: [0, 255, 255],  # Cyan (Urban land)
        1: [255, 255, 0],  # Yellow (Agriculture land)
        2: [255, 0, 255],  # Purple (Rangeland)
        3: [0, 255, 0],    # Green (Forest land)
        4: [0, 0, 255],    # Blue (Water)
        5: [255, 255, 255],# White (Barren land)
        6: [0, 0, 0],      # Black (Unknown)
    }
    
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for label, color in color_map.items():
        color_mask[mask == label] = color
    
    return color_mask

def plot_comparison(predictions, masks, idx, filename):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    pred_color = mask_to_color(predictions)
    true_color = mask_to_color(masks)

    axs[0].imshow(pred_color)
    axs[0].set_title(f'Predicted Mask {idx} - {filename}')
    axs[0].axis('off')

    axs[1].imshow(true_color)
    axs[1].set_title(f'True Mask {idx} - {filename}')
    axs[1].axis('off')

    plt.show()

img_dir_valid = "hw1_data/p2_data/validation"

transform_valid = transforms.Compose([
    transforms.Resize(512),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

valid_dataset = [imgfile for imgfile in os.listdir(img_dir_valid) if imgfile.endswith(".jpg")]
valid_dataset.sort()
valid_data = ImageDataset(img_dir_valid, valid_dataset, read_masks(img_dir_valid), transform_valid)
valid_loader = DataLoader(dataset=valid_data, batch_size=4, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

# model = Deeplabv3_MobilenetV3_Model()
# model.load_state_dict(torch.load("part2/B_best_iou.pt"))
# model = model.to(device)
# # summary(model, (3, 512, 512))

# val_prediction_list, val_masks_list = [], []
# model.eval()
# with torch.no_grad():
#     for i, (images, masks) in enumerate(valid_loader):
#         test_images = images.to(torch.float32).to(device)
#         test_masks = masks.to(torch.float32).to(device)

#         outputs = model(test_images)
#         prediction = torch.max(outputs, 1)[1]

#         val_prediction_list.extend(prediction.cpu().numpy().flatten())
#         val_masks_list.extend(masks.cpu().numpy().flatten())

# val_iou = mean_iou_score(np.array(val_prediction_list), np.array(val_masks_list))


# model.eval()
# with torch.no_grad():
#     batch_count = 0
#     for i, (images, masks) in enumerate(valid_loader):

#         test_images = images.to(torch.float32).to(device)
#         test_masks = masks.to(torch.float32).to(device)

#         outputs = model(test_images)
#         prediction = torch.max(outputs, 1)[1]

#         for j in range(images.size(0)):
#             pred_mask = prediction[j].cpu().numpy()
#             true_mask = masks[j].cpu().numpy()
#             filename = valid_dataset[i * valid_loader.batch_size + j]
#             plot_comparison(pred_mask, true_mask, i * valid_loader.batch_size + j, filename)

#         batch_count += 1

#         if batch_count == 5:
#             break


model = Deeplabv3_MobilenetV3_Model()
# model.load_state_dict(torch.load("part2/B_EPOCH_1.pt"))
# model.load_state_dict(torch.load("part2/B_EPOCH_75.pt"))
model.load_state_dict(torch.load("part2/B_EPOCH_150.pt"))
model = model.to(device)
model.eval()

with torch.no_grad():
    for i, (images, masks) in enumerate(valid_loader):
        test_images = images.to(torch.float32).to(device)
        test_masks = masks.to(torch.float32).to(device)
        outputs = model(test_images)
        prediction = torch.max(outputs, 1)[1]
        for j in range(images.size(0)):
            filename = valid_dataset[i * valid_loader.batch_size + j]
            if filename in ['0013_sat.jpg', '0062_sat.jpg', '0104_sat.jpg']:
                img_path = f'hw1_data/p2_data/validation/{filename}'
                img = Image.open(img_path)
                true_mask = masks[j].cpu().numpy()
                pred_mask = prediction[j].cpu().numpy()
                plt.imshow(img)
                # plt.imshow(mask_to_color(true_mask), alpha=0.3)
                plt.imshow(mask_to_color(pred_mask), alpha=0.3)
                plt.axis('off')
                plt.show()