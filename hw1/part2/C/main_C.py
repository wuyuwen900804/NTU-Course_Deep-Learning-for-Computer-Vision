import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

image = cv2.imread('hw1_data/p2_data/validation/0050_sat.jpg')
# image = cv2.imread('hw1_data/p2_data/validation/0130_sat.jpg')
# image = cv2.imread('hw1_data/p2_data/validation/0256_sat.jpg')
image = cv2.resize(image, None, fx=0.5, fy=0.5)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

sam_checkpoint = 'part2/C/sam_vit_h_4b8939.pth'
model_type = 'vit_h'

device = 'cuda'

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image)

plt.figure(figsize=(16,16))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show()