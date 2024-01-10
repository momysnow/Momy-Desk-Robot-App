from transformers import AutoImageProcessor, DPTForDepthEstimation
import torch
import numpy as np
from PIL import Image
import requests
import matplotlib.pyplot as plt
import time

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

image_processor = AutoImageProcessor.from_pretrained("facebook/dpt-dinov2-small-kitti")
model = DPTForDepthEstimation.from_pretrained("facebook/dpt-dinov2-small-kitti")

# Prepare image for the model
inputs = image_processor(images=image, return_tensors="pt")

# Measure time for prediction
start_time = time.time()
with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth
elapsed_time = time.time() - start_time
print(f"Time elapsed for prediction: {elapsed_time} seconds")

# Interpolate to original size
prediction = torch.nn.functional.interpolate(
    predicted_depth.unsqueeze(1),
    size=image.size[::-1],
    mode="bicubic",
    align_corners=False,
)

# Visualize the prediction
output = prediction.squeeze().cpu().numpy()
formatted = (output * 255 / np.max(output)).astype("uint8")
depth = Image.fromarray(formatted)

# Display the original image and the predicted depth map side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Original Image
axes[0].imshow(image)
axes[0].set_title('Original Image')

# Predicted Depth Map
depth_plot = axes[1].imshow(depth, cmap='viridis')  # You can choose any colormap here
axes[1].set_title('Predicted Depth Map')

# Add colorbar for the depth map
cbar = plt.colorbar(depth_plot, ax=axes[1], fraction=0.046, pad=0.04)
cbar.set_label('Depth Value')

plt.show()
