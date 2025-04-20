from PIL import Image
import numpy as np

img = Image.open(r"C:\Users\Admin\Desktop\QGIS\FINAL FILES\biodiversity_dataset png 512\annotations_single_channel\train\biodiversity_0004.png")

# Check the image mode like RGB or single channel
print("Image mode:", img.mode)  

# Check shape via NumPy
arr = np.array(img)
print("Array shape:", arr.shape)
#      (height, width, 4) -> 4 channels (often RGBA)
#      (height, width)    -> 1 channel (grayscale)
