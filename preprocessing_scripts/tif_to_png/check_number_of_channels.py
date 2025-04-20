from PIL import Image
import numpy as np

img = Image.open(r"path/to/your/image/here")

print("Image mode:", img.mode)  

# Check shape via NumPy
arr = np.array(img)
print("Array shape:", arr.shape)
#      (height, width, 4) -> 4 channels (RGBA)
#      (height, width)    -> 1 channel (grayscale)
