import os
import rasterio
import numpy as np

tif_dir = r"directory/with/tifs/to/fill"

def fill_black_pixels(data):
    """Fill black pixels with next valid pixel value"""
    height, width, bands = data.shape
    filled = data.copy()
    
    # Create mask for black pixels
    black_mask = np.all(data == 0, axis=2)
    

    for y in range(height):
        last_valid_pixel = None
        # Forward pass - fill with previous valid pixel
        for x in range(width):
            if not black_mask[y, x]:
                last_valid_pixel = data[y, x]
            elif last_valid_pixel is not None:
                filled[y, x] = last_valid_pixel
    return filled

# Process each TIF file
for tif_file in os.listdir(tif_dir):
    if tif_file.endswith('.tif'):
        tif_path = os.path.join(tif_dir, tif_file)
        
        with rasterio.open(tif_path) as src:
            data = src.read()
            profile = src.profile
            
            # Rearrange to (height, width, bands)
            data_hwc = np.moveaxis(data, 0, -1)
            
            # Fill black pixels with next valid value
            filled_hwc = fill_black_pixels(data_hwc)
            
            # Back to (bands, height, width)
            filled_data = np.moveaxis(filled_hwc, -1, 0)
            
            # Save to new file
            output_path = tif_path.replace('.tif', '_filled.tif')
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(filled_data)
                
        print(f"Processed: {tif_file}")

print("All files processed!")
