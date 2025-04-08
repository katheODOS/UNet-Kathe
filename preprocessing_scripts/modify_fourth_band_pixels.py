import rasterio
import numpy as np

def modify_zero_pixels(input_path, output_path):
    """Modify pixels with band 4 value of 0 to specified RGB values"""
    
    # New RGB values to set
    new_r, new_g, new_b = 11, 246, 210
    
    with rasterio.open(input_path) as src:
        # Read all bands
        data = src.read()
        
        # Check if we have at least 4 bands
        if data.shape[0] < 4:
            raise ValueError("Input file must have at least 4 bands")
            
        # Find pixels where band 4 equals 0
        zero_mask = (data[3] == 0)
        
        # Modify RGB values where mask is True
        data[0][zero_mask] = new_r
        data[1][zero_mask] = new_g
        data[2][zero_mask] = new_b
        
        # Create new file with same properties as source
        profile = src.profile
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data)

if __name__ == "__main__":
    # Specific input file path
    input_path = r"C:\Users\Admin\Desktop\QGIS\FINAL FILES\big mask true colors.tif"
    
    # Create output path by adding '_modified' before extension
    output_path = input_path.rsplit('.', 1)[0] + '_modified.tif'
    
    try:
        modify_zero_pixels(input_path, output_path)
        print(f"Successfully modified file. Saved as: {output_path}")
    except Exception as e:
        print(f"Error: {str(e)}")
