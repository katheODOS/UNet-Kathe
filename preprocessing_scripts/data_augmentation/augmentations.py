import os
import numpy as np
import tifffile
import imgaug.augmenters as iaa
import imgaug.parameters as iap
from tqdm import tqdm

base_img_dir = r"C:\Users\Admin\Desktop\QGIS\test retiling\512x512 50 percent overlap\Dataset DSA\imgs"
base_mask_dir = r"C:\Users\Admin\Desktop\QGIS\test retiling\512x512 50 percent overlap\Dataset DSA\masks"

class DeterministicColor(iap.StochasticParameter):
    def __init__(self, color):
        self.color = np.uint8(color)

    def _draw_samples(self, size, random_state):
        assert size[-1] == 3  
        arr = np.zeros(size, dtype=np.uint8)
        arr[..., :] = self.color
        return arr

FILL_COLOR = [11, 246, 210]

FOLDER_AUGMENTATIONS = {
    "geometric": {
        "folders": ["3_primary_secondary_label", "12_primary_label"],
        "image_transform": iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Affine(
                order=0,
                scale=(0.95, 1.05),
                translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)},
                rotate=(-20, 20),
                cval=DeterministicColor(FILL_COLOR),
                mode='constant'
            )
        ]),
        # For geometric augmentations, mask transform is the same as image transform to avoid mask RGB value modification
        "mask_transform": iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Affine(
                order=0,
                scale=(0.95, 1.05),
                translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)},
                rotate=(-20, 20),
                cval=DeterministicColor(FILL_COLOR),
                mode='constant'
            )
        ])
    },
    "photometric": {
        "folders": ["2_primary_label", "6_primary_secondary_label"],
        "image_transform": iaa.Sequential([
            iaa.LinearContrast((0.95, 1.05)),
            iaa.Multiply((0.95, 1.05))
        ]),
        # For photometric augmentations masks do NOT get transformed
        "mask_transform": None  
    },
    "combined": {
        "folders": ["4_primary_secondary_label", "9_any_position", "13_secondary_label"],
        "image_transform": iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Affine(
                order=0,
                scale=(0.95, 1.05),
                translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)},
                rotate=(-20, 20),
                cval=DeterministicColor(FILL_COLOR),
                mode='constant'
            ),
            iaa.LinearContrast((0.95, 1.05))
        ]),
        # Here masks only receive the geometric transforms so they align with imgs
        "mask_transform": iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Affine(
                order=0,
                scale=(0.95, 1.05),
                translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)},
                rotate=(-20, 20),
                cval=DeterministicColor(FILL_COLOR),
                mode='constant'
            )
        ])
    },
    "rotations": {
        "folders": [
            "2_primary_label", "3_primary_secondary_label",
            "4_primary_secondary_label", "6_primary_secondary_label",
            "9_any_position", "12_primary_label", "13_secondary_label"
        ],
        "transforms": [
            iaa.Affine(order=0,rotate=90, cval=DeterministicColor(FILL_COLOR), mode='constant'),
            iaa.Affine(order=0,rotate=180, cval=DeterministicColor(FILL_COLOR), mode='constant'),
            iaa.Affine(order=0,rotate=270, cval=DeterministicColor(FILL_COLOR), mode='constant')
        ]
    }
}

def load_image(path):
    """Load TIF image and ensure proper formatting"""
    image = tifffile.imread(path)
    
    if image.dtype != np.uint8:
        if image.max() <= 1.0: 
            image = (image * 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)
    
    return image

def save_image(path, image):
    # Make sure we're saving as uint8
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
        
    tifffile.imwrite(path, image)

def load_mask(path):
    """Load TIF mask with consistent formatting"""
    mask = tifffile.imread(path)
    if mask.dtype != np.uint8:
        if mask.max() <= 1.0:
            mask = (mask * 255).astype(np.uint8)
        else:
            mask = np.clip(mask, 0, 255).astype(np.uint8)
    
    return mask

def save_mask(path, mask):
    """Save TIF mask with consistent formatting"""
 
    if mask.dtype != np.uint8:
        mask = np.clip(mask, 0, 255).astype(np.uint8)
        
    tifffile.imwrite(path, mask)

def augment_folder_pair(img_folder, mask_folder, aug_type):
    """Augment all images in a folder pair with specified augmentation type"""
    img_dir = os.path.join(base_img_dir, img_folder)
    mask_dir = os.path.join(base_mask_dir, mask_folder)
    
    img_augment_dir, mask_augment_dir = ensure_augment_directories(img_folder)
    
    if not (os.path.exists(img_dir) and os.path.exists(mask_dir)):
        print(f"Skipping {img_folder} - directories not found")
        return
    
    img_files = set(f for f in os.listdir(img_dir) 
                   if f.endswith('.tif') and os.path.isfile(os.path.join(img_dir, f)))
    mask_files = set(f for f in os.listdir(mask_dir) 
                    if f.endswith('.tif') and os.path.isfile(os.path.join(mask_dir, f)))
    common_files = img_files.intersection(mask_files)
    
    print(f"\nProcessing {img_folder} with {aug_type} augmentations")
    print(f"Found {len(common_files)} matching files")
    
    if aug_type == "rotations":
        for filename in tqdm(common_files, desc=f"Rotating {img_folder}"):
            img_path = os.path.join(img_dir, filename)
            mask_path = os.path.join(mask_dir, filename)
            
            ext = os.path.splitext(filename)[1]
            
            image = load_image(img_path)
            mask = load_mask(mask_path)
            
            for i, transform in enumerate(FOLDER_AUGMENTATIONS[aug_type]['transforms']):
                angle = 90 * (i + 1)
                
                # Make transform deterministic 
                det_transform = transform.to_deterministic()
                
                # Apply the SAME deterministic transform to both image and mask
                aug_image = det_transform(image=image)
                aug_mask = det_transform(image=mask)
                
                # Generate output filenames with same extension as input
                base_name = os.path.splitext(filename)[0]
                aug_img_name = f"{base_name}_rot{angle}{ext}"
                aug_mask_name = f"{base_name}_rot{angle}{ext}"
                
                # Save augmented images to augment directories
                save_image(os.path.join(img_augment_dir, aug_img_name), aug_image)
                save_mask(os.path.join(mask_augment_dir, aug_mask_name), aug_mask)
    else:
        for filename in tqdm(common_files):
            img_path = os.path.join(img_dir, filename)
            mask_path = os.path.join(mask_dir, filename)
            
            ext = os.path.splitext(filename)[1]
            
            image = load_image(img_path)
            mask = load_mask(mask_path)
            
            image_transform = FOLDER_AUGMENTATIONS[aug_type]['image_transform']
            mask_transform = FOLDER_AUGMENTATIONS[aug_type]['mask_transform']
            
            for i in range(3):
                aug_image = image.copy()  # Default to unchanged
                if image_transform is not None:
                    det_image_transform = image_transform.to_deterministic()
                    aug_image = det_image_transform(image=image)
                
                # Apply mask transformation (if any)
                # For photometric augmentations, mask_transform might be None
                aug_mask = mask.copy()  # Default to unchanged
                if mask_transform is not None:
                    det_mask_transform = mask_transform.to_deterministic()
                    aug_mask = det_mask_transform(image=mask)
                
                base_name = os.path.splitext(filename)[0]
                aug_img_name = f"{base_name}_aug{i+1}{ext}"
                aug_mask_name = f"{base_name}_aug{i+1}{ext}"
                
                # Save augmented images to augment directories
                save_image(os.path.join(img_augment_dir, aug_img_name), aug_image)
                save_mask(os.path.join(mask_augment_dir, aug_mask_name), aug_mask)

def main():
    print("Starting augmentation process...")
    
    # First apply regular augmentations
    for aug_type, config in FOLDER_AUGMENTATIONS.items():
        if aug_type != "rotations":  # Skip rotations for now
            for folder in config['folders']:
                augment_folder_pair(folder, folder, aug_type)
    
    # Then apply rotations to all folders
    for folder in FOLDER_AUGMENTATIONS["rotations"]["folders"]:
        augment_folder_pair(folder, folder, "rotations")
    
    print("\nAugmentation complete!")

if __name__ == "__main__":
    main()