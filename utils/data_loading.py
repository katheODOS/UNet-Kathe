import logging
import numpy as np
import torch
from PIL import Image
from functools import partial
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple"""
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        # RGB to class mapping
        self.rgb_classes = {
            hex_to_rgb("27b341"): 1,    # Will map to 1
            hex_to_rgb("e657c4"): 2,    # Will map to 2
            hex_to_rgb("fc7ebb"): 3,    # Will map to 3
            hex_to_rgb("ffcf4a"): 4,    # Will map to 4
            hex_to_rgb("fa3e77"): 5,    # Will map to 5
            hex_to_rgb("fa9441"): 6,    # Will map to 6
            hex_to_rgb("adadad"): 7,    # Will map to 7
            hex_to_rgb("ffc17a"): 9,    # Will map to 8
            hex_to_rgb("a8e854"): 12,   # Will map to 9
            hex_to_rgb("d9d9d9"): 13    # Will map to 10
        }

        # Create a contiguous mapping while preserving 0 as background/ignore
        unique_classes = sorted(list(self.rgb_classes.values()))
        self.class_map = {old_class: idx + 1 for idx, old_class in enumerate(unique_classes)}
        # Total number of classes including background (0)
        self.n_classes = len(unique_classes) + 1

        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        # Predefined valid classes
        self.valid_classes = list(self.rgb_classes.values())

        self.ids = [splitext(file)[0] for file in listdir(images_dir) 
                    if isfile(join(images_dir, file)) and not file.startswith('.')]
        
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        
        # Scan and log unique mask values (optional, but kept for compatibility)
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(self._unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def _unique_mask_values(self, idx, mask_dir, mask_suffix):
        """Helper method to find unique values in masks"""
        mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
        mask = np.asarray(load_image(mask_file))
        if mask.ndim == 2:
            return np.unique(mask)
        elif mask.ndim == 3:
            mask = mask.reshape(-1, mask.shape[-1])
            return np.unique(mask, axis=0)
        else:
            raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img, scale, is_mask=False):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        
        img_ndarray = np.asarray(pil_img)
        
        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))

            img_ndarray = img_ndarray / 255 if img_ndarray.max() > 1 else img_ndarray

        return img_ndarray

    def _preprocess_image(self, pil_img):
        return self.preprocess(pil_img, self.scale, is_mask=False)

    def _preprocess_mask(self, pil_mask):
        """Preprocess mask image"""
        w, h = pil_mask.size
        newW, newH = int(self.scale * w), int(self.scale * h)
        pil_mask = pil_mask.resize((newW, newH), resample=Image.NEAREST)
        
        mask = np.asarray(pil_mask)
        
        # Initialize mask with zeros (background/ignore)
        processed_mask = np.zeros((newH, newW), dtype=np.int64)
        
        # Handle RGB mask
        if mask.ndim == 3 and mask.shape[2] == 3:
            for rgb, class_id in self.rgb_classes.items():
                # Create a boolean mask for pixels matching the current RGB value
                class_mask = np.all(mask == rgb, axis=-1)
                processed_mask[class_mask] = self.class_map[class_id]  # Map to contiguous class index
        
        # Handle single-channel mask
        elif mask.ndim == 2:
            for orig_class, mapped_class in self.rgb_classes.items():
                mask_indices = mask == mapped_class
                processed_mask[mask_indices] = self.class_map[mapped_class]  # Map to contiguous class index
        
        return processed_mask

    def __getitem__(self, idx):
        name = self.ids[idx]
        
        # Find image and mask files
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        
        # Load images
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        # Preprocess
        img = self._preprocess_image(img)
        mask = self._preprocess_mask(mask)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')