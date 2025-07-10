from pathlib import Path
import re
import logging

def rename_folders():
    """Add missing batch size to checkpoint folder names"""
    checkpoint_dir = Path('./checkpoints')
    renamed = 0
    skipped = 0
    
    logging.basicConfig(level=logging.INFO)
    
    # Regular expression to match folder pattern and extract components
    pattern = r'^([A-Z]+)L(\d+e-\d+)W(\d+e-\d+)(B\d+)?E(\d+)$'
    
    for folder in checkpoint_dir.iterdir():
        if not folder.is_dir():
            continue
            
        match = re.match(pattern, folder.name)
        if match:
            dataset, lr, wd, batch, epochs = match.groups()
            
            # Skip if already has batch size
            if batch:
                logging.info(f"Skipping {folder.name} - already has batch size")
                skipped += 1
                continue
            
            # Insert B2 before E unless it's an 'A' dataset (which uses B1)
            batch_size = "B1" if dataset.startswith('A') else "B2"
            new_name = f"{dataset}L{lr}W{wd}{batch_size}E{epochs}"
            
            try:
                folder.rename(checkpoint_dir / new_name)
                logging.info(f"Renamed: {folder.name} -> {new_name}")
                renamed += 1
            except Exception as e:
                logging.error(f"Error renaming {folder.name}: {str(e)}")
                
    logging.info(f"\nSummary:")
    logging.info(f"Renamed: {renamed} folders")
    logging.info(f"Skipped: {skipped} folders (already had batch size)")

if __name__ == '__main__':
    rename_folders()
