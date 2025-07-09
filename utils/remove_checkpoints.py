import os
import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def remove_checkpoint_files():
    """
    Remove checkpoint (.pth) files from folders listed in remove_checkpoints.txt
    """
    # Get the project root directory (2 levels up from this script)
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    project_root = current_dir.parent
    
    # Paths to checkpoints directory and removal list
    checkpoints_dir = project_root / 'checkpoints'
    removal_list_path = project_root / 'remove_checkpoints.txt'
    
    # Check if paths exist
    if not checkpoints_dir.exists():
        logging.error(f"Checkpoints directory not found: {checkpoints_dir}")
        return
    
    if not removal_list_path.exists():
        logging.error(f"Removal list not found: {removal_list_path}")
        return
    
    # Read folders to remove from the list
    try:
        with open(removal_list_path, 'r') as f:
            folders_to_clean = [line.strip() for line in f if line.strip()]
    except Exception as e:
        logging.error(f"Error reading removal list: {e}")
        return
    
    logging.info(f"Found {len(folders_to_clean)} folders to clean in the removal list")
    
    # List all available checkpoint folders for debugging
    available_folders = [folder.name for folder in checkpoints_dir.iterdir() if folder.is_dir()]
    logging.info(f"Available checkpoint folders: {len(available_folders)}")
    
    # Print the first few entries from both lists for comparison
    if folders_to_clean:
        logging.info(f"First 3 folders in removal list: {folders_to_clean[:3]}")
    if available_folders:
        logging.info(f"First 3 available folders: {available_folders[:3]}")
    
    # Check for case sensitivity or whitespace issues
    matching_folders = set(available_folders).intersection(set(folders_to_clean))
    logging.info(f"Found {len(matching_folders)} matching folders")
    
    # Track statistics
    total_deleted = 0
    folders_processed = 0
    
    # Iterate through all subdirectories in the checkpoints directory
    for folder_path in checkpoints_dir.iterdir():
        if not folder_path.is_dir():
            continue
            
        folder_name = folder_path.name
        
        # Check if this folder should be cleaned
        if folder_name in folders_to_clean:
            logging.info(f"Processing folder: {folder_name}")
            
            # Find and delete all .pth files
            pth_files = list(folder_path.glob('*.pth'))
            
            if not pth_files:
                logging.info(f"  No .pth files found in {folder_name}")
                continue
                
            for pth_file in pth_files:
                try:
                    pth_file.unlink()
                    logging.info(f"  Deleted: {pth_file.name}")
                    total_deleted += 1
                except Exception as e:
                    logging.error(f"  Failed to delete {pth_file}: {e}")
            
            folders_processed += 1
    
    logging.info(f"Cleanup complete. Processed {folders_processed} folders, deleted {total_deleted} checkpoint files.")

if __name__ == "__main__":
    remove_checkpoint_files()
