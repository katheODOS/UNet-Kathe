import os
from pathlib import Path
import logging

def cleanup_empty_checkpoints(delete_checkpoints=False):
    """
    Remove checkpoint folders that don't contain any .pth files or meaningful output
    
    Parameters:
    - delete_checkpoints: If True, will also delete .pth files in folders
    """
    checkpoint_dir = Path('./checkpoints')
    removed_folders = 0
    removed_files = 0
    kept = 0
    skipped = 0
    
    logging.basicConfig(level=logging.INFO)
    
    # Read folders to remove from file if it exists
    folders_to_remove = []
    remove_file = Path('./remove_checkpoints.txt')
    if remove_file.exists():
        with open(remove_file, 'r') as f:
            folders_to_remove = [line.strip() for line in f.readlines() if line.strip()]
        logging.info(f"Found {len(folders_to_remove)} folders to remove from {remove_file}")
    
    for folder in checkpoint_dir.iterdir():
        if not folder.is_dir():
            continue
            
        # Check if this folder is in the removal list
        folder_name = folder.name
        folder_in_remove_list = folder_name in folders_to_remove or str(folder) in folders_to_remove
        
        try:
            pth_files = [file for file in folder.iterdir() if file.suffix == '.pth']
            has_checkpoints = len(pth_files) > 0
            has_output = (folder / 'output.txt').exists()
            
            # Delete .pth files if the folder is in remove_checkpoints.txt or if delete_checkpoints is True
            if (folder_in_remove_list or delete_checkpoints) and has_checkpoints:
                for pth_file in pth_files:
                    try:
                        pth_file.unlink()
                        logging.info(f"Removed checkpoint file: {pth_file}")
                        removed_files += 1
                    except Exception as e:
                        logging.error(f"Error removing file {pth_file}: {str(e)}")
                # After deleting all checkpoints, this folder no longer has checkpoints
                has_checkpoints = False
            
            # Only keep folders that have either checkpoints or output files
            if not (has_checkpoints or has_output):
                try:
                    # Remove all files in folder
                    for file in folder.iterdir():
                        file.unlink()
                    # Remove folder
                    folder.rmdir()
                    logging.info(f"Removed empty folder: {folder}")
                    removed_folders += 1
                except Exception as e:
                    # Only report error if this folder was explicitly requested to be removed
                    if folder_in_remove_list:
                        logging.error(f"Error removing folder {folder}: {str(e)}")
                    else:
                        logging.debug(f"Skipping inaccessible folder (not in removal list): {folder}")
                        skipped += 1
            else:
                kept += 1
        except PermissionError:
            # Handle permission errors when trying to iterate directory contents
            if folder_in_remove_list:
                logging.error(f"Permission denied when accessing folder {folder}")
            else:
                logging.debug(f"Skipping inaccessible folder (not in removal list): {folder}")
                skipped += 1
    
    log_message = f"Cleanup complete. Removed {removed_folders} empty folders"
    log_message += f" and {removed_files} checkpoint files"
    log_message += f", kept {kept} folders, skipped {skipped} inaccessible folders."
    logging.info(log_message)

if __name__ == '__main__':
    # You can set this to True if you want to delete checkpoint files from all folders,
    # but it's not necessary since we're now deleting .pth files from folders in remove_checkpoints.txt
    cleanup_empty_checkpoints(delete_checkpoints=False)
