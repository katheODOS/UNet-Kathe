import os
from pathlib import Path
import logging

def cleanup_empty_checkpoints():
    """Remove checkpoint folders that don't contain any .pth files or meaningful output"""
    checkpoint_dir = Path('./checkpoints')
    removed = 0
    kept = 0
    
    logging.basicConfig(level=logging.INFO)
    
    for folder in checkpoint_dir.iterdir():
        if not folder.is_dir():
            continue
            
        has_checkpoints = any(file.suffix == '.pth' for file in folder.iterdir())
        has_output = (folder / 'output.txt').exists()
        
        # Only keep folders that have either checkpoints or output files
        if not (has_checkpoints or has_output):
            try:
                # Remove all files in folder
                for file in folder.iterdir():
                    file.unlink()
                # Remove folder
                folder.rmdir()
                logging.info(f"Removed empty folder: {folder}")
                removed += 1
            except Exception as e:
                logging.error(f"Error removing folder {folder}: {str(e)}")
        else:
            kept += 1
    
    logging.info(f"Cleanup complete. Removed {removed} empty folders, kept {kept} folders.")

if __name__ == '__main__':
    cleanup_empty_checkpoints()
