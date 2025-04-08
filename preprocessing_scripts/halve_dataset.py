import os
import random

# Cian request: see how long it takes to train the model on one epoch with half the training set
base_dir = r"C:\Users\Admin\Desktop\QGIS\FINAL FILES\biodiversity_dataset png 512 halved"
folders = {
    'annotations': os.path.join(base_dir, 'annotations', 'train'),
    'single_channel': os.path.join(base_dir, 'annotations_single_channel', 'train'),
    'images': os.path.join(base_dir, 'images', 'train')
}

def get_files_to_delete(directory, delete_fraction=0.5):
    """Get random list of files to delete"""
    all_files = [f for f in os.listdir(directory) if f.endswith('.png')]
    num_to_delete = int(len(all_files) * delete_fraction)
    return random.sample(all_files, num_to_delete)

def delete_files(files_to_delete):
    """Delete files from all three directories"""
    deleted_count = {folder: 0 for folder in folders.keys()}
    errors = []
    
    for filename in files_to_delete:
        for folder_name, folder_path in folders.items():
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    deleted_count[folder_name] += 1
                else:
                    errors.append(f"File not found: {file_path}")
            except Exception as e:
                errors.append(f"Error deleting {file_path}: {str(e)}")
    
    return deleted_count, errors

def main():
    # Verify directories exist
    for folder_path in folders.values():
        if not os.path.exists(folder_path):
            print(f"Error: Directory not found: {folder_path}")
            return

    # Get list of files to delete
    files_to_delete = get_files_to_delete(folders['annotations'])
    print(f"Selected {len(files_to_delete)} files for deletion")
    
    # Delete files
    deleted_count, errors = delete_files(files_to_delete)
    
    # Print results
    print("\nDeletion Summary:")
    for folder, count in deleted_count.items():
        print(f"{folder}: {count} files deleted")
    
    if errors:
        print("\nErrors encountered:")
        for error in errors:
            print(error)
    
    # Save list of deleted files
    log_file = os.path.join(base_dir, 'deleted_files_log.txt')
    with open(log_file, 'w') as f:
        f.write('\n'.join(files_to_delete))
    print(f"\nList of deleted files saved to: {log_file}")

if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    main()
