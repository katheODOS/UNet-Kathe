import os
import shutil

# List of folders to move
target_folders = [
    "BSTL1e-08W1e-07B2E40", "BSTL1e-08W1e-06B4E40", "BSTL1e-08W1e-06B2E45",
    "BSTL1e-08W1e-08B2E45", "BSTL1e-08W1e-07B2E35", "BSTL1e-08W1e-06B2E25",
    "BSTL1e-08W1e-08B2E40", "BSTL1e-08W1e-06B4E35", "BSTL1e-08W1e-06B4E30",
    "BSTL1e-08W1e-08B2E35", "BSTL1e-08W1e-06B2E35", "BSTL1e-08W1e-07B2E25",
    "BSTL1e-08W1e-08B2E30", "BSTL1e-08W1e-07B2E30", "BSTL1e-08W1e-08B2E20",
    "BSTL1e-08W1e-06B4E25", "BSTL1e-08W1e-09B4E25", "BSTL1e-08W1e-06B2E40",
    "BSTL1e-08W1e-06B2E30", "BSTL1e-08W1e-08B4E20", "BSTL1e-08W1e-08B4E25",
    "BSTL1e-08W1e-06B2E20", "BSTL1e-08W1e-09B2E25"
]

# Define source and destination paths
source_dir = "checkpoints"
dest_dir = os.path.join("checkpoints", "best_runs_scale_0.5")

# Create destination directory if it doesn't exist
os.makedirs(dest_dir, exist_ok=True)

# Move matching folders
moved_count = 0
for folder in target_folders:
    source_path = os.path.join(source_dir, folder)
    dest_path = os.path.join(dest_dir, folder)
    
    try:
        if os.path.exists(source_path):
            shutil.move(source_path, dest_path)
            print(f"Moved: {folder}")
            moved_count += 1
        else:
            print(f"Not found: {folder}")
    except Exception as e:
        print(f"Error moving {folder}: {str(e)}")

print(f"\nMoved {moved_count} folders successfully")
