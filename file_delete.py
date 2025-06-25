import sys
from pathlib import Path

# List of file extensions to delete (include the dot)
EXTENSIONS = [".pyc", ".log", ".tmp", ".pth"]

# Optionally get the folder to clean from command line, or use project root if not specified
PROJECT_ROOT = Path(__file__).resolve().parents[0]
FOLDER_TO_CLEAN = sys.argv[1] if len(sys.argv) > 1 else ""
TARGET_FOLDER = PROJECT_ROOT / FOLDER_TO_CLEAN if FOLDER_TO_CLEAN else PROJECT_ROOT

# Step 1: Collect files (files only!) to delete
files_to_delete = []
for ext in EXTENSIONS:
    for p in TARGET_FOLDER.rglob(f'*{ext}'):
        if p.is_file():  # Make absolutely sure it's a file
            files_to_delete.append(p)

if not files_to_delete:
    print("No files found matching the given extensions.")
    sys.exit(0)

# Step 2: Show the files to the user
print(f"\nFiles that will be deleted in: {TARGET_FOLDER}\n")
for f in files_to_delete:
    print(f"  {f}")

# Step 3: Ask for confirmation
answer = input("\nAre you sure you want to delete ALL the files listed above? (yes/no): ").strip().lower()
if answer != "yes":
    print("Aborted. No files were deleted.")
    sys.exit(0)

# Step 4: Proceed with deletion
deleted_count = 0
for file in files_to_delete:
    try:
        file.unlink()
        print(f"Deleted: {file}")
        deleted_count += 1
    except Exception as e:
        print(f"Error deleting {file}: {e}")

print(f"\nTotal files deleted: {deleted_count}")
