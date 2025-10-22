from pathlib import Path
from zipfile import ZipFile

def extract(zip_path):
    zip_path = Path(zip_path)
    if not zip_path.is_file() or zip_path.suffix != '.zip':
        raise ValueError("The provided path is not a valid zip file.")
    extract_dir = zip_path.parent

    try:
        print(f"Extracting {zip_path.name}...")
        with ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)
        
        print(f"Extracted contents of {zip_path} successfully")
    except Exception as e:
        print(f"Error extracting {zip_path}: {e}")
        raise

if __name__ == "__main__":
    current_dir = Path('.')
    zip_files = list(current_dir.glob('*.zip'))
    if not zip_files:
        print("No zip files found in the current directory.")
    else:
        extract(zip_files[0])