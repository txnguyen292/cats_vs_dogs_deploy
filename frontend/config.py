from pathlib import Path

file_dir = Path(__file__).resolve().parent

class CONFIG:
    data = file_dir.parent / "data"
    models = file_dir.parent / "models"
    src = file_dir.parent / "src"
    

if __name__ == "__main__":
    pass
