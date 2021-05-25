from pathlib import Path

file_dir = Path(__file__).resolve().parent

class CONFIG:
    data = file_dir / "data"
    models = file_dir / "models"
    src = file_dir / "src"
    

if __name__ == "__main__":
    print(CONFIG.models)
    print(CONFIG.data)