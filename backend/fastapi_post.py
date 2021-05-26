import requests
import json
from config import CONFIG


if __name__ == "__main__":
    files = {"file": open(CONFIG.data / "cat1.jpg", "rb")}
    url = "http://127.0.0.1:8000/predict"
    r = requests.post(url, files=files)
    print(r.content)
    print(r.text)
    pred = r.json()[0]
    print(pred["class"])