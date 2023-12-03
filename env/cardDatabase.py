import json

class CardDatabase:
    with open("../assets/id2cardinfo.json", "r") as f:
        id2cardInfo = json.load(f)