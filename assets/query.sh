#!/bin/bash

cards=$(python -c "import json; f = open('deck.json'); print(tuple(set(json.load(f)['cards'])))")

# Query card attributes
sqlite3 cards.cdb ".mode json" "SELECT * FROM datas WHERE id IN $cards"

# Query card descriptions
sqlite3 cards.cdb ".mode json" "SELECT * FROM texts WHERE id IN $cards"
