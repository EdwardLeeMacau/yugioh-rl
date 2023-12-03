import json
import sqlite3

def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

def main():
    # read deck.json to get the cards ids
    with open("./deck.json", "r") as f:
        deck = json.load(f)

    cards_id = deck['cards']

    # connect to the cards.cdb
    cards_db = sqlite3.connect("./cards.cdb")
    cards_db.row_factory = dict_factory
    cursor = cards_db.cursor()

    # merge the info for each card from the deck 
    id2info = {}
    for id in cards_id:
        cursor.execute(f'SELECT * FROM datas WHERE id={id}')
        data_row = cursor.fetchone()

        cursor.execute(f'SELECT * FROM texts WHERE id={id}')
        text_row = cursor.fetchone()

        info = data_row | text_row
        id2info[id] = info
    
    # save the info as json file
    with open("id2cardinfo.json", "w") as f:
        json.dump(id2info, f, indent=4)
    
if __name__ == '__main__':
    main()