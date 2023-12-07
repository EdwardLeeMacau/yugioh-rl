#!/bin/bash

# Parameters:
# number of accounts to create
n=8

# Replace game.db with the .game.db.backup
cp .game.db.backup game.db
if [ $? -ne 0 ]
then
    echo "Failed to restore game.db from .game.db.backup"
    exit 1
fi

# Schema of tables `accounts` and `decks`.
#
# CREATE TABLE accounts (
#         id INTEGER NOT NULL,
#         name VARCHAR NOT NULL,
#         password VARCHAR NOT NULL,
#         email VARCHAR(50),
#         created DATETIME,
#         last_logged_in DATETIME,
#         language VARCHAR NOT NULL,
#         encoding VARCHAR NOT NULL,
#         is_admin BOOLEAN NOT NULL,
#         duel_rules INTEGER NOT NULL,
#         banlist VARCHAR(50) NOT NULL,
#         ip_address VARCHAR(100) NOT NULL,
#         banned BOOLEAN NOT NULL,
#         PRIMARY KEY (id),
#         CHECK (is_admin IN (0, 1)),
#         CHECK (banned IN (0, 1))
# );
# CREATE UNIQUE INDEX ix_accounts_name ON accounts (name);
#
# Example:
# [
#     {
#         "id": 1,
#         "name": "Player1",
#         "password": "$pbkdf2-sha256$29000$V2oNodT6fy8lZCyl1Nq7Fw$14CbnvllF.BjOOnVLnw6V5Ipx4CgkpDH09YUvY8ugPA",
#         "email": "player1@gmail.com",
#         "created": "2023-11-25 07:38:52",
#         "last_logged_in": "2023-11-25 07:38:52",
#         "language": "en",
#         "encoding": "utf-8",
#         "is_admin": 0,
#         "duel_rules": 5,
#         "banlist": "tcg",
#         "ip_address": "172.17.0.1",
#         "banned": 0
#     }
# ]
#
# CREATE TABLE decks (
#     id INTEGER NOT NULL,
#     account_id INTEGER NOT NULL,
#     name VARCHAR COLLATE "NOCASE" NOT NULL,
#     content VARCHAR NOT NULL,
#     public BOOLEAN NOT NULL,
#     PRIMARY KEY (id),
#     FOREIGN KEY(account_id) REFERENCES accounts (id),
#     CHECK (public IN (0, 1))
# );

# Create new accounts
#
#   Player: should be capitalized
# Password: case sensitive
now=$(sqlite3 game.db "SELECT datetime();")
contents=$(jq -c . < deck.json)
for i in $(seq 1 256);
do
    player=Player$i
    passwd=$(python3 sha256.py -p player$i)
    email=player$i@gmail.com

    sqlite3 game.db "INSERT INTO accounts \
        (id, name, password, email, created, last_logged_in, language, encoding, is_admin, duel_rules, banlist, ip_address, banned) \
        VALUES ('$i', '$player', '$passwd', '$email', '$now', '$now', 'en', 'utf-8', 0, 5, 'tcg', '172.17.0.1', 0)";

    sqlite3 game.db "INSERT INTO decks \
        (id, account_id, name, content, public) \
        VALUES ($i, $i, 'YGO04', '$contents', 1)";
done
