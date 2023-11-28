# Assets

## Usages of Files

|       File        |            Descriptions            |
| :---------------: | :--------------------------------: |
|    `cards.cdb`    |   Card content and effects (en)    |
|    `deck.json`    |     Deck for YGO04 game format     |
| `.game.db.backup` | Backup of an empty server database |

## Steps to Configure Environment

```bash
# Change current directory to `assets`.
cd assets

# Execute `init.sh` to obtain a configured `game.db`
bash init.sh

# Copy `game.db` and `cards.cdb` to `mud-server/`
cp game.db ../mud-server
cp cards.cdb ../mud-server/locale/en

# Apply Git patch to enable unlimited ban-list and solve compatibility issues.
cd ../mud-server
git am "../assets/*.patch"

# Build a docker image for the MUD server.
# Refer to mud-server/readme.md for details.

# Run a docker container with mounting cards.cdb and game.db
docker run --rm -p 4000:4000/tcp \
    -v ./game.db:/usr/src/app/game.db \
    -it --name=yugioh yugioh
```
