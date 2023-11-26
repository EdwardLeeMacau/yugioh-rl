# Assets

## Usages of Files

|    File     |            Descriptions            |
| :---------: | :--------------------------------: |
|  `game.db`  | Backup of an empty server database |
| `deck.json` |     Deck for YGO04 game rules      |

## Steps to Configure Environment

```bash
# Change current directory to `assets`.
cd assets

# Execute `init.sh` to obtain a configured `game.db`
bash init.sh

# Copy `game.db` and `cards.cdb` to `mud-server/`
cp cards.db game.db ../mud-server

# Apply Git patch to enable unlimited ban-list and solve compatibility issues.
cd ../mud-server
git am "../assets/*.patch"

# Build a docker image for the MUD server.
# Refer to mud-server/readme.md for details.

# Run a docker container with mounting cards.cdb and game.db
docker run --rm -p 4000:4000/tcp \
    -v ./cards.cdb:/usr/src/app/locale/en/cards.cdb \
    -v ./game.db:/usr/src/app/game.db \
    -it --name=yugioh yugioh
```
