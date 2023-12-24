# Yu-Gi-Oh RL

## Game Design

The game is based on YGO04 environment with these revision:

- Card effects depends on the version of YGOPro-core.
- Replace the cards with effect "檢索" or "檢查對方手牌" to other monsters, spells and traps.
  - 74191942 (Painful choice) => 44095762 (Mirror Force)
  - 32807846 (Reinforcement of the Army) => 44095762 (Mirror Force)
  - 42829885 (The Forceful Sentry) => 71413901 (Breaker the Magical Warrior)
  - 17375316 (Confiscation) => 69162969 (Lightning Vortex)

## Setup

1. Prepare `game.db`

    ```bash
    cd assets; sh init.sh; cd ..
    ```

2. Deploy databases to Yu-Gi-Oh MUD server

    ```bash
    git-lfs pull
    cp assets/game.db yugioh-game/game.db;
    cp assets/cards.cdb yugioh-game/locale/en/cards.cdb;
    ```

3. Build docker image

    ```bash
    cd yugioh-game;
    docker build . -t yugioh;
    ```

4. Run MUD server inside docker container

    ```bash
    cd yugioh-game;

    docker run --rm -p 4000:4000/tcp -it \
        -v $(pwd)/ygo:/usr/src/app/ygo \
        -v $(pwd)/ygo.py:/usr/src/app/ygo.py \
        --name yugioh \
        yugioh

    # Confirm the server is reachable with telnet.
    telnet localhost 4000
    ```

5. Train an agent with `train_PPO.py`.

    ```bash
    python3 train_PPO.py
    ```

6. Evaluate the agent with `eval.py`

    ```bash
    python eval.py --model-path <model-path>
    ```

## Sources

- tspivey/yugioh-game. We adapt the server based on Commit-ID `314f15ef5275f6f4d22101c3d5f00c62560b5a88`.
