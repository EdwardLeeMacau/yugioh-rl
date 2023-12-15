# Yu-Gi-Oh RL

## Project Architecture

`Game`, `Account`, and `Player` aim to manage the resource to interact with MUD server. They are just adapters, and do not hold any game decision making issues.

`Policy` is the abstract class for decision making. You can think `Policy.react()` is the function $\pi(a_t|s_t)$ in mathematical representation.

## Game Design

The game is based on YGO04 environment with these revision:

- Card effects depends on the version of YGOPro-core.
- Replace the cards with effect "檢索" or "檢查對方手牌" to other monsters, spells and traps.
  - 74191942 (Painful choice) => 44095762 (Mirror Force)
  - 32807846 (Reinforcement of the Army) => 44095762 (Mirror Force)
  - 42829885 (The Forceful Sentry) => 71413901 (Breaker the Magical Warrior)
  - 17375316 (Confiscation) => 69162969 (Lightning Vortex)

## Run Experiments in Parallel

Modify `env/account.py` before running the training script.

For example
- Parallel training in 32 games (see argument `parallel` in `env_config.py`), needs 32 games * 2 = 64 accounts
- Evaluate with 32 environments (see argument `num_resources` in `eval.play_game_multiple_times()`), needs 32 games * 2 = 64 accounts

Total 128 accounts are allocated by single training script.

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
        yugioh \
        /bin/bash

    # Execute the script inside container
    python3 ygo.py
    ```

5. Test with telnet (manual)

    ```bash
    telnet localhost 4000
    ```

    or with `main.py` (random agent)

    ```bash
    python3 main.py
    ```

## Sources

- tspivey/yugioh-game. We adapt the server based on Commit-ID `314f15ef5275f6f4d22101c3d5f00c62560b5a88`.
