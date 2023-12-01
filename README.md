# Yu-Gi-Oh RL

## Project Architecture

`Game`, `Account`, and `Player` aim to manage the resource to interact with MUD server. They are just adapters, and do not hold any game decision making issues.

`Policy` is the abstract class for decision making. You can think `Policy.react()` is the function $\pi(a_t|s_t)$ in mathematical representation.

## Setup

1. Prepare `game.db`

    ```bash
    cd assets; sh init.sh; cd ..
    ```

2. Deploy databases to Yu-Gi-Oh MUD server

    ```bash
    cp assets/game.db yugioh-game/game.db;
    cp assets/cards.cdb yugioh-game/locale/en/cards.cdb;
    ```

3. Build docker image

    ```bash
    cd yugioh-game;
    docker build . -t yugioh;
    ```

4. Run docker container

    ```bash
    cd yugioh-game;

    docker run --rm -p 4000:4000/tcp -it \
        -v ./ygo:/usr/src/app/ygo \
        -v ./ygo.py:/usr/src/app/ygo.py \
        --name yugioh
        yugioh \
        /bin/bash

    # Execute the script inside container
    python3 ygo.py
    ```

5. Test with telnet (manual) or `main.py` (random agent)

    ```bash
    cd ygo_interface;

    python3 main.py
    ```
