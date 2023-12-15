import json
from typing import Dict, List, Tuple

from ygo.constants import LOCATION
from ygo.duel import Duel
from ygo.player import Player

def _dump_card_sequence(
        duel: Duel, index: int, location: LOCATION, n: int = 0, hiding: bool = True
    ) -> List[Tuple[int, int]]:
    """ Dump the cards in the given location into a list of card sequences.

    Parameters
    ----------
    n : int
        The size of location. Pad [None, 0] if the location is empty.
    """
    cards = duel.get_cards_in_location(index, location)

    ret = [(None, 0) for _ in range(n)]
    for card in cards:
        code = 0 if hiding and card.position & (0x2 | 0x8) else card.code
        ret[card.sequence] = (code, card.position, )

    return ret

def _dump_state(duel: Duel, player: Player) -> Dict:
    # !! Card is not serializable by JSON emitter, thus keep card code for simplicity !!
    index: int = player.duel_player
    return {
        'phase': duel.current_phase,
        'turn': duel.players[duel.tp] == player,
        'player': {
            'lp': duel.lp[index],
            'deck': len(duel.get_cards_in_location(index, LOCATION.DECK)),
            'hand': [card.code for card in duel.get_cards_in_location(index, LOCATION.HAND)],
            'monster': _dump_card_sequence(duel, index, LOCATION.MZONE, 5, hiding=False),
            'spell': _dump_card_sequence(duel, index, LOCATION.SZONE, 5, hiding=False),
            'grave': [card.code for card in duel.get_cards_in_location(index, LOCATION.GRAVE)],
            'removed': [card.code for card in duel.get_cards_in_location(index, LOCATION.REMOVED)],
        },
        "opponent": {
            'lp': duel.lp[1 - index],
            'deck': len(duel.get_cards_in_location(1 - index, LOCATION.DECK)),
            'hand': len(duel.get_cards_in_location(1 - index, LOCATION.HAND)),
            'monster': _dump_card_sequence(duel, 1 - index, LOCATION.MZONE, 5),
            'spell': _dump_card_sequence(duel, 1 - index, LOCATION.SZONE, 5),
            'grave': [card.code for card in duel.get_cards_in_location(1 - index, LOCATION.GRAVE)],
            'removed': [card.code for card in duel.get_cards_in_location(1 - index, LOCATION.REMOVED)],
        },
    }

# See ygo.constants.PHASES
def dump_game_info(duel: Duel, player: Player, **kwargs) -> str:
    """ Pack the current state of the duel into a JSON string.
    Then, wrap the message by adding separators '|' before and after the JSON string.
    """
    return f"|{json.dumps({ 'state': _dump_state(duel, player), **kwargs })}|"
