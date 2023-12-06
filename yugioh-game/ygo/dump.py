import json
from typing import Callable, Dict, List, Optional, Tuple

from ygo.constants import LOCATION
from ygo.duel import Duel
from ygo.player import Player

def _dump_card_sequence(
        duel: Duel, index: int, location: LOCATION, n: int = 0, hiding: bool = True
    ) -> List[Tuple[int, int]]:
    """ Dump the cards in the given location into a list of card sequences. """
    cards = duel.get_cards_in_location(index, location)

    ret = [(None, 0) for _ in range(n)]
    for card in cards:
        code = 0 if hiding and card.position & (0x2 | 0x8) else card.code
        ret[card.sequence] = (code, card.position, )

    return ret

def _dump_state(duel: Duel, player: Player) -> Dict:
    index: int = player.duel_player
    return {
        'phase': duel.current_phase,
        'remain_normal_summon': duel.remain_normal_summon[index],
        # See: Duel.show_score()
        'score': {
            'player': {
                'lp': duel.lp[index],
                'hand': len(duel.get_cards_in_location(index, LOCATION.HAND)),
                'deck': len(duel.get_cards_in_location(index, LOCATION.DECK)),
                'grave': [card.code for card in duel.get_cards_in_location(index, LOCATION.GRAVE)],
                'removed': [card.code for card in duel.get_cards_in_location(index, LOCATION.REMOVED)],
            },
            'opponent': {
                'lp': duel.lp[1 - index],
                'hand': len(duel.get_cards_in_location(1 - index, LOCATION.HAND)),
                'deck': len(duel.get_cards_in_location(1 - index, LOCATION.DECK)),
                'grave': [card.code for card in duel.get_cards_in_location(1 - index, LOCATION.GRAVE)],
                'removed': [card.code for card in duel.get_cards_in_location(1 - index, LOCATION.REMOVED)],
            },
        },
        # !! Card is not serializable by JSON emitter, thus keep card code for simplicity !!
        # See: Card.__init__()
        # See: Duel.show_cards_in_location(player, index, LOCATION.HAND, False)
        'hand': [card.code for card in duel.get_cards_in_location(index, LOCATION.HAND)],
        # See: Duel.show_table()
        'table': {
            'player': {
                'monster': _dump_card_sequence(duel, index, LOCATION.MZONE, 5, hiding=False),
                'spell': _dump_card_sequence(duel, index, LOCATION.SZONE, 5, hiding=False)
            },
            "opponent": {
                'monster': _dump_card_sequence(duel, 1 - index, LOCATION.MZONE, 5),
                'spell': _dump_card_sequence(duel, 1 - index, LOCATION.SZONE, 5)
            }
        },
    }

# See ygo.constants.PHASES
def dump_game_info(duel: Duel, player: Player, **kwargs) -> str:
    """ Pack the current state of the duel into a JSON string.
    Then, wrap the message by adding separators '|' before and after the JSON string.
    """
    return f"|{json.dumps({ 'state': _dump_state(duel, player), **kwargs })}|"
