from typing import Callable, Dict, Optional

from ygo.constants import LOCATION
from ygo.duel import Duel
from ygo.player import Player

def _dump_state(duel: Duel, player: Player) -> Dict:
    index: int = player.duel_player
    return {
        'phase': duel.current_phase,
        # See: Duel.show_score()
        'score': {
            'player': {
                'lp': duel.lp[index],
                'hand': len(duel.get_cards_in_location(index, LOCATION.HAND)),
                'deck': len(duel.get_cards_in_location(index, LOCATION.DECK)),
                'grave': len(duel.get_cards_in_location(index, LOCATION.GRAVE)),
                'removed': len(duel.get_cards_in_location(index, LOCATION.REMOVED)),
            },
            'opponent': {
                'lp': duel.lp[1 - index],
                'hand': len(duel.get_cards_in_location(1 - index, LOCATION.HAND)),
                'deck': len(duel.get_cards_in_location(1 - index, LOCATION.DECK)),
                'grave': len(duel.get_cards_in_location(1 - index, LOCATION.GRAVE)),
                'removed': len(duel.get_cards_in_location(1 - index, LOCATION.REMOVED)),
            },
        },
        # !! Card is not serializable by JSON emitter, thus keep card code for simplicity !!
        # See: Card.__init__()
        # See: Duel.show_cards_in_location(player, index, LOCATION.HAND, False)
        'hand': [card.code for card in duel.get_cards_in_location(index, LOCATION.HAND)],
        # See: Duel.show_table()
        'table': {
            'player': {
                'monster': [
                    (card.code, card.position, )
                        for card in duel.get_cards_in_location(index, LOCATION.MZONE)
                ],
                'spell': [
                    (card.code, card.position, )
                        for card in duel.get_cards_in_location(index, LOCATION.SZONE)
                ],
            },
            "opponent": {
                'monster': [
                    ((0 if card.position & (0x2 | 0x8) else card.code), card.position)
                        for card in duel.get_cards_in_location(1 - index, LOCATION.MZONE)
                ],
                'spell': [
                    ((0 if card.position & (0x2 | 0x8) else card.code), card.position)
                        for card in duel.get_cards_in_location(1 - index, LOCATION.SZONE)
                ],
            }
        },
    }

# See ygo.constants.PHASES
def dump(duel: Duel, player: Player, requirement: Optional[str] = None, **kwargs) -> Dict:
    """ Pack the current state of the duel into a JSON string. """
    return { 'state': _dump_state(duel, player), **kwargs }
