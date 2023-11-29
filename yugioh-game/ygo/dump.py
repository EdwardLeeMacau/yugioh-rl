from typing import Callable, Dict

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
                'monster': [card.code for card in duel.get_cards_in_location(index, LOCATION.MZONE)],
                'spell': [card.code for card in duel.get_cards_in_location(index, LOCATION.SZONE)],
            },
            "opponent": {
                'monster': [card.code for card in duel.get_cards_in_location(1 - index, LOCATION.MZONE)],
                'spell': [card.code for card in duel.get_cards_in_location(1 - index, LOCATION.SZONE)],
            }
        },
    }

def _dump_idle_options(duel: Duel, player: Player) -> Dict:
    """ Pack the current valid action of the duel into a JSON string. """
    return {
        # Summonable in attack position
        'summonable': [card.get_spec(player) for card in duel.summonable],
        # Summonable in defense position
        'mset': [card.get_spec(player) for card in duel.idle_mset],
        # Special summonable
        'spsummon': [card.get_spec(player) for card in duel.spsummon],
        # Activatable
        'activate': [card.get_spec(player) for card in duel.idle_activate],
        # Re-positionable
        'repos': [card.get_spec(player) for card in duel.repos],
        # Settable
        'set': [card.get_spec(player) for card in duel.idle_set],
        # To next phase
        'to_bp': duel.to_bp,
        'to_ep': duel.to_ep,
    }

def _dump_battle_options(duel: Duel, player: Player) -> Dict:
    return {
        'attackable': [card.get_spec(player) for card in duel.attackable],
        'activatable': [card.get_spec(player) for card in duel.activatable],
        'to_m2': duel.to_m2,
        'to_ep': duel.to_ep,
    }

def _dump(*args, **kwargs) -> Dict:
    return {}

def _dump_options(phase: int) -> Callable:
    """ Pack the current valid action of the duel into a JSON string. """
    return {
        0x4: _dump_idle_options,
        0x8: _dump_battle_options,
    }.get(phase, _dump)

# See ygo.constants.PHASES
def dump(duel: Duel, player: Player, **kwargs) -> Dict:
    """ Pack the current state of the duel into a JSON string. """
    state = {
        'state': _dump_state(duel, player),
        '?': _dump_options(duel.current_phase)(duel, player),
    }

    # Concatenate the keyword arguments. The later one will override the former one.
    state.update(kwargs)

    return state