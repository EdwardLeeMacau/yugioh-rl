import json
from itertools import chain
from twisted.internet import reactor

from ygo.duel_reader import DuelReader
from ygo.dump import dump_game_info
from ygo.utils import process_duel
from ygo.parsers.duel_parser import DuelParser

def display_battle_menu(self, pl):
	pl.notify(pl._("Battle menu:"))
	if self.attackable:
		pl.notify(pl._("a: Attack."))
	if self.activatable:
		pl.notify(pl._("c: activate."))
	if self.to_m2:
		pl.notify(pl._("m: Main phase 2."))
	if self.to_ep:
		pl.notify(pl._("e: End phase."))
	def r(caller):
		if caller.text == 'a' and self.attackable:
			self.battle_attack(caller.connection.player)
		elif caller.text == 'c' and self.activatable:
			self.battle_activate(caller.connection.player)
		elif caller.text == 'e' and self.to_ep:
			self.set_responsei(3)
			reactor.callLater(0, process_duel, self)
		elif caller.text == 'm' and self.to_m2:
			self.set_responsei(2)
			reactor.callLater(0, process_duel, self)
		else:
			pl.notify(pl._("Invalid option."))
			return self.display_battle_menu(pl)

	options = []
	if self.attackable:
		options.append('a')
	if self.activatable:
		options.append('c')
	if self.to_m2:
		options.append('m')
	if self.to_ep:
		options.append('e')
	pl.notify(dump_game_info(
		self, pl, **{ 'actions': {
			'requirement': 'BATTLE',
			'options': options,
			'targets': {
				'a': [(card.get_spec(pl), card.code) for card in self.attackable],
				'c': [(card.get_spec(pl), card.code) for card in self.activatable],
			}
		}}
	))
	pl.notify(DuelReader, r, no_abort=pl._("Invalid command."), prompt=pl._("Select an option:"), restore_parser=DuelParser)

METHODS = {'display_battle_menu': display_battle_menu}
