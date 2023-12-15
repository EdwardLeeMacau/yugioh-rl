import io
import natsort
from typing import Dict, List
from twisted.internet import reactor

from ygo.constants import AMOUNT_RACES, RACES_OFFSET
from ygo.dump import dump_game_info
from ygo.duel_reader import DuelReader
from ygo.parsers.duel_parser import DuelParser
from ygo.utils import process_duel

def msg_announce_race(self, data):
	data = io.BytesIO(data[1:])
	player = self.read_u8(data)
	count = self.read_u8(data)
	avail = self.read_u32(data)
	self.cm.call_callbacks('announce_race', player, count, avail)
	return data.read()

def announce_race(self, player: int, count, avail: int):
	"""
	Parameters
	----------
	self : ygo.duel.Duel

	player : int
		0 or 1

	count : int
		number of races to announce

	avail : int
		bit-mask
	"""
	pl = self.players[player]
	racemap: Dict[str, int] = {pl.strings['system'][RACES_OFFSET+i]: (1<<i) for i in range(AMOUNT_RACES)}
	avail_races: Dict[str, int] = {k: v for k, v in racemap.items() if avail & v}
	avail_races_keys: List[str] = natsort.natsorted(list(avail_races.keys()))

	def prompt():
		pl.notify("Type %d races separated by spaces." % count)
		for i, s in enumerate(avail_races_keys):
			pl.notify("%d: %s" % (i+1, s))

		self.players[self.agent].notify(dump_game_info(
			self, pl, recv=int(self.agent != player), **{ 'actions': {
				'requirement': 'ANNOUNCE_RACE', 'min': count, 'max': count,
				'options': [k for _, k in enumerate(avail_races_keys)],
				'targets': [],
			}}
		))
		pl.notify(DuelReader, r, no_abort="Invalid entry.", restore_parser=DuelParser)

	def error(text):
		pl.notify(text)
		pl.notify(DuelReader, r, no_abort="Invalid entry.", restore_parser=DuelParser)

	def r(caller):
		res: List[str] = []
		try:
			res = caller.text.split()
		except ValueError:
			return error("Invalid value.")

		if len(res) != count:
			return error("%d items required." % count)
		if len(res) != len(set(res)):
			return error("Duplicate values not allowed.")
		if any(i not in avail_races for i in res):
			return error("Invalid value.")
		result = 0
		for i in res:
			result |= avail_races[i]
		self.set_responsei(result)
		reactor.callLater(0, process_duel, self)

	prompt()

MESSAGES = {140: msg_announce_race}

CALLBACKS = {'announce_race': announce_race}
