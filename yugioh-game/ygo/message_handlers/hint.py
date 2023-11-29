import io
import json
from twisted.internet import reactor

from ygo.dump import dump
from ygo.utils import process_duel

def msg_hint(self, data):
	data = io.BytesIO(data[1:])
	msg = self.read_u8(data)
	player = self.read_u8(data)
	value = self.read_u32(data)
	self.cm.call_callbacks('hint', msg, player, value)
	return data.read()

def hint(self, msg, player, data):
	pl = self.players[player]
	op = self.players[1 - player]
	if msg == 3 and data in pl.strings['system']:
		self.players[player].notify(pl.strings['system'][data])
		self.players[player].notify(f'|{json.dumps(dump(self, pl, system=pl.strings["system"][data]))}|')
	elif msg == 6 or msg == 7 or msg == 8:
		reactor.callLater(0, process_duel, self)
	elif msg == 9:
		op.notify(op.strings['system'][1512] % data)
		reactor.callLater(0, process_duel, self)

MESSAGES = {2: msg_hint}

CALLBACKS = {'hint': hint}
