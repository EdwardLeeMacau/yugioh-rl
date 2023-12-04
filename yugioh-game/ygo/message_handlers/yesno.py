import io
from twisted.internet import reactor

from ygo.card import Card
from ygo.dump import dump_game_info
from ygo.parsers.yes_or_no_parser import yes_or_no_parser
from ygo.utils import process_duel

def msg_yesno(self, data):
	data = io.BytesIO(data[1:])
	player = self.read_u8(data)
	desc = self.read_u32(data)
	self.cm.call_callbacks('yesno', player, desc)
	return data.read()

def yesno(self, player, desc):
	pl = self.players[player]
	old_parser = pl.connection.parser
	def yes(caller):
		self.set_responsei(1)
		reactor.callLater(0, process_duel, self)
	def no(caller):
		self.set_responsei(0)
		reactor.callLater(0, process_duel, self)
	if desc > 10000:
		code = desc >> 4
		card = Card(code)
		opt = card.get_strings(pl)[desc & 0xf]
		if opt == '':
			opt = pl._('Unknown question from %s. Yes or no?')%(card.get_name(pl))
	else:
		opt = "String %d" % desc
		opt = pl.strings['system'].get(desc, opt)

	pl.notify(dump_game_info(
		self, pl, **{ '?': {
			'requirement': 'YESNO',
			'question': opt,
			'choices': ['y', 'n'],
			'type': 'spec',
			'min': 1,
			'max': 1,
		}}
	))
	pl.notify(yes_or_no_parser, opt, yes, no=no, restore_parser=old_parser)

MESSAGES = {13: msg_yesno}

CALLBACKS = {'yesno': yesno}
