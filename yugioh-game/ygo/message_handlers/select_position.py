import io
from twisted.internet import reactor

from ygo.card import Card
from ygo.dump import dump_game_info
from ygo.constants import POSITION
from ygo.duel_menu import DuelMenu
from ygo.parsers.duel_parser import DuelParser
from ygo.utils import process_duel

def msg_select_position(self, data):
	data = io.BytesIO(data[1:])
	player = self.read_u8(data)
	code = self.read_u32(data)
	card = Card(code)
	positions = POSITION(self.read_u8(data))
	self.cm.call_callbacks('select_position', player, card, positions)
	return data.read()

def select_position(self, player, card, positions):
	pl = self.players[player]
	m = DuelMenu(
		pl._("Select position for %s:") % (card.get_name(pl),),
		no_abort="Invalid option.", persistent=True,
		restore_parser=DuelParser
	)
	def set(caller, pos=None):
		self.set_responsei(pos)
		reactor.callLater(0, process_duel, self)

	choices = []
	if positions & POSITION.FACEUP_ATTACK:
		m.item(pl._("Face-up attack"))(lambda caller: set(caller, 1))
		choices.append(len(choices) + 1)
	if positions & POSITION.FACEDOWN_ATTACK:
		m.item(pl._("Face-down attack"))(lambda caller: set(caller, 2))
		choices.append(len(choices) + 1)
	if positions & POSITION.FACEUP_DEFENSE:
		m.item(pl._("Face-up defense"))(lambda caller: set(caller, 4))
		choices.append(len(choices) + 1)
	if positions & POSITION.FACEDOWN_DEFENSE:
		m.item(pl._("Face-down defense"))(lambda caller: set(caller, 8))
		choices.append(len(choices) + 1)

	self.players[self.agent].notify(dump_game_info(
		self, pl, recv=int(self.agent != player), **{ 'actions': {
			'requirement': 'SELECT', 'min': 1, 'max': 1,
			'targets': [],
			'options': list(map(lambda x: str(int(x)), choices)),
		}}
	))
	pl.notify(m)

MESSAGES = {19: msg_select_position}

CALLBACKS = {'select_position': select_position}
