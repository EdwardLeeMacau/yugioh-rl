# File to suppress unhandled messages

import io
import json

def msg_do_nothing(self, data):
	return b''

MESSAGES = {
	msg: msg_do_nothing for msg in (71, 72, 73, 74, )
}
