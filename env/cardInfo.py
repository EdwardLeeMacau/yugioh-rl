from enum import IntFlag, unique

from cardDatabase import CardDatabase


@unique
class TYPE(IntFlag):
	MONSTER = 0x1
	SPELL = 0x2
	TRAP = 0x4
	NORMAL = 0x10
	EFFECT = 0x20
	FUSION = 0x40
	RITUAL = 0x80
	TRAPMONSTER = 0x100
	SPIRIT = 0x200
	UNION = 0x400
	DUAL = 0x800
	TUNER = 0x1000
	SYNCHRO = 0x2000
	TOKEN = 0x4000
	QUICKPLAY = 0x10000
	CONTINUOUS = 0x20000
	EQUIP = 0x40000
	FIELD = 0x80000
	COUNTER = 0x100000
	FLIP = 0x200000
	TOON = 0x400000
	XYZ = 0x800000
	PENDULUM = 0x1000000
	SPSUMMON = 0x2000000
	LINK = 0x4000000
	# for this mud only
	EXTRA = XYZ | SYNCHRO | FUSION | LINK

class CardInfo:
    "A class for mapping card id to card information"

    def __init__(self, id: int) -> None:
        id2cardInfo = CardDatabase.id2cardInfo
        cardinfo = id2cardInfo[str(id)]

        self.id = id

        self.alias = cardinfo['alias']
        self.setcode = cardinfo['setcode']
        self.type = TYPE(cardinfo['type'])
        self.attack = cardinfo['atk']
        self.defense = cardinfo['def']
        self.level = cardinfo['level'] & 0xff
        self.race = cardinfo['race']
        self.attribute = cardinfo['attribute']
        self.category = cardinfo['category']

        self.name = cardinfo['name']
        self.desc = cardinfo['desc']
        self.strings = []
        for i in range(1, 17):
            self.strings.append(cardinfo[f"str{i}"])