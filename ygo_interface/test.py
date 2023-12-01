import os
import random
import sqlite3

from _duel import ffi, lib

# create in-memory database
def create_in_memory_database():
    cdb = sqlite3.connect(':memory:')
    cdb.row_factory = sqlite3.Row
    cdb.create_function('UPPERCASE', 1, lambda s: s.upper())
    cdb.execute('ATTACH ? AS new', ('cards.cdb',))
    cdb.execute('CREATE TABLE datas AS SELECT * FROM new.datas WHERE id<100000000')
    cdb.execute('CREATE TABLE texts AS SELECT * FROM new.texts WHERE id<100000000')
    cdb.execute('DETACH new')
    cdb.execute('CREATE UNIQUE INDEX idx_datas_id ON datas (id)')
    cdb.execute('CREATE UNIQUE INDEX idx_texts_id ON texts (id)')

    cursor = cdb.execute('SELECT * FROM datas LIMIT 1')
    row = cursor.fetchone()
    columns = row.keys()
    cursor.close()

    cdb.execute('ATTACH ? as new', ('cards.cdb',))
    cursor = cdb.execute('SELECT * FROM new.datas LIMIT 1')
    row = cursor.fetchone()
    new_columns = row.keys()
    cursor.close()

    new_columns = [c for c in new_columns if c in columns]
    cdb.execute('INSERT OR REPLACE INTO datas ({0}) SELECT {0} FROM new.datas WHERE id<100000000'.format(', '.join(new_columns)))
    cdb.execute('INSERT OR REPLACE INTO texts SELECT * FROM new.texts WHERE id<100000000')
    cdb.commit()
    cdb.execute('DETACH new')

    return cdb

@ffi.def_extern()
def card_reader_callback(code, data):
    cd = data[0]
    row = database.execute('select * from datas where id=?', (code,)).fetchone()
    cd.code = code
    cd.alias = row['alias']
    cd.setcode = row['setcode']
    cd.type = row['type']
    cd.level = row['level'] & 0xff
    cd.lscale = (row['level'] >> 24) & 0xff
    cd.rscale = (row['level'] >> 16) & 0xff
    cd.attack = row['atk']
    cd.defense = row['def']
    if cd.type & 0x4000000:
        cd.link_marker = cd.defense
        cd.defense = 0
    else:
        cd.link_marker = 0
    cd.race = row['race']
    cd.attribute = row['attribute']
    return 0

scriptbuf = ffi.new('char[131072]')
@ffi.def_extern()
def script_reader_callback(name, lenptr):
    fn = ffi.string(name)
    if not os.path.exists(fn):
        lenptr[0] = 0
        return ffi.NULL
    s = open(fn, 'rb').read()
    buf = ffi.buffer(scriptbuf)
    buf[0:len(s)] = s
    lenptr[0] = len(s)
    return ffi.cast('byte *', scriptbuf)

database = create_in_memory_database()
def main():
    buf = ffi.new('char[]', 4096)
    seed = random.randint(0, 2**32 - 1)
    cards = [
        52927340,
        52927340,
        53143898,
        53143898,
        53143898,
        14558127,
        14558127,
        14558127,
        89538537,
        89538537,
        23434538,
        23434538,
        23434538,
        25533642,
        25533642,
        25533642,
        2295440,
        2295440,
        18144506,
        35261759,
        35261759,
        35261759,
        68462976,
        68462976,
        23924608,
        23924608,
        35146019,
        35146019,
        27541563,
        27541563,
        27541563,
        53936268,
        53936268,
        53936268,
        61740673,
        61740673,
        40605147,
        40605147,
        41420027,
        41420027,
    ]
    options = 0

    lib.set_card_reader(lib.card_reader_callback)
    lib.set_script_reader(lib.script_reader_callback)
    print(f'Set callbacks')

    duel = lib.create_duel(seed)
    print(f'Created duel with seed {seed}')

    lib.set_player_info(duel, 0, 8000, 5, 1)
    lib.set_player_info(duel, 1, 8000, 5, 1)
    print(f'Set player info')

    for c in cards:
        # void new_card(
        #   ptr pduel, uint32 code, uint8 owner, uint8 playerid, uint8 location, uint8 sequence, uint8 position
        # );
        lib.new_card(duel, c, 0, 0, 0x1, 0, 0x8)
        lib.new_card(duel, c, 1, 1, 0x1, 0, 0x8)
    print(f'Added cards')

    lib. (duel, options)
    print(f'Started duel')

    deck = [lib.query_field_count(duel, playerid, 0x1) for playerid in range(2)]
    print(f'Player 0 has {deck[0]} cards in deck. Player 1 has {deck[1]} cards in deck.')

    res = lib.process(duel)
    print(f'{res=}')

    msg = lib.get_message(duel, ffi.cast('byte *', buf))
    msg = ffi.unpack(buf, msg)
    print(f'{msg=}')

    lib.end_duel(duel)
    print(f'Ended duel')

if __name__ == '__main__':
    main()
