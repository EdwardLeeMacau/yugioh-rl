from env.state import StateMachine

def test_idle():
    sm = StateMachine.from_dict({
        '?': {
            'activate': ['h4'],
            'mset': ['h1', 'h2'],
            'repos': [],
            'requirement': 'IDLE',
            'set': ['h3', 'h4', 'h5'],
            'spsummon': [],
            'summonable': ['h1', 'h2'],
            'to_bp': False,
            'to_ep': True
        }
    })

    assert set(sm.list_valid_actions()) == set(['h1', 'h2', 'h3', 'h4', 'h5', 'e'])
    assert sm.step('h1') == False

    assert set(sm.list_valid_actions()) == set(['s', 'm'])
    assert sm.step('s') == True

    assert sm.to_string() == 'h1\r\ns'

def test_battle():
    sm = StateMachine.from_dict({
        '?': {
            'activatable': [],
            'attackable': ['m1'],
            'requirement': 'BATTLE',
            'to_ep': True,
            'to_m2': True
        }
    })

    assert set(sm.list_valid_actions()) == set(['m1', 'm', 'e'])
    assert sm.step('m1') == False

    assert set(sm.list_valid_actions()) == set(['a'])
    assert sm.step('a') == True

    assert sm.to_string() == 'a\r\nm1'

def test_select():
    sm = StateMachine.from_dict({
        '?': {
            'choices': ['y', 'n'],
            'max': 1,
            'min': 1,
            'requirement': 'SELECT',
            'type': 'spec'
        }
    })

    assert set(sm.list_valid_actions()) == set(['y', 'n'])
    assert sm.step('y') == True
    assert sm.to_string() == 'y'

def test_select_indices():
    sm = StateMachine.from_dict({
        '?': {
            'choices': [18036057, 63749102, 39507162, 97077563, 42829885, 71044499, 5318639],
            'max': 1,
            'min': 1,
            'requirement': 'SELECT'
        }
    })

    assert set(sm.list_valid_actions()) == set(['1', '2', '3', '4', '5', '6', '7'])
    assert sm.step('1') == True
    assert sm.to_string() == '1'

def test_select_indices_batch():
    sm = StateMachine.from_dict({
        '?': {
            'choices': [7572887, 63749102, 77585513],
            'max': 2,
            'min': 2,
            'requirement': 'SELECT'
        }
    })

    assert set(sm.list_valid_actions()) == set(['1', '2', '3'])
    assert sm.step('1') == False

    assert set(sm.list_valid_actions()) == set(['2', '3'])
    assert sm.step('2') == True
    assert sm.to_string() == '1 2'

def test_select_indices_for_each():
    sm = StateMachine.from_dict({
        '?': {
            'choices': [7572887, 63749102, 77585513],
            'foreach': 1,
            'max': 2,
            'min': 2,
            'requirement': 'SELECT'
        }
    })

    assert set(sm.list_valid_actions()) == set(['1', '2', '3'])
    assert sm.step('1') == True
    assert sm.to_string() == '1'
