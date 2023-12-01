from dataclasses import dataclass
from queue import Queue
from typing import Dict, List

@dataclass
class Account:
    host: str
    port: int
    username: str
    password: str

# Create a global pool to manage the accounts
# User should allocate / free accounts through the APIs
_POOL = Queue(maxsize=8)
for i in range(1, 9):
    _POOL.put(Account(
        host="cubone.csie.org", port=4002,
        username=f"player{i}", password=f"player{i}"
    ))

def allocate() -> Account:
    """ Allocate an account from the account pool. """
    return _POOL.get()

def free(account: Account):
    """ Free the accounts back to the account pool. """
    _POOL.put(account, block=False)
