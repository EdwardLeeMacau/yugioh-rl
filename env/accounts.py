from dataclasses import dataclass
from queue import Queue


@dataclass
class Account:
    host: str
    port: int
    username: str
    password: str

# Create a global pool to manage the accounts
# User should allocate / free accounts through the APIs
n = 128
_POOL = Queue(maxsize=n)
for i in range(1, n + 1):
    _POOL.put(Account(
        host="localhost", port=4000,
        username=f"player{i}", password=f"player{i}"
    ))

def allocate() -> Account:
    """ Allocate an account from the account pool. """
    return _POOL.get()

def free(account: Account):
    """ Free the accounts back to the account pool. """
    _POOL.put(account, block=False)
