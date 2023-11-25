import argparse

from passlib.hash import pbkdf2_sha256

parser = argparse.ArgumentParser(description='Password hashing utility')
parser.add_argument('-p', '--password', required=True, help='Password to hash')

args = parser.parse_args()
print(pbkdf2_sha256.hash(args.password))
