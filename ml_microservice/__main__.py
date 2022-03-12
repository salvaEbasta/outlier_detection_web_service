import os
import sys

WDIR = os.path.split(__file__)[0]
if WDIR not in sys.path:
    sys.path.append(WDIR)

from ml_microservice import RPCservice

if __name__ == "__main__":
    RPCservice.build_app().run()