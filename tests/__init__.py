import os, sys
ml_micro_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ml_micro_dir not in sys.path:
    sys.path.append(ml_micro_dir)

TEST_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data")