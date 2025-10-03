import os, sys

# Ensure 'src' is on sys.path so packages under it are importable without PYTHONPATH
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from representation_learning.exp.train import main

if __name__ == '__main__':
    main()


