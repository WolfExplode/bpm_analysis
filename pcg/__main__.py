"""`python -m pcg` — see pcg/cli.py."""
import sys

from pcg.cli import main

sys.exit(main(sys.argv[1:]))
