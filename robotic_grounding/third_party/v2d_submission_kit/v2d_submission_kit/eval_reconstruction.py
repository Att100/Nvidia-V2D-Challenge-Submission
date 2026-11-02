#!/usr/bin/env python3
"""Prepare a Track 1 reconstruction Parquet for the Kaggle leaderboards.

This entry point uses the same schema and validation as tools/pack_reconstruction.py.
Run ``python3 eval_reconstruction.py --help`` for the required episode files and options.
"""

from __future__ import annotations

import sys

from tools.pack_reconstruction import main


if __name__ == "__main__":
    sys.argv.insert(1, "track1")
    main()
