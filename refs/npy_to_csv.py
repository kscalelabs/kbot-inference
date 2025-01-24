#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path

import colorlogging
import numpy as np

logger = logging.getLogger(__name__)


def convert_npy_to_csv(npy_file: Path) -> None:
    """Convert a NumPy array file to CSV format."""
    try:
        # Load the NumPy array
        data = np.load(npy_file)

        # Flatten final dimensions.
        data = data.reshape(-1, data.shape[-1])

        # Create output path with .csv extension
        csv_file = npy_file.with_suffix(".csv")

        # Save as CSV
        np.savetxt(csv_file, data, delimiter=",")
        logger.info("Successfully converted %s to %s", npy_file, csv_file)

    except Exception as e:
        logger.error("Error converting %s: %s", npy_file, str(e))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert NumPy array files to CSV format"
    )
    parser.add_argument(
        "npy_files",
        type=str,
        nargs="+",
        help="One or more .npy files to convert",
    )
    args = parser.parse_args()

    colorlogging.configure()

    # Process each input file
    for npy_file in args.npy_files:
        npy_path = Path(npy_file)
        if not npy_path.exists():
            logger.warning("File not found: %s", npy_file)
            continue
        if npy_path.suffix != ".npy":
            logger.warning("File is not a .npy file: %s", npy_file)
            continue

        convert_npy_to_csv(npy_path)


if __name__ == "__main__":
    # python -m refs.npy_to_csv
    main()
