"""Acquisition instructions for the PISA 2022 data.

The OECD does not provide a stable direct download URL for the student
questionnaire file, and the published archive is distributed as SPSS. This
script states the manual steps and verifies a prepared extract rather than
silently fetching a file whose provenance cannot be checked.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SOURCE_PAGE = "https://www.oecd.org/en/data/datasets/pisa-2022-database.html"

INSTRUCTIONS = f"""
PISA 2022 data acquisition
==========================

1. Open the OECD PISA 2022 database page:

   {SOURCE_PAGE}

2. Download the student questionnaire data file (SPSS format),
   CY08MSP_STU_QQQ.sav. The archive is approximately 1.5 GB.

3. Filter to the United Kingdom and write a CSV extract:

     import pandas as pd
     frame = pd.read_spss("CY08MSP_STU_QQQ.sav")
     frame[frame["CNT"] == "GBR"].to_csv("data/uk_pisa_2022.csv", index=False)

4. Place the extract at data/uk_pisa_2022.csv.

5. Verify it:

     python scripts/download_data.py --verify data/uk_pisa_2022.csv

The data is not committed to this repository and is not redistributed here.
"""

REQUIRED = ["W_FSTUWT", "ESCS", "REGION", *[f"PV{i}MATH" for i in range(1, 11)]]


def verify(path: Path) -> int:
    if not path.exists():
        print(f"error: {path} does not exist", file=sys.stderr)
        return 1

    import pandas as pd

    header = pd.read_csv(path, nrows=0)
    missing = [c for c in REQUIRED if c not in header.columns]
    if missing:
        print(f"error: extract is missing required columns: {missing}", file=sys.stderr)
        return 1

    rows = sum(1 for _ in open(path, encoding="utf-8", errors="replace")) - 1
    print(f"extract verified: {rows} rows, {len(header.columns)} columns")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PISA 2022 data acquisition helper")
    parser.add_argument("--verify", type=Path, help="Verify a prepared extract")
    args = parser.parse_args(argv)

    if args.verify is not None:
        return verify(args.verify)

    print(INSTRUCTIONS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
