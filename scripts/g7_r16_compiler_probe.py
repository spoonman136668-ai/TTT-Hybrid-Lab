"""G7-R16: read-only current compiler discovery, original exact source custody.

Returns 0 compiler present, 3 compiler absent, 2 custody block, 1 unexpected.
No root, sudo, apt, CUDA or model invocation.
"""
from __future__ import annotations
import argparse
import sys

import g7_r15_compiler_repair_forward as native


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--expected-sha", required=True)
    args = p.parse_args()
    try:
        native.source_custody(args.expected_sha)
        found = native.selected_compiler()
        if found is None:
            print("G7_R16_COMPILER=ABSENT")
            return 3
        print("G7_R16_COMPILER=PRESENT")
        return 0
    except native.Blocked as exc:
        print("G7_R16_BLOCKED=" + str(exc))
        return 2
    except Exception as exc:
        print("G7_R16_FAIL=" + type(exc).__name__)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
