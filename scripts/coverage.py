"""Prints line coverage for the package sources, read from `xccov view --report --json` on stdin.

The xccov summary counts the test files too, since they share a binary with the code under test.
"""

import json
import sys

report = json.load(sys.stdin)
files = {f["path"]: f for target in report["targets"] for f in target["files"] if "/Sources/" in f["path"]}
covered = sum(f["coveredLines"] for f in files.values())
total = sum(f["executableLines"] for f in files.values())
for _, f in sorted(files.items()):
    print(f"{f['name']:24} {100 * f['lineCoverage']:5.1f}%  ({f['coveredLines']}/{f['executableLines']})")
print(f"{'total':24} {100 * covered / total:5.1f}%  ({covered}/{total})")
