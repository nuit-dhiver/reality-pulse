#!/usr/bin/env python3
"""Print the UDID of an available iOS simulator.

Usage: pick-simulator.py iPhone

Picks the last available device whose name starts with the given prefix on the
newest installed iOS runtime, so workflows do not have to hard-code simulator
names that change with every Xcode release.
"""

import json
import re
import subprocess
import sys


def runtime_version(identifier):
    """Sort key for a runtime identifier such as ...SimRuntime.iOS-18-2."""
    match = re.search(r"iOS-([0-9-]+)$", identifier)
    if not match:
        return ()
    return tuple(int(part) for part in match.group(1).split("-"))


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: pick-simulator.py <device name prefix>")

    prefix = sys.argv[1]
    listing = subprocess.run(
        ["xcrun", "simctl", "list", "devices", "available", "--json"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    devices = json.loads(listing)["devices"]

    runtimes = sorted(
        (identifier for identifier in devices if "SimRuntime.iOS-" in identifier),
        key=runtime_version,
        reverse=True,
    )

    for identifier in runtimes:
        matches = [
            device
            for device in devices[identifier]
            if device.get("isAvailable", True) and device["name"].startswith(prefix)
        ]
        if matches:
            print(matches[-1]["udid"])
            return

    sys.exit(f"No available {prefix} simulator found.")


if __name__ == "__main__":
    main()
