#!/usr/bin/env python3
"""
Script that prints the location of a GitHub user.
"""

import sys
import requests
import time


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit()

    url = sys.argv[1]
    res = requests.get(url)

    if res.status_code == 404:
        print("Not found")
        sys.exit()

    if res.status_code == 403:
        reset = res.headers.get("X-RateLimit-Reset")
        if reset:
            reset_time = int(reset) - int(time.time())
            minutes = reset_time // 60
            print(f"Reset in {minutes} min")
        else:
            print("Reset in unknown time")
        sys.exit()

    data = res.json()
    location = data.get("location")

    if location:
        print(location)
    else:
        print("Not found")
