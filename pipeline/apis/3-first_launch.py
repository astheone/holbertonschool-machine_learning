#!/usr/bin/env python3
"""
Script that displays the first upcoming SpaceX launch with:
- launch name
- date (local time)
- rocket name
- launchpad name and locality
"""

import requests


if __name__ == "__main__":
    # 1. Get upcoming launches
    url = "https://api.spacexdata.com/v4/launches/upcoming"
    launches = requests.get(url).json()

    # 2. Sort by date_unix (ascending)
    launches.sort(key=lambda x: x.get("date_unix", float("inf")))

    # 3. Pick first launch
    first = launches[0]

    launch_name = first.get("name")
    launch_date = first.get("date_local")
    rocket_id = first.get("rocket")
    launchpad_id = first.get("launchpad")

    # 4. Get rocket name
    rocket = requests.get(
        f"https://api.spacexdata.com/v4/rockets/{rocket_id}"
    ).json()
    rocket_name = rocket.get("name")

    # 5. Get launchpad name + locality
    launchpad = requests.get(
        f"https://api.spacexdata.com/v4/launchpads/{launchpad_id}"
    ).json()
    launchpad_name = launchpad.get("name")
    locality = launchpad.get("locality")

    # 6. Final output format
    print(f"{launch_name} ({launch_date}) {rocket_name} - "
          f"{launchpad_name} ({locality})")
