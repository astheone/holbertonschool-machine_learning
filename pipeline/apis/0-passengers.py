#!/usr/bin/env python3
"""
Module that returns list of ships that can hold given number of passengers
using the SWAPI API.
"""

import requests

def availableShips(passengerCount):
    """
    Returns list of starships that can carry at least passengerCount passengers.
    Handles pagination properly.
    """
    url = "https://swapi-api.alx-tools.com/api/starships/"
    result = []

    while url:
        r = requests.get(url)
        data = r.json()

        for ship in data.get("results", []):
            passengers = ship.get("passengers", "0")

            # Clean passengers value: remove commas, handle 'unknown', 'n/a', etc.
            if passengers.isdigit() or passengers.replace(",", "").isdigit():
                num = int(passengers.replace(",", ""))
                if num >= passengerCount:
                    result.append(ship["name"])

        url = data.get("next")  # pagination

    return result
