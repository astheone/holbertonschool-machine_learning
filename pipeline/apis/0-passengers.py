#!/usr/bin/env python3
"""
Module that returns a list of starships that can hold a given number of passengers.
"""

import requests

def availableShips(passengerCount):
    """
    Returns a list of starships that can carry at least passengerCount passengers.
    Handles pagination of the SWAPI API.
    """
    url = "https://swapi-api.alx-tools.com/api/starships/"
    result = []

    while url:
        r = requests.get(url)
        data = r.json()

        for ship in data.get("results", []):
            passengers = ship.get("passengers", "0")

            # Clean passengers value: remove commas, ignore 'unknown', etc.
            clean = passengers.replace(",", "")

            if clean.isdigit():
                num = int(clean)
                if num >= passengerCount:
                    result.append(ship.get("name"))

        url = data.get("next")  # pagination

    return result
