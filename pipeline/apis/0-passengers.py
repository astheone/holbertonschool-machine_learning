#!/usr/bin/env python3
"""
Module that returns a list of starships that can hold
a given number of passengers from the SWAPI API.
"""

import requests


def availableShips(passengerCount):
    """
    Returns a list of starships that can carry at least
    passengerCount passengers.
    Handles SWAPI pagination.
    """
    url = "https://swapi-api.alx-tools.com/api/starships/"
    ships = []

    while url:
        response = requests.get(url)
        data = response.json()

        for ship in data.get("results", []):
            p = ship.get("passengers", "0").replace(",", "")
            if p.isdigit():
                if int(p) >= passengerCount:
                    ships.append(ship.get("name"))

        url = data.get("next")

    return ships
