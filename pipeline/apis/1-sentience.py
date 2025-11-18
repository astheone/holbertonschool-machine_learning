#!/usr/bin/env python3
"""
Module that returns the list of names of the home planets
of all sentient species using the SWAPI API.
"""

import requests


def sentientPlanets():
    """
    Returns a list of planet names that are homeworlds
    of species classified or designated as sentient.
    """
    url = "https://swapi-api.alx-tools.com/api/species/"
    planets = []

    while url:
        r = requests.get(url)
        data = r.json()

        for sp in data.get("results", []):
            classification = sp.get("classification", "").lower()
            designation = sp.get("designation", "").lower()

            # Check if species is sentient
            if classification == "sentient" or designation == "sentient":
                home = sp.get("homeworld")
                if home:
                    planet_data = requests.get(home).json()
                    name = planet_data.get("name")
                    if name:
                        planets.append(name)

        # Pagination
        url = data.get("next")

    return planets
