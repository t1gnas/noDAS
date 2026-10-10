UNVERIFIED_ID = 9

CLASS_STYLE = {
    "background": ("Quiet", "#8E8E93"),
    "car": ("Vehicle", "#FF9F0A"),
    "construction": ("Construction", "#FFD60A"),
    "fence": ("Fence contact", "#FF453A"),
    "longboard": ("Longboard", "#64D2FF"),
    "manipulation": ("Cable handling", "#BF5AF2"),
    "openclose": ("Manhole opened", "#FF375F"),
    "running": ("Running", "#5E5CE6"),
    "walk": ("Walking", "#0A84FF"),
}

UNVERIFIED = {"id": UNVERIFIED_ID, "name": "unverified", "label": "Unverified activity", "color": "#AEAEB2"}

SCENARIOS = (
    {"id": "vehicle", "title": "Vehicle on road", "subtitle": "Cars pass over a 1.2 km buried cable", "icon": "car", "recording": "auto4_2023-06-06T145055+0100"},
    {"id": "walking", "title": "Person walking", "subtitle": "Footsteps along the cable route", "icon": "walk", "recording": "walking2_2023-04-17T122153+0100"},
    {"id": "fence", "title": "Fence intrusion", "subtitle": "Someone climbs and shakes the fence", "icon": "fence", "recording": "fence_2023-04-17T123117+0100"},
    {"id": "construction", "title": "Construction work", "subtitle": "Heavy machinery digging near the line", "icon": "construction", "recording": "reconstruction1_2023-06-06T150714+0100"},
    {"id": "running", "title": "Person running", "subtitle": "A runner sprints past the sensor", "icon": "running", "recording": "running_2023-04-17T122413+0100"},
    {"id": "manhole", "title": "Manhole opened", "subtitle": "A cover is lifted and closed again", "icon": "manhole", "recording": "openingclosing_2023-04-17T122658+0100"},
)


def class_list(names: list[str]) -> list[dict]:
    return [{"id": i, "name": n, "label": CLASS_STYLE[n][0], "color": CLASS_STYLE[n][1]} for i, n in enumerate(names)] + [UNVERIFIED]


def find_scenario(scenario_id: str) -> dict:
    match = [s for s in SCENARIOS if s["id"] == scenario_id]
    assert match, f"unknown scenario {scenario_id!r}; choose from {[s['id'] for s in SCENARIOS]}"
    return match[0]
