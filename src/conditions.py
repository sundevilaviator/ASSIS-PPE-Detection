"""
ASSIS conditional PPE policy layer.

Implements docs/PPE_TAXONOMY.md section 4. This module is deliberately
independent of the detector: it decides which PPE classes are REQUIRED
right now, given weather and time of day. The detector (utils.py) only
reports what it OBSERVES. Keeping these separate means the requirement
logic can be corrected without retraining, and the detector's accuracy
can be evaluated without conflating it with policy assumptions.

Data source: METAR, fetched from the NOAA Aviation Weather Center's free
public API. No API key required.
    https://aviationweather.gov/data/api/#/Data/dataMetars
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone

import requests

AWC_METAR_URL = "https://aviationweather.gov/api/data/metar"

# Thresholds are placeholders - not sourced from a specific regulation.
# Tune against ramp supervisor input before relying on them operationally.
HIGH_WIND_KT = 25
LOW_TEMP_C = 4
BLOWING_DUST_CODES = {"DU", "SA", "PO", "BLDU", "BLSA"}
PRECIP_CODES = {"RA", "SN", "DZ", "PL", "GR", "GS", "IC", "FZRA", "FZDZ"}


@dataclass
class WeatherConditions:
    station: str
    observed_at: datetime | None
    wind_speed_kt: float | None
    temp_c: float | None
    precip: bool
    blowing_debris: bool
    icing_risk: bool
    raw_metar: str

    @property
    def is_stale(self) -> bool:
        """METAR is periodic (hourly, + SPECI on significant change).

        An observation older than ~90 minutes should not be trusted for
        a real-time policy decision - treat as unknown / fail open.
        """
        if self.observed_at is None:
            return True
        age = datetime.now(timezone.utc) - self.observed_at
        return age.total_seconds() > 90 * 60


@dataclass
class PPERequirements:
    vest: bool
    hearing_protection: bool
    eye_protection: bool
    gloves: bool
    footwear: bool
    reasons: dict  # class -> human-readable reason for the determination


def fetch_metar(station_icao: str, timeout: float = 5.0) -> WeatherConditions | None:
    """Fetch and parse the latest METAR for a station.

    Returns None on any network/parse failure - callers must handle this
    as "conditions unknown", not as "conditions are calm". See
    default_requirements() for the fail-safe behaviour.
    """
    try:
        resp = requests.get(
            AWC_METAR_URL,
            params={"ids": station_icao, "format": "json", "hours": 2},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return None
        latest = data[0]
    except (requests.RequestException, ValueError, KeyError, IndexError):
        return None

    raw = latest.get("rawOb", "")
    observed_at = None
    if latest.get("obsTime"):
        observed_at = datetime.fromtimestamp(latest["obsTime"], tz=timezone.utc)

    wind_speed = latest.get("wspd")
    temp_c = latest.get("temp")

    wx_string = latest.get("wxString", "") or ""
    precip = any(code in wx_string for code in PRECIP_CODES)
    blowing_debris = any(code in wx_string for code in BLOWING_DUST_CODES)
    icing_risk = temp_c is not None and temp_c <= 2 and (precip or "FZ" in wx_string)

    return WeatherConditions(
        station=station_icao,
        observed_at=observed_at,
        wind_speed_kt=wind_speed,
        temp_c=temp_c,
        precip=precip,
        blowing_debris=blowing_debris,
        icing_risk=icing_risk,
        raw_metar=raw,
    )


def default_requirements() -> PPERequirements:
    """Fail-safe requirement set when conditions are unknown or stale.

    Fails toward the SAFER assumption for each class: if we cannot confirm
    conditions, assume the stricter requirement rather than the relaxed one.
    Vest and footwear are unconditional regardless of weather, so they are
    unaffected by this fallback.
    """
    return PPERequirements(
        vest=True,
        hearing_protection=True,
        eye_protection=True,  # fail toward requiring, not exempting
        gloves=False,  # gloves has no unconditional baseline; leave advisory
        footwear=True,
        reasons={
            "vest": "Always required, independent of conditions.",
            "hearing_protection": "Always required near active ramp operations.",
            "eye_protection": "Conditions unknown or stale METAR - failing toward required.",
            "gloves": "Conditions unknown - no baseline requirement to fall back on.",
            "footwear": "Always required, independent of conditions.",
        },
    )


def determine_requirements(weather: WeatherConditions | None) -> PPERequirements:
    """Apply docs/PPE_TAXONOMY.md section 4.1 to current conditions."""
    if weather is None or weather.is_stale:
        return default_requirements()

    reasons = {
        "vest": "Always required, independent of conditions.",
        "footwear": "Always required, independent of conditions.",
        "hearing_protection": "Required near active ramp/engine operations (not confirmable from METAR alone).",
    }

    eye_required = weather.blowing_debris or (
        weather.wind_speed_kt is not None and weather.wind_speed_kt >= HIGH_WIND_KT
    )
    reasons["eye_protection"] = (
        f"Required: blowing debris or wind {weather.wind_speed_kt} kt >= {HIGH_WIND_KT} kt threshold."
        if eye_required
        else f"Not indicated by METAR (wind {weather.wind_speed_kt} kt, no blowing debris reported)."
    )

    gloves_required = (
        weather.temp_c is not None and weather.temp_c <= LOW_TEMP_C
    ) or weather.icing_risk
    reasons["gloves"] = (
        f"Required: temperature {weather.temp_c}C <= {LOW_TEMP_C}C or icing risk."
        if gloves_required
        else f"Not indicated by METAR (temperature {weather.temp_c}C)."
    )

    return PPERequirements(
        vest=True,
        hearing_protection=True,
        eye_protection=eye_required,
        gloves=gloves_required,
        footwear=True,
        reasons=reasons,
    )


_METAR_LOOKS_LIKE = re.compile(r"^[A-Z]{4}$")


def is_valid_icao(code: str) -> bool:
    return bool(_METAR_LOOKS_LIKE.match(code.strip().upper()))
