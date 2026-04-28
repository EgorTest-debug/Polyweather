"""
City configuration — airports, not city centers.
Every Polymarket weather market resolves on a specific airport station.
We pick "second tier" cities with decent liquidity ($30-70K/day) but less bot competition.
"""

CITIES = {
    # bias_correction: how much to ADD to ECMWF to match Wunderground resolution
    # Calculated from 14 days of ECMWF vs Polymarket winner data (April 2026)

    # ── Asia ──────────────────────────────────────────────
    "seoul": {
        "name":      "Seoul",
        "lat":       37.4691,
        "lon":       126.4505,
        "station":   "RKSI",        # Incheon International
        "unit":      "C",
        "timezone":  "Asia/Seoul",
        "bias_correction": 0.8,     # ECMWF bias=-0.8, MAE=1.0
    },
    "tokyo": {
        "name":      "Tokyo",
        "lat":       35.7647,
        "lon":       140.3864,
        "station":   "RJTT",        # Haneda
        "unit":      "C",
        "timezone":  "Asia/Tokyo",
        "bias_correction": 1.3,     # ECMWF bias=-1.3, MAE=1.4
    },
    "singapore": {
        "name":      "Singapore",
        "lat":       1.3502,
        "lon":       103.9940,
        "station":   "WSSS",        # Changi
        "unit":      "C",
        "timezone":  "Asia/Singapore",
        "bias_correction": 2.0,     # ECMWF bias=-2.0, MAE=2.0
    },
    "tel-aviv": {
        "name":      "Tel Aviv",
        "lat":       32.0114,
        "lon":       34.8867,
        "station":   "LLBG",        # Ben Gurion
        "unit":      "C",
        "timezone":  "Asia/Jerusalem",
        "bias_correction": 0.0,     # No data yet — conservative estimate
    },

    # ── Europe ────────────────────────────────────────────
    "helsinki": {
        "name":      "Helsinki",
        "lat":       60.3172,
        "lon":       24.9633,
        "station":   "EFHK",        # Helsinki-Vantaa
        "unit":      "C",
        "timezone":  "Europe/Helsinki",
        "bias_correction": 1.1,     # ECMWF bias=-1.1, MAE=1.3
    },
    "ankara": {
        "name":      "Ankara",
        "lat":       40.1281,
        "lon":       32.9951,
        "station":   "LTAC",        # Esenboğa
        "unit":      "C",
        "timezone":  "Europe/Istanbul",
        "bias_correction": 1.3,     # No data yet — conservative estimate
    },

    # ── Americas ──────────────────────────────────────────
    "toronto": {
        "name":      "Toronto",
        "lat":       43.6772,
        "lon":       -79.6306,
        "station":   "CYYZ",        # Pearson International
        "unit":      "C",
        "timezone":  "America/Toronto",
        "bias_correction": 0.0,     # ECMWF bias=+0.3, MAE=1.4 — best calibrated
    },
    "sao-paulo": {
        "name":      "Sao Paulo",
        "lat":       -23.4356,
        "lon":       -46.4731,
        "station":   "SBGR",        # Guarulhos
        "unit":      "C",
        "timezone":  "America/Sao_Paulo",
        "bias_correction": 1.0,     # No data yet — conservative estimate
    },
    "buenos-aires": {
        "name":      "Buenos Aires",
        "lat":       -34.8222,
        "lon":       -58.5358,
        "station":   "SAEZ",        # Ezeiza
        "unit":      "C",
        "timezone":  "America/Argentina/Buenos_Aires",
        "bias_correction": 0.7,     # No data yet — conservative estimate
    },

    # ── Oceania ───────────────────────────────────────────
    "wellington": {
        "name":      "Wellington",
        "lat":       -41.3272,
        "lon":       174.8052,
        "station":   "NZWN",        # Wellington International
        "unit":      "C",
        "timezone":  "Pacific/Auckland",
        "bias_correction": 0.9,     # No data yet — conservative estimate
    },
}

MONTHS = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]
