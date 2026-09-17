CHANNELS_8 = [
    "L_O_DSI", "L_D_DSI", "L_O_DSphi", "L_D_DSphi",
    "R_O_DSI", "R_D_DSI", "R_O_DSphi", "R_D_DSphi",
]
# Intensity-only (drop all DSphi) when drop_suspect_channels=true.
SUSPECT = ["L_O_DSphi", "L_D_DSphi", "R_O_DSphi", "R_D_DSphi"]
CHANNELS_6 = [c for c in CHANNELS_8 if c not in SUSPECT]  # actually intensity-only count
CHANNELS_INTENSITY = CHANNELS_6

PAIRS_8 = [
    ("L_O_DSI", "R_O_DSI"),
    ("L_D_DSI", "R_D_DSI"),
    ("L_O_DSphi", "R_O_DSphi"),
    ("L_D_DSphi", "R_D_DSphi"),
]
PAIRS_6 = [("L_O_DSI", "R_O_DSI"), ("L_D_DSI", "R_D_DSI")]
PAIRS_INTENSITY = PAIRS_6

DOMAINS = ["R", "F", "L"]
CONDS = ["RW", "FW", "LW", "RP", "FP", "LP"]
