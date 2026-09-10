"""Gameplay constants extracted from hardcoded magic numbers."""

# ── Combat ──────────────────────────────────────────────────────
MELEE_RANGE_THRESHOLD: int = 30  # attack_range <= this → melee
MAX_MELEE_ENGAGEMENTS: int = 6  # max simultaneous melee attackers per target
CHASE_DISTANCE: float = 180.0  # attack-move pursuit range (world units)
COMBAT_CELL_SIZE: float = 64.0  # spatial hash grid cell for combat targeting
FACILITY_ATTACK_BUFFER: float = 12.0  # extra range for attacking facilities
MAX_TARGET_SAMPLES: int = 4  # max candidates sampled per spatial hash cell

# ── Facilities & Structures ─────────────────────────────────────
BRIDGE_RADIUS: float = 160.0  # bridge crossing/build radius (world units)
BRIDGE_DECK_HALF_WIDTH: float = 36.0  # half-width of the physical walkable bridge deck
FACILITY_INTERACTION_RADIUS: float = 144.0  # enter/embark range (world units)
BUILD_PROXIMITY: float = 48.0  # engineer build proximity (world units)
FACILITY_EJECT_RADIUS: float = 34.0  # ejection radius when facility destroyed
DEFAULT_DESTROYED_BOAT_DAMAGE: float = 40.0

# ── Guard ───────────────────────────────────────────────────────
GUARD_INTERCEPT_DISTANCE: float = 160.0  # guard intercept range from commander
GUARD_FOLLOW_OFFSET_X: float = 34.0
GUARD_FOLLOW_OFFSET_Y: float = 28.0
GUARD_FOLLOW_RESQ: float = 80.0**2  # squared repositioning threshold

# ── Movement & Formation ────────────────────────────────────────
FORMATION_SPACING: float = 18.0  # formation spacing for move commands
ARRIVAL_DISTANCE: float = 1.0  # movement arrival threshold
RIVER_ROUTE_SAMPLES: int = 48  # samples used to detect a river-crossing order

# ── Collision ───────────────────────────────────────────────────
COLLISION_CELL_SIZE: float = 20.0  # spatial hash grid cell for collision
COLLISION_MAX_SAMPLES: int = 6  # max neighbor samples per unit
COLLISION_OVERLAP_FACTOR: float = 0.78
COLLISION_PUSH_FACTOR: float = 0.35
COLLISION_MAX_PUSH: float = 2.0

# ── Recruit & Golden Angle ─────────────────────────────────────
GOLDEN_ANGLE: float = 2.399963
RECRUIT_BASE_RADIUS: float = 24.0
RECRUIT_GROWTH: float = 5.0
DISEMBARK_BASE_RADIUS: float = 30.0
DISEMBARK_GROWTH: float = 4.0

# ── Army Composition ────────────────────────────────────────────
COMPOSITION_INFANTRY: int = 280
COMPOSITION_SCOUT: int = 60
COMPOSITION_ENGINEER: int = 50
COMPOSITION_ASSASSIN: int = 30
COMPOSITION_TOTAL: int = 495  # sum of above (excludes 1 commander + 4 guards)

# ── UI / Selection ──────────────────────────────────────────────
UNIT_CLICK_RADIUS_SQ: int = 12**2  # squared click radius for unit selection (screen px)
ENEMY_CLICK_RADIUS_SQ: int = 14**2  # squared click radius for enemy targeting (screen px)
