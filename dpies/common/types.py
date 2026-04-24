from enum import IntEnum


class ActionMode(IntEnum):
    KEEP = 0
    STOP = 1
    PROCEED = 2
    LANE_CHANGE_LEFT = 3
    LANE_CHANGE_RIGHT = 4
    MERGE = 5
    NUDGE_LEFT = 6
    NUDGE_RIGHT = 7
    CREEP = 8


class EvidenceType(IntEnum):
    DYNAMIC_AGENT = 0
    CONFLICT_POINT = 1
    GAP = 2
    MAP_RULE = 3
    LOW_TTC_RISK = 4
    PADDING = 5


ACTION_STATE_DIM = 6
ACTION_META_DIM = 8
EGO_DIM = 8
AGENT_DIM = 8
MAP_DIM = 4
EVIDENCE_DIM = 32
QUERY_DIM = 24
