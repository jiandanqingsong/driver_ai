"""Dataset constants for State Farm Distracted Driver Detection."""

STATE_FARM_CLASS_TO_IDX = {
    "c0": 0,
    "c1": 1,
    "c2": 2,
    "c3": 3,
    "c4": 4,
    "c5": 5,
    "c6": 6,
    "c7": 7,
    "c8": 8,
    "c9": 9,
}

STATE_FARM_IDX_TO_CLASS = {idx: cls for cls, idx in STATE_FARM_CLASS_TO_IDX.items()}

STATE_FARM_CLASS_NAMES = [
    "safe_driving",
    "texting_right",
    "talking_phone_right",
    "texting_left",
    "talking_phone_left",
    "operating_radio",
    "drinking",
    "reaching_behind",
    "hair_and_makeup",
    "talking_to_passenger",
]

STATE_FARM_CLASS_NAMES_ZH = [
    "安全驾驶",
    "右手发短信",
    "右手打电话",
    "左手发短信",
    "左手打电话",
    "操作中控设备",
    "喝水",
    "伸手取后排物品",
    "整理头发或化妆",
    "与乘客交谈",
]
