import pytest

from stratum2.config import GOLIATH_308, DOME_29


def test_goliath_308_sentinels():
    assert len(GOLIATH_308) == 308
    assert GOLIATH_308[41] == "right_wrist"
    assert GOLIATH_308[62] == "left_wrist"
    assert GOLIATH_308[9] == "left_hip"
    assert GOLIATH_308[10] == "right_hip"
    assert GOLIATH_308[69] == "neck"


def test_dome_29_sentinels():
    assert len(DOME_29) == 29
    assert DOME_29[0] == "Background"
    assert DOME_29[3] == "Face_Neck"
    assert DOME_29[22] == "Torso"
