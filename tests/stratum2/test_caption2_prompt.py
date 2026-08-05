from stratum2.pipeline.caption2 import build_prompt


def test_caption2_prompt_builder():
    mock_determinations = {
        "schema_version": 2,
        "subject": {"n_detections": 1, "detector_anomaly": "none"},
        "subject_extent": {"h_position": "center"},
        "body_parts_visible": [
            {"part": "face", "pixel_frac": 0.05, "kp_conf": 0.9},
            {"part": "torso", "pixel_frac": 0.20, "kp_conf": 0.9},
        ],
        "orientation": {"upright_deg": 178.5},
        "relations": ["left arm extended upward", "hands together"],
    }

    prompt = build_prompt(mock_determinations)

    # Assert determinations block is present
    assert "exactly one primary subject detected" in prompt
    assert "subject horizontal position: center" in prompt
    assert "visible body regions: face, torso" in prompt
    assert "torso upright angle: 178.5 degrees" in prompt

    # Assert relations appear as bullets
    assert "* left arm extended upward" in prompt
    assert "* hands together" in prompt

    # Assert instructions
    assert "NEVER contradict them" in prompt
    assert "ADD what the determinations omit" in prompt


def test_caption2_prompt_omits_detector_disagreement_from_caption_content():
    prompt = build_prompt(
        {
            "schema_version": 2,
            "subject": {"n_detections": 2, "detector_anomaly": "extra_detections(2)"},
        }
    )

    assert "detector anomaly" not in prompt.lower()
    assert "extra_detections" not in prompt
    assert "exactly one primary subject detected" not in prompt


def test_stratum1_caption_prompt_untouched():
    from stratum.config import CAPTION_PROMPT

    # Assert it hasn't been modified to match the new one
    assert "DETERMINATIONS" not in CAPTION_PROMPT
    assert "Generate a single, dense paragraph" in CAPTION_PROMPT
    assert "objective, and descriptive tone" in CAPTION_PROMPT
