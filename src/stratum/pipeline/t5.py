"""T5 text encoding pipeline — hidden states and attention masks for captions."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from stratum.config import CAPTION_FILE, T5_HIDDEN_FILE, T5_MASK_FILE, T5_MODEL_ID


def eprint(*args: object, **kwargs: object) -> None:
    print(*args, file=sys.stderr, **kwargs)


def load_t5_tokenizer(model_id: str | None = None):
    """Load T5 tokenizer.

    Uses :data:`~stratum.config.T5_MODEL_ID` when *model_id* is ``None``.
    """
    from transformers import AutoTokenizer

    if model_id is None:
        model_id = T5_MODEL_ID
    return AutoTokenizer.from_pretrained(model_id)


def load_t5_encoder(model_id: str | None = None):
    """Load T5 encoder model in float16.

    Uses :data:`~stratum.config.T5_MODEL_ID` when *model_id* is ``None``.
    """
    from transformers import T5EncoderModel
    import torch

    if model_id is None:
        model_id = T5_MODEL_ID

    model = T5EncoderModel.from_pretrained(model_id, torch_dtype=torch.float16)
    model.eval()
    return model


def compute_t5_hidden_states(
    caption: str,
    tokenizer,
    encoder,
) -> np.ndarray | None:
    """Encode *caption* to T5 hidden states.

    Returns a ``(512, 1024)`` float16 array, or ``None`` on error.
    """
    try:
        import torch

        inputs = tokenizer(
            caption,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        device = next(encoder.parameters()).device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state

        result = hidden_states.squeeze(0).cpu().numpy().astype(np.float16)

        if result.shape != (512, 1024):
            eprint(
                f"warning: T5 hidden state shape mismatch: "
                f"got {result.shape}, expected (512, 1024)"
            )
            return None

        return result

    except Exception as e:
        eprint(f"warning: T5 encoding failed: {e}")
        return None


def compute_t5_attention_mask(tokenizer, caption: str) -> list[int]:
    """Tokenize *caption* and return a 512-length attention mask.

    Each element is ``1`` for valid tokens and ``0`` for padding.
    """
    tokens = tokenizer(
        caption,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    mask: list[int] = tokens["attention_mask"][0].tolist()

    assert len(mask) == 512, f"Expected mask length 512, got {len(mask)}"
    assert all(v in (0, 1) for v in mask), "Mask must contain only 0s and 1s"

    return mask


def process(
    image_dir: Path,
    tokenizer,
    encoder,
    device,
) -> bool:
    """Read caption, compute T5 hidden states + mask, and save to *image_dir*.

    Reads :data:`~stratum.config.CAPTION_FILE` from *image_dir*, then writes
    :data:`~stratum.config.T5_HIDDEN_FILE` and :data:`~stratum.config.T5_MASK_FILE`.

    Returns ``True`` on success, ``False`` on failure.
    """
    caption_path = image_dir / CAPTION_FILE
    if not caption_path.exists():
        eprint(f"warning: caption file not found: {caption_path}")
        return False

    try:
        caption = caption_path.read_text(encoding="utf-8").strip()
    except Exception as e:
        eprint(f"warning: failed to read caption {caption_path}: {e}")
        return False

    if not caption:
        eprint(f"warning: empty caption in {caption_path}")
        return False

    hidden = compute_t5_hidden_states(caption, tokenizer, encoder)
    if hidden is None:
        return False

    mask = compute_t5_attention_mask(tokenizer, caption)

    image_dir.mkdir(parents=True, exist_ok=True)
    np.save(image_dir / T5_HIDDEN_FILE, hidden)
    np.save(image_dir / T5_MASK_FILE, np.asarray(mask, dtype=np.int32))

    return True
