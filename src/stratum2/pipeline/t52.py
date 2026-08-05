"""T5 text encoding pipeline for caption2.

Produces ``t52_hidden.npy`` and ``t52_mask.npy``.
"""

import sys
from pathlib import Path
import numpy as np


def eprint(*args: object, **kwargs: object) -> None:
    print(*args, file=sys.stderr, **kwargs)


def process(
    image_path: Path, output_dir: Path, tokenizer=None, encoder=None, **kwargs
) -> bool:
    hidden_path = output_dir / "t52_hidden.npy"
    mask_path = output_dir / "t52_mask.npy"

    if hidden_path.exists() and mask_path.exists():
        return True

    caption_path = output_dir / "caption2.txt"
    if not caption_path.exists():
        eprint(f"warning: t52 skipped for {image_path}: caption2.txt not found")
        return False

    caption = caption_path.read_text().strip()

    # Delegate the actual math to stratum1's t5 module (we share the encoder and tokenizer)
    from stratum.pipeline.t5 import compute_t5_hidden_states

    res = compute_t5_hidden_states(caption, tokenizer, encoder)
    if res is None:
        eprint(f"warning: t52 compute failed for {image_path}")
        return False

    # We must also get the mask. Let's see how stratum1 does it...
    # Wait, compute_t5_hidden_states returns just the hidden states.
    # We'll just run the tokenizer here to get the mask.
    inputs = tokenizer(
        caption,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )

    hidden = res
    mask = inputs["attention_mask"][0].astype(np.uint8)

    np.save(str(hidden_path), hidden)
    np.save(str(mask_path), mask)

    return True
