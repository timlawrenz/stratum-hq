"""Sapiens2 308-keypoint pose estimation — top-down with DETR person detector.

Produces ``pose2.npy`` — (N, 308, 3) float32 array where each person has
308 keypoints with (x, y, confidence) per keypoint.

Requires a person detector (DETR ResNet-101) for bounding box proposals.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from stratum2.config import POSE2_FILE


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def _get_detector(device: str, det_checkpoint: str):
    """Load DETR person detector (cached)."""
    import torch
    from transformers import DetrForObjectDetection, DetrImageProcessor

    processor = DetrImageProcessor.from_pretrained(det_checkpoint)
    model = DetrForObjectDetection.from_pretrained(det_checkpoint).eval().to(device)
    return processor, model


def _detect_persons(
    image_bgr: np.ndarray,
    processor,
    detector_model,
    device,
    box_threshold: float = 0.3,
    nms_threshold: float = 0.3,
) -> np.ndarray:
    """Detect persons in BGR image. Returns bboxes as (N, 4) array [x1, y1, x2, y2]."""
    import cv2
    import torch

    # The NMS import is from the sapiens pose module
    from sapiens.pose.evaluators import nms

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    from PIL import Image

    pil_img = Image.fromarray(image_rgb)
    inputs = processor(images=pil_img, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = detector_model(**inputs)

    target_sizes = torch.tensor([image_rgb.shape[:2]], device=device)
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=box_threshold
    )[0]

    person_mask = results["labels"] == 1  # COCO person class
    boxes = results["boxes"][person_mask].cpu().numpy()
    scores = results["scores"][person_mask].cpu().numpy().reshape(-1, 1)
    bboxes = np.concatenate([boxes, scores], axis=1)

    if len(bboxes) == 0:
        h, w = image_rgb.shape[:2]
        return np.array([[0, 0, w - 1, h - 1]], dtype=np.float32)

    bboxes = bboxes[nms(bboxes, nms_threshold), :4]  # strip scores
    return bboxes


def process(
    image_path: Path,
    output_dir: Path,
    pose_model,
    device,
    aspect_bucket: str | None = None,
    *,
    det_checkpoint: str | None = None,
) -> bool:
    """Run Sapiens2 308-keypoint pose estimation and save ``pose2.npy``.

    Args:
        det_checkpoint: Path to DETR snapshot directory. Defaults to
            ``SAPIENS2_CACHE_DIR / "detector" / "detr-resnet-101-dc5"``.

    Returns ``True`` on success, ``False`` on failure.
    """
    try:
        import cv2
        import torch

        from stratum2.config import SAPIENS2_CACHE_DIR

        image = cv2.imread(str(image_path))  # BGR
        if image is None:
            eprint(f"warning: cannot read {image_path}")
            return False

        # --- Person detection ---
        if det_checkpoint is None:
            local_det_dir = SAPIENS2_CACHE_DIR / "detector" / "detr-resnet-101-dc5"
            if local_det_dir.exists():
                det_checkpoint = str(local_det_dir)
            else:
                from stratum2.config import POSE_DETECTOR_REPO
                det_checkpoint = POSE_DETECTOR_REPO
        processor, detector_model = _get_detector(device, det_checkpoint)
        bboxes = _detect_persons(image, processor, detector_model, device)

        # --- Pose estimation per person ---
        all_keypoints = []
        for bbox in bboxes:
            data_info = dict(
                img=image,
                bbox=bbox[None],  # shape (1, 4)
                bbox_score=np.ones(1, dtype=np.float32),
            )
            data = pose_model.pipeline(data_info)
            data = pose_model.data_preprocessor(data)
            inputs = data["inputs"].to(device)

            with torch.no_grad():
                pred = pose_model(inputs)  # B × K × heatmap_H × heatmap_W

            pred = pred.cpu().numpy()
            # Decode via UDPHeatmap codec
            ds = data["data_samples"]
            if isinstance(ds, list) and len(ds) > 0: ds = ds[0]
            metainfo = getattr(ds, "metainfo", ds.get("meta", {}) if isinstance(ds, dict) else {})
            keypoints_i, scores_i = pose_model.codec.decode(pred[0])
            # Transform from crop coords to image coords
            input_size = np.asarray(metainfo["input_size"], dtype=np.float32)
            bbox_center = np.asarray(metainfo["bbox_center"], dtype=np.float32)
            bbox_scale = np.asarray(metainfo["bbox_scale"], dtype=np.float32)
            keypoints_i = (
                keypoints_i / input_size * bbox_scale
                + bbox_center
                - 0.5 * bbox_scale
            )
            kp_with_conf = np.concatenate(
                [keypoints_i[0], scores_i[0][:, None]], axis=-1
            )  # (308, 3)
            all_keypoints.append(kp_with_conf)

        result = (
            np.array(all_keypoints, dtype=np.float32)
            if all_keypoints
            else np.zeros((0, 308, 3), dtype=np.float32)
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(output_dir / POSE2_FILE), result)
        return True

    except Exception as exc:
        eprint(f"warning: pose2 failed for {image_path}: {exc}")
        return False
