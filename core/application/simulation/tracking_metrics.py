from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True, slots=True)
class TrackingMetrics:
    mota: float
    motp: float
    idf1: float
    true_positives: int
    false_positives: int
    false_negatives: int
    identity_switches: int
    total_ground_truth: int


def _iou(first: Sequence[float], second: Sequence[float]) -> float:
    intersection_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union > 0 else 0.0


def evaluate_tracking_sequence(
    ground_truth_frames: Sequence[Sequence[dict[str, object]]],
    prediction_frames: Sequence[Sequence[dict[str, object]]],
    *,
    iou_threshold: float = 0.5,
) -> TrackingMetrics:
    if len(ground_truth_frames) != len(prediction_frames):
        raise ValueError("ground-truth and prediction frame counts must match")
    if not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold must be in (0, 1]")

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    identity_switches = 0
    total_ground_truth = 0
    matched_ious = []
    last_prediction_by_ground_truth: dict[object, object] = {}
    identity_pair_counts: dict[tuple[object, object], int] = {}
    total_predictions = 0

    for truths, predictions in zip(ground_truth_frames, prediction_frames):
        total_ground_truth += len(truths)
        total_predictions += len(predictions)
        if not truths or not predictions:
            false_negatives += len(truths)
            false_positives += len(predictions)
            continue

        costs = np.ones((len(truths), len(predictions)), dtype=np.float64)
        overlaps = np.zeros_like(costs)
        for truth_index, truth in enumerate(truths):
            for prediction_index, prediction in enumerate(predictions):
                if truth.get("class") != prediction.get("class"):
                    continue
                overlap = _iou(truth["bbox"], prediction["bbox"])
                overlaps[truth_index, prediction_index] = overlap
                if overlap >= iou_threshold:
                    costs[truth_index, prediction_index] = 1.0 - overlap

        truth_indices, prediction_indices = linear_sum_assignment(costs)
        matches = [
            (truth_index, prediction_index)
            for truth_index, prediction_index in zip(truth_indices, prediction_indices)
            if overlaps[truth_index, prediction_index] >= iou_threshold
        ]
        true_positives += len(matches)
        false_negatives += len(truths) - len(matches)
        false_positives += len(predictions) - len(matches)

        for truth_index, prediction_index in matches:
            truth = truths[truth_index]
            prediction = predictions[prediction_index]
            truth_id = truth.get("track_id", truth.get("object_id"))
            prediction_id = prediction.get("track_id")
            if truth_id is None or prediction_id is None:
                raise ValueError("truth objects need object_id/track_id and predictions need track_id")
            previous_prediction_id = last_prediction_by_ground_truth.get(truth_id)
            if previous_prediction_id is not None and previous_prediction_id != prediction_id:
                identity_switches += 1
            last_prediction_by_ground_truth[truth_id] = prediction_id
            pair = (truth_id, prediction_id)
            identity_pair_counts[pair] = identity_pair_counts.get(pair, 0) + 1
            matched_ious.append(float(overlaps[truth_index, prediction_index]))

    if identity_pair_counts:
        ground_truth_ids = sorted({pair[0] for pair in identity_pair_counts}, key=str)
        prediction_ids = sorted({pair[1] for pair in identity_pair_counts}, key=str)
        id_costs = np.zeros((len(ground_truth_ids), len(prediction_ids)), dtype=np.float64)
        for truth_index, truth_id in enumerate(ground_truth_ids):
            for prediction_index, prediction_id in enumerate(prediction_ids):
                id_costs[truth_index, prediction_index] = -identity_pair_counts.get(
                    (truth_id, prediction_id), 0
                )
        truth_indices, prediction_indices = linear_sum_assignment(id_costs)
        id_true_positives = sum(
            identity_pair_counts.get(
                (ground_truth_ids[truth_index], prediction_ids[prediction_index]), 0
            )
            for truth_index, prediction_index in zip(truth_indices, prediction_indices)
        )
    else:
        id_true_positives = 0

    id_false_negatives = total_ground_truth - id_true_positives
    id_false_positives = total_predictions - id_true_positives
    id_denominator = 2 * id_true_positives + id_false_positives + id_false_negatives
    mota = (
        1.0 - (false_negatives + false_positives + identity_switches) / total_ground_truth
        if total_ground_truth
        else 1.0
    )
    motp = float(np.mean(matched_ious)) if matched_ious else 0.0
    idf1 = 2 * id_true_positives / id_denominator if id_denominator else 1.0
    return TrackingMetrics(
        mota=mota,
        motp=motp,
        idf1=idf1,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        identity_switches=identity_switches,
        total_ground_truth=total_ground_truth,
    )