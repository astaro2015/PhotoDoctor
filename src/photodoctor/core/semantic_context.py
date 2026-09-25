from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class SemanticContext:
    classification: str
    people_context: str
    face_count: int
    archival_likelihood: float
    preservation_priority: str
    confidence: float
    reasons: list[str]

    def to_raw(self) -> dict[str, object]:
        return {
            "classification": self.classification,
            "people_context": self.people_context,
            "face_count": self.face_count,
            "archival_likelihood": self.archival_likelihood,
            "preservation_priority": self.preservation_priority,
            "confidence": self.confidence,
            "reasons": list(self.reasons),
            "method": "semantic_context_rules_v1",
        }


def analyze_semantic_context(
    *,
    face_count: int,
    face_detector_confidence: float,
    tone_class: str,
    tone_confidence: float,
    surface_candidate_count: int,
    surface_confidence: float,
) -> SemanticContext:
    face_count = max(0, int(face_count))
    people_context = "none" if face_count == 0 else ("single" if face_count == 1 else "group")
    reasons: list[str] = []

    archival = 0.0
    if tone_class == "sepia":
        archival += 0.58 * float(np.clip(tone_confidence, 0.0, 1.0))
        reasons.append("sepia_tone")
    elif tone_class == "monochrome":
        archival += 0.24 * float(np.clip(tone_confidence, 0.0, 1.0))
        reasons.append("monochrome_tone")

    if surface_candidate_count >= 8:
        surface_factor = float(np.clip((surface_candidate_count - 6) / 30.0, 0.0, 1.0))
        archival += 0.24 * max(surface_factor, float(np.clip(surface_confidence, 0.0, 1.0)) * 0.45)
        reasons.append("surface_ageing_candidates")

    archival = float(np.clip(archival, 0.0, 0.92))

    if face_count > 0 and archival >= 0.50:
        classification = "archival_portrait"
        preservation_priority = "faces_and_original_character"
        confidence = float(np.clip(0.42 + 0.28 * archival + 0.22 * face_detector_confidence, 0.0, 0.86))
    elif face_count >= 2:
        classification = "group_portrait"
        preservation_priority = "faces"
        confidence = float(np.clip(0.48 + 0.34 * face_detector_confidence, 0.0, 0.82))
        reasons.append("multiple_faces")
    elif face_count == 1:
        classification = "portrait"
        preservation_priority = "face"
        confidence = float(np.clip(0.50 + 0.32 * face_detector_confidence, 0.0, 0.82))
        reasons.append("single_face")
    else:
        classification = "general_photo"
        preservation_priority = "global"
        # A face detector can miss small/profile faces, so no-face semantic confidence is intentionally limited.
        confidence = float(np.clip(0.38 + 0.16 * face_detector_confidence, 0.0, 0.58))
        reasons.append("no_confident_faces")

    if face_count > 0:
        reasons.append("faces_detected")
    if archival >= 0.50:
        reasons.append("archival_context")

    return SemanticContext(
        classification=classification,
        people_context=people_context,
        face_count=face_count,
        archival_likelihood=archival,
        preservation_priority=preservation_priority,
        confidence=confidence,
        reasons=reasons,
    )
