"""Feedback collection and storage."""

import json
from datetime import datetime
from pathlib import Path


FEEDBACK_DIR = Path(__file__).parent.parent / "feedback"
FEEDBACK_FILE = FEEDBACK_DIR / "feedback.jsonl"


def save_feedback(
    description: str,
    generated_code: str,
    rating: int,
    code_quality: list[str],
    visual_quality: list[str],
    guide_suggestions: str,
    additional_notes: str = "",
) -> None:
    """
    Save feedback to JSONL file.

    Args:
        description: The visualization description used
        generated_code: The code that was generated
        rating: Overall rating 1-5
        code_quality: List of checked items (correct, readable, follows_guide)
        visual_quality: List of checked items (clear, appropriate_type, good_colors)
        guide_suggestions: Free-text suggestions for guide improvements
        additional_notes: Any other notes
    """
    FEEDBACK_DIR.mkdir(exist_ok=True)

    feedback_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "description": description,
        "generated_code": generated_code,
        "rating": rating,
        "code_quality": code_quality,
        "visual_quality": visual_quality,
        "guide_suggestions": guide_suggestions,
        "additional_notes": additional_notes,
    }

    with open(FEEDBACK_FILE, "a") as f:
        f.write(json.dumps(feedback_entry) + "\n")


def load_feedback() -> list[dict]:
    """Load all feedback entries."""
    if not FEEDBACK_FILE.exists():
        return []

    entries = []
    with open(FEEDBACK_FILE) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def get_feedback_summary() -> dict:
    """Get summary statistics of feedback."""
    entries = load_feedback()
    if not entries:
        return {"count": 0}

    ratings = [e["rating"] for e in entries]
    return {
        "count": len(entries),
        "avg_rating": sum(ratings) / len(ratings),
        "min_rating": min(ratings),
        "max_rating": max(ratings),
    }
