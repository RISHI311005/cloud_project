"""Recommendation logic for food waste predictions."""

from __future__ import annotations

from typing import Tuple


def get_recommendation(
    predicted_leftover_kg: float,
    attendance_ratio: float,
    is_holiday: int,
    special_event: int,
    meal_type: str,
) -> Tuple[str, str]:
    """Return waste risk level and action message based on prediction and context."""
    risk = "Low"
    message = "Waste is expected to be low. Continue normal preparation."

    if predicted_leftover_kg < 5:
        risk = "Low"
        message = "Waste is expected to be low. Continue normal preparation."
    elif predicted_leftover_kg <= 10:
        risk = "Medium"
        message = "Moderate surplus expected. Reduce preparation slightly."
    else:
        risk = "High"
        message = "High surplus likely. Recommend redistribution to NGO/food bank."

    # Context-aware adjustments
    if attendance_ratio < 0.9:
        message += " Attendance is lower than expected, consider reducing portions."
    elif attendance_ratio > 1.1:
        message += " Attendance is higher than expected, monitor demand closely."

    if is_holiday or special_event:
        message += " Demand can be volatile due to holiday/event effects."

    if meal_type == "breakfast":
        message += " Breakfast demand is typically lighter; prep conservatively."
    elif meal_type == "dinner":
        message += " Dinner portions can vary; consider flexible serving sizes."

    if risk == "High":
        message += " Reduce preparation by 10-15% and plan redistribution."
    elif risk == "Medium":
        message += " Reduce preparation by about 5%."

    return risk, message
