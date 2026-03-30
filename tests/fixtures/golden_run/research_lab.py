"""Isolated research surface for the golden fixture."""

from __future__ import annotations

import logging

from research_protocol import DuplicateExperimentError, validate_lab_state_integrity


LOGGER = logging.getLogger(__name__)


# RESEARCH_SURFACE_STATE_START
LAB_STATE = {
    "surface_version": 1,
    "accepted_experiments": [],
    "recent_kept_families": [],
}
# RESEARCH_SURFACE_STATE_END

try:
    validate_lab_state_integrity(LAB_STATE)
except DuplicateExperimentError as exc:
    LOGGER.critical("Invalid golden LAB_STATE: %s", exc)
    raise
