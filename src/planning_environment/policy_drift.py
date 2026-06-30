"""Policy and context drift helpers for planning scenario analysis."""

from copy import deepcopy
from typing import Any, Dict, Optional


def _copy_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return a defensive copy of a scenario configuration."""
    return deepcopy(config)


def _attach_effective_period(config: Dict[str, Any], effective_period: Optional[Any]) -> Dict[str, Any]:
    """Attach an optional effective period to a scenario configuration."""
    if effective_period is not None:
        config["effective_period"] = effective_period
    return config


def apply_service_level_change(
    planning_config: Dict[str, Any],
    new_service_level_target: float,
    effective_period: Optional[Any] = None,
) -> Dict[str, Any]:
    """Return a scenario with a changed service level target.

    Service-level drift matters when business policy becomes more or less
    aggressive. It can change safety stock and inventory targets even if the
    underlying forecast is unchanged.

    This helps separate forecast instability from policy-driven planning signal
    changes in the paper.
    """
    updated = _copy_config(planning_config)
    updated["service_level_target"] = float(new_service_level_target)
    return _attach_effective_period(updated, effective_period)


def apply_shortage_cost_change(
    planning_config: Dict[str, Any],
    new_shortage_cost_rate: float,
    effective_period: Optional[Any] = None,
) -> Dict[str, Any]:
    """Return a scenario with a changed shortage cost rate.

    Shortage-cost drift represents changes in service priorities, stockout
    penalties, or customer impact. It matters because the same forecast may lead
    to different executable plans under a different cost environment.

    This supports research scenarios where policy and context drift interact
    with forecast-driven instability.
    """
    updated = _copy_config(planning_config)
    updated["shortage_cost_rate"] = float(new_shortage_cost_rate)
    return _attach_effective_period(updated, effective_period)


def apply_lead_time_change(
    planning_config: Dict[str, Any],
    new_lead_time_periods: int,
    effective_period: Optional[Any] = None,
) -> Dict[str, Any]:
    """Return a scenario with a changed operational lead time.

    Lead-time drift changes how quickly plans can be executed. It matters for
    the planning-infrastructure gap because longer lead times reduce the
    operation's ability to absorb rapid forecast changes.

    The function only updates scenario metadata in this first skeleton.
    """
    updated = _copy_config(planning_config)
    updated["lead_time_periods"] = int(new_lead_time_periods)
    return _attach_effective_period(updated, effective_period)


def apply_execution_capacity_change(
    planning_config: Dict[str, Any],
    new_execution_capacity: float,
    effective_period: Optional[Any] = None,
) -> Dict[str, Any]:
    """Return a scenario with a changed execution capacity.

    Execution-capacity drift represents infrastructure upgrades, staffing
    constraints, supplier limits, or system outages. It matters because the same
    planning signal can be executable in one context and infeasible in another.

    This scenario mechanism lets later experiments stress-test stability-aware
    decision policies under changing operational constraints.
    """
    updated = _copy_config(planning_config)
    updated["execution_capacity"] = float(new_execution_capacity)
    return _attach_effective_period(updated, effective_period)
