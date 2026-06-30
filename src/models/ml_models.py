"""Machine learning model registry placeholders."""

from typing import Dict


def get_model_registry() -> Dict[str, object]:
    """Return the machine learning model registry.

    The registry is intentionally empty in the first skeleton. Future work can
    add tree-based, linear, or probabilistic models after the planning utility
    framework is stable.

    Keeping this module explicit prevents the repository from becoming a model
    leaderboard before the research question is operationally defined.
    """
    return {}
