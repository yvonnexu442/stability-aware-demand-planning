"""Simple ensemble weighting helpers for candidate forecasts."""

from typing import Dict, Mapping

from decision_layer.selectors import normalize_weights


def inverse_loss_weights(model_losses: Mapping[str, float], epsilon: float = 1e-8) -> Dict[str, float]:
    """Return normalized inverse-loss ensemble weights.

    Inverse-loss weighting is a transparent soft composition baseline. It gives
    lower-loss models more influence while avoiding hard switches between single
    selected models.

    Later experiments can compare whether smoother weights reduce switching
    burden without hiding planning signal volatility.
    """
    raw_weights = {model_name: 1.0 / max(float(loss), float(epsilon)) for model_name, loss in model_losses.items()}
    return normalize_weights(raw_weights)


def blend_with_previous_weights(
    current_weights: Mapping[str, float],
    previous_weights: Mapping[str, float],
    stability_alpha: float,
) -> Dict[str, float]:
    """Blend current ensemble weights with previous weights.

    Weight blending is a soft stability mechanism. It matters because an
    ensemble can reduce hard model switches but still change too quickly for
    monitoring and governance processes.

    A lower `stability_alpha` places more weight on the previous composition and
    therefore creates smoother model responsibility over time.
    """
    all_models = set(current_weights).union(set(previous_weights))
    blended = {}
    for model_name in all_models:
        current = float(current_weights.get(model_name, 0.0))
        previous = float(previous_weights.get(model_name, 0.0))
        blended[model_name] = float(stability_alpha) * current + (1.0 - float(stability_alpha)) * previous
    return normalize_weights(blended)
