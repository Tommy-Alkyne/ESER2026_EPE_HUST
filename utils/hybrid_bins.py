DEFAULT_CLS_THRESHOLDS = (0.05, 0.3, 0.55, 0.68)


def validate_cls_thresholds(thresholds):
    values = tuple(float(value) for value in thresholds)
    if not values:
        raise ValueError("At least one classification threshold is required")
    if any(value <= 0.0 or value >= 1.0 for value in values):
        raise ValueError("Classification thresholds must lie strictly inside (0, 1)")
    if any(left >= right for left, right in zip(values, values[1:])):
        raise ValueError("Classification thresholds must be strictly increasing")
    return values


def bin_center_ratios_from_thresholds(thresholds):
    """Return interval midpoints for [0, *thresholds, 1]."""
    values = validate_cls_thresholds(thresholds)
    edges = (0.0,) + values + (1.0,)
    return tuple(0.5 * (left + right) for left, right in zip(edges, edges[1:]))
