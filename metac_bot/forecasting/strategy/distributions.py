"""Probability distributions for deliberating forecast agents.

Quantile forecasts define a piecewise-linear CDF between their stated points.
Closed tails are linear to the question bounds; open tails are exponential with
scale equal to the nearest interval between stated quantiles. These assumptions
are part of the forecast, not consequences of the quantiles alone.
"""

from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Callable

from forecasting_tools import DateQuestion, NumericQuestion


@dataclass(frozen=True)
class ContinuousDistribution:
    """An evaluable continuous CDF with finite numerical integration brackets."""

    cdf: Callable[[float], float]
    lower: float
    upper: float

    def quantile(self, probability: float) -> float:
        if not 0 < probability < 1:
            raise ValueError("Quantile probability must be strictly between 0 and 1")
        left, right = self.lower, self.upper
        width = right - left
        while self.cdf(left) > probability:
            left -= width
            width *= 2
        width = right - left
        while self.cdf(right) < probability:
            right += width
            width *= 2
        for _ in range(80):
            midpoint = left + (right - left) / 2
            if midpoint == left or midpoint == right:
                break
            if self.cdf(midpoint) < probability:
                left = midpoint
            else:
                right = midpoint
        return left + (right - left) / 2


def distribution_from_percentiles(
    question: NumericQuestion | DateQuestion, prediction: dict
) -> ContinuousDistribution:
    """Construct a full CDF from validated quantiles and explicit tail rules."""
    from ..implementation.domain import build_prediction

    # Also validates the question-specific JSON shape and platform constraints.
    model = build_prediction(question, prediction)
    points = [(float(p.value), float(p.percentile)) for p in model.declared_percentiles]
    if len(points) < 2 or any(not 0 < p < 1 for _, p in points):
        raise ValueError("Distribution quantiles must lie strictly inside (0, 1)")
    if any(not math.isfinite(x) or not math.isfinite(p) for x, p in points):
        raise ValueError("Distribution quantiles must be finite")
    lower = (
        question.lower_bound.timestamp()
        if isinstance(question, DateQuestion)
        else float(question.lower_bound)
    )
    upper = (
        question.upper_bound.timestamp()
        if isinstance(question, DateQuestion)
        else float(question.upper_bound)
    )
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        raise ValueError("Question bounds must be finite and increasing")
    if not question.open_lower_bound:
        if points[0][0] <= lower:
            raise ValueError("First quantile must be above the closed lower bound")
        points.insert(0, (lower, 0.0))
    if not question.open_upper_bound:
        if points[-1][0] >= upper:
            raise ValueError("Last quantile must be below the closed upper bound")
        points.append((upper, 1.0))
    xs = [point[0] for point in points]
    lower_scale = xs[1] - xs[0]
    upper_scale = xs[-1] - xs[-2]

    def cdf(value: float) -> float:
        if value < xs[0]:
            return points[0][1] * math.exp((value - xs[0]) / lower_scale)
        if value >= xs[-1]:
            return 1 - (1 - points[-1][1]) * math.exp((xs[-1] - value) / upper_scale)
        index = bisect_right(xs, value) - 1
        x0, p0 = points[index]
        x1, p1 = points[index + 1]
        return p0 + (p1 - p0) * (value - x0) / (x1 - x0)

    return ContinuousDistribution(cdf, min(lower, xs[0]), max(upper, xs[-1]))


def pool_distributions(
    distributions: list[ContinuousDistribution], weights: list[float]
) -> ContinuousDistribution:
    if not distributions or len(distributions) != len(weights):
        raise ValueError("A weight is required for each distribution")
    if any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError("Weights must be finite and nonnegative")
    total = sum(weights)
    if total <= 0 or not math.isfinite(total):
        raise ValueError("Weights must have positive finite total")
    normalized = [w / total for w in weights]
    return ContinuousDistribution(
        lambda x: sum(w * d.cdf(x) for w, d in zip(normalized, distributions)),
        min(d.lower for d in distributions),
        max(d.upper for d in distributions),
    )


def distribution_prediction(
    question: NumericQuestion | DateQuestion, distribution: ContinuousDistribution
) -> dict:
    """Serialize a pooled CDF as quantiles for the existing platform interface."""
    percentiles = []
    for i in range(1, 100):
        probability = i / 100
        value = distribution.quantile(probability)
        if isinstance(question, DateQuestion):
            value = datetime.fromtimestamp(value, timezone.utc).isoformat()
        percentiles.append({"percentile": probability, "value": value})
    return {"percentiles": percentiles}


def cdf_distance_bounds(
    first: ContinuousDistribution,
    second: ContinuousDistribution,
    *,
    error: float = 0.001,
) -> tuple[float, float]:
    """Bound the sup CDF distance, including unbounded tails, within `error`.

    Monotonicity bounds the difference within each interval. The unexamined
    tails each contain at most error/4 probability in either distribution.
    """
    import heapq

    if not 0 < error < 1:
        raise ValueError("Distance error must lie strictly between 0 and 1")
    tail = error / 4
    left = min(first.quantile(tail), second.quantile(tail))
    right = max(first.quantile(1 - tail), second.quantile(1 - tail))
    best = 0.0
    heap: list[tuple[float, float, float, float, float, float, float]] = []

    def add(a: float, b: float, fa: float, fb: float, ga: float, gb: float) -> None:
        nonlocal best
        best = max(best, abs(fa - ga), abs(fb - gb))
        upper = max(abs(fa - gb), abs(fb - ga))
        heapq.heappush(heap, (-upper, a, b, fa, fb, ga, gb))

    add(
        left,
        right,
        first.cdf(left),
        first.cdf(right),
        second.cdf(left),
        second.cdf(right),
    )
    while heap and max(-heap[0][0], tail) - best > error:
        _, a, b, fa, fb, ga, gb = heapq.heappop(heap)
        middle = a + (b - a) / 2
        if middle == a or middle == b:
            raise ValueError(
                "CDF distance cannot be bounded at floating-point precision"
            )
        fm, gm = first.cdf(middle), second.cdf(middle)
        add(a, middle, fa, fm, ga, gm)
        add(middle, b, fm, fb, gm, gb)
    return best, max(best, tail, -heap[0][0] if heap else 0.0)
