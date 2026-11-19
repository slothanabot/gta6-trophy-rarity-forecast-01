"""GTA VI trophy earn-rate forecaster (make.fun gta6-trophy-rarity-forecast-01).

Model: a saturating growth curve per trophy.  Each trophy's earn rate is
modelled as a logistic curve in saturating time since launch, rising
toward a tier-specific ceiling:

    rate(t) = CEIL[tier] * sigmoid(a + b * u(t)),  u(t) = h / (h + 48h)

Story trophies climb fast early and level off; rare and collectible ones
climb slowly and top out low.  The slope b is estimated from the trophy's
own history where the history actually moves, and otherwise shrunk toward
its tier's pooled slope (platinum/gold/silver/bronze grow at very different
speeds).  The intercept a is anchored on the trophy's last observed rate,
so a flat launch-night history still yields a sensible curve.

Why this beats a flat forecast: Exophase refreshes its counts on its own
schedule, so a read shows rates as of the last refresh, and the tracked
player base grows enormously in week one.  Launch night proved it: the
04:30 UTC refresh cut every rate but the 100% one as tracked players jumped
638 -> 3,868.  A forecast that just repeats the last read with a narrow
band is wrong the moment a refresh lands, which is why the house baseline
scores ~1.51 on the practice.  This entry therefore forecasts real growth
toward each tier's ceiling and carries an honestly wide spread.

Spread: quantiles come from a logit-scale sigma combining (a) a base
dispersion floor that prices in unseen refreshes, (b) the trophy's own
residual scatter around its fitted curve, and (c) a horizon term that
grows with sqrt(days to target), so the 4.5-day final forecast is far
wider than a 90-minute one.  The lower tail is stretched a little further
than the upper one: dilution from new tracked players can always drag a
rate down, while growth toward the ceiling caps the upside.

Deterministic, standard library only, no network.  Reads every *.in in
$MAKEFUN_INPUT_DIR (else /input/) and writes one <id>.out per case to
$MAKEFUN_OUTPUT_DIR (else /output/).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
NORM_Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
          0.6744897501960817, 1.6448536269514722]

# Tier priors.  CEIL is the saturating earn rate (percent) a trophy of
# that tier approaches; SLOPE is the prior growth rate in logit units per
# unit of the saturating time feature u(t) = h / (h + HALF_LIFE_H); TYPICAL
# is the fallback last-rate used when a trophy has no usable read at all.
TIER = {
    "platinum": {"ceil": 2.0, "slope": 0.5, "typical": 0.6},
    "gold": {"ceil": 10.0, "slope": 1.2, "typical": 0.8},
    "silver": {"ceil": 30.0, "slope": 1.8, "typical": 0.7},
    "bronze": {"ceil": 85.0, "slope": 2.2, "typical": 0.8},
}
DEFAULT_TIER = {"ceil": 40.0, "slope": 1.8, "typical": 0.8}

# Time feature: u(t) = hours / (hours + HALF_LIFE_H).  Growth is fast in
# the first day or two after launch and then saturates -- a trophy does
# not keep climbing on a log-time straight line for a whole week, or a
# 4.5-day forecast would explode to absurd rates.
HALF_LIFE_H = 48.0

# Spread (logit units).  BASE prices in unseen Exophase refreshes; HORIZON
# scales with sqrt(days to target); LO/HI stretch the tails asymmetrically.
BASE_SIGMA = 0.70
HORIZON_SIGMA = 0.55
# Short-horizon refresh dilution: launch night showed one refresh cutting
# every rate as tracked players jumped 638 -> 3,868.  Over a 90-minute
# horizon the next refresh can only dilute, while over 4.5 days real
# earning dominates, so this term decays with the horizon in days.
DILUTION_LOGIT = 0.35
DILUTION_DECAY_DAYS = 0.75
LOW_STRETCH = 1.15
HIGH_STRETCH = 1.00
# Pseudo-observations anchoring a trophy's fitted slope to its tier prior.
SHRINK_K = 4.0
MIN_PCT = 0.01
MAX_PCT = 99.99


def _clamp_pct(value):
    if value != value or value in (math.inf, -math.inf):
        return None
    return min(MAX_PCT, max(MIN_PCT, float(value)))


def _logit(pct):
    p = _clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def _sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _round4(value):
    """Round to 4 decimals, half away from zero for positive values."""
    scaled = value * 10000.0
    whole = math.floor(scaled + 0.5)
    out = whole / 10000.0
    return int(out) if float(out).is_integer() else out


def _parse_time(text):
    if not text:
        return None
    try:
        moment = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def _launch_moment(case):
    game = case.get("game") or {}
    release = _parse_time(game.get("release_date"))
    if release is not None:
        return release
    reads = case.get("history") or []
    if reads:
        first = _parse_time(reads[0].get("at"))
        if first is not None:
            return first
    return _parse_time(case.get("at"))


def _time_u(moment, launch):
    """Saturating time feature: 0 at launch, -> 1 as hours grow large."""
    if moment is None or launch is None:
        return 0.0
    hours = max(0.0, (moment - launch).total_seconds() / 3600.0)
    return hours / (hours + HALF_LIFE_H)


def _median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _fit_line(points):
    """Least-squares y = a + b*x over [(x, y), ...]; None if degenerate."""
    n = len(points)
    if n < 2:
        return None
    mean_x = sum(p[0] for p in points) / n
    mean_y = sum(p[1] for p in points) / n
    sxx = sum((p[0] - mean_x) ** 2 for p in points)
    if sxx < 1e-12:
        return None
    sxy = sum((p[0] - mean_x) * (p[1] - mean_y) for p in points)
    slope = sxy / sxx
    return (mean_y - slope * mean_x, slope)


def _tier_curve_slopes(series_by_trophy, prior):
    """Median fitted slope per tier, falling back to the tier prior.

    Only series that actually moved count: a flat launch-night history
    fits a slope of exactly 0 for every trophy, which is evidence of no
    movement yet, not evidence that nothing will ever grow.  When no
    trophy in the tier has moved, the tier prior is the slope.
    """
    slopes = []
    for series in series_by_trophy:
        if not series:
            continue
        ys = [p[1] for p in series]
        if max(ys) - min(ys) < 0.1:
            continue
        fit = _fit_line(series)
        if fit is not None and len({round(p[0], 6) for p in series}) >= 3:
            slopes.append(fit[1])
    if not slopes:
        return prior
    pooled = _median(slopes)
    if pooled != pooled or pooled in (math.inf, -math.inf):
        return prior
    return max(-0.5, min(2.5, pooled))


def _forecast_trophy(series, last_rate, tier, x_target, days, tier_slope):
    """Forecast one trophy's five quantiles from its transformed history.

    series: [(u(t), logit(rate / ceil)), ...] in time order.
    Returns [q05, q25, q50, q75, q95] in percent.
    """
    prior = TIER.get(tier, DEFAULT_TIER)
    ceil = prior["ceil"]
    last = _clamp_pct(last_rate)
    if last is None:
        last = prior["typical"]
    # A trophy already earned by (almost) everyone stays there.
    if last >= 99.5:
        return [99.5, 99.8, 99.95, 100.0, 100.0]

    fit = _fit_line(series) if len(series) >= 2 else None
    distinct_x = len({round(p[0], 6) for p in series})
    # A series that never moved carries no slope information at all: its
    # least-squares slope is 0 by construction, so do not let it drag the
    # estimate toward zero -- use the tier slope instead.
    moved = bool(series) and (max(p[1] for p in series)
                              - min(p[1] for p in series) >= 0.1)
    if fit is not None and distinct_x >= 3 and moved:
        _, slope_hat = fit
        weight = distinct_x / (distinct_x + SHRINK_K)
        slope = weight * slope_hat + (1.0 - weight) * tier_slope
        slope = max(-0.5, min(4.0, slope))
        # Intercept re-anchored on the curve, then pulled to the last read.
        intercept = _median([y - slope * x for x, y in series])
        resid = [y - (intercept + slope * x) for x, y in series]
        scatter = _median([abs(r) for r in resid]) * 1.4826
    else:
        slope = tier_slope
        if series:
            intercept = _median([y - slope * x for x, y in series])
        else:
            intercept = _logit(last / ceil * 100.0) - slope * 0.0
        scatter = 0.0
    # Anchor the curve exactly on the last read: the freshest fact wins.
    if series:
        x_last, y_last = series[-1]
        intercept = y_last - slope * x_last
    else:
        x_last = 0.0

    centre = (intercept + slope * x_target
              - DILUTION_LOGIT * math.exp(-days / DILUTION_DECAY_DAYS))
    median = ceil * _sigmoid(centre)
    median = _clamp_pct(median)
    if median is None:
        median = last

    sigma = math.sqrt(BASE_SIGMA ** 2
                      + (HORIZON_SIGMA * math.sqrt(max(0.0, days))) ** 2
                      + min(scatter, 1.5) ** 2)
    med_logit = _logit(median)
    quantiles = []
    for z in NORM_Z:
        stretch = LOW_STRETCH if z < 0 else HIGH_STRETCH
        value = 100.0 * _sigmoid(med_logit + z * sigma * stretch)
        quantiles.append(_round4(_clamp_pct(value)))
    # Keep the row monotone after rounding.
    for i in range(1, len(quantiles)):
        if quantiles[i] < quantiles[i - 1]:
            quantiles[i] = quantiles[i - 1]
    return quantiles


def _forecast_case(case):
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    launch = _launch_moment(case)
    checkpoint = _parse_time(case.get("at"))
    horizon = case.get("horizon_minutes")
    if horizon is None and checkpoint is not None:
        target = _parse_time(case.get("target"))
        horizon = (target - checkpoint).total_seconds() / 60.0 if target else 0.0
    days = max(0.0, float(horizon or 0.0) / 1440.0)
    target_moment = _parse_time(case.get("target"))
    if target_moment is None and checkpoint is not None:
        from datetime import timedelta
        target_moment = checkpoint + timedelta(minutes=float(horizon or 0.0))
    x_target = _time_u(target_moment, launch)

    # Transformed history per trophy, plus each trophy's last read rate.
    series_by_trophy = []
    last_rates = []
    for idx, trophy in enumerate(trophies):
        prior = TIER.get(trophy.get("tier"), DEFAULT_TIER)
        ceil = prior["ceil"]
        series = []
        last_rate = None
        for read in history:
            rates = read.get("rates") or []
            if idx >= len(rates):
                continue
            rate = rates[idx]
            if rate is None:
                continue
            rate = _clamp_pct(rate)
            if rate is None:
                continue
            last_rate = rate
            # logit(rate / ceil): the curve's own scale, capped so a rate
            # at/above the ceiling does not blow up the transform.
            frac = min(0.999, max(0.001, rate / ceil))
            series.append((_time_u(_parse_time(read.get("at")), launch),
                           math.log(frac / (1.0 - frac))))
        series_by_trophy.append(series)
        last_rates.append(last_rate)

    # Tier pooled slopes, for shrinkage of trophies with thin histories.
    by_tier = {}
    for tier in {t.get("tier") for t in trophies}:
        group = [s for t, s in zip(trophies, series_by_trophy)
                 if t.get("tier") == tier and len(s) >= 2]
        prior = TIER.get(tier, DEFAULT_TIER)["slope"]
        by_tier[tier] = _tier_curve_slopes(group, prior)

    rows = []
    for trophy, series, last_rate in zip(trophies, series_by_trophy,
                                         last_rates):
        tier = trophy.get("tier")
        quantiles = _forecast_trophy(series, last_rate, tier, x_target,
                                     days, by_tier.get(tier, TIER.get(
                                         tier, DEFAULT_TIER)["slope"]))
        rows.append({"id": trophy["id"], "q": quantiles})
    return {"trophies": rows}


def _fallback_case(case):
    """Last-read rates with a wide spread: never leave a case unwritten."""
    rows = []
    history = case.get("history") or []
    last = history[-1].get("rates") if history else []
    for idx, trophy in enumerate(case.get("trophies") or []):
        rate = None
        if idx < len(last or []):
            rate = _clamp_pct(last[idx])
        if rate is None:
            rate = TIER.get(trophy.get("tier"), DEFAULT_TIER)["typical"]
        centre = _logit(rate)
        quantiles = []
        for z in NORM_Z:
            quantiles.append(_round4(_clamp_pct(
                100.0 * _sigmoid(centre + z * 1.5))))
        for i in range(1, len(quantiles)):
            if quantiles[i] < quantiles[i - 1]:
                quantiles[i] = quantiles[i - 1]
        rows.append({"id": trophy["id"], "q": quantiles})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            forecast = _forecast_case(case)
        except Exception:
            try:
                case = json.loads(path.read_text(encoding="utf-8"))
                forecast = _fallback_case(case)
            except Exception:
                continue
        out_path = OUTPUT_DIR / (path.stem + ".out")
        out_path.write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
