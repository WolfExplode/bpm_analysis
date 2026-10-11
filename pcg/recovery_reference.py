"""Simplified first-five-minute stop-and-sit recovery approximation."""
from math import exp, isfinite


# Adjusted cumulative-drop associations, Hughes/Chaturvedi (2017), Tables S1–S4.
# https://pmc.ncbi.nlm.nih.gov/articles/PMC5439512/
AGE_SEX_COEFFICIENTS = {
    30: (-0.18, 2.15),
    60: (-0.18, 3.98),
    120: (-0.13, 6.04),
    180: (-0.12, 6.74),
}

# Ratios of cohort means, not fitted individual recovery fractions.
# Abu Hanifah et al. (2013), Table 1: doi:10.1371/journal.pone.0082893.
ADOLESCENT_FRACTIONS = {
    "male": {60: 0.55, 120: 0.68},
    "female": {60: 0.43, 120: 0.57},
}
# Castner et al. (2013), normal-weight cohort: doi:10.1155/2013/384167.
# Selecting 8–10 years around its mean age 9.4 is an app heuristic, not a
# published applicability boundary. No obesity classification is inferred.
CHILD_FRACTIONS = {60: 0.64}

# Muro et al. (2003), doi/source https://www.jstage.jst.go.jp/article/hbs/2/1/2_49/_pdf.
# Band centers are chosen interpolation anchors; Medium is an assumed midpoint.
FITNESS_ANCHORS = ((29, 57, 37), (49.5, 85, 46), (69.5, 108, 56))


def _fitness_tau(age, fitness_level):
    """Linear age interpolation of sedentary/active tau; hold end anchors fixed."""
    def at(anchor):
        _, low, high = anchor
        return {"low": low, "medium": (low + high) / 2, "high": high}[fitness_level]

    if age <= FITNESS_ANCHORS[0][0]:
        return at(FITNESS_ANCHORS[0])
    for left, right in zip(FITNESS_ANCHORS, FITNESS_ANCHORS[1:]):
        if age <= right[0]:
            weight = (age - left[0]) / (right[0] - left[0])
            return at(left) + weight * (at(right) - at(left))
    return at(FITNESS_ANCHORS[-1])


def estimate_seated_recovery(start_bpm, resting_hr, seconds, *, age=None, reported_sex=None,
                            fitness_level="unknown"):
    """Proposed heuristic HR, not a population percentile or fitted personal curve.

    HR = R + (S-R)*(0.25 + 0.75*exp(-t/50)), restricted to 0–300 s.
    Rounded assumptions are informed by Barak et al. Table 2, seated inactive
    recovery, doi/source https://www.jssm.org/volume10/iss2/cap/jssm-10-369.pdf.
    Marker time zero is assumed to be exertion cessation; resting HR is seated.

    Optional age/sex checkpoint sensitivity: D = Dbase + a*(age-20)
    + b*(female-1), anchored to a 20-year-old woman per the research document.
    This transfers associations across protocols; it is not a validated curve.
    Adults use 16 < age < 70. Children 8–10 use a pooled-sex 60-s reserve
    fraction; adolescents 12–14 use sex-specific 60/120-s fractions. Pediatric
    HR = S - fraction*(S-R), assuming peak is near cessation for adolescents.
    Other pediatric ages and times return None. No interpolation or
    adjusted 300-s coefficient is supported. Return None for unsupported times
    or drops outside [0, S-R], rather than silently clamping the result.

    Selected adult fitness (18–79 y) replaces the age/sex checkpoint correction
    with the same residual curve using an interpolated activity-group tau.
    Low=sedentary, High=habitually active, Medium=chosen midpoint, not measured
    fitness thresholds. Age 18 uses the youngest adult anchor as an explicit
    one-year extrapolation. Sexes are pooled; pediatric estimates ignore fitness.
    """
    if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not isfinite(v)
           for v in (start_bpm, resting_hr, seconds)):
        raise ValueError("Starting BPM, seated resting HR and time must be finite numbers.")
    if resting_hr <= 0 or start_bpm < resting_hr:
        raise ValueError("Seated resting HR must be positive and no greater than starting BPM.")
    if not 0 <= seconds <= 300:
        raise ValueError("Seated recovery approximation covers the first 300 seconds only.")
    if fitness_level not in ("unknown", "low", "medium", "high"):
        raise ValueError("Choose Unknown, Low, Medium or High fitness level.")
    baseline = resting_hr + (start_bpm - resting_hr) * (0.25 + 0.75 * exp(-seconds / 50))
    if age is None and reported_sex is None and fitness_level == "unknown":
        return baseline
    if not isinstance(age, (int, float)) or isinstance(age, bool) or not isfinite(age):
        raise ValueError("Enter age to calculate the age/sex estimate.")
    if age <= 0:
        raise ValueError("Age/sex estimates require a positive age less than 70.")
    if 8 <= age <= 10:
        fractions = CHILD_FRACTIONS
    elif 12 <= age <= 14:
        if reported_sex not in ADOLESCENT_FRACTIONS:
            raise ValueError("Select Male or Female to calculate the age/sex estimate.")
        fractions = ADOLESCENT_FRACTIONS[reported_sex]
    else:
        fractions = None
    if age <= 16:
        if fractions is None or start_bpm == resting_hr:
            return None
        if seconds == 0:
            return start_bpm
        fraction = fractions.get(seconds)
        return start_bpm - fraction * (start_bpm - resting_hr) if fraction is not None else None
    if fitness_level != "unknown":
        if not 18 <= age <= 79:
            return None
        tau = _fitness_tau(age, fitness_level)
        return resting_hr + (start_bpm - resting_hr) * (0.25 + 0.75 * exp(-seconds / tau))
    if age >= 70:
        raise ValueError("Age/sex estimates require a positive age less than 70.")
    if reported_sex not in ("male", "female"):
        raise ValueError("Select Male or Female to calculate the age/sex estimate.")
    if seconds == 0:
        return start_bpm
    coefficients = AGE_SEX_COEFFICIENTS.get(seconds)
    if coefficients is None:
        return None
    age_coefficient, female_coefficient = coefficients
    drop = (start_bpm - baseline + age_coefficient * (age - 20)
            + female_coefficient * (int(reported_sex == "female") - 1))
    if not 0 <= drop <= start_bpm - resting_hr:
        return None
    return start_bpm - drop
