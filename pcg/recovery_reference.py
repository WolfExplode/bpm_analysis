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


def estimate_seated_recovery(start_bpm, resting_hr, seconds, *, age=None, reported_sex=None):
    """Proposed heuristic HR, not a population percentile or fitted personal curve.

    HR = R + (S-R)*(0.25 + 0.75*exp(-t/50)), restricted to 0–300 s.
    Rounded assumptions are informed by Barak et al. Table 2, seated inactive
    recovery, doi/source https://www.jssm.org/volume10/iss2/cap/jssm-10-369.pdf.
    Marker time zero is assumed to be exertion cessation; resting HR is seated.

    Optional age/sex checkpoint sensitivity: D = Dbase + a*(age-20)
    + b*(female-1), anchored to a 20-year-old woman per the research document.
    This transfers associations across protocols; it is not a validated curve.
    Both demographics are required, with 16 < age < 70. No interpolation or
    adjusted 300-s coefficient is supported. Return None for unsupported times
    or drops outside [0, S-R], rather than silently clamping the result.
    """
    if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not isfinite(v)
           for v in (start_bpm, resting_hr, seconds)):
        raise ValueError("Starting BPM, seated resting HR and time must be finite numbers.")
    if resting_hr <= 0 or start_bpm < resting_hr:
        raise ValueError("Seated resting HR must be positive and no greater than starting BPM.")
    if not 0 <= seconds <= 300:
        raise ValueError("Seated recovery approximation covers the first 300 seconds only.")
    baseline = resting_hr + (start_bpm - resting_hr) * (0.25 + 0.75 * exp(-seconds / 50))
    if age is None and reported_sex is None:
        return baseline
    if not isinstance(age, (int, float)) or isinstance(age, bool) or not isfinite(age):
        raise ValueError("Enter age to calculate the age/sex estimate.")
    if not 16 < age < 70:
        raise ValueError("Age/sex estimates cover ages greater than 16 and less than 70 only.")
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
