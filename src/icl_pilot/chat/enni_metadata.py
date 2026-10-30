"""Literal ENNI participant metadata; folders never supply a diagnosis or age."""
from __future__ import annotations

import re


_AGE = re.compile(r"(\d+);(\d{1,2})(?:\.(\d{0,2}))?")
_CHI = re.compile(r"^@ID:[ \t]*[^|\n]*\|[^|\n]*\|CHI\|([^|\n]*)\|([^|\n]*)\|([^|\n]*)\|", re.MULTILINE)


def completed_age_months(value: str) -> int:
    """Use completed CHAT years/months. Days never round a child into a new year."""
    match = _AGE.fullmatch(str(value).strip())
    if not match:
        raise ValueError("Unknown or malformed CHAT age: {!r}".format(value))
    years, months, days = match.groups()
    if int(months) >= 12 or (days and int(days) > 31):
        raise ValueError("Invalid CHAT age: {!r}".format(value))
    return 12 * int(years) + int(months)


def approximate_age_months(value: str) -> float:
    """Days / 30 breaks age-matching ties only; cohorts use completed months."""
    months = completed_age_months(value)
    days = _AGE.fullmatch(str(value).strip()).group(3)
    return months + int(days or 0) / 30.0


def read_chi_metadata(text: str) -> dict:
    values = {tuple(s.strip() for s in match.groups()) for match in _CHI.finditer(text.lstrip("\ufeff"))}
    if len(values) > 1:
        raise ValueError("Conflicting CHI @ID headers; participant identity is ambiguous.")
    age, sex, group = next(iter(values)) if values else ("", "", "")
    issues = []
    try:
        months = completed_age_months(age)
    except ValueError:
        months = None
        issues.append("missing_or_invalid_chi_age")
    if group not in {"TD", "SLI"}:
        issues.append("missing_or_invalid_chi_group")
    return {"Age_Raw": age, "Age_Month": months, "Sex": sex, "Group": group, "issues": issues}
