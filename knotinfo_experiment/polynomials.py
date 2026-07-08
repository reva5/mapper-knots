"""Parse KnotInfo Laurent polynomial strings into coefficient vectors."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

PolyKind = Literal["alexander", "jones", "homfly"]

_UNIV_TERM = re.compile(
    r"^(?P<sign>[+-])?"
    r"(?:(?P<coeff>\d+)\*)?"
    r"(?:(?P<var>t|v)(?:\^(?P<exp>\(-?\d+\)|-?\d+))?|(?P<const>\d+))?$"
)
_DIVISION = re.compile(
    r"(?:(?P<coeff>\d+)\*)?/(?P<var>t|v)(?:\^(?P<exp>\(-?\d+\)|-?\d+))?"
)
_Z_TERM = re.compile(
    r"^(?P<sign>[+-])?"
    r"(?:(?P<coeff>\d+)\*)?"
    r"(?:v(?:\^(?P<vexp>\(-?\d+\)|-?\d+))?)?"
    r"\*?"
    r"z(?:\^(?P<zexp>\(-?\d+\)|-?\d+))?$"
)


@dataclass(frozen=True)
class ExponentGrid1D:
    """Shared 1D exponent axis for a polynomial family."""

    exponents: tuple[int, ...]

    @property
    def dim(self) -> int:
        return len(self.exponents)

    def index(self, exp: int) -> int:
        return self.exponents.index(exp)


@dataclass(frozen=True)
class ExponentGrid2D:
    """Shared (v, z) exponent grid for HOMFLYPT."""

    v_exponents: tuple[int, ...]
    z_exponents: tuple[int, ...]

    @property
    def dim(self) -> int:
        return len(self.v_exponents) * len(self.z_exponents)

    def index(self, v_exp: int, z_exp: int) -> int:
        vi = self.v_exponents.index(v_exp)
        zi = self.z_exponents.index(z_exp)
        return vi * len(self.z_exponents) + zi


def _parse_int(raw: str) -> int:
    return int(raw.strip("()"))


def _normalize_poly_text(text: str) -> str:
    s = text.strip()
    if not s or s.lower() in {"?", "unknown", "na", "n/a"}:
        return ""
    s = s.replace(" ", "")
    s = _expand_division_form(s)
    if not s[0] in "+-":
        s = "+" + s
    return s


def _expand_division_form(text: str) -> str:
    """Rewrite KnotInfo ``a/t^b`` division notation as ``a*t^(-b)``."""

    def repl(match: re.Match[str]) -> str:
        coeff = match.group("coeff") or "1"
        var = match.group("var")
        exp_raw = match.group("exp")
        exp = _parse_int(exp_raw) if exp_raw else 1
        return f"{coeff}*{var}^(-{exp})"

    return _DIVISION.sub(repl, text)


def _split_addends(text: str) -> list[str]:
    """Split on top-level +/-; keep signs; respect parentheses."""
    s = text.replace(" ", "")
    if not s:
        return []
    terms: list[str] = []
    i = 0
    n = len(s)
    while i < n:
        sign = "+"
        if s[i] in "+-":
            sign = s[i]
            i += 1
        depth = 0
        start = i
        while i < n:
            ch = s[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch in "+-" and depth == 0 and i > start:
                break
            i += 1
        terms.append(sign + s[start:i])
    return terms


def _parse_univariate_terms(text: str, var: str) -> dict[int, float] | None:
    """Parse Laurent polynomial in one variable (t or v)."""
    s = _normalize_poly_text(text)
    if not s:
        return None
    coeffs: dict[int, float] = {}
    for part in _split_addends(s):
        m = _UNIV_TERM.match(part)
        if not m:
            return None
        sign = -1.0 if m.group("sign") == "-" else 1.0
        const_str = m.group("const")
        if const_str is not None:
            exp = 0
            coeff = float(const_str)
        else:
            coeff_str = m.group("coeff")
            coeff = float(coeff_str) if coeff_str else 1.0
            var_token = m.group("var")
            if var_token is None:
                exp = 0
            else:
                if var_token != var:
                    return None
                exp_raw = m.group("exp")
                exp = _parse_int(exp_raw) if exp_raw else 1
        coeffs[exp] = coeffs.get(exp, 0.0) + sign * coeff
    return coeffs or None


def _trim_near_zero(coeffs: dict[int, float], tol: float = 1e-12) -> dict[int, float]:
    return {e: c for e, c in coeffs.items() if abs(c) > tol}


def _is_symmetric(coeffs: dict[int, float], tol: float = 1e-9) -> bool:
    for exp, val in coeffs.items():
        if abs(coeffs.get(-exp, 0.0) - val) > tol:
            return False
    return True


def symmetrize_alexander(coeffs: dict[int, float]) -> dict[int, float] | None:
    """
    Normalize Alexander polynomial to symmetric form Δ(t)=Δ(1/t).

    KnotInfo gives Alexander up to ±t^k; choose a shift making coefficients
    satisfy a_e = a_{-e}.
    """
    if not coeffs:
        return None
    min_e, max_e = min(coeffs), max(coeffs)
    best: dict[int, float] | None = None
    best_span = None
    for shift in range(min_e - max_e, max_e - min_e + 1):
        shifted = _trim_near_zero({e + shift: c for e, c in coeffs.items()})
        if not shifted or not _is_symmetric(shifted):
            continue
        span = max(shifted) - min(shifted)
        if best is None or span < best_span:
            best = shifted
            best_span = span
    return best


def parse_alexander(text: str) -> dict[int, float] | None:
    raw = _parse_univariate_terms(text, "t")
    if raw is None:
        return None
    return symmetrize_alexander(_trim_near_zero(raw))


def parse_jones(text: str) -> dict[int, float] | None:
    raw = _parse_univariate_terms(text, "t")
    if raw is None:
        return None
    return _trim_near_zero(raw)


def _strip_addend_wrapper(addend: str) -> str:
    """Drop a leading sign and optional outer parentheses."""
    s = addend.strip()
    if s and s[0] in "+-":
        s = s[1:]
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    return s


def _parse_homfly_addend(addend: str) -> dict[tuple[int, int], float] | None:
    addend = addend.strip()
    if not addend:
        return None

    sign = -1.0 if addend.startswith("-") else 1.0
    body = addend[1:] if addend[0] in "+-" else addend

    if "z" not in body:
        inner = _strip_addend_wrapper(body)
        v_coeffs = _parse_univariate_terms(inner, "v")
        if v_coeffs is None:
            return None
        return {(v, 0): sign * c for v, c in v_coeffs.items()}

    z_pos = body.index("z")
    before = body[:z_pos].rstrip("*")
    after = body[z_pos + 1 :]
    if after.startswith("^"):
        if after.startswith("^("):
            close = after.index(")")
            z_exp = _parse_int(after[2:close])
        else:
            z_exp = int(after[1:])
    elif after == "":
        z_exp = 1
    else:
        return None

    if before in {"", "+", "-"}:
        v_sign = -1.0 if before == "-" else 1.0
        return {(0, z_exp): sign * v_sign}

    v_part = _strip_addend_wrapper(before)
    v_coeffs = _parse_univariate_terms(v_part, "v")
    if v_coeffs is None:
        return None
    return {(v_exp, z_exp): sign * coeff for v_exp, coeff in v_coeffs.items()}


def parse_homfly(text: str) -> dict[tuple[int, int], float] | None:
    """Parse KnotInfo HOMFLY column (HOMFLYPT in v, z)."""
    s = _normalize_poly_text(text)
    if not s:
        return None
    merged: dict[tuple[int, int], float] = {}
    for addend in _split_addends(s):
        part = _parse_homfly_addend(addend)
        if part is None:
            return None
        for key, val in part.items():
            merged[key] = merged.get(key, 0.0) + val
    merged = _trim_near_zero({k: v for k, v in merged.items()})
    return merged or None


def vectorize_1d(coeffs: dict[int, float], grid: ExponentGrid1D) -> list[float]:
    return [coeffs.get(exp, 0.0) for exp in grid.exponents]


def vectorize_2d(coeffs: dict[tuple[int, int], float], grid: ExponentGrid2D) -> list[float]:
    out: list[float] = []
    for v_exp in grid.v_exponents:
        for z_exp in grid.z_exponents:
            out.append(coeffs.get((v_exp, z_exp), 0.0))
    return out


def build_grid_1d(all_coeffs: list[dict[int, float]]) -> ExponentGrid1D:
    exps: set[int] = set()
    for coeffs in all_coeffs:
        exps.update(coeffs.keys())
    return ExponentGrid1D(exponents=tuple(sorted(exps)))


def build_grid_2d(all_coeffs: list[dict[tuple[int, int], float]]) -> ExponentGrid2D:
    v_exps: set[int] = set()
    z_exps: set[int] = set()
    for coeffs in all_coeffs:
        for v, z in coeffs:
            v_exps.add(v)
            z_exps.add(z)
    return ExponentGrid2D(
        v_exponents=tuple(sorted(v_exps)),
        z_exponents=tuple(sorted(z_exps)),
    )
