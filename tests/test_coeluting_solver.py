"""Unit tests for the joint coeluting multi-charge solver."""

from decimal import Decimal

import pytest

from app.solver import (
    VERDICT_AMBIGUOUS,
    VERDICT_UNIQUE,
    VERDICT_UNRESOLVED,
    CoelutingSolver,
    Peak,
    SearchSpaceExceededError,
)

S = Decimal("1.003355")


def make_peaks(spec: list[tuple[str, int]]) -> list[Peak]:
    return [Peak(index=i, mz=Decimal(mz), intensity=it) for i, (mz, it) in enumerate(spec)]


def pair_around(mass: str, charge: int, intensities=(100, 100)) -> list[tuple[str, int]]:
    m0 = Decimal(mass) / charge
    return [
        (str(m0), intensities[0]),
        (str(m0 + S / charge), intensities[1]),
    ]


def solve(spec, required, tol, mass_tol, allowed=None):
    peaks = make_peaks(spec)
    return CoelutingSolver(
        peaks,
        allowed_charges=allowed if allowed is not None else required,
        required_charges=required,
        tolerance=Decimal(tol),
        mass_tolerance=Decimal(mass_tol),
    ).solve()


def merge_specs(*specs):
    merged: dict[str, int] = {}
    for spec in specs:
        for mz, it in spec:
            merged[mz] = merged.get(mz, 0) + it
    return [(mz, merged[mz]) for mz in sorted(merged, key=Decimal)]


def test_two_charges_same_neutral_mass_is_unique():
    spec = merge_specs(pair_around("1000", 1), pair_around("1000", 2))
    result = solve(spec, [1, 2], "0.0001", "0.001")
    assert result.verdict == VERDICT_UNIQUE
    assert result.cluster_count == 2
    assert result.explained_peak_count == 4
    assert result.explained_intensity == 400
    assert {c.charge for c in result.primary} == {1, 2}
    assert result.mass_lower == Decimal("999.999")
    assert result.mass_upper == Decimal("1000.001")


def test_common_mass_boundary_is_inclusive():
    # z=1 first peak gives M=1000.0010; z=2 first peak gives M=1000.0000.
    # Intervals touch exactly at 1000.0005 when mass_tolerance == 0.0005.
    spec = merge_specs(
        pair_around("1000.0010", 1),
        [("500.0000", 100), ("500.5016775", 100)],
    )
    result = solve(spec, [1, 2], "0.0001", "0.0005")
    assert result.verdict == VERDICT_UNIQUE
    assert result.mass_lower == result.mass_upper == Decimal("1000.0005")
    miss = solve(spec, [1, 2], "0.0001", "0.000499")
    assert miss.verdict == VERDICT_UNRESOLVED


def test_unresolved_when_intervals_disjoint():
    spec = merge_specs(pair_around("1000", 1), pair_around("1010", 2))
    result = solve(spec, [1, 2], "0.0001", "0.5")
    assert result.verdict == VERDICT_UNRESOLVED
    assert result.primary == ()
    assert result.secondary is None
    assert (result.explained_intensity, result.explained_peak_count, result.cluster_count) == (0, 0, 0)


def test_unresolved_when_only_overlapping_clusters_exist():
    # One pair of peaks that (with a wide m/z tolerance) is legal at both
    # charges: any two clusters would share peaks, so no complete candidate.
    spec = [("500.0000", 10), ("500.5016775", 10)]
    result = solve(spec, [1, 2], "0.6", "300")
    assert result.verdict == VERDICT_UNRESOLVED


def test_global_tradeoff_rejects_independent_best_clusters():
    # z=1 strongest envelope sits at M=900; z=2 strongest at M=1000.  Picking
    # both independently maximises neither jointly because their mass
    # intervals do not intersect.  The joint optimum sacrifices intensity.
    spec = merge_specs(
        [("450.0000", 1), ("450.5016775", 1)],          # z=2 @ M=900, weak
        [("500.0000", 1000), ("500.5016775", 1000)],    # z=2 @ M=1000, strong
        [("900.0000", 1000), ("901.003355", 1000)],     # z=1 @ M=900, strong
        [("1000.0000", 10), ("1001.003355", 10)],       # z=1 @ M=1000, weak
    )
    result = solve(spec, [1, 2], "0.0001", "1")
    assert result.verdict == VERDICT_UNIQUE
    # Joint optimum: strong z=2 @ 1000 + weak z=1 @ 1000 = 2020.
    assert result.explained_intensity == 2020
    by_charge = {c.charge: c.peak_indices for c in result.primary}
    z1_first = Decimal(spec[by_charge[1][0]][0])
    z2_first = Decimal(spec[by_charge[2][0]][0])
    assert z1_first == Decimal("1000.0000")
    assert z2_first == Decimal("500.0000")


def test_peak_count_breaks_equal_intensity_ties():
    spec = merge_specs(
        # z=2 option A: pair at M=1000 (2 peaks, intensity 200)
        [("500.0000000", 100), ("500.5016775", 100)],
        # z=2 option B: triplet at M=1000.2 (3 peaks, intensity 200)
        [("500.1000000", 50), ("500.6016775", 100), ("501.1033550", 50)],
        pair_around("1000", 1),
    )
    result = solve(spec, [1, 2], "0.0001", "0.5")
    assert result.verdict == VERDICT_UNIQUE
    assert result.explained_intensity == 400
    assert result.explained_peak_count == 5
    z2 = next(c for c in result.primary if c.charge == 2)
    assert z2.peak_indices == (1, 3, 4)


def test_ambiguous_joint_candidates_yield_witness():
    # Two distinct z=2 envelopes at M=1000.0000 and M=1000.0010, both
    # compatible with the same z=1 envelope and equal on every objective.
    spec = merge_specs(
        [("500.0000", 100), ("500.5016775", 100)],
        [("500.0005", 100), ("500.5021775", 100)],
        pair_around("1000", 1),
    )
    result = solve(spec, [1, 2], "0.0001", "0.001")
    assert result.verdict == VERDICT_AMBIGUOUS
    assert result.secondary is not None
    primary = {c.peak_indices for c in result.primary}
    witness = {c.peak_indices for c in result.secondary}
    assert primary != witness
    z2_pairs = {tuple(c.peak_indices for c in result.primary if c.charge == 2),
                tuple(c.peak_indices for c in result.secondary if c.charge == 2)}
    assert z2_pairs == {((0, 2),), ((1, 3),)}
    for candidate in (result.primary, result.secondary):
        assert sum(c.explained_intensity for c in candidate) == result.explained_intensity
        centers = [c.charge * Decimal(spec[c.peak_indices[0]][0]) for c in candidate]
        lo = max(centers) - Decimal("0.001")
        hi = min(centers) + Decimal("0.001")
        assert lo <= hi


def test_three_and_four_required_charges():
    for k in (3, 4):
        raw = []
        mass = Decimal("2000")
        for z in range(1, k + 1):
            raw.append((mass / z, z))
            raw.append((mass / z + S / z, z))
        raw.sort()
        spec = [(str(m), z) for m, z in raw]
        result = solve(spec, list(range(1, k + 1)), "0.0000001", "0.000001")
        assert result.verdict == VERDICT_UNIQUE
        assert result.cluster_count == k
        assert {c.charge for c in result.primary} == set(range(1, k + 1))
        used = [i for c in result.primary for i in c.peak_indices]
        assert len(used) == len(set(used)) == 2 * k


def test_required_charge_order_is_normalised():
    spec = merge_specs(pair_around("1000", 1), pair_around("1000", 2))
    a = solve(spec, [2, 1], "0.0001", "0.001")
    b = solve(spec, [1, 2], "0.0001", "0.001")
    assert a == b


def test_witness_cannot_share_peaks():
    # On three peaks every legal z=1/z=2 pair shares a peak, so the joint
    # constraint yields UNRESOLVED rather than merging the per-charge best.
    spec = [("300.0", 10), ("300.6", 10), ("301.0", 10)]
    result = solve(spec, [1, 2], "0.5", "600")
    assert result.verdict == VERDICT_UNRESOLVED


def test_search_budget_guard():
    spec = merge_specs(pair_around("1000", 1), pair_around("1000", 2))
    peaks = make_peaks(spec)
    with pytest.raises(SearchSpaceExceededError):
        CoelutingSolver(
            peaks, [1, 2], [1, 2], Decimal("0.0001"), Decimal("0.001"),
            max_search_ops=0,
        ).solve()


def test_invalid_construction_arguments():
    peaks = make_peaks(merge_specs(pair_around("1000", 1), pair_around("1000", 2)))
    with pytest.raises(ValueError):
        CoelutingSolver(peaks, [1, 2], [1], Decimal("0.1"), Decimal("1"))
    with pytest.raises(ValueError):
        CoelutingSolver(peaks, [1, 2], [1, 2, 3, 4, 5], Decimal("0.1"), Decimal("1"))
    with pytest.raises(ValueError):
        CoelutingSolver(peaks, [1, 2], [1, 1], Decimal("0.1"), Decimal("1"))
    with pytest.raises(ValueError):
        CoelutingSolver(peaks, [1], [1, 2], Decimal("0.1"), Decimal("1"))
    with pytest.raises(ValueError):
        CoelutingSolver(peaks, [1, 2], [1, 2], Decimal("0.1"), Decimal("-1"))
    with pytest.raises(ValueError):
        CoelutingSolver(peaks, [1, 2], [1, 2], Decimal("-0.1"), Decimal("1"))
