"""Exact, deterministic deconvolution of overlapping isotope peak clusters.

A *cluster* is a set of 2-6 peaks sharing one charge state ``z`` whose
adjacent m/z spacings each deviate from ``1.003355 / z`` by no more than the
given tolerance.  Every peak may belong to at most one cluster.

This module performs an *exhaustive* search over all legal combinations of
mutually disjoint clusters and optimises the objectives lexicographically:

1. maximise the total explained intensity;
2. maximise the number of explained peaks;
3. minimise the number of clusters.

No greedy nearest-peak or strongest-candidate-first heuristics are used: a
dynamic program over peak bitmasks explores the complete search space, and
ties on all three objectives are detected by enumerating a second optimal
witness.  All arithmetic is done with :class:`decimal.Decimal`, so results
are exact and reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable, Sequence

#: Exact isotope spacing (mass difference of 13C vs 12C) in Dalton.
ISOTOPE_SPACING = Decimal("1.003355")

MIN_CLUSTER_SIZE = 2
MAX_CLUSTER_SIZE = 6

VERDICT_UNIQUE = "UNIQUE"
VERDICT_AMBIGUOUS = "AMBIGUOUS"
VERDICT_UNRESOLVED = "UNRESOLVED"

#: Default budget of cluster-expansion operations for the exact search.  The
#: search is always exhaustive; the budget only guards the service against
#: pathological inputs (e.g. a tolerance as wide as the isotope spacing
#: itself) where the NP-hard set-packing search space explodes.  Exceeding it
#: raises :class:`SearchSpaceExceededError` instead of hanging.
DEFAULT_MAX_SEARCH_OPS = 20_000_000


class SearchSpaceExceededError(RuntimeError):
    """The exact search exceeded the configured work budget."""

# Objective tuples are (explained_intensity, explained_peak_count, -cluster_count).
# Plain tuple comparison then implements the required lexicographic order:
# intensity first, then explained peak count, then fewest clusters.
Objective = tuple[int, int, int]


@dataclass(frozen=True)
class Peak:
    """One input peak. ``index`` is its position in the submitted list."""

    index: int
    mz: Decimal
    intensity: int


@dataclass(frozen=True)
class Cluster:
    """A legal isotope cluster: 2-6 peaks at one charge state."""

    charge: int
    peak_indices: tuple[int, ...]
    mask: int
    explained_intensity: int

    @property
    def size(self) -> int:
        return len(self.peak_indices)

    def canonical_key(self) -> tuple:
        return (self.peak_indices[0], self.charge, self.peak_indices)


@dataclass(frozen=True)
class DeconvolutionResult:
    verdict: str
    explained_intensity: int
    explained_peak_count: int
    cluster_count: int
    #: Canonically sorted clusters of the primary optimal solution.
    primary: tuple[Cluster, ...]
    #: A second, distinct optimal solution (only for AMBIGUOUS verdicts).
    secondary: tuple[Cluster, ...] | None


@dataclass(frozen=True)
class CoelutingResult:
    """Outcome of the co-eluting multi-charge confirmation search."""

    verdict: str
    explained_intensity: int
    explained_peak_count: int
    cluster_count: int
    #: Canonically sorted clusters of the primary optimal combination.
    primary: tuple[Cluster, ...]
    #: Common neutral-mass intersection of the primary combination
    #: ``(lower, upper)``; ``None`` when UNRESOLVED.
    primary_mass_interval: tuple[Decimal, Decimal] | None
    #: A second, distinct optimal combination (only for AMBIGUOUS verdicts).
    secondary: tuple[Cluster, ...] | None
    #: Common neutral-mass intersection of the secondary combination.
    secondary_mass_interval: tuple[Decimal, Decimal] | None


def generate_clusters(
    peaks: Sequence[Peak],
    charges: Iterable[int],
    tolerance: Decimal,
) -> list[Cluster]:
    """Enumerate every legal cluster for every allowed charge state.

    The spacing test is evaluated exactly: ``|Δmz·z − 1.003355| ≤ tol·z``
    is equivalent to ``|Δmz − 1.003355/z| ≤ tol`` but needs no division.
    """
    mzs = [p.mz for p in peaks]
    intensities = [p.intensity for p in peaks]
    n = len(mzs)
    clusters: list[Cluster] = []
    for charge in sorted(set(charges)):
        threshold = tolerance * charge
        # adjacency[i] = peaks j > i whose spacing from i matches 1.003355/z.
        adjacency: list[list[int]] = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                delta = mzs[j] - mzs[i]
                deviation = delta * charge - ISOTOPE_SPACING
                if deviation > threshold:
                    break  # m/z strictly increasing: later j deviate even more
                if deviation >= -threshold:
                    adjacency[i].append(j)
        # A cluster is a chain i1 < i2 < ... < ik (2 <= k <= 6) of
        # adjacent matches; extend chains depth-first.
        chain: list[int] = []

        def visit() -> None:
            if len(chain) >= MIN_CLUSTER_SIZE:
                mask = 0
                total = 0
                for idx in chain:
                    mask |= 1 << idx
                    total += intensities[idx]
                clusters.append(
                    Cluster(
                        charge=charge,
                        peak_indices=tuple(chain),
                        mask=mask,
                        explained_intensity=total,
                    )
                )
            if len(chain) == MAX_CLUSTER_SIZE:
                return
            for nxt in adjacency[chain[-1]]:
                chain.append(nxt)
                visit()
                chain.pop()

        for start in range(n):
            chain.append(start)
            visit()
            chain.pop()
    clusters.sort(key=lambda c: (c.peak_indices[0], c.charge, c.peak_indices))
    return clusters


def canon_solution(solution: tuple[Cluster, ...]) -> tuple[Cluster, ...]:
    """Sort a solution's clusters into canonical order."""
    return tuple(sorted(solution, key=lambda c: c.canonical_key()))


def solution_sort_key(solution: tuple[Cluster, ...]) -> tuple:
    """Deterministic total order over canonically sorted solutions."""
    return tuple((c.peak_indices, c.charge) for c in solution)


class Deconvolver:
    """Exhaustive exact solver over disjoint isotope-peak clusters."""

    def __init__(
        self,
        peaks: Sequence[Peak],
        charges: Iterable[int],
        tolerance: Decimal,
        max_search_ops: int = DEFAULT_MAX_SEARCH_OPS,
    ) -> None:
        peaks = tuple(peaks)
        if not peaks:
            raise ValueError("at least one peak is required")
        charges = tuple(sorted(set(charges)))
        if not charges:
            raise ValueError("at least one charge state is required")
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        self._peaks = peaks
        self._charges = charges
        self._tolerance = tolerance
        self._max_search_ops = max_search_ops
        self._search_ops = 0
        self._full_mask = (1 << len(peaks)) - 1
        self.clusters: tuple[Cluster, ...] = tuple(
            generate_clusters(self._peaks, self._charges, self._tolerance)
        )
        by_peak: list[list[Cluster]] = [[] for _ in peaks]
        for cluster in self.clusters:
            for idx in cluster.peak_indices:
                by_peak[idx].append(cluster)
        # Deterministic bucket order so witness enumeration is reproducible.
        self._clusters_by_peak: tuple[tuple[Cluster, ...], ...] = tuple(
            tuple(sorted(bucket, key=lambda c: (c.charge, c.peak_indices)))
            for bucket in by_peak
        )
        self._best_memo: dict[int, Objective] = {}

    # ------------------------------------------------------------------ #
    # Exhaustive optimisation (exact DP over used-peak bitmasks)
    # ------------------------------------------------------------------ #

    def _best(self, used_mask: int) -> Objective:
        """Best achievable objective over the peaks still free in ``used_mask``."""
        memo = self._best_memo
        cached = memo.get(used_mask)
        if cached is not None:
            return cached
        free = self._full_mask & ~used_mask
        if free == 0:
            result: Objective = (0, 0, 0)
        else:
            first = (free & -free).bit_length() - 1
            # Option A: leave `first` unexplained.
            best = self._best(used_mask | (1 << first))
            # Option B: cover `first` with each legal cluster that fits.
            for cluster in self._clusters_by_peak[first]:
                self._search_ops += 1
                if self._search_ops > self._max_search_ops:
                    raise SearchSpaceExceededError(
                        "exact deconvolution search exceeded the configured "
                        f"work budget ({self._max_search_ops} operations); "
                        "narrow the tolerance or the charge set"
                    )
                if cluster.mask & used_mask:
                    continue
                ci, cp, cc = self._best(used_mask | cluster.mask)
                candidate = (
                    ci + cluster.explained_intensity,
                    cp + cluster.size,
                    cc - 1,
                )
                if candidate > best:
                    best = candidate
            result = best
        memo[used_mask] = result
        return result

    # ------------------------------------------------------------------ #
    # Optimal-witness enumeration (for tie detection / second witness)
    # ------------------------------------------------------------------ #

    def _witnesses(self, used_mask: int, limit: int) -> list[tuple[Cluster, ...]]:
        """Up to ``limit`` distinct optimal solutions from ``used_mask``."""
        if limit <= 0:
            return []
        target = self._best(used_mask)
        free = self._full_mask & ~used_mask
        if free == 0:
            return [()]
        first = (free & -free).bit_length() - 1
        found: list[tuple[Cluster, ...]] = []
        skip_mask = used_mask | (1 << first)
        if self._best(skip_mask) == target:
            for tail in self._witnesses(skip_mask, limit):
                found.append(tail)
                if len(found) >= limit:
                    return found[:limit]
        for cluster in self._clusters_by_peak[first]:
            if cluster.mask & used_mask:
                continue
            ci, cp, cc = self._best(used_mask | cluster.mask)
            if (ci + cluster.explained_intensity, cp + cluster.size, cc - 1) != target:
                continue
            for tail in self._witnesses(used_mask | cluster.mask, limit - len(found)):
                found.append((cluster,) + tail)
                if len(found) >= limit:
                    return found[:limit]
        return found

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #

    def solve(self) -> DeconvolutionResult:
        intensity, peak_count, neg_clusters = self._best(0)
        if peak_count == 0:
            return DeconvolutionResult(
                verdict=VERDICT_UNRESOLVED,
                explained_intensity=0,
                explained_peak_count=0,
                cluster_count=0,
                primary=(),
                secondary=None,
            )
        canonical = {canon_solution(w) for w in self._witnesses(0, 2)}
        ordered = sorted(canonical, key=solution_sort_key)
        primary = ordered[0]
        secondary = ordered[1] if len(ordered) > 1 else None
        return DeconvolutionResult(
            verdict=VERDICT_UNIQUE if secondary is None else VERDICT_AMBIGUOUS,
            explained_intensity=intensity,
            explained_peak_count=peak_count,
            cluster_count=-neg_clusters,
            primary=primary,
            secondary=secondary,
        )


class CoelutingDeconvolver:
    """Exhaustive confirmation of a co-eluting multi-charge envelope.

    A *feasible combination* picks **exactly one** legal cluster per required
    charge state; the clusters are pairwise disjoint, and their neutral-mass
    estimates — ``first-peak m/z × charge`` — must all agree within
    ``mass_tolerance``: the per-cluster intervals ``[M − tol, M + tol]`` must
    share a common point, i.e. ``max(M) − min(M) ≤ 2·mass_tolerance``.

    The search enumerates the feasible combinations *directly* (it does not
    run the ordinary deconvolution and filter its verdict) and optimises the
    same lexicographic objectives: (1) maximise explained intensity,
    (2) maximise explained peak count, (3) minimise cluster count.  Ties on
    all three objectives yield an AMBIGUOUS verdict with a distinct second
    witness.  All arithmetic is exact :class:`decimal.Decimal`.
    """

    def __init__(
        self,
        peaks: Sequence[Peak],
        charges: Iterable[int],
        tolerance: Decimal,
        required_charges: Iterable[int],
        mass_tolerance: Decimal,
        max_search_ops: int = DEFAULT_MAX_SEARCH_OPS,
    ) -> None:
        peaks = tuple(peaks)
        if not peaks:
            raise ValueError("at least one peak is required")
        charges = tuple(sorted(set(charges)))
        if not charges:
            raise ValueError("at least one charge state is required")
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        required = tuple(required_charges)
        if not required:
            raise ValueError("at least one required charge state is required")
        if len(set(required)) != len(required):
            raise ValueError("required charge states must be unique")
        unknown = [z for z in required if z not in charges]
        if unknown:
            raise ValueError(
                f"required charge states must belong to the allowed charges: {unknown}"
            )
        if mass_tolerance < 0:
            raise ValueError("mass tolerance must be non-negative")
        self._peaks = peaks
        self._required = tuple(sorted(required))
        self._mass_tolerance = mass_tolerance
        self._max_search_ops = max_search_ops
        self._search_ops = 0
        # Neutral-mass estimate of every legal cluster: first-peak m/z × charge.
        # Candidate clusters per required charge, in canonical order.
        self._mass_of: dict[Cluster, Decimal] = {}
        candidate_lists: list[tuple[Cluster, ...]] = []
        for charge in self._required:
            candidates = generate_clusters(peaks, (charge,), tolerance)
            for cluster in candidates:
                self._mass_of[cluster] = peaks[cluster.peak_indices[0]].mz * charge
            candidate_lists.append(tuple(candidates))
        self._candidates: tuple[tuple[Cluster, ...], ...] = tuple(candidate_lists)

    # ------------------------------------------------------------------ #
    # Exhaustive enumeration of feasible combinations
    # ------------------------------------------------------------------ #

    def solve(self) -> CoelutingResult:
        best: Objective | None = None
        witnesses: list[tuple[Cluster, ...]] = []
        chosen: list[Cluster] = []
        two_tol = self._mass_tolerance * 2
        levels = self._candidates

        def visit(
            level: int,
            used_mask: int,
            run_lo: Decimal | None,
            run_hi: Decimal | None,
            acc_intensity: int,
            acc_peaks: int,
        ) -> None:
            nonlocal best
            if level == len(levels):
                # Cluster count is fixed (= number of required charges); the
                # third objective component is kept for a uniform ordering.
                objective: Objective = (acc_intensity, acc_peaks, -len(levels))
                combo = tuple(chosen)
                if best is None or objective > best:
                    best = objective
                    witnesses.clear()
                    witnesses.append(combo)
                elif objective == best and len(witnesses) < 2:
                    witnesses.append(combo)
                return
            for cluster in levels[level]:
                self._search_ops += 1
                if self._search_ops > self._max_search_ops:
                    raise SearchSpaceExceededError(
                        "co-eluting deconvolution search exceeded the configured "
                        f"work budget ({self._max_search_ops} operations); "
                        "narrow the tolerance or the charge set"
                    )
                if cluster.mask & used_mask:
                    continue
                mass = self._mass_of[cluster]
                lo = mass if run_lo is None or mass < run_lo else run_lo
                hi = mass if run_hi is None or mass > run_hi else run_hi
                if hi - lo > two_tol:
                    continue  # neutral-mass intervals cannot intersect
                chosen.append(cluster)
                visit(
                    level + 1,
                    used_mask | cluster.mask,
                    lo,
                    hi,
                    acc_intensity + cluster.explained_intensity,
                    acc_peaks + cluster.size,
                )
                chosen.pop()

        visit(0, 0, None, None, 0, 0)
        if best is None:
            return CoelutingResult(
                verdict=VERDICT_UNRESOLVED,
                explained_intensity=0,
                explained_peak_count=0,
                cluster_count=0,
                primary=(),
                primary_mass_interval=None,
                secondary=None,
                secondary_mass_interval=None,
            )
        intensity, peak_count, neg_clusters = best
        canonical = sorted({canon_solution(w) for w in witnesses}, key=solution_sort_key)
        primary = canonical[0]
        secondary = canonical[1] if len(canonical) > 1 else None
        return CoelutingResult(
            verdict=VERDICT_UNIQUE if secondary is None else VERDICT_AMBIGUOUS,
            explained_intensity=intensity,
            explained_peak_count=peak_count,
            cluster_count=-neg_clusters,
            primary=primary,
            primary_mass_interval=self._mass_interval(primary),
            secondary=secondary,
            secondary_mass_interval=(
                self._mass_interval(secondary) if secondary is not None else None
            ),
        )

    def _mass_interval(self, combo: tuple[Cluster, ...]) -> tuple[Decimal, Decimal]:
        """Common intersection of the combo's neutral-mass tolerance intervals."""
        masses = [self._mass_of[cluster] for cluster in combo]
        return (
            max(masses) - self._mass_tolerance,
            min(masses) + self._mass_tolerance,
        )
