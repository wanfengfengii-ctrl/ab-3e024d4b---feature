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


def generate_clusters_for_charge(
    charge: int,
    mzs: Sequence[Decimal],
    intensities: Sequence[int],
    tolerance: Decimal,
) -> list[Cluster]:
    """Enumerate every legal 2-6 peak cluster at one charge state.

    The spacing test is evaluated exactly: ``|Δmz·z − 1.003355| ≤ tol·z``
    is equivalent to ``|Δmz − 1.003355/z| ≤ tol`` but needs no division.
    """
    n = len(mzs)
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
    found: list[Cluster] = []
    chain: list[int] = []

    def visit() -> None:
        if len(chain) >= MIN_CLUSTER_SIZE:
            mask = 0
            total = 0
            for idx in chain:
                mask |= 1 << idx
                total += intensities[idx]
            found.append(
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
    return found

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
        self.clusters: tuple[Cluster, ...] = tuple(self._generate_clusters())
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
    # Cluster generation
    # ------------------------------------------------------------------ #

    def _generate_clusters(self) -> list[Cluster]:
        """Enumerate every legal cluster for every allowed charge state.

        The spacing test is evaluated exactly: ``|Δmz·z − 1.003355| ≤ tol·z``
        is equivalent to ``|Δmz − 1.003355/z| ≤ tol`` but needs no division.
        """
        mzs = [p.mz for p in self._peaks]
        intensities = [p.intensity for p in self._peaks]
        n = len(mzs)
        clusters: list[Cluster] = []
        for charge in self._charges:
            clusters.extend(
                generate_clusters_for_charge(
                    charge, mzs, intensities, self._tolerance
                )
            )
        clusters.sort(key=lambda c: (c.peak_indices[0], c.charge, c.peak_indices))
        return clusters

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
        canonical = {self._canon(w) for w in self._witnesses(0, 2)}
        ordered = sorted(canonical, key=self._solution_key)
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

    @staticmethod
    def _canon(solution: tuple[Cluster, ...]) -> tuple[Cluster, ...]:
        return tuple(sorted(solution, key=lambda c: c.canonical_key()))

    @staticmethod
    def _solution_key(solution: tuple[Cluster, ...]) -> tuple:
        return tuple((c.peak_indices, c.charge) for c in solution)


# ---------------------------------------------------------------------- #
# Coeluting multi-charge confirmation
# ---------------------------------------------------------------------- #

#: A complete candidate is a mapping ``depth -> chosen cluster``, represented
#: as a tuple in the (ascending) required-charge order.
Candidate = tuple[Cluster, ...]


@dataclass(frozen=True)
class CoelutingResult:
    verdict: str
    explained_intensity: int
    explained_peak_count: int
    cluster_count: int
    #: Canonically sorted clusters of the primary optimal candidate.
    primary: tuple[Cluster, ...]
    #: A second, distinct optimal candidate (only for AMBIGUOUS verdicts).
    secondary: tuple[Cluster, ...] | None
    #: Inclusive common neutral-mass intersection [lower, upper].
    mass_lower: Decimal
    mass_upper: Decimal


def candidate_objective(candidate: Candidate) -> Objective:
    return (
        sum(c.explained_intensity for c in candidate),
        sum(c.size for c in candidate),
        -len(candidate),
    )


class CoelutingSolver:
    """Exhaustive solver for jointly confirmed coeluting charge states.

    Every complete candidate chooses exactly one non-overlapping cluster per
    required charge state such that the neutral-mass tolerance intervals
    derived from each cluster's first peak share a common (inclusive)
    intersection.  The search over complete candidates is exhaustive and
    joint -- clusters are never chosen independently and filtered afterwards.
    Objectives are optimised lexicographically: explained intensity, then
    explained peak count, then number of clusters (fixed here, retained for
    symmetry with the ordinary solver).
    """

    def __init__(
        self,
        peaks: Sequence[Peak],
        allowed_charges: Iterable[int],
        required_charges: Iterable[int],
        tolerance: Decimal,
        mass_tolerance: Decimal,
        max_search_ops: int = DEFAULT_MAX_SEARCH_OPS,
    ) -> None:
        peaks = tuple(peaks)
        if not peaks:
            raise ValueError("at least one peak is required")
        required = tuple(sorted(required_charges))
        if not (2 <= len(required) <= 4):
            raise ValueError("required_charges must contain 2 to 4 charge states")
        if len(set(required)) != len(required):
            raise ValueError("required_charges must not contain duplicates")
        allowed = frozenset(allowed_charges)
        if any(z not in allowed for z in required):
            raise ValueError("every required charge must be in the allowed charge set")
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        if mass_tolerance < 0:
            raise ValueError("mass_tolerance must be non-negative")
        self._peaks = peaks
        self._intensities = [p.intensity for p in peaks]
        self._mass_tolerance = mass_tolerance
        self._max_search_ops = max_search_ops
        self._search_ops = 0
        self._required = required  # ascending distinct positive ints
        mzs = [p.mz for p in peaks]
        intensities = [p.intensity for p in peaks]
        options: list[list[Cluster]] = []
        for z in required:
            bucket = generate_clusters_for_charge(z, mzs, intensities, tolerance)
            # Deterministic order: by first peak, then peak-index tuple.
            bucket.sort(key=lambda c: (c.peak_indices[0], c.peak_indices))
            options.append(bucket)
        self._options = options

    def _tick(self) -> None:
        self._search_ops += 1
        if self._search_ops > self._max_search_ops:
            raise SearchSpaceExceededError(
                "exact coeluting search exceeded the configured work budget "
                f"({self._max_search_ops} operations); narrow the tolerance, "
                "the mass tolerance or the charge set"
            )

    def _interval(
        self, cluster: Cluster, lo: Decimal, hi: Decimal
    ) -> tuple[Decimal, Decimal] | None:
        """Intersect the running mass window with the cluster's interval."""
        center = cluster.charge * self._peaks[cluster.peak_indices[0]].mz
        new_lo = max(lo, center - self._mass_tolerance)
        new_hi = min(hi, center + self._mass_tolerance)
        return (new_lo, new_hi) if new_lo <= new_hi else None

    def _find_best(
        self,
        depth: int,
        used_mask: int,
        lo: Decimal,
        hi: Decimal,
        intensity: int,
        peak_count: int,
        free_intensity: int,
        best: list[tuple[int, int]],
    ) -> None:
        """Pass 1: exhaustive joint search for the optimal objective.

        ``free_intensity`` is the total intensity of peaks not yet used; the
        optimistic upper bound ``intensity + free_intensity`` prunes branches
        that cannot beat the incumbent intensity.  The cluster count is fixed
        (one cluster per required charge), so it is not part of the key.
        """
        if intensity + free_intensity < best[0][0]:
            return
        if depth == len(self._required):
            candidate = (intensity, peak_count)
            if candidate > best[0]:
                best[0] = candidate
            return
        for cluster in self._options[depth]:
            self._tick()
            if cluster.mask & used_mask:
                continue
            window = self._interval(cluster, lo, hi)
            if window is None:
                continue
            self._find_best(
                depth + 1,
                used_mask | cluster.mask,
                window[0],
                window[1],
                intensity + cluster.explained_intensity,
                peak_count + cluster.size,
                free_intensity - cluster.explained_intensity,
                best,
            )

    def _collect(
        self,
        depth: int,
        used_mask: int,
        lo: Decimal,
        hi: Decimal,
        intensity: int,
        peak_count: int,
        free_intensity: int,
        target: tuple[int, int],
        chosen: list[Cluster],
        out: list[Candidate],
        limit: int,
    ) -> None:
        """Pass 2: collect up to ``limit`` candidates at the optimum.

        This is a bounded, fully exhaustive traversal of the same joint
        candidate space -- never a filter applied to an ordinary deconvolution.
        """
        if len(out) >= limit or intensity + free_intensity < target[0]:
            return
        if depth == len(self._required):
            if (intensity, peak_count) == target:
                out.append(tuple(chosen))
            return
        for cluster in self._options[depth]:
            self._tick()
            if cluster.mask & used_mask:
                continue
            window = self._interval(cluster, lo, hi)
            if window is None:
                continue
            chosen.append(cluster)
            self._collect(
                depth + 1,
                used_mask | cluster.mask,
                window[0],
                window[1],
                intensity + cluster.explained_intensity,
                peak_count + cluster.size,
                free_intensity - cluster.explained_intensity,
                target,
                chosen,
                out,
                limit,
            )
            chosen.pop()
            if len(out) >= limit:
                return

    def solve(self) -> CoelutingResult:
        total_intensity = sum(self._intensities)
        best: list[tuple[int, int]] = [(-1, -1)]
        self._find_best(
            0,
            0,
            Decimal("-Infinity"),
            Decimal("Infinity"),
            0,
            0,
            total_intensity,
            best,
        )
        if best[0][0] < 0:
            return CoelutingResult(
                verdict=VERDICT_UNRESOLVED,
                explained_intensity=0,
                explained_peak_count=0,
                cluster_count=0,
                primary=(),
                secondary=None,
                mass_lower=Decimal(0),
                mass_upper=Decimal(0),
            )

        found: list[Candidate] = []
        self._collect(
            0,
            0,
            Decimal("-Infinity"),
            Decimal("Infinity"),
            0,
            0,
            total_intensity,
            best[0],
            [],
            found,
            limit=2,
        )
        canonical = {
            tuple(sorted(cand, key=lambda c: c.canonical_key())) for cand in found
        }
        ordered = sorted(
            canonical,
            key=lambda cand: tuple((c.peak_indices, c.charge) for c in cand),
        )
        primary = ordered[0]
        secondary = ordered[1] if len(ordered) > 1 else None
        primary_obj = candidate_objective(primary)

        lower = max(
            c.charge * self._peaks[c.peak_indices[0]].mz - self._mass_tolerance
            for c in primary
        )
        upper = min(
            c.charge * self._peaks[c.peak_indices[0]].mz + self._mass_tolerance
            for c in primary
        )
        return CoelutingResult(
            verdict=VERDICT_UNIQUE if secondary is None else VERDICT_AMBIGUOUS,
            explained_intensity=primary_obj[0],
            explained_peak_count=primary_obj[1],
            cluster_count=-primary_obj[2],
            primary=primary,
            secondary=secondary,
            mass_lower=lower,
            mass_upper=upper,
        )
