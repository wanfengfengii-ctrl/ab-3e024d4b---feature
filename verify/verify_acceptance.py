#!/usr/bin/env python3
"""One-shot acceptance suite against a live Isotope Deconvolution API.

Usage (Docker Compose, from the repository root):

    docker compose run --rm verify

or against any reachable instance:

    API_BASE_URL=http://localhost:8000 python verify/verify_acceptance.py

Every check talks to the real HTTP API.  The process exits 0 only if all
checks pass.
"""

from __future__ import annotations

import os
import sys
import time

import httpx

BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")
HEALTH_URL = f"{BASE_URL}/health"
DECONVOLVE_URL = f"{BASE_URL}/api/v1/deconvolve"
COELUTING_URL = f"{BASE_URL}/api/v1/deconvolve/coeluting"

_checks = 0
_failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    global _checks
    _checks += 1
    if condition:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}" + (f"  -- {detail}" if detail else ""))
        _failures.append(name)


def wait_for_api(timeout_s: float = 60.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(HEALTH_URL, timeout=3.0)
            if resp.status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(1.0)
    return False


def post(client: httpx.Client, payload: dict) -> httpx.Response:
    return client.post(DECONVOLVE_URL, json=payload, timeout=30.0)


def post_coeluting(client: httpx.Client, payload: dict) -> httpx.Response:
    return client.post(COELUTING_URL, json=payload, timeout=30.0)


def cluster_index_sets(solution_clusters: list[dict]) -> set[tuple[int, ...]]:
    return {tuple(c["peak_indices"]) for c in solution_clusters}


# ---------------------------------------------------------------------- #
# Scenarios
# ---------------------------------------------------------------------- #


def scenario_health(client: httpx.Client) -> None:
    print("[health]")
    resp = client.get(HEALTH_URL, timeout=5.0)
    check("GET /health returns 200", resp.status_code == 200, f"got {resp.status_code}")
    body = resp.json() if resp.status_code == 200 else {}
    check("health body reports ok", body.get("status") == "ok", repr(body))


def scenario_unique(client: httpx.Client) -> dict:
    print("[unique verdict]")
    payload = {
        "peaks": [
            {"mz": "400.000000", "intensity": 500},
            {"mz": "500.000000", "intensity": 1000},
            {"mz": "501.003355", "intensity": 800},
            {"mz": "502.006710", "intensity": 600},
            {"mz": "503.010065", "intensity": 400},
            {"mz": "700.000000", "intensity": 50},
        ],
        "charges": [1],
        "tolerance": "0.0005",
    }
    resp = post(client, payload)
    check("unique: 200", resp.status_code == 200, f"got {resp.status_code}: {resp.text}")
    body = resp.json()
    check("unique: verdict UNIQUE", body.get("verdict") == "UNIQUE", body.get("verdict", ""))
    obj = body.get("objectives", {})
    check(
        "unique: objectives (2800 / 4 / 1)",
        (obj.get("explained_intensity"), obj.get("explained_peak_count"), obj.get("cluster_count"))
        == (2800, 4, 1),
        repr(obj),
    )
    clusters = body.get("clusters", [])
    check("unique: one cluster", len(clusters) == 1, repr(clusters))
    if clusters:
        check(
            "unique: cluster is charge 1 over peaks [1,2,3,4]",
            clusters[0].get("charge") == 1 and clusters[0].get("peak_indices") == [1, 2, 3, 4],
            repr(clusters[0]),
        )
    unexplained = [p["index"] for p in body.get("unexplained_peaks", [])]
    check("unique: unexplained peaks are [0, 5]", unexplained == [0, 5], repr(unexplained))
    check("unique: no second witness", body.get("second_witness") is None)
    return payload


def scenario_determinism(client: httpx.Client, payload: dict) -> None:
    print("[determinism]")
    bodies = {post(client, payload).text for _ in range(3)}
    check("identical input yields byte-identical responses", len(bodies) == 1)


def scenario_ambiguous(client: httpx.Client) -> None:
    print("[ambiguous verdict + second witness]")
    # d(0,1)=0.6 and d(0,2)=1.0 both lie within 0.5 of 1.003355, d(1,2)=0.4 does
    # not.  Clusters {0,1} and {0,2} tie on (intensity 300, 2 peaks, 1 cluster).
    payload = {
        "peaks": [
            {"mz": "300.0", "intensity": 100},
            {"mz": "300.6", "intensity": 200},
            {"mz": "301.0", "intensity": 200},
        ],
        "charges": [1],
        "tolerance": "0.5",
    }
    resp = post(client, payload)
    check("ambiguous: 200", resp.status_code == 200, f"got {resp.status_code}: {resp.text}")
    body = resp.json()
    check("ambiguous: verdict AMBIGUOUS", body.get("verdict") == "AMBIGUOUS", body.get("verdict", ""))
    witness = body.get("second_witness")
    check("ambiguous: second witness present", witness is not None)
    if witness:
        primary_sets = cluster_index_sets(body.get("clusters", []))
        witness_sets = cluster_index_sets(witness.get("clusters", []))
        check(
            "ambiguous: witness differs from primary",
            primary_sets != witness_sets,
            f"primary={primary_sets} witness={witness_sets}",
        )
        check(
            "ambiguous: witnesses are {{0,1}} and {{0,2}}",
            primary_sets | witness_sets == {(0, 1), (0, 2)},
            f"primary={primary_sets} witness={witness_sets}",
        )
        w_intensity = sum(c["explained_intensity"] for c in witness.get("clusters", []))
        obj = body.get("objectives", {})
        check(
            "ambiguous: witness matches primary objectives",
            w_intensity == obj.get("explained_intensity")
            and len(witness.get("clusters", [])) == obj.get("cluster_count"),
            f"witness_intensity={w_intensity} objectives={obj}",
        )


def scenario_unresolved(client: httpx.Client) -> None:
    print("[unresolved verdict]")
    payload = {
        "peaks": [
            {"mz": "100.0", "intensity": 10},
            {"mz": "100.3", "intensity": 20},
            {"mz": "100.6", "intensity": 30},
        ],
        "charges": [1],
        "tolerance": "0.001",
    }
    resp = post(client, payload)
    check("unresolved: 200", resp.status_code == 200, f"got {resp.status_code}: {resp.text}")
    body = resp.json()
    check("unresolved: verdict UNRESOLVED", body.get("verdict") == "UNRESOLVED", body.get("verdict", ""))
    check("unresolved: no clusters", body.get("clusters") == [])
    check(
        "unresolved: all peaks unexplained",
        [p["index"] for p in body.get("unexplained_peaks", [])] == [0, 1, 2],
    )
    obj = body.get("objectives", {})
    check(
        "unresolved: zero objectives",
        (obj.get("explained_intensity"), obj.get("explained_peak_count"), obj.get("cluster_count"))
        == (0, 0, 0),
        repr(obj),
    )


def scenario_intensity_before_peak_count(client: httpx.Client) -> None:
    print("[lexicographic objectives: intensity beats peak count]")
    # X = {1,3} at z=1 explains 200 over 2 peaks; Y = {0,1,2} at z=3 explains
    # 102 over 3 peaks.  They conflict on peak 1 and Y's leftover peaks cannot
    # recombine, so the intensity-first lexicographic order must prefer X.
    payload = {
        "peaks": [
            {"mz": "500.6689033", "intensity": 1},
            {"mz": "501.003355", "intensity": 100},
            {"mz": "501.3378067", "intensity": 1},
            {"mz": "502.006710", "intensity": 100},
        ],
        "charges": [1, 2, 3],
        "tolerance": "0.0001",
    }
    resp = post(client, payload)
    check("lexico: 200", resp.status_code == 200, f"got {resp.status_code}: {resp.text}")
    body = resp.json()
    obj = body.get("objectives", {})
    check(
        "lexico: intensity-first optimum (200 / 2 / 1)",
        (obj.get("explained_intensity"), obj.get("explained_peak_count"), obj.get("cluster_count"))
        == (200, 2, 1),
        repr(obj),
    )
    clusters = body.get("clusters", [])
    check(
        "lexico: chosen cluster is {1,3} at z=1",
        len(clusters) == 1
        and clusters[0].get("peak_indices") == [1, 3]
        and clusters[0].get("charge") == 1,
        repr(clusters),
    )
    check("lexico: verdict UNIQUE", body.get("verdict") == "UNIQUE", body.get("verdict", ""))
    check(
        "lexico: unexplained peaks are [0, 2]",
        [p["index"] for p in body.get("unexplained_peaks", [])] == [0, 2],
    )


def scenario_charge_two(client: httpx.Client) -> None:
    print("[charge-state spacing]")
    base = {
        "peaks": [
            {"mz": "700.0000000", "intensity": 10},
            {"mz": "700.5016775", "intensity": 20},
        ],
        "tolerance": "0.0000001",
    }
    resp = post(client, {**base, "charges": [2]})
    body = resp.json()
    clusters = body.get("clusters", [])
    check(
        "z=2: pair at 1.003355/2 spacing forms a cluster",
        resp.status_code == 200
        and body.get("verdict") == "UNIQUE"
        and len(clusters) == 1
        and clusters[0].get("charge") == 2
        and clusters[0].get("peak_indices") == [0, 1],
        resp.text,
    )
    resp = post(client, {**base, "charges": [1]})
    check(
        "z=1: same pair is not an isotope spacing",
        resp.status_code == 200 and resp.json().get("verdict") == "UNRESOLVED",
        resp.text,
    )


def scenario_tolerance_boundary(client: httpx.Client) -> None:
    print("[tolerance boundary is inclusive]")
    base = {
        "peaks": [
            {"mz": "400.000000", "intensity": 5},
            {"mz": "401.003855", "intensity": 7},  # deviation exactly 0.0005
        ],
        "charges": [1],
    }
    resp = post(client, {**base, "tolerance": "0.0005"})
    check(
        "boundary: deviation == tolerance is accepted",
        resp.status_code == 200 and resp.json().get("verdict") == "UNIQUE",
        resp.text,
    )
    resp = post(client, {**base, "tolerance": "0.0004999"})
    check(
        "boundary: deviation > tolerance is rejected",
        resp.status_code == 200 and resp.json().get("verdict") == "UNRESOLVED",
        resp.text,
    )


def scenario_cluster_size_cap(client: httpx.Client) -> None:
    print("[cluster size capped at 6]")
    spacing = "1.003355"
    from decimal import Decimal

    mz = Decimal("900.000000")
    peaks = []
    for i in range(7):
        peaks.append({"mz": str(mz), "intensity": 1})
        mz += Decimal(spacing)
    payload = {"peaks": peaks, "charges": [1], "tolerance": "0.0001"}
    resp = post(client, payload)
    body = resp.json()
    obj = body.get("objectives", {})
    check(
        "7-chain: all 7 peaks explained by 2 clusters (cap at 6 forces a split)",
        resp.status_code == 200
        and (obj.get("explained_peak_count"), obj.get("cluster_count")) == (7, 2),
        resp.text,
    )
    sizes = [len(c["peak_indices"]) for c in body.get("clusters", [])]
    check("7-chain: every cluster has at most 6 peaks", all(s <= 6 for s in sizes), repr(sizes))
    check(
        "7-chain: multiple equal splits exist -> AMBIGUOUS with witness",
        body.get("verdict") == "AMBIGUOUS" and body.get("second_witness") is not None,
        body.get("verdict", ""),
    )


def scenario_full_scale(client: httpx.Client) -> None:
    print("[36 peaks, 6 disjoint chains]")
    from decimal import Decimal

    spacing = Decimal("1.003355")
    peaks = []
    for k in range(6):
        mz = Decimal(200 + 100 * k)
        for i in range(6):
            peaks.append({"mz": str(mz), "intensity": 100 + 10 * k + i})
            mz += spacing
    payload = {"peaks": peaks, "charges": [1], "tolerance": "0.0001"}
    expected_intensity = sum(p["intensity"] for p in peaks)
    resp = post(client, payload)
    body = resp.json()
    obj = body.get("objectives", {})
    check(
        "36 peaks: UNIQUE, 6 clusters, everything explained",
        resp.status_code == 200
        and body.get("verdict") == "UNIQUE"
        and (obj.get("explained_intensity"), obj.get("explained_peak_count"), obj.get("cluster_count"))
        == (expected_intensity, 36, 6)
        and body.get("unexplained_peaks") == [],
        resp.text[:400],
    )


def scenario_validation(client: httpx.Client) -> None:
    print("[validation: field-locatable 422, never a verdict]")
    good_peaks = [
        {"mz": "500.000000", "intensity": 100},
        {"mz": "501.003355", "intensity": 90},
    ]
    cases = {
        "too few peaks": {"peaks": good_peaks[:1], "charges": [1], "tolerance": "0.001"},
        "too many peaks": {
            "peaks": [{"mz": str(100 + i), "intensity": 1} for i in range(37)],
            "charges": [1],
            "tolerance": "0.001",
        },
        "mz not increasing": {
            "peaks": [
                {"mz": "501.003355", "intensity": 90},
                {"mz": "500.000000", "intensity": 100},
            ],
            "charges": [1],
            "tolerance": "0.001",
        },
        "zero intensity": {
            "peaks": [good_peaks[0], {"mz": "501.003355", "intensity": 0}],
            "charges": [1],
            "tolerance": "0.001",
        },
        "fractional intensity": {
            "peaks": [good_peaks[0], {"mz": "501.003355", "intensity": 1.5}],
            "charges": [1],
            "tolerance": "0.001",
        },
        "non-positive mz": {
            "peaks": [{"mz": "0", "intensity": 1}, {"mz": "1.003355", "intensity": 1}],
            "charges": [1],
            "tolerance": "0.001",
        },
        "empty charges": {"peaks": good_peaks, "charges": [], "tolerance": "0.001"},
        "zero charge": {"peaks": good_peaks, "charges": [0], "tolerance": "0.001"},
        "duplicate charges": {"peaks": good_peaks, "charges": [1, 1], "tolerance": "0.001"},
        "negative tolerance": {"peaks": good_peaks, "charges": [1], "tolerance": "-0.1"},
        "missing tolerance": {"peaks": good_peaks, "charges": [1]},
        "unknown field": {"peaks": good_peaks, "charges": [1], "tolerance": "0.001", "debug": True},
    }
    for name, payload in cases.items():
        resp = post(client, payload)
        ok_status = resp.status_code == 422
        body = resp.json() if ok_status else {}
        fields = body.get("error", {}).get("fields", [])
        located = all(f.get("loc") for f in fields) and len(fields) > 0
        check(
            f"validation[{name}]: 422 with located fields, no verdict",
            ok_status and located and "verdict" not in body,
            f"status={resp.status_code} body={resp.text[:300]}",
        )


# ---------------------------------------------------------------------- #
# Co-eluting multi-charge confirmation scenarios
# ---------------------------------------------------------------------- #

# A single precursor of neutral mass 500 observed at z=1 and z=2.
COELUTING_UNIQUE_PEAKS = [
    {"mz": "250.000000", "intensity": 400},
    {"mz": "250.5016775", "intensity": 300},
    {"mz": "500.000000", "intensity": 1000},
    {"mz": "501.003355", "intensity": 800},
]

# The strong z=1 envelope (mass 500) is mass-inconsistent with the only z=2
# envelope (mass 600); the feasible optimum must trade intensity for a common
# neutral mass and pick the weak z=1 envelope at mass 600.
COELUTING_TRADEOFF_PEAKS = [
    {"mz": "300.000000", "intensity": 500},
    {"mz": "300.5016775", "intensity": 400},
    {"mz": "500.000000", "intensity": 1000},
    {"mz": "501.003355", "intensity": 800},
    {"mz": "600.000000", "intensity": 100},
    {"mz": "601.003355", "intensity": 90},
]

# Two equal-quality z=1 envelopes (mass 600 and 602) both fit the z=2
# envelope (mass 600) within mass_tolerance 1.0.
COELUTING_AMBIGUOUS_PEAKS = [
    {"mz": "300.000000", "intensity": 10},
    {"mz": "300.5016775", "intensity": 10},
    {"mz": "600.000000", "intensity": 100},
    {"mz": "601.003355", "intensity": 100},
    {"mz": "602.000000", "intensity": 100},
    {"mz": "603.003355", "intensity": 100},
]


def scenario_coeluting_unique(client: httpx.Client) -> None:
    print("[coeluting: unique two-charge envelope]")
    payload = {
        "peaks": COELUTING_UNIQUE_PEAKS,
        "charges": [1, 2],
        "tolerance": "0.0001",
        "required_charges": [1, 2],
        "mass_tolerance": "0.5",
    }
    resp = post_coeluting(client, payload)
    check("coeluting unique: 200", resp.status_code == 200, f"got {resp.status_code}: {resp.text}")
    body = resp.json()
    check("coeluting unique: verdict UNIQUE", body.get("verdict") == "UNIQUE", body.get("verdict", ""))
    obj = body.get("objectives", {})
    check(
        "coeluting unique: objectives (2500 / 4 / 2)",
        (obj.get("explained_intensity"), obj.get("explained_peak_count"), obj.get("cluster_count"))
        == (2500, 4, 2),
        repr(obj),
    )
    clusters = body.get("clusters", [])
    check(
        "coeluting unique: canonical clusters z=2 [0,1] then z=1 [2,3]",
        [(c.get("charge"), c.get("peak_indices")) for c in clusters] == [(2, [0, 1]), (1, [2, 3])],
        repr(clusters),
    )
    check(
        "coeluting unique: common mass interval [499.5, 500.5]",
        body.get("common_mass_interval") == {"lower": "499.500000", "upper": "500.500000"},
        repr(body.get("common_mass_interval")),
    )
    check("coeluting unique: no unexplained peaks", body.get("unexplained_peaks") == [])
    check("coeluting unique: no second witness", body.get("second_witness") is None)
    summary = body.get("input_summary", {})
    check(
        "coeluting unique: input summary echoes required charges and mass tolerance",
        summary.get("required_charges") == [1, 2] and summary.get("mass_tolerance") == "0.5",
        repr(summary),
    )


def scenario_coeluting_mass_boundary(client: httpx.Client) -> None:
    print("[coeluting: common-mass boundary is inclusive]")
    # Neutral-mass estimates 500.4 (z=2) and 500.0 (z=1): spread exactly 0.4.
    base = {
        "peaks": [
            {"mz": "250.200000", "intensity": 10},
            {"mz": "250.7016775", "intensity": 20},
            {"mz": "500.000000", "intensity": 30},
            {"mz": "501.003355", "intensity": 40},
        ],
        "charges": [1, 2],
        "tolerance": "0.0001",
        "required_charges": [1, 2],
    }
    resp = post_coeluting(client, {**base, "mass_tolerance": "0.2"})
    body = resp.json()
    check(
        "coeluting boundary: spread == 2*mass_tolerance is accepted (degenerate interval)",
        resp.status_code == 200
        and body.get("verdict") == "UNIQUE"
        and body.get("common_mass_interval") == {"lower": "500.200000", "upper": "500.200000"},
        resp.text,
    )
    resp = post_coeluting(client, {**base, "mass_tolerance": "0.1999"})
    check(
        "coeluting boundary: spread > 2*mass_tolerance is UNRESOLVED",
        resp.status_code == 200 and resp.json().get("verdict") == "UNRESOLVED",
        resp.text,
    )


def scenario_coeluting_global_tradeoff(client: httpx.Client) -> None:
    print("[coeluting: global trade-off beats per-charge optimum]")
    payload = {"peaks": COELUTING_TRADEOFF_PEAKS, "charges": [1, 2], "tolerance": "0.0001"}
    resp = post_coeluting(client, {**payload, "required_charges": [1, 2], "mass_tolerance": "0.3"})
    body = resp.json()
    obj = body.get("objectives", {})
    check(
        "coeluting tradeoff: feasible optimum (1090 / 4 / 2), not the mass-inconsistent 2890",
        resp.status_code == 200
        and body.get("verdict") == "UNIQUE"
        and (obj.get("explained_intensity"), obj.get("explained_peak_count"), obj.get("cluster_count"))
        == (1090, 4, 2)
        and [(c.get("charge"), c.get("peak_indices")) for c in body.get("clusters", [])]
        == [(2, [0, 1]), (1, [4, 5])],
        resp.text,
    )
    # Compatibility regression: the plain endpoint on the same peaks is
    # unaffected and still returns its unconstrained optimum.
    resp = post(client, payload)
    obj = resp.json().get("objectives", {})
    check(
        "coeluting tradeoff: plain endpoint unchanged (2890 / 6 / 3 on same peaks)",
        resp.status_code == 200
        and (obj.get("explained_intensity"), obj.get("explained_peak_count"), obj.get("cluster_count"))
        == (2890, 6, 3),
        resp.text,
    )


def scenario_coeluting_ambiguous(client: httpx.Client) -> None:
    print("[coeluting: ambiguous verdict + distinct witness]")
    payload = {
        "peaks": COELUTING_AMBIGUOUS_PEAKS,
        "charges": [1, 2],
        "tolerance": "0.0001",
        "required_charges": [1, 2],
        "mass_tolerance": "1.0",
    }
    resp = post_coeluting(client, payload)
    check("coeluting ambiguous: 200", resp.status_code == 200, f"got {resp.status_code}: {resp.text}")
    body = resp.json()
    check(
        "coeluting ambiguous: verdict AMBIGUOUS",
        body.get("verdict") == "AMBIGUOUS",
        body.get("verdict", ""),
    )
    witness = body.get("second_witness")
    check("coeluting ambiguous: second witness present", witness is not None)
    if witness:
        primary_sets = cluster_index_sets(body.get("clusters", []))
        witness_sets = cluster_index_sets(witness.get("clusters", []))
        check(
            "coeluting ambiguous: witness differs from primary",
            primary_sets != witness_sets,
            f"primary={primary_sets} witness={witness_sets}",
        )
        check(
            "coeluting ambiguous: witnesses are {(0,1),(2,3)} and {(0,1),(4,5)}",
            primary_sets | witness_sets == {(0, 1), (2, 3), (4, 5)},
            f"primary={primary_sets} witness={witness_sets}",
        )
        check(
            "coeluting ambiguous: witness carries its own common mass interval",
            witness.get("common_mass_interval") == {"lower": "601.000000", "upper": "601.000000"},
            repr(witness.get("common_mass_interval")),
        )
        w_intensity = sum(c["explained_intensity"] for c in witness.get("clusters", []))
        check(
            "coeluting ambiguous: witness matches primary objectives",
            w_intensity == body.get("objectives", {}).get("explained_intensity")
            and len(witness.get("clusters", [])) == body.get("objectives", {}).get("cluster_count"),
            f"witness_intensity={w_intensity}",
        )


def scenario_coeluting_unresolved(client: httpx.Client) -> None:
    print("[coeluting: unresolved when masses cannot intersect]")
    payload = {
        "peaks": [
            {"mz": "250.000000", "intensity": 10},
            {"mz": "250.5016775", "intensity": 10},
            {"mz": "510.000000", "intensity": 10},
            {"mz": "511.003355", "intensity": 10},
        ],
        "charges": [1, 2],
        "tolerance": "0.0001",
        "required_charges": [1, 2],
        "mass_tolerance": "0.1",
    }
    resp = post_coeluting(client, payload)
    body = resp.json()
    check(
        "coeluting unresolved: verdict UNRESOLVED with empty clusters and null interval",
        resp.status_code == 200
        and body.get("verdict") == "UNRESOLVED"
        and body.get("clusters") == []
        and body.get("common_mass_interval") is None
        and body.get("second_witness") is None,
        resp.text,
    )
    check(
        "coeluting unresolved: all peaks unexplained",
        [p["index"] for p in body.get("unexplained_peaks", [])] == [0, 1, 2, 3],
    )


def scenario_coeluting_three_charges(client: httpx.Client) -> None:
    print("[coeluting: three charge states]")
    payload = {
        "peaks": [
            {"mz": "200.000000", "intensity": 5},
            {"mz": "200.3344517", "intensity": 5},
            {"mz": "300.000000", "intensity": 7},
            {"mz": "300.5016775", "intensity": 6},
            {"mz": "600.000000", "intensity": 11},
            {"mz": "601.003355", "intensity": 10},
        ],
        "charges": [1, 2, 3],
        "tolerance": "0.000001",
        "required_charges": [1, 2, 3],
        "mass_tolerance": "0.5",
    }
    resp = post_coeluting(client, payload)
    body = resp.json()
    obj = body.get("objectives", {})
    check(
        "coeluting 3-charge: UNIQUE, 3 clusters, everything explained",
        resp.status_code == 200
        and body.get("verdict") == "UNIQUE"
        and (obj.get("explained_intensity"), obj.get("explained_peak_count"), obj.get("cluster_count"))
        == (44, 6, 3)
        and [c.get("charge") for c in body.get("clusters", [])] == [3, 2, 1]
        and body.get("common_mass_interval") == {"lower": "599.500000", "upper": "600.500000"},
        resp.text,
    )


def scenario_coeluting_validation(client: httpx.Client) -> None:
    print("[coeluting validation: field-locatable 422, never a verdict]")
    good = {
        "peaks": COELUTING_UNIQUE_PEAKS,
        "charges": [1, 2],
        "tolerance": "0.0001",
        "required_charges": [1, 2],
        "mass_tolerance": "0.5",
    }
    cases = {
        "required charge missing": {k: v for k, v in good.items() if k != "required_charges"},
        "mass tolerance missing": {k: v for k, v in good.items() if k != "mass_tolerance"},
        "too few required charges": {**good, "required_charges": [1]},
        "too many required charges": {
            **good,
            "charges": [1, 2, 3, 4, 5],
            "required_charges": [1, 2, 3, 4, 5],
        },
        "duplicate required charges": {**good, "required_charges": [1, 1]},
        "required charge not allowed": {**good, "required_charges": [1, 3]},
        "non-positive required charge": {**good, "required_charges": [0, 1]},
        "fractional required charge": {**good, "required_charges": [1.5, 2]},
        "negative mass tolerance": {**good, "mass_tolerance": "-0.1"},
        "unknown field": {**good, "debug": True},
    }
    for name, payload in cases.items():
        resp = post_coeluting(client, payload)
        ok_status = resp.status_code == 422
        body = resp.json() if ok_status else {}
        fields = body.get("error", {}).get("fields", [])
        located = all(f.get("loc") for f in fields) and len(fields) > 0
        check(
            f"coeluting validation[{name}]: 422 with located fields, no verdict",
            ok_status and located and "verdict" not in body,
            f"status={resp.status_code} body={resp.text[:300]}",
        )
    # Compatibility regression: the legacy schema must not accept the new
    # co-eluting-only fields.
    resp = post(
        client,
        {
            "peaks": COELUTING_UNIQUE_PEAKS,
            "charges": [1, 2],
            "tolerance": "0.0001",
            "required_charges": [1, 2],
        },
    )
    check(
        "compat: plain deconvolve still rejects co-eluting-only fields with 422",
        resp.status_code == 422 and "verdict" not in resp.json(),
        f"status={resp.status_code} body={resp.text[:300]}",
    )


# ---------------------------------------------------------------------- #


def main() -> int:
    print(f"Acceptance target: {BASE_URL}")
    if not wait_for_api():
        print("FATAL: API did not become healthy within 60s")
        return 1
    with httpx.Client() as client:
        scenario_health(client)
        payload = scenario_unique(client)
        scenario_determinism(client, payload)
        scenario_ambiguous(client)
        scenario_unresolved(client)
        scenario_intensity_before_peak_count(client)
        scenario_charge_two(client)
        scenario_tolerance_boundary(client)
        scenario_cluster_size_cap(client)
        scenario_full_scale(client)
        scenario_validation(client)
        scenario_coeluting_unique(client)
        scenario_coeluting_mass_boundary(client)
        scenario_coeluting_global_tradeoff(client)
        scenario_coeluting_ambiguous(client)
        scenario_coeluting_unresolved(client)
        scenario_coeluting_three_charges(client)
        scenario_coeluting_validation(client)
    print(f"\n{_checks - len(_failures)}/{_checks} checks passed")
    if _failures:
        print("FAILED checks:")
        for name in _failures:
            print(f"  - {name}")
        return 1
    print("ALL ACCEPTANCE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
