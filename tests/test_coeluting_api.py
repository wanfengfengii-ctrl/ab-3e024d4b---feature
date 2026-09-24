"""API-level tests for the coeluting multi-charge endpoint."""

from decimal import Decimal

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
URL = "/api/v1/deconvolve/coeluting"
LEGACY_URL = "/api/v1/deconvolve"


def post(payload: dict):
    return client.post(URL, json=payload)


def pair_around(mass: str, charge: int, intensity: int = 100) -> list[dict]:
    m0 = Decimal(mass) / charge
    return [
        {"mz": str(m0), "intensity": intensity},
        {"mz": str(m0 + Decimal("1.003355") / charge), "intensity": intensity},
    ]


def merge(*pairs: list[dict]) -> list[dict]:
    by_mz: dict[str, int] = {}
    for pair in pairs:
        for p in pair:
            by_mz[p["mz"]] = by_mz.get(p["mz"], 0) + p["intensity"]
    return [
        {"mz": mz, "intensity": it}
        for mz, it in sorted(by_mz.items(), key=lambda kv: Decimal(kv[0]))
    ]


GOOD = merge(pair_around("1000", 1), pair_around("1000", 2))
GOOD_PAYLOAD = {
    "peaks": GOOD,
    "charges": [1, 2],
    "required_charges": [1, 2],
    "tolerance": "0.0001",
    "mass_tolerance": "0.001",
}


def test_unique_coeluting_end_to_end():
    resp = post(GOOD_PAYLOAD)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["verdict"] == "UNIQUE"
    assert body["objectives"] == {
        "explained_intensity": 400,
        "explained_peak_count": 4,
        "cluster_count": 2,
    }
    assert sorted(c["charge"] for c in body["clusters"]) == [1, 2]
    for cluster in body["clusters"]:
        assert "neutral_mass" in cluster
        assert len(cluster["peak_indices"]) == 2
    assert body["unexplained_peaks"] == []
    assert body["common_mass"] == {"lower": "999.999", "upper": "1000.001"}
    assert body["second_witness"] is None
    summary = body["input_summary"]
    assert summary["required_charges"] == [1, 2]
    assert summary["mass_tolerance"] == "0.001"
    assert summary["isotope_spacing"] == "1.003355"


def test_response_is_deterministic():
    bodies = {post(GOOD_PAYLOAD).text for _ in range(3)}
    assert len(bodies) == 1


def test_clusters_are_canonically_sorted():
    # z=2 envelope (m/z ~500) must be listed before z=1 (m/z ~1000).
    body = post(GOOD_PAYLOAD).json()
    charges_in_order = [c["charge"] for c in body["clusters"]]
    assert charges_in_order == [2, 1]
    first_mz = [Decimal(c["peaks"][0]["mz"]) for c in body["clusters"]]
    assert first_mz == sorted(first_mz)


def test_common_mass_boundary_touch():
    peaks = merge(
        pair_around("1000.0010", 1),
        [{"mz": "500.0000", "intensity": 100}, {"mz": "500.5016775", "intensity": 100}],
    )
    payload = {**GOOD_PAYLOAD, "peaks": peaks}
    touch = post({**payload, "mass_tolerance": "0.0005"})
    assert touch.status_code == 200
    assert touch.json()["verdict"] == "UNIQUE"
    assert touch.json()["common_mass"] == {"lower": "1000.0005", "upper": "1000.0005"}
    miss = post({**payload, "mass_tolerance": "0.000499"})
    assert miss.status_code == 200
    assert miss.json()["verdict"] == "UNRESOLVED"
    assert miss.json()["common_mass"] is None
    assert miss.json()["objectives"] == {
        "explained_intensity": 0,
        "explained_peak_count": 0,
        "cluster_count": 0,
    }


def test_global_tradeoff():
    peaks = merge(
        [{"mz": "450.0000", "intensity": 1}, {"mz": "450.5016775", "intensity": 1}],
        [{"mz": "500.0000", "intensity": 1000}, {"mz": "500.5016775", "intensity": 1000}],
        [{"mz": "900.0000", "intensity": 1000}, {"mz": "901.003355", "intensity": 1000}],
        [{"mz": "1000.0000", "intensity": 10}, {"mz": "1001.003355", "intensity": 10}],
    )
    payload = {
        "peaks": peaks,
        "charges": [1, 2],
        "required_charges": [1, 2],
        "tolerance": "0.0001",
        "mass_tolerance": "1",
    }
    body = post(payload).json()
    assert body["verdict"] == "UNIQUE"
    assert body["objectives"]["explained_intensity"] == 2020
    chosen = {c["charge"]: c["peak_indices"] for c in body["clusters"]}
    assert chosen[1] == [6, 7]
    assert chosen[2] == [2, 3]


def test_ambiguous_returns_distinct_witness():
    peaks = merge(
        [{"mz": "500.0000", "intensity": 100}, {"mz": "500.5016775", "intensity": 100}],
        [{"mz": "500.0005", "intensity": 100}, {"mz": "500.5021775", "intensity": 100}],
        pair_around("1000", 1),
    )
    body = post(GOOD_PAYLOAD | {"peaks": peaks}).json()
    assert body["verdict"] == "AMBIGUOUS"
    witness = body["second_witness"]
    assert witness is not None
    primary_z2 = {
        tuple(c["peak_indices"]) for c in body["clusters"] if c["charge"] == 2
    }
    witness_z2 = {
        tuple(c["peak_indices"]) for c in witness["clusters"] if c["charge"] == 2
    }
    assert primary_z2 != witness_z2
    assert primary_z2 | witness_z2 == {(0, 2), (1, 3)}
    # The witness carries its own common mass interval.
    assert witness["common_mass"]["lower"] <= witness["common_mass"]["upper"]
    for solution in (body, witness):
        seen = [i for c in solution["clusters"] for i in c["peak_indices"]]
        assert len(seen) == len(set(seen))


def test_three_charges():
    raw = []
    mass = Decimal("2000")
    for z in (1, 2, 3):
        raw.append((mass / z, z))
        raw.append((mass / z + Decimal("1.003355") / z, z))
    raw.sort()
    peaks = [{"mz": str(m), "intensity": z} for m, z in raw]
    payload = {
        "peaks": peaks,
        "charges": [1, 2, 3],
        "required_charges": [3, 1, 2],  # unsorted on purpose
        "tolerance": "0.0000001",
        "mass_tolerance": "0.000001",
    }
    body = post(payload).json()
    assert body["verdict"] == "UNIQUE"
    assert sorted(c["charge"] for c in body["clusters"]) == [1, 2, 3]
    assert body["input_summary"]["required_charges"] == [1, 2, 3]


def test_search_budget_exceeded_returns_503(monkeypatch):
    import app.main as main_module

    monkeypatch.setattr(main_module, "MAX_SEARCH_OPS", 1)
    resp = post({**GOOD_PAYLOAD, "tolerance": "1", "mass_tolerance": "100"})
    assert resp.status_code == 503
    body = resp.json()
    assert body["error"]["code"] == "SEARCH_SPACE_EXCEEDED"
    assert "verdict" not in body


def test_invalid_inputs_are_422_with_located_fields():
    cases = {
        "too few required": (["required_charges"], {**GOOD_PAYLOAD, "required_charges": [1]}),
        "too many required": (
            ["required_charges"],
            {
                **GOOD_PAYLOAD,
                "charges": [1, 2, 3, 4, 5],
                "required_charges": [1, 2, 3, 4, 5],
            },
        ),
        "duplicate required": (
            ["required_charges"],
            {**GOOD_PAYLOAD, "required_charges": [1, 1]},
        ),
        "required not in charges": (
            ["required_charges"],
            {**GOOD_PAYLOAD, "charges": [1], "required_charges": [1, 2]},
        ),
        "zero required charge": (
            ["required_charges.0"],
            {**GOOD_PAYLOAD, "charges": [0, 1], "required_charges": [0, 1]},
        ),
        "fractional required charge": (
            ["required_charges.0"],
            {**GOOD_PAYLOAD, "required_charges": [1.5, 2]},
        ),
        "negative mass tolerance": (
            ["mass_tolerance"],
            {**GOOD_PAYLOAD, "mass_tolerance": "-0.001"},
        ),
        "non-finite mass tolerance": (
            ["mass_tolerance"],
            {**GOOD_PAYLOAD, "mass_tolerance": "NaN"},
        ),
        "missing mass tolerance": (
            ["mass_tolerance"],
            {k: v for k, v in GOOD_PAYLOAD.items() if k != "mass_tolerance"},
        ),
        "missing required charges": (
            ["required_charges"],
            {k: v for k, v in GOOD_PAYLOAD.items() if k != "required_charges"},
        ),
        "unknown field": (["debug"], {**GOOD_PAYLOAD, "debug": 1}),
        "duplicate allowed charges": (
            ["charges"],
            {**GOOD_PAYLOAD, "charges": [1, 1]},
        ),
        "negative mz tolerance": (
            ["tolerance"],
            {**GOOD_PAYLOAD, "tolerance": "-0.001"},
        ),
    }
    for name, (expected_prefixes, payload) in cases.items():
        resp = post(payload)
        assert resp.status_code == 422, f"{name}: {resp.status_code} {resp.text}"
        body = resp.json()
        assert "verdict" not in body, name
        locs = [f["loc"] for f in body["error"]["fields"]]
        assert locs, name
        assert any(
            any(loc == prefix or loc.startswith(prefix) for prefix in expected_prefixes)
            for loc in locs
        ), f"{name}: {locs}"


def test_legacy_endpoint_semantics_unchanged():
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
    resp = client.post(LEGACY_URL, json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {
        "verdict",
        "objectives",
        "clusters",
        "unexplained_peaks",
        "second_witness",
        "input_summary",
    }
    assert body["verdict"] == "UNIQUE"
    assert body["objectives"] == {
        "explained_intensity": 2800,
        "explained_peak_count": 4,
        "cluster_count": 1,
    }
    assert body["clusters"][0]["peak_indices"] == [1, 2, 3, 4]
    assert "neutral_mass" not in body["clusters"][0]
    assert "common_mass" not in body
    assert set(body["input_summary"]) == {
        "peak_count",
        "charges",
        "tolerance",
        "isotope_spacing",
    }
    # Extra coeluting-only fields must not leak into the legacy request.
    resp = client.post(LEGACY_URL, json={**payload, "required_charges": [1, 2]})
    assert resp.status_code == 422
