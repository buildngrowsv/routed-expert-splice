"""Verify the 7B knowledge-transfer readout from the Studio's saved scores."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


REVISION = "4982b3395c2cace1c4295be8e0a47d93b099218a"
COUNTRY_QUESTIONS = {"birth_country", "employer_country", "university_country"}
DIRECT_QUESTIONS = {"city", "employer", "instrument", "sport", "university"}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, help="Studio architecture queue runs directory")
    args = parser.parse_args()
    checksums = {}

    def fingerprint(path):
        content = path.read_bytes()
        checksums[str(path.relative_to(args.runs))] = sha256(content).hexdigest()
        return content

    sweep = args.runs / "prog-e1-sweep-7b"
    ranks = args.runs / "prog-e1-ranks-7b"
    for directory in (sweep, ranks):
        receipt = json.loads(fingerprint(directory / "receipt.json"))
        assert receipt["job"] == directory.name and receipt["revision"] == REVISION
        assert receipt["status"] == "done" and receipt["exit"] == 0
    selected = json.loads(fingerprint(ranks / "result.json"))
    first = json.loads(fingerprint(sweep / "result.json"))
    rank_name = selected["best"]
    assert rank_name == "L20-r256-s0"
    arms = {
        "map_seed_0": ranks / "configs" / f"{rank_name}.npz",
        "map_seed_1": ranks / "configs" / "L20-r256-s1.npz",
        "map_seed_2": ranks / "configs" / "L20-r256-s2.npz",
        "text_relay": sweep / "configs" / "control-relay.npz",
        "host_alone": sweep / "configs" / "host_alone.npz",
        "donor_alone": sweep / "configs" / "donor_alone.npz",
        "lora": sweep / "configs" / "control-lora.npz",
        "untrained_donor_rank64": sweep / "configs" / "control-base-donor-L20-r64.npz",
    }
    with np.load(arms["map_seed_0"]) as trace:
        people, kind = trace["person"], trace["kind"]
    assert set(kind) == DIRECT_QUESTIONS | COUNTRY_QUESTIONS
    direct = ~np.isin(kind, sorted(COUNTRY_QUESTIONS))
    for mask, expected in ((direct, 10), (~direct, 3)):
        people_in_scope, count = np.unique(people[mask], return_counts=True)
        assert len(people_in_scope) == 2000 and np.all(count == expected)

    counts = {}
    for arm, path in arms.items():
        fingerprint(path)
        with np.load(path) as trace:
            np.testing.assert_array_equal(trace["person"], people)
            np.testing.assert_array_equal(trace["kind"], kind)
            closed, open_ = trace["closed"], trace["open"]
            assert np.isin(closed, (0, 1)).all() and np.isin(open_, (0, 1)).all()
            counts[arm] = dict(direct=int(closed[direct].sum()), country=int(closed[~direct].sum()),
                               direct_top_token=int(open_[direct].sum()))

    expected = {
        "map_seed_0": (selected["held_direct"], selected["held_composition"]),
        "text_relay": (first["control_relay"]["direct"], first["control_relay"]["composition"]),
        "host_alone": (first["host_alone"]["held_direct"], first["host_alone"]["held_composition"]),
        "donor_alone": (first["donor_alone"]["held_direct"], first["donor_alone"]["held_composition"]),
        "untrained_donor_rank64": (first["control_base_donor"], first["control_base_donor_composition"]),
    }
    for arm, (direct_rate, country_rate) in expected.items():
        np.testing.assert_allclose([counts[arm]["direct"] / 20000, counts[arm]["country"] / 6000],
                                   [direct_rate, country_rate])
    np.testing.assert_allclose(counts["lora"]["direct"] / 20000, first["control_lora"])
    seed_rates = [counts[f"map_seed_{i}"]["direct"] / 20000 for i in range(3)]
    np.testing.assert_allclose(seed_rates, selected["held_direct_seeds"])
    np.testing.assert_allclose(np.mean(seed_rates), selected["held_direct_seed_mean"])
    donor_floor = 0.6 * counts["donor_alone"]["direct"] / 20000
    assert max(seed_rates) < donor_floor and selected["decision"] == "transplant_partial"
    assert counts["map_seed_0"]["country"] < counts["text_relay"]["country"]
    wrong_person = json.loads(fingerprint(ranks / "configs" / f"{rank_name}.json"))
    assert wrong_person["wrong_direct"] == selected["wrong_person_direct"]
    print(json.dumps(dict(revision=REVISION, people=2000, direct_questions=20000, country_questions=6000,
                          decision=selected["decision"], counts=counts, direct_seed_mean=float(np.mean(seed_rates)),
                          donor_floor=donor_floor, wrong_person_direct_reported=wrong_person["wrong_direct"],
                          sha256=checksums), indent=2))


if __name__ == "__main__":
    main()
