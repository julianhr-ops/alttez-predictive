from __future__ import annotations

import json
import os
import re
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from unittest import mock

from alttez.features.elo import tournament_k_factor
from alttez.features.klement import _classify_competition as history_competition_class
from alttez.ingestion import espn_results
from alttez.ingestion.espn_competitions import (
    ESPN_DEFAULT_LEAGUES,
    HISTORICAL_TOURNAMENT_BY_SLUG,
    LEGACY_LEAGUES,
    tournament_name,
)
from alttez.ingestion.live_matches import classify_competition, parse_espn_records
from alttez.ingestion.providers import espn as espn_provider
from alttez.ingestion.providers.espn import extract_league_meta, resolve_espn_leagues, stamp_espn_event

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
BASE_K = 30.0

# slug -> (name stored, Elo K, competition class) as the historical dataset gives them
HISTORICAL = {
    "uefa.nations": ("UEFA Nations League", 30.0, "continental"),
    "concacaf.nations.league": ("CONCACAF Nations League", 30.0, "continental"),
    "fifa.worldq.uefa": ("FIFA World Cup qualification", 40.0, "world_cup"),
    "fifa.worldq.conmebol": ("FIFA World Cup qualification", 40.0, "world_cup"),
    "fifa.worldq.concacaf": ("FIFA World Cup qualification", 40.0, "world_cup"),
    "fifa.worldq.afc": ("FIFA World Cup qualification", 40.0, "world_cup"),
    "fifa.worldq.caf": ("FIFA World Cup qualification", 40.0, "world_cup"),
    "fifa.wcq.ply": ("FIFA World Cup qualification", 40.0, "world_cup"),
    "caf.nations": ("African Cup of Nations", 30.0, "other_international"),
    "caf.nations_qual": ("African Cup of Nations qualification", 40.0, "qualifiers"),
    "afc.cupq": ("AFC Asian Cup qualification", 40.0, "qualifiers"),
    "aff.championship": ("ASEAN Championship", 30.0, "other_international"),
    "global.gulf_cup": ("Gulf Cup", 30.0, "other_international"),
    "afc.saff.championship": ("SAFF Cup", 30.0, "other_international"),
    "global.finalissima": ("CONMEBOL\u2013UEFA Cup of Champions", 30.0, "other_international"),
}


def _stored_match(fixture: str, slug: str):
    payload = json.loads((FIXTURES / fixture).read_text(encoding="utf-8"))
    meta = extract_league_meta(payload, slug)
    records = [stamp_espn_event(event, meta) for event in payload["events"]]
    matches, _ = parse_espn_records(records, datetime(2026, 11, 18, tzinfo=timezone.utc), date(2025, 10, 11))
    assert len(matches) == 1
    return matches[0]


class EspnRealPayloadTests(unittest.TestCase):
    def test_world_cup_qualifier_is_stored_like_the_historical_qualifier(self) -> None:
        match = _stored_match("espn_worldq_uefa_scoreboard.json", "fifa.worldq.uefa")
        self.assertEqual(match.tournament_name, "FIFA World Cup qualification")
        self.assertEqual(tournament_k_factor(match.tournament_name, BASE_K), 40.0)
        self.assertEqual(match.competition_class, history_competition_class("FIFA World Cup qualification", None))
        self.assertEqual((match.home_team, match.away_team), ("Spain", "Georgia"))
        self.assertEqual(match.status, "finished")
        self.assertFalse(match.is_neutral)

    def test_africa_cup_of_nations_is_not_classed_as_continental(self) -> None:
        match = _stored_match("espn_caf_nations_scoreboard.json", "caf.nations")
        self.assertEqual(match.tournament_name, "African Cup of Nations")
        self.assertEqual(tournament_k_factor(match.tournament_name, BASE_K), 30.0)
        self.assertEqual(match.competition_class, history_competition_class("African Cup of Nations", None))
        self.assertEqual(match.competition_class, "other_international")

    def test_slugs_without_a_mapping_keep_the_name_espn_sends(self) -> None:
        payload = {"leagues": [{"name": "International Friendly", "slug": "fifa.friendly", "season": {"year": 2026}}]}
        self.assertEqual(extract_league_meta(payload, "fifa.friendly")["name"], "International Friendly")
        self.assertEqual(tournament_name("fifa.world", "FIFA World Cup"), "FIFA World Cup")


class HistoricalNameMappingTests(unittest.TestCase):
    def test_every_new_slug_is_covered(self) -> None:
        self.assertEqual(set(HISTORICAL), set(HISTORICAL_TOURNAMENT_BY_SLUG))

    def test_stored_name_k_factor_and_class_match_the_historical_dataset(self) -> None:
        for slug, (name, k_factor, competition_class) in HISTORICAL.items():
            with self.subTest(slug=slug):
                stored = tournament_name(slug, "whatever ESPN calls it")
                self.assertEqual(stored, name)
                self.assertEqual(tournament_k_factor(stored, BASE_K), k_factor)
                self.assertEqual(classify_competition(stored, None), competition_class)
                self.assertEqual(history_competition_class(stored, None), competition_class)

    def test_stage_slugs_seen_on_espn_do_not_change_the_class(self) -> None:
        stages = ("group-stage", "third-round", "fourth-round", "preliminary-phase", "final", "quarterfinals")
        for slug, (name, _, competition_class) in HISTORICAL.items():
            for stage in stages:
                with self.subTest(slug=slug, stage=stage):
                    self.assertEqual(classify_competition(name, stage), competition_class)


class LeagueListAgreementTests(unittest.TestCase):
    def test_both_espn_readers_use_the_shared_list(self) -> None:
        self.assertEqual(espn_provider.ESPN_DEFAULT_LEAGUES, ESPN_DEFAULT_LEAGUES)
        self.assertEqual(espn_results.ESPN_DEFAULT_LEAGUES, ESPN_DEFAULT_LEAGUES)

    def test_legacy_leagues_stay_first_and_no_slug_repeats(self) -> None:
        self.assertEqual(ESPN_DEFAULT_LEAGUES[: len(LEGACY_LEAGUES)], LEGACY_LEAGUES)
        self.assertEqual(len(set(ESPN_DEFAULT_LEAGUES)), len(ESPN_DEFAULT_LEAGUES))

    def test_unset_env_resolves_to_the_shared_list(self) -> None:
        with mock.patch.dict(os.environ):
            os.environ.pop("ALTTEZ_ESPN_LEAGUES", None)
            self.assertEqual(resolve_espn_leagues(None), list(ESPN_DEFAULT_LEAGUES))

    def test_env_template_and_workflow_carry_the_same_list(self) -> None:
        expected = list(ESPN_DEFAULT_LEAGUES)
        env_text = (ROOT / ".env.example").read_text(encoding="utf-8")
        workflow_text = (ROOT / ".github" / "workflows" / "daily_process.yml").read_text(encoding="utf-8")
        env_values = re.findall(r"^ALTTEZ_ESPN_LEAGUES=(\S+)$", env_text, re.MULTILINE)
        workflow_values = re.findall(r"^\s*ALTTEZ_ESPN_LEAGUES:\s*(\S+)\s*$", workflow_text, re.MULTILINE)
        self.assertEqual([v.split(",") for v in env_values], [expected])
        self.assertEqual([v.split(",") for v in workflow_values], [expected])


if __name__ == "__main__":
    unittest.main()
