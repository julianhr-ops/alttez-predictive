from __future__ import annotations

import json
import os
import re
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from unittest import mock

from alttez.features.elo import tournament_k_factor
from alttez.ingestion import espn_results
from alttez.ingestion.espn_competitions import (
    ESPN_DEFAULT_LEAGUES,
    HISTORICAL_TOURNAMENT_BY_SLUG,
    LEGACY_LEAGUES,
    tournament_name,
)
from alttez.ingestion.live_matches import classify_competition, infer_effective_is_neutral, parse_espn_records
from alttez.ingestion.providers import espn as espn_provider
from alttez.ingestion.providers.espn import extract_league_meta, resolve_espn_leagues, stamp_espn_event

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
BASE_K = 30.0

# slug -> (name stored, Elo K, competition class) as the historical dataset gives them
HISTORICAL = {
    "uefa.nations": ("UEFA Nations League", 30.0, "continental"),
    "concacaf.nations.league": ("CONCACAF Nations League", 30.0, "continental"),
    "fifa.worldq.uefa": ("FIFA World Cup qualification", 40.0, "qualifiers"),
    "fifa.worldq.conmebol": ("FIFA World Cup qualification", 40.0, "qualifiers"),
    "fifa.worldq.concacaf": ("FIFA World Cup qualification", 40.0, "qualifiers"),
    "fifa.worldq.afc": ("FIFA World Cup qualification", 40.0, "qualifiers"),
    "fifa.worldq.caf": ("FIFA World Cup qualification", 40.0, "qualifiers"),
    "fifa.wcq.ply": ("FIFA World Cup qualification", 40.0, "qualifiers"),
    "caf.nations": ("African Cup of Nations", 30.0, "other_international"),
    "caf.nations_qual": ("African Cup of Nations qualification", 40.0, "qualifiers"),
    "afc.cupq": ("AFC Asian Cup qualification", 40.0, "qualifiers"),
    "aff.championship": ("ASEAN Championship", 30.0, "other_international"),
    "global.gulf_cup": ("Gulf Cup", 30.0, "other_international"),
    "afc.saff.championship": ("SAFF Cup", 30.0, "other_international"),
    "global.finalissima": ("CONMEBOL\u2013UEFA Cup of Champions", 30.0, "other_international"),
}


def _stored_match(fixture: str, slug: str, target_day: date = date(2025, 10, 11)):
    payload = json.loads((FIXTURES / fixture).read_text(encoding="utf-8"))
    meta = extract_league_meta(payload, slug)
    records = [stamp_espn_event(event, meta) for event in payload["events"]]
    matches, _ = parse_espn_records(records, datetime(2026, 11, 18, tzinfo=timezone.utc), target_day)
    assert len(matches) == 1
    return matches[0]


class EspnRealPayloadTests(unittest.TestCase):
    def test_world_cup_qualifier_is_stored_like_the_historical_qualifier(self) -> None:
        match = _stored_match("espn_worldq_uefa_scoreboard.json", "fifa.worldq.uefa")
        self.assertEqual(match.tournament_name, "FIFA World Cup qualification")
        self.assertEqual(tournament_k_factor(match.tournament_name, BASE_K), 40.0)
        self.assertEqual(match.competition_class, "qualifiers")
        self.assertEqual((match.home_team, match.away_team), ("Spain", "Georgia"))
        self.assertEqual(match.status, "finished")
        self.assertFalse(match.is_neutral)

    def test_africa_cup_of_nations_is_not_classed_as_continental(self) -> None:
        match = _stored_match("espn_caf_nations_scoreboard.json", "caf.nations", date(2025, 12, 23))
        self.assertEqual(match.tournament_name, "African Cup of Nations")
        self.assertEqual(tournament_k_factor(match.tournament_name, BASE_K), 30.0)
        self.assertEqual(match.competition_class, "other_international")

    def test_slugs_without_a_mapping_keep_the_name_espn_sends(self) -> None:
        payload = {"leagues": [{"name": "International Friendly", "slug": "fifa.friendly", "season": {"year": 2026}}]}
        self.assertEqual(extract_league_meta(payload, "fifa.friendly")["name"], "International Friendly")
        self.assertEqual(tournament_name("fifa.world", "FIFA World Cup"), "FIFA World Cup")


class HistoricalNameMappingTests(unittest.TestCase):
    def test_every_new_slug_is_covered(self) -> None:
        self.assertEqual(set(HISTORICAL), set(HISTORICAL_TOURNAMENT_BY_SLUG))

    def test_stored_name_k_factor_and_class_per_slug(self) -> None:
        for slug, (name, k_factor, competition_class) in HISTORICAL.items():
            with self.subTest(slug=slug):
                stored = tournament_name(slug, "whatever ESPN calls it")
                self.assertEqual(stored, name)
                self.assertEqual(tournament_k_factor(stored, BASE_K), k_factor)
                self.assertEqual(classify_competition(stored, None), competition_class)

    def test_stage_slugs_seen_on_espn_do_not_change_the_class(self) -> None:
        stages = ("group-stage", "third-round", "fourth-round", "preliminary-phase", "final", "quarterfinals")
        for slug, (name, _, competition_class) in HISTORICAL.items():
            for stage in stages:
                with self.subTest(slug=slug, stage=stage):
                    self.assertEqual(classify_competition(name, stage), competition_class)


class QualifierAndFinalsVenueTests(unittest.TestCase):
    def test_world_cup_finals_names_keep_the_finals_class(self) -> None:
        for name, stage in (("FIFA World Cup 2026", "group-stage"), ("FIFA World Cup", "final")):
            with self.subTest(name=name):
                self.assertEqual(classify_competition(name, stage), "world_cup")

    def test_qualifier_played_at_home_is_not_neutral_and_not_the_finals(self) -> None:
        match = _stored_match("espn_worldq_uefa_scoreboard.json", "fifa.worldq.uefa")
        self.assertEqual(match.venue_country_name, "Spain")
        self.assertEqual(match.competition_class, "qualifiers")
        self.assertFalse(
            infer_effective_is_neutral(
                tournament_name=match.tournament_name,
                competition_class=match.competition_class,
                home_team=match.home_team,
                away_team=match.away_team,
                venue_country_name=match.venue_country_name,
            )
        )

    def test_qualifier_without_venue_metadata_is_not_neutral(self) -> None:
        for stored_class in ("qualifiers", "world_cup"):
            with self.subTest(competition_class=stored_class):
                self.assertFalse(
                    infer_effective_is_neutral(
                        tournament_name="FIFA World Cup qualification",
                        competition_class=stored_class,
                        home_team="Spain",
                        away_team="Georgia",
                        venue_country_name=None,
                    )
                )

    def test_qualifier_at_a_neutral_provider_venue_stays_neutral(self) -> None:
        self.assertTrue(
            infer_effective_is_neutral(
                tournament_name="FIFA World Cup qualification",
                competition_class="qualifiers",
                home_team="Spain",
                away_team="Georgia",
                venue_country_name="Portugal",
                provider_is_neutral=True,
            )
        )

    def test_world_cup_final_is_still_neutral_and_the_finals_class(self) -> None:
        match = _stored_match("espn_world_cup_final_scoreboard.json", "fifa.world", date(2026, 7, 19))
        self.assertEqual(match.tournament_name, "FIFA World Cup")
        self.assertEqual(match.competition_class, "world_cup")
        self.assertEqual(match.venue_country_name, "USA")
        self.assertTrue(match.is_neutral)

    def test_world_cup_without_venue_country_is_neutral_unless_a_host_plays(self) -> None:
        neutral = infer_effective_is_neutral(
            tournament_name="FIFA World Cup 2026", competition_class="world_cup", home_team="Spain", away_team="Argentina"
        )
        host = infer_effective_is_neutral(
            tournament_name="FIFA World Cup 2026", competition_class="world_cup", home_team="Mexico", away_team="Argentina"
        )
        self.assertTrue(neutral)
        self.assertFalse(host)

    def test_single_host_finals_are_neutral_when_the_venue_is_a_third_country(self) -> None:
        match = _stored_match("espn_caf_nations_scoreboard.json", "caf.nations", date(2025, 12, 23))
        self.assertEqual((match.home_team, match.away_team, match.venue_country_name), ("DR Congo", "Benin", "Morocco"))
        self.assertTrue(match.is_neutral)

    def test_single_host_rule_applies_to_the_gulf_cup_and_not_to_the_host_side(self) -> None:
        def neutral(venue: str) -> bool:
            return infer_effective_is_neutral(
                tournament_name="Gulf Cup",
                competition_class="other_international",
                home_team="Yemen",
                away_team="Qatar",
                venue_country_name=venue,
            )

        self.assertTrue(neutral("Kuwait"))
        self.assertFalse(neutral("Qatar"))

    def test_single_host_rule_leaves_other_competitions_alone(self) -> None:
        for name, competition_class in (
            ("UEFA Nations League", "continental"),
            ("ASEAN Championship", "other_international"),
            ("African Cup of Nations qualification", "qualifiers"),
            ("CONCACAF Nations League", "continental"),
        ):
            with self.subTest(name=name):
                self.assertFalse(
                    infer_effective_is_neutral(
                        tournament_name=name,
                        competition_class=competition_class,
                        home_team="Benin",
                        away_team="Congo DR",
                        venue_country_name="Morocco",
                    )
                )


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
