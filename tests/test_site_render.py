import json
import shutil
import subprocess
import tempfile
import threading
import unittest
import urllib.request
from functools import partial
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from alttez.site import render
from alttez.site.render import SiteDataError, render_site

FIXTURES = Path(__file__).parent / "fixtures" / "site"
NODE = shutil.which("node") or r"C:\Program Files\nodejs\node.exe"
PAGE_FILES = ("index.html", "nba.html", "soccer.html", "method.html")
TEAMS = ("Brazil", "Argentina", "France", "Côte d'Ivoire", "Türkiye", "</script><b>x")
DRAW = 0.27


def outcome_triple(rating_i: float, rating_j: float, home_bonus: float) -> list[int]:
    """Win, draw, loss of team i in units of 1/10000, summing to 10000."""
    expected = 1 / (1 + 10 ** ((rating_j - rating_i - home_bonus) / 400))
    draw = DRAW * (1 - abs(2 * expected - 1))
    parts = [expected - draw / 2, draw, 1 - expected - draw / 2]
    units = [round(p * 10000) for p in parts]
    units[units.index(max(units))] += 10000 - sum(units)
    return units


def make_ratings() -> dict:
    n = len(TEAMS)
    ratings = [2100.0 - 90 * k for k in range(n)]
    neutral: list[int] = []
    home: list[int] = []
    for i in range(n):
        for j in range(n):
            neutral += [0, 0, 0] if i == j else outcome_triple(ratings[i], ratings[j], 0)
            home += [0, 0, 0] if i == j else outcome_triple(ratings[i], ratings[j], 100)
    return {
        "generated_at": "2026-11-18T06:00:00Z",
        "as_of": "2026-11-18",
        "window_years": 2,
        "source": "martj42/international_results",
        "n": n,
        "teams": [{"name": name, "rating": ratings[k], "rank": k + 1} for k, name in enumerate(TEAMS)],
        "neutral": neutral,
        "home": home,
    }


def write_data(directory: Path, **overrides) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for source in FIXTURES.glob("*.json"):
        shutil.copyfile(source, directory / source.name)
    docs = {"soccer_ratings.json": make_ratings(), **overrides}
    for name, doc in docs.items():
        (directory / name).write_text(json.dumps(doc), encoding="utf-8")


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class LinkCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links: list[tuple[str, str, str]] = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ("href", "src") and value:
                self.links.append((tag, key, value))


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


class SiteTestCase(unittest.TestCase):
    def build(self, **overrides) -> Path:
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        write_data(tmp / "in", **overrides)
        render_site(tmp / "in", tmp / "out")
        return tmp / "out"

    @staticmethod
    def page(out: Path, name: str) -> str:
        return (out / name).read_text(encoding="utf-8")


class BuildTests(SiteTestCase):
    def test_pages_assets_and_data_are_written(self):
        out = self.build()
        for name in PAGE_FILES:
            self.assertTrue((out / name).is_file(), name)
        self.assertTrue((out / "assets" / "calc.js").is_file())
        self.assertTrue((out / "assets" / "style.css").is_file())
        for name in render.DATA_FILES:
            self.assertTrue((out / "data" / name).is_file(), name)

    def test_same_data_gives_same_bytes(self):
        first = self.build()
        second = self.build()
        for name in PAGE_FILES:
            self.assertEqual(self.page(first, name), self.page(second, name), name)

    def test_missing_data_file_names_the_file(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        write_data(tmp / "in")
        (tmp / "in" / "nba_recent.json").unlink()
        with self.assertRaisesRegex(SiteDataError, "nba_recent.json"):
            render_site(tmp / "in", tmp / "out")

    def test_data_already_in_output_is_not_copied_onto_itself(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        write_data(tmp / "site" / "data")
        render_site(tmp / "site" / "data", tmp / "site")
        self.assertTrue((tmp / "site" / "index.html").is_file())


class ContentTests(SiteTestCase):
    def test_dynamic_values_are_escaped(self):
        out = self.build()
        nba = self.page(out, "nba.html")
        self.assertIn("ESPN BET &lt;b&gt;", nba)
        self.assertNotIn("ESPN BET <b>", nba)

    def test_headline_numbers_come_from_the_data(self):
        index = self.page(self.build(), "index.html")
        for expected in ("0.2113 vs 0.2087", "-0.0728", "[-0.0836, -0.0623]", "10,967", "40 of 44", "42"):
            self.assertIn(expected, index)

    def test_soccer_gain_without_interval_note_still_shows_the_difference(self):
        validation = load_fixture("soccer_validation.json")
        for stage in validation["stages"]:
            stage["notes"] = []
        index = self.page(self.build(**{"soccer_validation.json": validation}), "index.html")
        self.assertIn("-0.0728", index)
        self.assertNotIn("[-0.0836", index)

    def test_upcoming_row_shows_served_model_market_and_home_line(self):
        nba = self.page(self.build(), "nba.html")
        self.assertIn("NYK at BOS", nba)
        self.assertIn("Elo + market blend", nba)
        self.assertIn("65.2%", nba)
        self.assertIn("BOS -4.5", nba)
        self.assertIn("DEN at LAL", nba)
        self.assertIn("Elo + Platt", nba)

    def test_no_games_shows_message_with_newest_completed(self):
        upcoming = load_fixture("nba_upcoming.json")
        upcoming["games"] = []
        nba = self.page(self.build(**{"nba_upcoming.json": upcoming}), "nba.html")
        self.assertIn("No upcoming NBA games", nba)
        self.assertIn("2026-11-17", nba)
        self.assertNotIn("<caption>Upcoming NBA games", nba)

    def test_no_games_and_no_completed_game_still_explains(self):
        upcoming = load_fixture("nba_upcoming.json")
        upcoming.update(games=[], newest_completed=None)
        nba = self.page(self.build(**{"nba_upcoming.json": upcoming}), "nba.html")
        self.assertIn("holds no completed games", nba)

    def test_scoreboard_without_paired_interval_says_when_it_appears(self):
        board = load_fixture("nba_scoreboard.json")
        board["paired_brier_diff"] = None
        nba = self.page(self.build(**{"nba_scoreboard.json": board}), "nba.html")
        self.assertIn("paired interval appears once 30", nba)
        self.assertNotIn("Paired Brier difference", nba)

    def test_scoreboard_with_paired_interval_reads_it(self):
        nba = self.page(self.build(), "nba.html")
        self.assertIn("+0.0026 [-0.0041, +0.0094]", nba)
        self.assertIn("no measurable difference", nba)
        self.assertIn("Predicted before tip-off: 40 of 44", nba)
        self.assertIn("First prediction on 2026-10-22", nba)

    def test_empty_scoreboard_has_no_blank_table(self):
        board = load_fixture("nba_scoreboard.json")
        board.update(
            n_graded=0,
            n_market=0,
            served={"brier": None, "accuracy": None},
            market={"brier": None, "accuracy": None},
            paired_brier_diff=None,
            first_prediction_at=None,
            last_graded_at=None,
            predicted_before_tipoff={"n": 0, "of": 0},
        )
        out = self.build(**{"nba_scoreboard.json": board})
        self.assertIn("No live predictions have been graded yet", self.page(out, "nba.html"))
        self.assertIn(">n/a<", self.page(out, "index.html"))

    def test_validation_takeaway_names_best_and_the_poisson_gap(self):
        soccer = self.page(self.build(), "soccer.html")
        self.assertIn("Lowest Brier score is Elo ordinal draw at 0.5127", soccer)
        self.assertIn("beats the legacy Poisson model by 0.0728 Brier", soccer)

    def test_ranking_lists_top_25_at_most(self):
        soccer = self.page(self.build(), "soccer.html")
        self.assertIn("Top 6 national teams by Elo rating", soccer)

    def test_footer_attribution_on_every_page(self):
        out = self.build()
        for name in PAGE_FILES:
            html = self.page(out, name)
            self.assertIn("Schedule and betting-line data: sportsdataverse/hoopR-nba-data (CC BY 4.0).", html)
            self.assertIn("martj42/international_results (CC0)", html)


class EmbeddingTests(SiteTestCase):
    def test_small_ratings_are_embedded_and_cannot_close_the_script_tag(self):
        soccer = self.page(self.build(), "soccer.html")
        self.assertEqual(soccer.count('id="ratings-data"'), 1)
        self.assertNotIn("</script><b>x", soccer)
        payload = soccer.split('id="ratings-data">', 1)[1].split("</script>", 1)[0]
        self.assertEqual(json.loads(payload)["teams"][5]["name"], "</script><b>x")

    def test_large_ratings_are_loaded_by_fetch(self):
        with mock.patch.object(render, "EMBED_LIMIT_BYTES", 10):
            soccer = self.page(self.build(), "soccer.html")
        self.assertNotIn("ratings-data", soccer)
        self.assertIn("assets/calc.js", soccer)


class ServedSiteTests(SiteTestCase):
    def test_pages_are_served_and_leak_no_external_requests(self):
        out = self.build()
        handler = partial(QuietHandler, directory=str(out))
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 5)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = f"http://127.0.0.1:{server.server_address[1]}/"
        for name in PAGE_FILES + ("assets/style.css", "assets/calc.js", "data/soccer_ratings.json"):
            with urllib.request.urlopen(base + name, timeout=10) as response:
                self.assertEqual(response.status, 200, name)
                body = response.read().decode("utf-8")
            if name.endswith(".html"):
                collector = LinkCollector()
                collector.feed(body)
                for tag, key, value in collector.links:
                    if value.startswith(("http://", "https://")):
                        self.assertEqual(tag, "a", f"{name} loads {value}")
                        self.assertTrue(value.startswith(render.REPO_URL), f"{name} links to {value}")
                    self.assertFalse(tag == "script" and key == "src" and ":" in value, f"{name} {value}")
                self.assertNotIn("@import", body)


@unittest.skipUnless(Path(NODE).exists() or shutil.which(NODE), "node is not installed")
class CalculatorTests(SiteTestCase):
    SCRIPT = """
const fs = require('fs');
const calc = require(process.argv[2]);
const html = fs.readFileSync(process.argv[3], 'utf8');
const payload = html.split('id="ratings-data">')[1].split('</script>')[0];
const data = JSON.parse(payload);
const [i, j] = [Number(process.argv[4]), Number(process.argv[5])];
console.log(JSON.stringify({
  neutral: calc.matchup(data, i, j, 'neutral'),
  a_home: calc.matchup(data, i, j, 'a_home'),
  b_home: calc.matchup(data, i, j, 'b_home'),
  same: calc.matchup(data, i, i, 'neutral'),
  search: calc.filterTeams(data.teams, 'cote'),
  none: calc.filterTeams(data.teams, 'zzz'),
  pct: calc.percent(5230),
}));
"""

    def test_lookup_returns_the_right_triple_for_every_venue(self):
        out = self.build()
        ratings = make_ratings()["teams"]
        i, j = 1, 3
        script = out.parent / "check.js"
        script.write_text(self.SCRIPT, encoding="utf-8")
        run = subprocess.run(
            [NODE, str(script), str(out / "assets" / "calc.js"), str(out / "soccer.html"), str(i), str(j)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        got = json.loads(run.stdout)

        ri, rj = ratings[i]["rating"], ratings[j]["rating"]
        neutral = outcome_triple(ri, rj, 0)
        a_home = outcome_triple(ri, rj, 100)
        b_home_for_j = outcome_triple(rj, ri, 100)
        self.assertEqual(got["neutral"], {"a": neutral[0], "draw": neutral[1], "b": neutral[2]})
        self.assertEqual(got["a_home"], {"a": a_home[0], "draw": a_home[1], "b": a_home[2]})
        self.assertEqual(got["b_home"], {"a": b_home_for_j[2], "draw": b_home_for_j[1], "b": b_home_for_j[0]})
        self.assertGreater(got["a_home"]["a"], got["neutral"]["a"])
        self.assertGreater(got["b_home"]["b"], got["neutral"]["b"])
        for triple in (got["neutral"], got["a_home"], got["b_home"]):
            self.assertEqual(sum(triple.values()), 10000)
        self.assertIsNone(got["same"])
        self.assertEqual(got["search"], [3])
        self.assertEqual(got["none"], [])
        self.assertEqual(got["pct"], "52.3%")


if __name__ == "__main__":
    unittest.main()
