"""Render the static ALTTEZ showcase from the six JSON data files.

``render_site(data_dir, out_dir)`` writes the HTML pages, copies the assets and
mirrors the data files into ``out_dir/data``. Pages are ``string.Template`` files
in ``site/templates``. Every dynamic value is escaped, either by ``fmt.esc`` or by
the table builder, which escapes any cell that is not marked as ``Html``.
"""

from __future__ import annotations

import base64
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any

from alttez.site import fmt

ROOT = Path(__file__).resolve().parents[2]
SITE_DIR = ROOT / "site"
TEMPLATE_DIR = SITE_DIR / "templates"
ASSET_DIR = SITE_DIR / "assets"

REPO_URL = "https://github.com/julianhr-ops/alttez-predictive"
SITE_URL = "https://julianhr-ops.github.io/alttez-predictive/"
SOCCER_SOURCE_NOTE = "martj42/international_results (CC0)"

DATA_FILES = (
    "nba_upcoming.json",
    "nba_scoreboard.json",
    "nba_recent.json",
    "nba_validation.json",
    "soccer_ratings.json",
    "soccer_validation.json",
)
EMBED_LIMIT_BYTES = 1_500_000
RANKING_ROWS = 25
MIN_PAIRED_GAMES = 30

_FAVICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
    '<rect width="32" height="32" rx="6" fill="#0b57d0"/>'
    '<path d="M8 24 16 8 24 24Z" fill="none" stroke="#fff" stroke-width="3" stroke-linejoin="round"/>'
    "</svg>"
)
FAVICON_URI = "data:image/svg+xml;base64," + base64.b64encode(_FAVICON_SVG.encode()).decode()

_CI_NOTE = re.compile(r"Brier\s+([+-]?\d+\.\d+)\s+\[([+-]?\d+\.\d+),\s*([+-]?\d+\.\d+)\]")
_NUM = ' class="num"'

PAGES = (
    ("index.html", "Overview"),
    ("nba.html", "NBA"),
    ("soccer.html", "Soccer"),
    ("method.html", "Method"),
)


class SiteDataError(RuntimeError):
    """A data file is missing or does not match the contract."""


class Html(str):
    """Markup that is already escaped and must not be escaped again."""


@dataclass(frozen=True)
class SiteData:
    upcoming: dict[str, Any]
    scoreboard: dict[str, Any]
    recent: dict[str, Any]
    nba_validation: dict[str, Any]
    ratings: dict[str, Any]
    soccer_validation: dict[str, Any]
    ratings_bytes: int


def load_data(data_dir: Path) -> SiteData:
    missing = [name for name in DATA_FILES if not (data_dir / name).is_file()]
    if missing:
        raise SiteDataError(f"missing data files in {data_dir}: {', '.join(missing)}")
    docs: dict[str, Any] = {}
    for name in DATA_FILES:
        try:
            docs[name] = json.loads((data_dir / name).read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SiteDataError(f"{name} is not valid JSON: {exc}") from exc
    return SiteData(
        upcoming=docs["nba_upcoming.json"],
        scoreboard=docs["nba_scoreboard.json"],
        recent=docs["nba_recent.json"],
        nba_validation=docs["nba_validation.json"],
        ratings=docs["soccer_ratings.json"],
        soccer_validation=docs["soccer_validation.json"],
        ratings_bytes=(data_dir / "soccer_ratings.json").stat().st_size,
    )


def table(
    caption: str,
    columns: list[tuple[str, bool]],
    rows: list[list[Any]],
    best_rows: frozenset[int] = frozenset(),
) -> Html:
    """Accessible table. ``columns`` is ``(label, numeric)``. The first cell of a row is its header."""

    def cell(value: Any) -> str:
        return value if isinstance(value, Html) else fmt.esc(value)

    head = "".join(
        f'<th scope="col"{_NUM if numeric else ""}>{fmt.esc(label)}</th>' for label, numeric in columns
    )
    body = []
    for index, row in enumerate(rows):
        cells = [f'<th scope="row">{cell(row[0])}</th>']
        for (_, numeric), value in zip(columns[1:], row[1:], strict=True):
            cells.append(f'<td{_NUM if numeric else ""}>{cell(value)}</td>')
        css = ' class="best"' if index in best_rows else ""
        body.append(f"<tr{css}>{''.join(cells)}</tr>")
    return Html(
        f'<div class="table-wrap"><table><caption>{fmt.esc(caption)}</caption>'
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
    )


def bullet_list(items: list[str]) -> Html:
    if not items:
        return Html("")
    return Html('<ul class="tight">' + "".join(f"<li>{fmt.esc(item)}</li>" for item in items) + "</ul>")


def notice(message: str) -> Html:
    return Html(f'<p class="notice" role="status">{fmt.esc(message)}</p>')


def prob_cell(p: float) -> Html:
    return Html(
        f'<div class="prob"><div class="bar" aria-hidden="true"><span style="width:{fmt.pct_width(p)}"></span></div>'
        f"<b>{fmt.probability(p)}</b></div>"
    )


def _model_name(value: str) -> str:
    return {"blend": "Elo + market blend", "elo_platt": "Elo + Platt"}.get(value, value)


def _best_index(rows: list[dict[str, Any]], key: str, lowest: bool) -> int:
    values = [row[key] for row in rows]
    return values.index(min(values) if lowest else max(values))


def validation_takeaway(rows: list[dict[str, Any]]) -> str:
    """One plain sentence or two, derived from the table rows only."""
    if not rows:
        return ""
    best = rows[_best_index(rows, "brier", lowest=True)]
    worst = rows[_best_index(rows, "brier", lowest=False)]
    text = (
        f"Lowest Brier score is {best['model']} at {fmt.brier(best['brier'])}. "
        f"Highest is {worst['model']} at {fmt.brier(worst['brier'])}."
    )
    poisson = next((r for r in rows if r["model"].lower().startswith("poisson raw")), None)
    elo_rows = [r for r in rows if r["model"].lower().startswith("elo")]
    if poisson and elo_rows:
        elo = min(elo_rows, key=lambda r: r["brier"])
        if elo["brier"] < poisson["brier"]:
            gap = poisson["brier"] - elo["brier"]
            text += f" {elo['model']} beats the legacy Poisson model by {gap:.4f} Brier."
    return text


def validation_table(caption: str, rows: list[dict[str, Any]], columns: list[tuple[str, str, Any]]) -> Html:
    """Model table. ``columns`` is ``(label, row key, formatter)``."""
    heads = [("Model", False)] + [(label, True) for label, _, _ in columns]
    body = [[row["model"]] + [formatter(row[key]) for _, key, formatter in columns] for row in rows]
    best = frozenset({_best_index(rows, "brier", lowest=True)}) if rows else frozenset()
    return table(caption, heads, body, best)


NBA_COLUMNS = [
    ("Brier", "brier", fmt.brier),
    ("Log loss", "log_loss", fmt.log_loss),
    ("Accuracy", "accuracy", fmt.accuracy),
    ("ECE", "ece", fmt.ece),
]
SOCCER_COLUMNS = [
    ("Brier", "brier", fmt.brier),
    ("RPS", "rps", fmt.brier),
    ("Log loss", "log_loss", fmt.log_loss),
    ("Accuracy", "accuracy", fmt.accuracy),
]


def _command_line(source_command: str | list[str]) -> str:
    commands = [source_command] if isinstance(source_command, str) else source_command
    return "".join(f"<code>{fmt.esc(command)}</code> " for command in commands).strip()


def nba_upcoming_html(upcoming: dict[str, Any]) -> Html:
    games = upcoming["games"]
    stamp = f'<p class="muted small">Updated {fmt.esc(fmt.utc_minute(upcoming["generated_at"]))}.</p>'
    if not games:
        newest = upcoming.get("newest_completed")
        tail = (
            f"The newest completed game in the data was played on {newest}."
            if newest
            else "The data holds no completed games yet."
        )
        message = (
            "No upcoming NBA games are scheduled right now. This is normal in the offseason or on days "
            f"without games. {tail}"
        )
        return Html(str(notice(message)) + stamp)
    rows = []
    for game in games:
        rows.append(
            [
                fmt.utc_minute(game["tipoff_utc"]),
                f"{game['away_abbr']} at {game['home_abbr']}",
                prob_cell(game["p_home_served"]),
                _model_name(game["served_model"]),
                fmt.probability(game["p_home_elo"]),
                fmt.probability(game["p_home_market"]),
                fmt.home_line(game["home_abbr"], game["market_spread"], game["market_source"]),
            ]
        )
    columns = [
        ("Tip-off", False),
        ("Matchup", False),
        ("Home win, served", False),
        ("Served model", False),
        ("Home win, Elo only", True),
        ("Home win, market", True),
        ("Home line (source)", False),
    ]
    return Html(str(table("Upcoming NBA games with the served home-win probability", columns, rows)) + stamp)


def nba_scoreboard_html(board: dict[str, Any]) -> Html:
    parts: list[str] = []
    if board["n_graded"] == 0:
        parts.append(str(notice("No live predictions have been graded yet. Scores appear after the first games end.")))
    else:
        rows = [
            ["Served prediction", fmt.brier(board["served"]["brier"]), fmt.accuracy(board["served"]["accuracy"])],
            ["Market close", fmt.brier(board["market"]["brier"]), fmt.accuracy(board["market"]["accuracy"])],
        ]
        columns = [("Source", False), ("Brier", True), ("Accuracy", True)]
        parts.append(str(table("Live scoreboard, served versus market", columns, rows)))
        parts.append(
            f"<p>{fmt.esc(fmt.count(board['n_graded']))} games graded, "
            f"{fmt.esc(fmt.count(board['n_market']))} of them with a market line. "
            "Lower Brier is better.</p>"
        )
        paired = board.get("paired_brier_diff")
        if paired:
            mean, lo, hi = paired["mean"], paired["lo"], paired["hi"]
            if lo > 0:
                verdict = "The interval is above 0, so the served model had the higher Brier score and the market did better."
            elif hi < 0:
                verdict = "The interval is below 0, so the served model had the lower Brier score."
            else:
                verdict = "The interval contains 0, so there is no measurable difference yet."
            parts.append(
                "<p>Paired Brier difference, served minus market, 95% interval "
                f"{fmt.esc(fmt.interval(mean, lo, hi))}. Negative means the served model had the lower Brier score. "
                f"{fmt.esc(verdict)}</p>"
            )
        elif board["n_market"] < MIN_PAIRED_GAMES:
            needed = MIN_PAIRED_GAMES - board["n_market"]
            parts.append(
                f'<p class="muted">No paired interval yet. It needs {MIN_PAIRED_GAMES} graded games with a market line '
                f"and there are {fmt.esc(fmt.count(board['n_market']))}, so {fmt.esc(fmt.count(needed))} more are needed.</p>"
            )
        else:
            parts.append('<p class="muted">The paired interval is not available in this data file.</p>')
    before = board["predicted_before_tipoff"]
    first = board.get("first_prediction_at")
    line = f"Predicted before tip-off: {fmt.count(before['n'])} of {fmt.count(before['of'])}."
    if first:
        line += f" First prediction on {fmt.utc_day(first)}."
    if board.get("last_graded_at"):
        line += f" Last graded {fmt.utc_minute(board['last_graded_at'])}."
    parts.append(f"<p>{fmt.esc(line)}</p>")
    parts.append(f'<p class="muted small">Updated {fmt.esc(fmt.utc_minute(board["generated_at"]))}.</p>')
    return Html("".join(parts))


def nba_recent_html(recent: dict[str, Any]) -> Html:
    games = recent["games"]
    if not games:
        return notice("No graded games yet.")
    rows = [
        [
            game["game_date"],
            f"{game['away_abbr']} at {game['home_abbr']}",
            f"{game['away_abbr']} {game['away_score']}, {game['home_abbr']} {game['home_score']}",
            "Home" if game["home_win"] else "Away",
            _model_name(game["served_model"]),
            fmt.probability(game["p_home_served"]),
            fmt.probability(game["p_home_market"]),
            fmt.brier(game["brier"]),
        ]
        for game in games
    ]
    columns = [
        ("Date", False),
        ("Matchup", False),
        ("Final score", False),
        ("Winner", False),
        ("Served model", False),
        ("Home win, served", True),
        ("Home win, market", True),
        ("Brier", True),
    ]
    return table(f"Last {len(games)} graded games, newest first", columns, rows)


def nba_validation_html(validation: dict[str, Any]) -> Html:
    rows = validation["rows"]
    intro = (
        f"<p>Test seasons {fmt.esc(validation['test_seasons'])}. {fmt.esc(fmt.count(validation['n_games']))} games, "
        f"{fmt.esc(fmt.count(validation['n_with_line']))} with a closing line. Reproduce with "
        f"{_command_line(validation['source_command'])}</p>"
    )
    tbl = validation_table("NBA walk-forward validation", rows, NBA_COLUMNS)
    takeaway = f'<p class="takeaway">{fmt.esc(validation_takeaway(rows))}</p>'
    return Html(intro + str(tbl) + takeaway + str(bullet_list(validation["findings"])))


def soccer_validation_html(validation: dict[str, Any]) -> Html:
    parts = []
    for stage in validation["stages"]:
        parts.append(f"<h3>{fmt.esc(stage['name'])}</h3>")
        parts.append(
            f"<p>Years {fmt.esc(stage['years'])}. {fmt.esc(fmt.count(stage['n_matches']))} identical matches.</p>"
        )
        caption = f"{stage['name']}, years {stage['years']}"
        parts.append(str(validation_table(caption, stage["rows"], SOCCER_COLUMNS)))
        parts.append(f'<p class="takeaway">{fmt.esc(validation_takeaway(stage["rows"]))}</p>')
        parts.append(str(bullet_list(stage["notes"])))
    parts.append(f"<p class=\"muted\">Reproduce with {_command_line(validation['source_command'])}</p>")
    return Html("".join(parts))


def ranking_html(ratings: dict[str, Any]) -> Html:
    rows = [[team["rank"], team["name"], fmt.rating(team["rating"])] for team in ratings["teams"][:RANKING_ROWS]]
    columns = [("Rank", False), ("Team", False), ("Elo rating", True)]
    return table(f"Top {len(rows)} national teams by Elo rating", columns, rows)


def _has_prefix(model: str, prefix: str) -> bool:
    return model.lower().startswith(prefix)


def soccer_gain(validation: dict[str, Any]) -> dict[str, Any] | None:
    """Elo ordinal minus legacy Poisson Brier on the held-out stage, with its interval when a note carries it."""
    for stage in validation["stages"]:
        elo = next((r for r in stage["rows"] if _has_prefix(r["model"], "elo ordinal")), None)
        poisson = next((r for r in stage["rows"] if _has_prefix(r["model"], "poisson raw")), None)
        if elo is None or poisson is None:
            continue
        gain: dict[str, Any] = {
            "diff": round(elo["brier"] - poisson["brier"], 4),
            "years": stage["years"],
            "n": stage["n_matches"],
            "interval": None,
        }
        for note in stage["notes"]:
            lowered = note.lower()
            match = _CI_NOTE.search(note)
            if match and "elo ordinal" in lowered and "poisson raw" in lowered:
                gain["interval"] = tuple(float(g) for g in match.groups())
                break
        return gain
    return None


def index_stats_html(data: SiteData) -> Html:
    board = data.scoreboard
    served, market = board["served"]["brier"], board["market"]["brier"]
    if served is None or market is None:
        nba_value, nba_note = fmt.MISSING, "No live graded games with a market line yet."
    else:
        nba_value = f"{fmt.brier(served)} vs {fmt.brier(market)}"
        nba_note = "Served prediction against the market close on live graded games. Lower is better."
    before = board["predicted_before_tipoff"]
    gain = soccer_gain(data.soccer_validation)
    if gain is None:
        soccer_value, soccer_note = fmt.MISSING, "The held-out stage is not in the validation file."
    else:
        soccer_value = fmt.signed(gain["diff"])
        if gain["interval"]:
            _, lo, hi = gain["interval"]
            ci = f"95% interval [{fmt.signed(lo)}, {fmt.signed(hi)}]."
        else:
            ci = "Interval is in the soccer validation report."
        soccer_note = (
            f"Brier, Elo ordinal rule minus legacy Poisson, years {gain['years']}. {ci} "
            "Negative means Elo ordinal is better."
        )
    nba_games = data.nba_validation["n_games"]
    total = nba_games + (gain["n"] if gain else 0)
    cards = [
        ("NBA served Brier against market close", nba_value, nba_note),
        (
            "NBA games graded live",
            fmt.count(board["n_graded"]),
            f"Predicted before tip-off: {fmt.count(before['n'])} of {fmt.count(before['of'])}.",
        ),
        ("Soccer gain over the Poisson model", soccer_value, soccer_note),
        (
            "Walk-forward test matches",
            fmt.count(total),
            f"{fmt.count(nba_games)} NBA games"
            + (f" and {fmt.count(gain['n'])} held-out soccer matches." if gain else "."),
        ),
    ]
    return Html(
        "".join(
            f'<div class="stat"><dt>{fmt.esc(label)}</dt><dd class="value">{fmt.esc(value)}</dd>'
            f'<dd class="note">{fmt.esc(note)}</dd></div>'
            for label, value, note in cards
        )
    )


def ratings_script(ratings: dict[str, Any], size: int) -> Html:
    """Embed the ratings so the page works from a file and needs no second request, when small enough."""
    if size >= EMBED_LIMIT_BYTES:
        return Html("")
    payload = json.dumps(ratings, separators=(",", ":")).replace("<", "\\u003c")
    return Html(f'<script type="application/json" id="ratings-data">{payload}</script>')


def _nav(current: str) -> str:
    items = []
    for file, label in PAGES:
        mark = ' aria-current="page"' if file == current else ""
        items.append(f'<li><a href="{file}"{mark}>{fmt.esc(label)}</a></li>')
    return "\n".join(items)


def _read_template(name: str) -> Template:
    return Template((TEMPLATE_DIR / name).read_text(encoding="utf-8"))


def _page(file: str, title: str, description: str, body: str, scripts: str = "") -> str:
    return _read_template("base.html").substitute(
        title=fmt.esc(title),
        description=fmt.esc(description),
        page_file="" if file == "index.html" else file,
        favicon=FAVICON_URI,
        nav=_nav(file),
        body=body,
        scripts=scripts,
        soccer_source=fmt.esc(SOCCER_SOURCE_NOTE),
        repo=REPO_URL,
        site_url=SITE_URL,
    )


def render_pages(data: SiteData) -> dict[str, str]:
    common = {"repo": REPO_URL}
    body = _read_template
    ratings = data.ratings
    pages = {
        "index.html": _page(
            "index.html",
            "ALTTEZ predictive engine",
            "Win probabilities for international football and NBA games, scored walk-forward "
            "against baselines. Negative results kept.",
            body("index.html").substitute(common, stats=index_stats_html(data)),
        ),
        "nba.html": _page(
            "nba.html",
            "NBA predictions and live scoreboard | ALTTEZ",
            "Upcoming NBA game probabilities, a live scoreboard against the betting line, and the walk-forward validation.",
            body("nba.html").substitute(
                common,
                upcoming=nba_upcoming_html(data.upcoming),
                scoreboard=nba_scoreboard_html(data.scoreboard),
                recent=nba_recent_html(data.recent),
                validation=nba_validation_html(data.nba_validation),
            ),
        ),
        "soccer.html": _page(
            "soccer.html",
            "Soccer validation and matchup calculator | ALTTEZ",
            "Walk-forward validation of the international football 1X2 model, an Elo ranking and a matchup calculator.",
            body("soccer.html").substitute(
                common,
                validation=soccer_validation_html(data.soccer_validation),
                ranking=ranking_html(ratings),
                as_of=fmt.esc(ratings["as_of"]),
                source=fmt.esc(ratings["source"]),
                ratings_data=ratings_script(ratings, data.ratings_bytes),
            ),
            scripts='<script src="assets/calc.js" defer></script>',
        ),
        "method.html": _page(
            "method.html",
            "Evaluation method | ALTTEZ",
            "How ALTTEZ evaluates models: walk-forward tests, baselines on identical games, paired bootstrap and kept negative results.",
            body("method.html").substitute(common),
        ),
    }
    return pages


def render_site(data_dir: Path, out_dir: Path) -> list[Path]:
    data = load_data(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, html in render_pages(data).items():
        path = out_dir / name
        path.write_text(html, encoding="utf-8", newline="\n")
        written.append(path)
    shutil.copytree(ASSET_DIR, out_dir / "assets", dirs_exist_ok=True)
    target = out_dir / "data"
    if data_dir.resolve() != target.resolve():
        target.mkdir(parents=True, exist_ok=True)
        for name in DATA_FILES:
            shutil.copyfile(data_dir / name, target / name)
    (out_dir / ".nojekyll").write_text("", encoding="utf-8")
    return written
