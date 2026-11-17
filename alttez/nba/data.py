"""Free NBA data: hoopR schedule and closing lines, plus the ESPN scoreboard.

Data sources and licences:

* hoopR-nba-data (sportsdataverse, CC BY 4.0, attribution required): the schedule
  master and the odds-API consensus closing lines.
* ESPN public scoreboard JSON (keyless): upcoming games with the DraftKings line.

The join key everywhere is the ESPN game id. Teams are identified by ESPN franchise
id, which is stable across relocations and renames (id 3 is the Hornets and later the
Pelicans, 25 the SuperSonics and later the Thunder, 29 the Grizzlies, and so on).

Spread convention: ``home_margin_line`` is the market-expected home margin, so a
positive value means the home team is favoured. It equals the hoopR column
``home_team_spread`` and the negation of ``home_point`` (the sportsbook "home line").
"""

from __future__ import annotations

import gzip
import io
import json
import os
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import IntEnum
from pathlib import Path
from typing import Any

import pandas as pd

SCHEDULE_URL = (
    "https://raw.githubusercontent.com/sportsdataverse/hoopR-nba-data/main/nba/schedules/nba_schedule_master.parquet"
)
CLOSING_LINES_URL = (
    "https://raw.githubusercontent.com/sportsdataverse/hoopR-nba-data/main/nba/betting_lines/closing_lines_odds_api.parquet"
)
ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
USER_AGENT = "Mozilla/5.0 (compatible; alttez-nba/1.0)"
DEFAULT_CACHE_DIR = Path("artifacts") / "nba"
CACHE_MAX_AGE_HOURS = 12.0
ATTRIBUTION = "Schedule and betting-line data: sportsdataverse/hoopR-nba-data (CC BY 4.0)."

NBA_TEAM_IDS = frozenset(range(1, 31))
LOCAL_DATE_OFFSET = pd.Timedelta(hours=10)


class NbaDataError(RuntimeError):
    """A download, cache or payload problem. Never swallowed into a fallback."""


class SeasonType(IntEnum):
    REGULAR = 2
    PLAYOFFS = 3
    PLAY_IN = 5


GAME_COLUMNS = (
    "game_id", "start_time", "game_date", "season", "season_type",
    "home_id", "away_id", "home_score", "away_score", "neutral",
)
LINE_COLUMNS = ("game_id", "home_margin_line", "total_line", "n_books")


@dataclass(frozen=True, slots=True)
class NbaSchedule:
    """Completed games (typed frame, one row per game) and season bookkeeping."""

    games: pd.DataFrame
    complete_seasons: tuple[int, ...]
    in_progress_season: int | None


@dataclass(frozen=True, slots=True)
class NbaScheduledGame:
    game_id: int
    start_time: datetime
    season: int
    season_type: int
    home_id: int
    away_id: int
    home_abbreviation: str
    away_abbreviation: str
    neutral: bool
    state: str
    home_score: int | None
    away_score: int | None
    home_margin_line: float | None
    total_line: float | None
    odds_provider: str | None


def _read_url(url: str, *, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise NbaDataError(f"download failed for {url}: {exc}") from exc
    if body[:2] == b"\x1f\x8b":
        try:
            body = gzip.decompress(body)
        except OSError as exc:
            raise NbaDataError(f"bad gzip body from {url}: {exc}") from exc
    return body


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def load_cached_parquet(
    url: str,
    cache_path: Path,
    *,
    max_age_hours: float = CACHE_MAX_AGE_HOURS,
    timeout: float = 60.0,
    now: float | None = None,
) -> pd.DataFrame:
    """Read a parquet file, refreshing the cache copy when older than ``max_age_hours``.

    The download is parsed before it replaces the cache, so a truncated or HTML body
    cannot poison it. Any failure raises :class:`NbaDataError`.
    """
    current = time.time() if now is None else now
    if cache_path.exists() and current - cache_path.stat().st_mtime < max_age_hours * 3600:
        return pd.read_parquet(cache_path)
    body = _read_url(url, timeout=timeout)
    try:
        frame = pd.read_parquet(io.BytesIO(body))
    except Exception as exc:
        raise NbaDataError(f"{url} did not return a readable parquet file: {exc}") from exc
    _atomic_write(cache_path, body)
    return frame


def local_game_date(start_utc: pd.Series) -> pd.Series:
    """US calendar date of a tip-off: UTC minus ten hours, so a 00:30Z game belongs to the evening before."""
    return (start_utc - LOCAL_DATE_OFFSET).dt.normalize()


def normalise_schedule(raw: pd.DataFrame) -> NbaSchedule:
    """Keep NBA-franchise games and return the completed ones with stable dtypes.

    Dropped: all-star and exhibition teams (id outside 1..30), duplicated ids,
    postponed games, and the 0-0 rows of the cancelled 1998-99 lockout games.
    """
    needed = {
        "id", "date", "season", "season_type", "status_type_completed", "status_type_state",
        "neutral_site", "home_id", "away_id", "home_score", "away_score",
    }
    missing = needed - set(raw.columns)
    if missing:
        raise NbaDataError(f"schedule is missing columns: {sorted(missing)}")
    frame = raw[raw["home_id"].isin(NBA_TEAM_IDS) & raw["away_id"].isin(NBA_TEAM_IDS)]
    frame = frame[frame["season_type"].isin([int(t) for t in SeasonType])].drop_duplicates("id")
    done = frame[frame["status_type_completed"].astype(bool)]
    done = done[(done["home_score"] != done["away_score"]) & (done[["home_score", "away_score"]].sum(axis=1) > 0)]
    start = pd.to_datetime(done["date"], utc=True).dt.tz_localize(None)
    games = pd.DataFrame(
        {
            "game_id": done["id"].astype("int64"),
            "start_time": start,
            "game_date": local_game_date(start),
            "season": done["season"].astype("int64"),
            "season_type": done["season_type"].astype("int64"),
            "home_id": done["home_id"].astype("int64"),
            "away_id": done["away_id"].astype("int64"),
            "home_score": done["home_score"].astype("int64"),
            "away_score": done["away_score"].astype("int64"),
            "neutral": done["neutral_site"].astype(bool),
        }
    )
    games = games.sort_values(["game_date", "start_time", "game_id"], kind="stable").reset_index(drop=True)
    if not games["game_id"].is_unique:
        raise NbaDataError("duplicate game ids after normalisation")

    pending = frame[frame["status_type_state"] == "pre"]
    pending_seasons = set(pending["season"].astype(int))
    playoff_seasons = set(games.loc[games["season_type"] == SeasonType.PLAYOFFS, "season"].astype(int))
    complete = tuple(sorted(playoff_seasons - pending_seasons))
    in_progress = max(pending_seasons, default=None)
    return NbaSchedule(games=games, complete_seasons=complete, in_progress_season=in_progress)


def normalise_closing_lines(raw: pd.DataFrame) -> pd.DataFrame:
    """Closing lines keyed by game id; ``home_margin_line > 0`` means the home team is favoured."""
    needed = {"game_id", "home_team_spread", "over_under", "n_books"}
    missing = needed - set(raw.columns)
    if missing:
        raise NbaDataError(f"closing lines are missing columns: {sorted(missing)}")
    lines = pd.DataFrame(
        {
            "game_id": raw["game_id"].astype("int64"),
            "home_margin_line": raw["home_team_spread"].astype(float),
            "total_line": raw["over_under"].astype(float),
            "n_books": raw["n_books"].astype("int64"),
        }
    )
    if not lines["game_id"].is_unique:
        raise NbaDataError("duplicate game ids in closing lines")
    if lines["home_margin_line"].isna().any():
        raise NbaDataError("closing lines contain missing spreads")
    return lines.reset_index(drop=True)


def attach_lines(games: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    """Left join: games without a closing line keep NaN in the line columns."""
    return games.merge(lines, on="game_id", how="left", validate="one_to_one")


def load_schedule(cache_dir: Path = DEFAULT_CACHE_DIR, **kwargs: Any) -> NbaSchedule:
    return normalise_schedule(load_cached_parquet(SCHEDULE_URL, cache_dir / "schedule.parquet", **kwargs))


def load_closing_lines(cache_dir: Path = DEFAULT_CACHE_DIR, **kwargs: Any) -> pd.DataFrame:
    return normalise_closing_lines(load_cached_parquet(CLOSING_LINES_URL, cache_dir / "closing_lines.parquet", **kwargs))


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise NbaDataError(f"non-numeric value in ESPN payload: {value!r}") from exc


def _home_margin_line(odds: dict[str, Any]) -> float | None:
    """ESPN's ``spread`` is the home sportsbook line (negative when the home team is favoured)."""
    spread = _number(odds.get("spread"))
    if spread is None:
        return None
    if spread != 0:
        home_favourite = (odds.get("homeTeamOdds") or {}).get("favorite")
        if home_favourite is not None and bool(home_favourite) != (spread < 0):
            raise NbaDataError(f"spread {spread} contradicts the favourite flags: {odds.get('details')!r}")
    return -spread + 0.0


def _parse_event(event: dict[str, Any]) -> NbaScheduledGame:
    try:
        competition = event["competitions"][0]
        sides = {c["homeAway"]: c for c in competition["competitors"]}
        home, away = sides["home"], sides["away"]
        state = competition["status"]["type"]["state"]
        start = datetime.strptime(event["date"], "%Y-%m-%dT%H:%MZ").replace(tzinfo=UTC)
        season = event["season"]
        odds_list = competition.get("odds") or []
        odds = odds_list[0] if odds_list else {}
        started = state != "pre"
        return NbaScheduledGame(
            game_id=int(event["id"]),
            start_time=start,
            season=int(season["year"]),
            season_type=int(season["type"]),
            home_id=int(home["team"]["id"]),
            away_id=int(away["team"]["id"]),
            home_abbreviation=str(home["team"]["abbreviation"]),
            away_abbreviation=str(away["team"]["abbreviation"]),
            neutral=bool(competition.get("neutralSite", False)),
            state=str(state),
            home_score=int(home["score"]) if started else None,
            away_score=int(away["score"]) if started else None,
            home_margin_line=_home_margin_line(odds) if odds else None,
            total_line=_number(odds.get("overUnder")) if odds else None,
            odds_provider=(odds.get("provider") or {}).get("name") if odds else None,
        )
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        if isinstance(exc, NbaDataError):
            raise
        raise NbaDataError(f"unexpected ESPN event shape (id={event.get('id') if isinstance(event, dict) else '?'}): {exc!r}") from exc


def parse_espn_scoreboard(payload: dict[str, Any]) -> list[NbaScheduledGame]:
    """Pure parse of one scoreboard response. Odds are only present on upcoming games."""
    if not isinstance(payload, dict) or "events" not in payload:
        raise NbaDataError("ESPN scoreboard payload has no 'events' key")
    return [_parse_event(event) for event in payload["events"]]


_last_request_at = 0.0


def _get_json(url: str, timeout: float) -> dict[str, Any]:
    return json.loads(_read_url(url, timeout=timeout))


def fetch_espn_scoreboard(
    day: date,
    *,
    retries: int = 3,
    backoff_seconds: float = 2.0,
    min_interval_seconds: float = 0.5,
    timeout: float = 30.0,
    get_json: Callable[[str, float], dict[str, Any]] = _get_json,
    sleep: Callable[[float], None] = time.sleep,
) -> list[NbaScheduledGame]:
    """One date per call (a range returns HTTP 400). Retries with exponential backoff, then raises."""
    global _last_request_at
    url = f"{ESPN_SCOREBOARD_URL}?dates={day:%Y%m%d}"
    last_error: Exception | None = None
    for attempt in range(retries):
        wait = _last_request_at + min_interval_seconds - time.monotonic()
        if wait > 0:
            sleep(wait)
        _last_request_at = time.monotonic()
        try:
            return parse_espn_scoreboard(get_json(url, timeout))
        except (NbaDataError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                sleep(backoff_seconds * 2**attempt)
    raise NbaDataError(f"ESPN scoreboard for {day} failed after {retries} attempts: {last_error}") from last_error
