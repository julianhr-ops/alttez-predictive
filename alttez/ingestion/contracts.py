from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


TEAM_METRIC_FIELDS: tuple[str, ...] = (
    "xg",
    "possession",
    "shots_total",
    "shots_on_target",
    "shots_off_target",
    "shots_blocked",
    "shots_inside_box",
    "shots_outside_box",
    "corners",
    "fouls",
    "yellow_cards",
    "red_cards",
    "offsides",
    "saves",
    "passes_completed",
    "pass_accuracy",
)

RAW_HARD_REQUIRED_FIELDS: tuple[str, ...] = (
    "provider",
    "external_match_id",
    "match_date",
    "tournament_name",
    "home_team",
    "away_team",
    "status",
)

CERTIFICATION_MIN_METRICS = 15
CERTIFICATION_MIN_COVERAGE_PCT = 45.0
MATCH_STATUSES = {"scheduled", "in_progress", "finished", "postponed", "cancelled"}


def _strip_text(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    return value


def _coerce_numeric(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        text = value.strip().replace("%", "")
        if not text:
            return None
        text = text.replace(",", ".")
        try:
            numeric = float(text)
        except ValueError:
            return value
        if numeric.is_integer():
            return int(numeric)
        return numeric
    return value


class MatchTeamMetrics(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    xg: float | None = Field(default=None, ge=0)
    possession: float | None = Field(default=None, ge=0, le=100)
    shots_total: int | None = Field(default=None, ge=0)
    shots_on_target: int | None = Field(default=None, ge=0)
    shots_off_target: int | None = Field(default=None, ge=0)
    shots_blocked: int | None = Field(default=None, ge=0)
    shots_inside_box: int | None = Field(default=None, ge=0)
    shots_outside_box: int | None = Field(default=None, ge=0)
    corners: int | None = Field(default=None, ge=0)
    fouls: int | None = Field(default=None, ge=0)
    yellow_cards: int | None = Field(default=None, ge=0)
    red_cards: int | None = Field(default=None, ge=0)
    offsides: int | None = Field(default=None, ge=0)
    saves: int | None = Field(default=None, ge=0)
    passes_completed: int | None = Field(default=None, ge=0)
    pass_accuracy: float | None = Field(default=None, ge=0, le=100)

    @field_validator(*TEAM_METRIC_FIELDS, mode="before")
    @classmethod
    def _normalize_metric_value(cls, value: Any) -> Any:
        return _coerce_numeric(value)

    @model_validator(mode="after")
    def _validate_relationships(self) -> "MatchTeamMetrics":
        if self.shots_total is not None:
            for metric_name in ("shots_on_target", "shots_off_target", "shots_blocked"):
                metric_value = getattr(self, metric_name)
                if metric_value is not None and metric_value > self.shots_total:
                    raise ValueError(f"{metric_name} cannot exceed shots_total")
        return self

    @classmethod
    def metric_names(cls) -> tuple[str, ...]:
        return TEAM_METRIC_FIELDS

    def available_metric_names(self) -> list[str]:
        return [field_name for field_name in TEAM_METRIC_FIELDS if getattr(self, field_name) is not None]

    def available_metric_count(self) -> int:
        return len(self.available_metric_names())


class RawMatchPayload(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    provider: str
    external_match_id: str
    match_date: date
    kickoff_at: datetime | None = None
    tournament_name: str
    home_team: str
    away_team: str
    status: str = "scheduled"
    home_score: int | None = None
    away_score: int | None = None
    season_label: str | None = None
    stage: str | None = None
    competition_class: str | None = None
    is_neutral: bool = False
    home_metrics: MatchTeamMetrics | None = None
    away_metrics: MatchTeamMetrics | None = None
    source_payload: dict[str, Any] = Field(default_factory=dict, alias="payload")
    parsed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator(
        "provider",
        "external_match_id",
        "tournament_name",
        "home_team",
        "away_team",
        "season_label",
        "stage",
        "competition_class",
        "status",
        mode="before",
    )
    @classmethod
    def _normalize_text_fields(cls, value: Any) -> Any:
        normalized = _strip_text(value)
        if isinstance(normalized, str) and not normalized:
            return None
        return normalized

    @field_validator("home_score", "away_score", mode="before")
    @classmethod
    def _normalize_scores(cls, value: Any) -> Any:
        return _coerce_numeric(value)

    @field_validator("status", mode="after")
    @classmethod
    def _normalize_status(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in MATCH_STATUSES:
            raise ValueError(f"Unsupported match status: {value}")
        return normalized

    @model_validator(mode="after")
    def _normalize_payload(self) -> "RawMatchPayload":
        if self.home_score is not None and self.home_score < 0:
            raise ValueError("home_score cannot be negative")
        if self.away_score is not None and self.away_score < 0:
            raise ValueError("away_score cannot be negative")
        if self.status == "finished" and (self.home_score is None or self.away_score is None):
            raise ValueError("finished matches require final scores")
        return self

    def hard_required_missing(self) -> list[str]:
        missing = []
        for field_name in RAW_HARD_REQUIRED_FIELDS:
            value = getattr(self, field_name)
            if value is None or (isinstance(value, str) and not value.strip()):
                missing.append(field_name)
        if self.status == "finished" and (self.home_score is None or self.away_score is None):
            if "home_score" not in missing:
                missing.append("home_score")
            if "away_score" not in missing:
                missing.append("away_score")
        return missing

    def hard_required_complete(self) -> bool:
        return not self.hard_required_missing()

    def available_metric_count(self) -> int:
        total = 0
        if self.home_metrics is not None:
            total += self.home_metrics.available_metric_count()
        if self.away_metrics is not None:
            total += self.away_metrics.available_metric_count()
        return total

    def total_metric_slots(self) -> int:
        return len(TEAM_METRIC_FIELDS) * 2

    def coverage_pct(self) -> float:
        total_slots = self.total_metric_slots()
        if total_slots <= 0:
            return 0.0
        return round((self.available_metric_count() / total_slots) * 100, 2)

    def certification_ready(
        self,
        min_metrics: int = CERTIFICATION_MIN_METRICS,
        min_coverage_pct: float = CERTIFICATION_MIN_COVERAGE_PCT,
    ) -> bool:
        return self.hard_required_complete() and self.available_metric_count() >= min_metrics and self.coverage_pct() >= min_coverage_pct


class CertifiedMatch(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    batch_id: str
    certified_at: datetime
    certified: Literal[True] = True
    certification_status: Literal["certified"] = "certified"
    provider: str
    external_match_id: str
    match_date: date
    kickoff_at: datetime | None = None
    tournament_name: str
    home_team: str
    away_team: str
    status: Literal["finished"]
    home_score: int
    away_score: int
    season_label: str | None = None
    stage: str | None = None
    competition_class: str | None = None
    is_neutral: bool = False
    home_metrics: MatchTeamMetrics | None = None
    away_metrics: MatchTeamMetrics | None = None
    source_payload: dict[str, Any] = Field(default_factory=dict, alias="payload")
    available_metric_count: int
    coverage_pct: float = Field(ge=0, le=100)
    hard_required_missing: list[str] = Field(default_factory=list)

    @classmethod
    def from_raw(
        cls,
        raw: RawMatchPayload,
        batch_id: str,
        *,
        certified_at: datetime | None = None,
        min_metrics: int = CERTIFICATION_MIN_METRICS,
        min_coverage_pct: float = CERTIFICATION_MIN_COVERAGE_PCT,
    ) -> "CertifiedMatch":
        if raw.status != "finished":
            raise ValueError(f"Only finished matches can be certified, got status={raw.status}")

        missing = raw.hard_required_missing()
        if missing:
            raise ValueError(f"Raw payload is missing hard required fields: {', '.join(missing)}")

        available_metric_count = raw.available_metric_count()
        coverage_pct = raw.coverage_pct()
        if available_metric_count < min_metrics:
            raise ValueError(
                f"Match has only {available_metric_count} available metrics, requires at least {min_metrics}"
            )
        if coverage_pct < min_coverage_pct:
            raise ValueError(
                f"Match coverage {coverage_pct:.2f}% is below the certification threshold of {min_coverage_pct:.2f}%"
            )

        return cls(
            batch_id=batch_id,
            certified_at=certified_at or datetime.now(timezone.utc),
            provider=raw.provider,
            external_match_id=raw.external_match_id,
            match_date=raw.match_date,
            kickoff_at=raw.kickoff_at,
            tournament_name=raw.tournament_name,
            home_team=raw.home_team,
            away_team=raw.away_team,
            status="finished",
            home_score=int(raw.home_score if raw.home_score is not None else 0),
            away_score=int(raw.away_score if raw.away_score is not None else 0),
            season_label=raw.season_label,
            stage=raw.stage,
            competition_class=raw.competition_class,
            is_neutral=raw.is_neutral,
            home_metrics=raw.home_metrics,
            away_metrics=raw.away_metrics,
            source_payload=raw.source_payload,
            available_metric_count=available_metric_count,
            coverage_pct=coverage_pct,
            hard_required_missing=missing,
        )


class BatchRejectedMatch(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    provider: str
    external_match_id: str
    match_date: date
    status: str
    reason: str
    coverage_pct: float
    missing_fields: list[str] = Field(default_factory=list)

    @classmethod
    def from_raw(cls, raw: RawMatchPayload, reason: str) -> "BatchRejectedMatch":
        return cls(
            provider=raw.provider,
            external_match_id=raw.external_match_id,
            match_date=raw.match_date,
            status=raw.status,
            reason=reason,
            coverage_pct=raw.coverage_pct(),
            missing_fields=raw.hard_required_missing(),
        )


class BatchCertification(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    batch_id: str
    target_date: date
    provider_used: str | None = None
    total_records: int
    certified_records: int
    rejected_records: int
    available_metric_count: int
    coverage_pct: float = Field(ge=0, le=100)
    certified: bool
    certification_status: Literal["certified", "quarantined", "skipped"]
    quarantine_reason: str | None = None
    certified_matches: list[CertifiedMatch] = Field(default_factory=list)
    rejected_matches: list[BatchRejectedMatch] = Field(default_factory=list)

    @classmethod
    def certify_batch(
        cls,
        raw_matches: Sequence[RawMatchPayload | dict[str, Any]],
        *,
        batch_id: str,
        target_date: date,
        provider_used: str | None = None,
        min_metrics: int = CERTIFICATION_MIN_METRICS,
        min_coverage_pct: float = CERTIFICATION_MIN_COVERAGE_PCT,
    ) -> "BatchCertification":
        certified_matches: list[CertifiedMatch] = []
        rejected_matches: list[BatchRejectedMatch] = []
        total_available_metrics = 0
        total_possible_metrics = 0

        for raw_match in raw_matches:
            raw_model = raw_match if isinstance(raw_match, RawMatchPayload) else RawMatchPayload.model_validate(raw_match)
            total_available_metrics += raw_model.available_metric_count()
            total_possible_metrics += raw_model.total_metric_slots()

            try:
                certified_matches.append(
                    CertifiedMatch.from_raw(
                        raw_model,
                        batch_id=batch_id,
                        min_metrics=min_metrics,
                        min_coverage_pct=min_coverage_pct,
                    )
                )
            except ValueError as exc:
                rejected_matches.append(BatchRejectedMatch.from_raw(raw_model, str(exc)))

        coverage_pct = 0.0
        if total_possible_metrics > 0:
            coverage_pct = round((total_available_metrics / total_possible_metrics) * 100, 2)

        certified = bool(certified_matches) and not rejected_matches
        if not raw_matches:
            certification_status = "skipped"
        elif certified:
            certification_status = "certified"
        else:
            certification_status = "quarantined"
        quarantine_reason = None
        if not certified:
            if not raw_matches:
                quarantine_reason = "Empty batch, nothing to certify."
            else:
                quarantine_reason = (
                    f"{len(rejected_matches)} record(s) rejected while certifying {len(raw_matches)} total record(s)."
                )

        return cls(
            batch_id=batch_id,
            target_date=target_date,
            provider_used=provider_used,
            total_records=len(raw_matches),
            certified_records=len(certified_matches),
            rejected_records=len(rejected_matches),
            available_metric_count=total_available_metrics,
            coverage_pct=coverage_pct,
            certified=certified,
            certification_status=certification_status,
            quarantine_reason=quarantine_reason,
            certified_matches=certified_matches,
            rejected_matches=rejected_matches,
        )

    def is_quarantined(self) -> bool:
        return not self.certified
