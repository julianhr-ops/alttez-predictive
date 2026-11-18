"""ESPN national-team competitions the pipeline ingests, and the tournament name each one is stored under.

The Elo K factor (``alttez.features.elo.tournament_k_factor``) and ``classify_competition`` read the
tournament name, and the ratings are trained on the historical dataset's names. ESPN words the same
competitions differently ("FIFA World Cup Qualifying - UEFA" against "FIFA World Cup qualification"), and
a qualifier named the ESPN way would get a final's K of 60 instead of 40, and "Africa Cup of Nations"
would be classed ``continental`` where the dataset's "African Cup of Nations" is ``other_international``.
So every competition added here is stored under the name the historical dataset uses.

``fifa.world`` and ``fifa.friendly`` keep the name ESPN sends, as they always did.
"""

from __future__ import annotations

LEGACY_LEAGUES: tuple[str, ...] = ("fifa.world", "fifa.friendly")

# ESPN slug -> tournament name of the historical results dataset (martj42/international_results).
HISTORICAL_TOURNAMENT_BY_SLUG: dict[str, str] = {
    "uefa.nations": "UEFA Nations League",
    "concacaf.nations.league": "CONCACAF Nations League",
    "fifa.worldq.uefa": "FIFA World Cup qualification",
    "fifa.worldq.conmebol": "FIFA World Cup qualification",
    "fifa.worldq.concacaf": "FIFA World Cup qualification",
    "fifa.worldq.afc": "FIFA World Cup qualification",
    "fifa.worldq.caf": "FIFA World Cup qualification",
    "fifa.wcq.ply": "FIFA World Cup qualification",
    "caf.nations": "African Cup of Nations",
    "caf.nations_qual": "African Cup of Nations qualification",
    "afc.cupq": "AFC Asian Cup qualification",
    "aff.championship": "ASEAN Championship",
    "global.gulf_cup": "Gulf Cup",
    "afc.saff.championship": "SAFF Cup",
    "global.finalissima": "CONMEBOL\u2013UEFA Cup of Champions",
}

ESPN_DEFAULT_LEAGUES: tuple[str, ...] = (*LEGACY_LEAGUES, *HISTORICAL_TOURNAMENT_BY_SLUG)


def tournament_name(league_slug: str, espn_name: str | None) -> str | None:
    """The name a competition is stored under: the historical dataset's, else ESPN's own."""
    return HISTORICAL_TOURNAMENT_BY_SLUG.get(league_slug, espn_name)
