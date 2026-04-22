"""Scoring system for Shadow Network: The Shadow War."""

from congestion_pricing.game.engine import GameState, RoundResult


def format_scores(state: GameState) -> dict[str, str]:
    """Format current scores for display."""
    return {
        "spy_master": f"{state.spy_master_cumulative:.0f} agents infiltrated",
        "security_chief": f"{state.security_chief_cumulative:.0f} agents detected",
        "round": f"{state.current_round}/{state.total_rounds}",
    }


def round_summary(result: RoundResult) -> dict[str, float]:
    """Extract key metrics from a round result for charting."""
    return {
        "round": result.round_number,
        "spy_score": result.spy_master_score,
        "chief_score": result.security_chief_score,
        "poa": result.price_of_anarchy,
        "total_risk": result.total_detection_risk,
        "nash_distance": result.nash_distance,
    }


def score_history(state: GameState) -> list[dict[str, float]]:
    """Build score history for charting across all rounds."""
    return [round_summary(r) for r in state.round_history]
