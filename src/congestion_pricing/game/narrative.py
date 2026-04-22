"""Narrative generator for Shadow Network: The Shadow War.

Generates briefing text, newspaper headlines, and educational
commentary from round results.
"""

import random

from congestion_pricing.game.engine import RoundResult, GameState


def round_briefing(result: RoundResult, state: GameState) -> str:
    """Generate a narrative briefing for the completed round.

    Args:
        result: The round's outcome
        state: Current game state

    Returns:
        Multi-line briefing text
    """
    lines = [_headline(result), "", _situation_report(result), "", _intel_analysis(result)]

    educational = _educational_note(result)
    if educational:
        lines.extend(["", educational])

    return "\n".join(lines)


def _headline(result: RoundResult) -> str:
    if result.agents_infiltrated > result.agents_detected * 2:
        templates = [
            "SECURITY BREACH: Major infiltration detected at facility",
            "SHADOW OPERATIVES STRIKE: Defenses overwhelmed",
            "INTELLIGENCE FAILURE: Majority of agents penetrated perimeter",
        ]
    elif result.agents_detected > result.agents_infiltrated * 2:
        templates = [
            "DRAGNET SUCCESS: Security forces intercept hostile agents",
            "PERIMETER HOLDS: Patrol deployment proves decisive",
            "COUNTERINTELLIGENCE TRIUMPH: Infiltration attempt repelled",
        ]
    else:
        templates = [
            "CONTESTED ENGAGEMENT: Both sides claim partial victory",
            "SHADOW WAR ESCALATES: Neither side gains clear advantage",
            "STALEMATE AT THE FACILITY: Losses mount on both sides",
        ]
    return f"--- {random.choice(templates)} ---"


def _situation_report(result: RoundResult) -> str:
    lines = [f"Round {result.round_number} Summary:"]
    lines.append(f"  Agents deployed: {result.agents_infiltrated + result.agents_detected}")
    lines.append(f"  Successfully infiltrated: {result.agents_infiltrated}")
    lines.append(f"  Detected and captured: {result.agents_detected}")

    if result.broker_active:
        lines.append(f"  Broker was active — profit: ${result.broker_profit:,.0f}")

    top_routes = sorted(result.route_flows.items(), key=lambda x: x[1], reverse=True)[:3]
    if top_routes:
        lines.append("  Most-used routes:")
        for route, count in top_routes:
            if count > 0:
                short = " → ".join(route.split(" → ")[:3]) + "..."
                lines.append(f"    {short}: {count:.0f} agents")

    return "\n".join(lines)


def _intel_analysis(result: RoundResult) -> str:
    lines = ["Intelligence Analysis:"]

    hottest = max(result.edge_risks.items(), key=lambda x: x[1])
    lines.append(f"  Highest-risk passage: {hottest[0][0]} → {hottest[0][1]} (risk: {hottest[1]:.1f})")

    busiest = max(result.edge_flows.items(), key=lambda x: x[1]) if result.edge_flows else None
    if busiest:
        lines.append(f"  Busiest passage: {busiest[0][0]} → {busiest[0][1]} ({busiest[1]:.0f} agents)")

    lines.append(f"  Total system detection risk: {result.total_detection_risk:,.0f}")
    lines.append(f"  Price of Anarchy: {result.price_of_anarchy:.2f}")

    return "\n".join(lines)


def _educational_note(result: RoundResult) -> str:
    if result.is_nash:
        return (
            "MODELING INSIGHT: This round reached a Nash equilibrium — "
            "neither player could improve their outcome by changing strategy alone. "
            "In game theory, this is a stable state, though not necessarily optimal for either side."
        )

    if result.price_of_anarchy > 1.3:
        return (
            f"MODELING INSIGHT: The Price of Anarchy is {result.price_of_anarchy:.2f} — "
            "agents routing selfishly cost the Spy Master "
            f"{(result.price_of_anarchy - 1) * 100:.0f}% more total risk than coordinated routing would. "
            "This is the mathematical cost of decentralized decision-making."
        )

    if result.broker_active:
        return (
            "MODELING INSIGHT: The Broker (a third strategic actor) has changed the "
            "equilibrium structure. In Stackelberg games, the leader (Broker) sets terms, "
            "and followers (agents) respond. The Broker's commission creates a wedge "
            "between supply and demand for elite operatives."
        )

    return ""


def game_over_briefing(state: GameState) -> str:
    """Generate end-of-game summary."""
    lines = [
        "=" * 50,
        "OPERATION COMPLETE",
        "=" * 50,
        "",
        f"Final Score:",
        f"  Spy Master: {state.spy_master_cumulative:.0f} agents infiltrated",
        f"  Security Chief: {state.security_chief_cumulative:.0f} agents detected",
        "",
    ]

    if state.winner == "Spy Master":
        lines.append("RESULT: The shadows prevailed. The Spy Master's network proved too elusive.")
    elif state.winner == "Security Chief":
        lines.append("RESULT: The facility held. The Security Chief's patrols dismantled the network.")
    else:
        lines.append("RESULT: A drawn conflict. Neither side achieved decisive advantage.")

    avg_poa = sum(r.price_of_anarchy for r in state.round_history) / len(state.round_history)
    nash_rounds = sum(1 for r in state.round_history if r.is_nash)
    lines.extend([
        "",
        "Campaign Statistics:",
        f"  Average Price of Anarchy: {avg_poa:.2f}",
        f"  Rounds at Nash Equilibrium: {nash_rounds}/{state.total_rounds}",
        f"  Total rounds played: {len(state.round_history)}",
    ])

    return "\n".join(lines)
