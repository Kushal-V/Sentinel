#!/usr/bin/env python3
"""
End-to-end crisis test harness for Sentinel.

Loads each test dataset, fires 2 crisis queries per dataset through the full
Dispatcher -> Specialist pipeline, and writes results to tests/crisis_results/.

Usage:
    python tests/test_crisis_e2e.py
"""

import json
import os
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import pandas as pd

from src.agents.orchestrator import AgentOrchestrator, CrisisEvent, DispatchRoute
from src.core.state_manager import FactoryDataManager
from src.tools.tool_registry import set_data_manager

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
RESULTS_DIR = PROJECT_ROOT / "tests" / "crisis_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Test datasets + 2 crises each
# ---------------------------------------------------------------------------
TEST_CASES: list[dict] = [
    # ── 1. Pharmaceutical Warehouse ──────────────────────────────────────
    {
        "dataset": "test_datasets/1_pharmaceutical_warehouse.csv",
        "label": "pharma",
        "crises": [
            {
                "id": "PHARMA-01-MOVER",
                "type": "SHIPPING_DELAY",
                "severity": "CRITICAL",
                "description": (
                    "Emergency shipment of 5,000 Flu Vaccines and 3,000 COVID Boosters "
                    "stuck at port customs — cryo shipping container lost refrigeration "
                    "4 hours ago. Vaccines from Serum Institute and Moderna are expiring "
                    "in under 45 days and current WH-CRYO-C stock is critically low. "
                    "We need immediate rerouting through air freight."
                ),
            },
            {
                "id": "PHARMA-02-KEEPER",
                "type": "INVENTORY_OVERFLOW",
                "severity": "HIGH",
                "description": (
                    "Warehouse WH-AMBIENT-B is at 92% capacity. Paracetamol 500mg has "
                    "48,500 units against 50,000 max capacity and a new shipment of "
                    "10,000 units is arriving tomorrow. We also have Ibuprofen 200mg "
                    "at 45,000/50,000. Need emergency reallocation or overflow storage."
                ),
            },
        ],
    },
    # ── 2. Electronics Manufacturing ─────────────────────────────────────
    {
        "dataset": "test_datasets/2_electronics_manufacturing.csv",
        "label": "electronics",
        "crises": [
            {
                "id": "ELEC-01-MAKER",
                "type": "SUPPLY_SHORTAGE",
                "severity": "CRITICAL",
                "description": (
                    "TSMC fab in Taiwan hit by earthquake — all ARM Cortex-M4 MCU and "
                    "Artix-7 FPGA shipments halted for 8+ weeks. Current stock of "
                    "CHIP-ARM-M4 is only 18,000 units against 80,000 capacity and "
                    "CHIP-FPGA-7K is down to 800 units. Production lines will stop "
                    "within 2 weeks without alternative sourcing."
                ),
            },
            {
                "id": "ELEC-02-MOVER",
                "type": "TRANSPORT_DISRUPTION",
                "severity": "HIGH",
                "description": (
                    "South Korea shipping lanes blocked due to typhoon — all Samsung SDI "
                    "battery shipments (BATT-LIPO-3000, BATT-LIPO-5000) delayed 3 weeks. "
                    "Current LiPo 3000mAh stock is only 1,200 of 25,000 capacity. "
                    "Need to reroute via air freight from alternate Samsung facility."
                ),
            },
        ],
    },
    # ── 3. Food Cold Chain ───────────────────────────────────────────────
    {
        "dataset": "test_datasets/3_food_cold_chain.csv",
        "label": "food",
        "crises": [
            {
                "id": "FOOD-01-KEEPER",
                "type": "COLD_CHAIN_FAILURE",
                "severity": "CRITICAL",
                "description": (
                    "DC-COLD-NORTH refrigeration unit failed overnight. All chilled "
                    "products at risk — Chicken Breast Fillet has 6 days shelf life, "
                    "Fresh Atlantic Salmon only 4 days, and Sashimi Grade Tuna expires "
                    "tomorrow. Need emergency stock transfer or markdown of 15,000+ kg "
                    "of perishables before spoilage."
                ),
            },
            {
                "id": "FOOD-02-MAKER",
                "type": "SUPPLY_SHORTAGE",
                "severity": "HIGH",
                "description": (
                    "Major avocado blight in Mexico — Mission Produce cannot fulfill "
                    "orders for 4 weeks. Current Hass Avocado stock is 8,500 units "
                    "with only 4 days shelf life remaining. Demand is 3,000/day. Also "
                    "banana supply from Ecuador disrupted — Cavendish stock at 5,500 kg "
                    "with 3-day shelf life. Need alternate sourcing immediately."
                ),
            },
        ],
    },
    # ── 4. Automotive Parts ──────────────────────────────────────────────
    {
        "dataset": "test_datasets/4_automotive_parts.csv",
        "label": "automotive",
        "crises": [
            {
                "id": "AUTO-01-MOVER",
                "type": "SHIPPING_DELAY",
                "severity": "CRITICAL",
                "description": (
                    "Brembo SpA factory in Italy caught fire — all brake component "
                    "shipments halted for 3 months. Front Brake Pads at 800 units "
                    "(reorder point 1000), Front Brake Calipers at 320 units. "
                    "Multiple vehicle platforms affected. Need emergency procurement "
                    "from alternate suppliers and expedited shipping."
                ),
            },
            {
                "id": "AUTO-02-KEEPER",
                "type": "INVENTORY_IMBALANCE",
                "severity": "HIGH",
                "description": (
                    "Winter tire season ending — we have 3,200 winter tires in stock "
                    "but demand has dropped to near zero. Meanwhile summer tire demand "
                    "is surging and 225/45R17 All-Season tires are at 1,100 units "
                    "with 500 reorder point. Need to rebalance storage allocation "
                    "and reduce winter tire holding costs."
                ),
            },
        ],
    },
]


def run_crisis(
    dm: FactoryDataManager,
    orch: AgentOrchestrator,
    crisis_cfg: dict,
) -> dict:
    """Run a single crisis through dispatcher + specialist and collect results."""
    crisis = CrisisEvent(
        event_id=crisis_cfg["id"],
        event_type=crisis_cfg["type"],
        severity=crisis_cfg["severity"],
        description=crisis_cfg["description"],
        affected_entities={},
    )

    result = {
        "crisis_id": crisis_cfg["id"],
        "description": crisis_cfg["description"],
        "severity": crisis_cfg["severity"],
        "dispatch": None,
        "agent_steps": [],
        "final_answer": None,
        "error": None,
        "duration_s": 0,
    }

    t0 = time.time()
    try:
        # Dispatch
        route: DispatchRoute = orch.dispatch(crisis=crisis)
        result["dispatch"] = {
            "selected_agent": route.selected_agent,
            "justification": route.delegation_justification,
            "urgency_tier": route.urgency_tier,
            "trust_override": route.trust_override_applied,
            "original_llm_choice": route.original_llm_choice,
        }
        print(f"    Dispatched to: {route.selected_agent.upper()} [{route.urgency_tier}]")

        # Run specialist
        for step in orch.run_specialist(
            route=route,
            crisis=crisis,
            chat_history=[],
        ):
            step_type = step.get("type", "")
            content = step.get("content", "")

            if step_type == "tool_call":
                tool_name = step.get("tool") or step.get("tool_name") or "?"
                print(f"    -> tool: {tool_name}")
                result["agent_steps"].append({
                    "type": "tool_call",
                    "tool_name": tool_name,
                    "args_preview": str(step.get("content", ""))[:200],
                })
            elif step_type == "tool_result":
                result["agent_steps"].append({
                    "type": "tool_result",
                    "preview": str(content)[:300],
                })
            elif step_type == "final_answer":
                result["final_answer"] = content
                print(f"    -> FINAL ANSWER received ({len(content)} chars)")
            elif step_type == "error":
                result["agent_steps"].append({"type": "error", "content": content})
                print(f"    -> ERROR: {content[:200]}")

    except Exception as exc:
        result["error"] = str(exc)
        print(f"    -> EXCEPTION: {exc}")

    result["duration_s"] = round(time.time() - t0, 2)
    return result


def evaluate_result(result: dict) -> dict:
    """Score a single crisis result for quality."""
    issues = []
    score = 100  # start perfect, deduct for problems

    fa = result.get("final_answer") or ""

    # 1. Did we get a final answer at all?
    if not fa:
        issues.append("NO_FINAL_ANSWER: Agent produced no final answer text.")
        score -= 40

    # 2. Does the answer contain the mitigation proposal format?
    if "--- MITIGATION PROPOSAL ---" not in fa:
        issues.append("MISSING_PROPOSAL_FORMAT: No '--- MITIGATION PROPOSAL ---' block found.")
        score -= 15

    # 3. Did the agent take actions (not "None" or "none")?
    if "ACTIONS TAKEN:" in fa:
        actions_section = fa.split("ACTIONS TAKEN:")[1].split("FINANCIAL")[0] if "FINANCIAL" in fa else fa.split("ACTIONS TAKEN:")[1][:500]
        if "none" in actions_section.lower() or "unable" in actions_section.lower():
            issues.append("NO_ACTIONS_TAKEN: Agent said it took no actions or was unable.")
            score -= 30

    # 4. Did the agent actually call propose_state_change?
    tool_calls = [s for s in result["agent_steps"] if s.get("type") == "tool_call"]
    tool_names = [s.get("tool_name", "") for s in tool_calls]
    if "propose_state_change" not in tool_names:
        issues.append("NO_PROPOSALS: Agent never called propose_state_change.")
        score -= 25

    # 5. Did the agent call get_dataset_schema first?
    if tool_names and tool_names[0] != "get_dataset_schema":
        issues.append("SCHEMA_NOT_FIRST: Agent did not call get_dataset_schema first.")
        score -= 10

    # 6. Did the agent call query_data?
    if "query_data" not in tool_names:
        issues.append("NO_QUERY: Agent never called query_data to look up current values.")
        score -= 15

    # 7. Check for "lack of data" or confusion
    lack_phrases = ["lack of data", "no data", "unable to determine", "insufficient data", "could not find"]
    for phrase in lack_phrases:
        if phrase in fa.lower():
            issues.append(f"DATA_CONFUSION: Agent mentioned '{phrase}' despite data being loaded.")
            score -= 20
            break

    # 8. Did the agent mention a financial impact?
    if "FINANCIAL IMPACT:" in fa:
        fi_section = fa.split("FINANCIAL IMPACT:")[1][:100]
        if "unknown" in fi_section.lower() or "n/a" in fi_section.lower():
            issues.append("UNKNOWN_FINANCIAL_IMPACT: Agent couldn't calculate financial impact.")
            score -= 5

    # 9. Error in pipeline?
    if result.get("error"):
        issues.append(f"PIPELINE_ERROR: {result['error'][:200]}")
        score -= 30

    # 10. Trust override bug?
    dispatch = result.get("dispatch") or {}
    if dispatch.get("trust_override") and dispatch.get("original_llm_choice"):
        just = dispatch.get("justification", "")
        if "0.95" in just and "below" in just.lower():
            issues.append("BOGUS_TRUST_OVERRIDE: Trust override triggered despite score being above threshold.")
            score -= 10

    return {
        "score": max(score, 0),
        "issues": issues,
        "issue_count": len(issues),
    }


def main():
    print("=" * 70)
    print("SENTINEL CRISIS E2E TEST HARNESS")
    print("=" * 70)

    all_results = []
    summary_lines = []

    for tc in TEST_CASES:
        dataset_path = PROJECT_ROOT / tc["dataset"]
        label = tc["label"]
        print(f"\n{'─' * 60}")
        print(f"DATASET: {label} ({dataset_path.name})")
        print(f"{'─' * 60}")

        # Load dataset into a fresh workspace
        df = pd.read_csv(dataset_path)
        ws_name = f"_test_{label}"
        dm = FactoryDataManager(workspace=ws_name)
        dm.update_inventory_data(df)
        dm.reset_for_new_dataset()
        set_data_manager(dm)

        # Create orchestrator
        orch = AgentOrchestrator(data_manager=dm)

        for crisis_cfg in tc["crises"]:
            print(f"\n  CRISIS: {crisis_cfg['id']}")
            print(f"  {crisis_cfg['description'][:100]}...")

            result = run_crisis(dm, orch, crisis_cfg)
            result["dataset"] = label

            # Evaluate
            evaluation = evaluate_result(result)
            result["evaluation"] = evaluation

            all_results.append(result)

            status = "PASS" if evaluation["score"] >= 60 else "FAIL"
            summary_lines.append(
                f"  [{status}] {result['crisis_id']:20s} | "
                f"Agent: {(result.get('dispatch') or {}).get('selected_agent', '?'):6s} | "
                f"Score: {evaluation['score']:3d}/100 | "
                f"Issues: {evaluation['issue_count']} | "
                f"Time: {result['duration_s']}s"
            )
            if evaluation["issues"]:
                for issue in evaluation["issues"]:
                    summary_lines.append(f"         -> {issue}")

            print(f"    Score: {evaluation['score']}/100 | Issues: {evaluation['issue_count']}")

    # Write detailed results
    results_file = RESULTS_DIR / "crisis_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nDetailed results written to: {results_file}")

    # Write summary
    summary_file = RESULTS_DIR / "summary.txt"
    summary_text = (
        "SENTINEL CRISIS E2E TEST SUMMARY\n"
        + "=" * 60 + "\n\n"
    )
    for tc in TEST_CASES:
        summary_text += f"Dataset: {tc['label']}\n"
        for line in summary_lines:
            if tc["label"].upper()[:4] in line.upper()[:30] or "  ->" in line[:10]:
                summary_text += line + "\n"
        summary_text += "\n"

    # Overall stats
    scores = [r["evaluation"]["score"] for r in all_results]
    avg_score = sum(scores) / len(scores) if scores else 0
    pass_count = sum(1 for s in scores if s >= 60)
    summary_text += f"\nOVERALL: {pass_count}/{len(scores)} passed | Avg score: {avg_score:.0f}/100\n"

    # Aggregate issues
    all_issues = []
    for r in all_results:
        all_issues.extend(r["evaluation"]["issues"])
    if all_issues:
        summary_text += "\nALL ISSUES:\n"
        for issue in all_issues:
            summary_text += f"  - {issue}\n"

    with open(summary_file, "w") as f:
        f.write(summary_text)

    print(f"Summary written to: {summary_file}")
    print(f"\n{'=' * 60}")
    print(f"OVERALL: {pass_count}/{len(scores)} passed | Avg score: {avg_score:.0f}/100")
    print(f"{'=' * 60}")

    # Print summary to console
    for line in summary_lines:
        print(line)

    return all_results


if __name__ == "__main__":
    results = main()
