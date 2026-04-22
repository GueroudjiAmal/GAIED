"""
extract_insights.py
"""

import ast
import json
from pathlib import Path

JSONL = Path(__file__).parent / "flowcept_buffer.jsonl"

records = []
with JSONL.open() as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

print(f"Total records: {len(records)}\n")

# ── Finding 1: LLM conclusion vs. actual XTB results ────────────────────────
print("=" * 70)
print("FINDING 1 — LLM ranking vs. XTB ground truth (Pipeline 2)")
print("=" * 70)

llm_records = [r for r in records if r.get("activity_id", "").startswith("argo_llm")]
p2_conclude = [
    r for r in llm_records
    if r.get("workflow_id", "").startswith("6dc24cfc")
    and not r.get("used", {}).get("has_tools")
    and "Synthesize" in r.get("used", {}).get("system_prompt", "")
]
assert len(p2_conclude) == 1, "Expected exactly one Pipeline-2 conclude LLM call"
llm_response = p2_conclude[0]["generated"]["response"]
print("\nLLM response (generated.response):")
for line in llm_response.splitlines():
    if line.strip():
        print(f"  {line}")

xtb_p2 = [
    r for r in records
    if r.get("activity_id") == "compute_ionization_energy"
    and r.get("workflow_id", "").startswith("6dc24cfc")
]
print("\nActual XTB results (generated.output), ranked by IE:")
rows = []
for r in xtb_p2:
    smiles = ast.literal_eval(r["used"]["input"])["smiles"]
    energy = r["generated"]["output"]
    rows.append((smiles, energy))
for smiles, energy in sorted(rows, key=lambda x: x[1], reverse=True):
    print(f"  {energy:.6f} eV  {smiles}")

# ── Finding 2: LangGraph node execution — critique/update missing ────────────
print()
print("=" * 70)
print("FINDING 2 — LangGraph nodes actually executed")
print("=" * 70)

lg_nodes = [r for r in records if r.get("subtype") == "langgraph_node"]
lg_nodes.sort(key=lambda r: r.get("started_at", 0))
print(f"\nTotal langgraph_node records: {len(lg_nodes)}")
print("\nNode name            | workflow        | duration (s)")
print("-" * 55)
for r in lg_nodes:
    node = r["activity_id"]
    wf = r.get("workflow_id", "")[:8]
    dur = round(r.get("ended_at", 0) - r.get("started_at", 0), 3)
    print(f"  {node:<20} {wf}      {dur}")

all_nodes = {r["activity_id"] for r in lg_nodes}
for expected in ("critique", "update"):
    present = expected in all_nodes
    print(f"\n'{expected}' node present in provenance: {present}")

# ── Finding 3: seed molecule never simulated ─────────────────────────────────
print()
print("=" * 70)
print("FINDING 3 — Seed molecule vs. simulated molecules (Pipeline 1)")
print("=" * 70)

wf_records = [r for r in records if r.get("type") == "workflow"]
p1_wf = next(r for r in wf_records if r.get("name", "").startswith("xtb-graph-CNC"))
print(f"\nWorkflow name (name field): {p1_wf['name']}")
print(f"Workflow ID:                {p1_wf['workflow_id']}")

p1_plan_llm = [
    r for r in llm_records
    if r.get("workflow_id", "").startswith("2910f922")
    and not r.get("used", {}).get("has_tools")
    and "tasked" in r.get("used", {}).get("system_prompt", "")
]
assert len(p1_plan_llm) == 1
print(f"\nLLM user_prompt (used.user_prompt):\n  {p1_plan_llm[0]['used']['user_prompt']}")

xtb_p1 = [
    r for r in records
    if r.get("activity_id") == "compute_ionization_energy"
    and r.get("workflow_id", "").startswith("2910f922")
]
simulated = [ast.literal_eval(r["used"]["input"])["smiles"] for r in xtb_p1]
seed = "CNC(N)=O"
print(f"\nAll simulated SMILES for Pipeline 1 ({len(simulated)} molecules):")
for s in simulated:
    print(f"  {s}")
print(f"\nSeed '{seed}' in simulated list: {seed in simulated}")
