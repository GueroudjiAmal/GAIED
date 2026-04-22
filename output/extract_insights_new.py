"""
extract_insights_new.py
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

t0 = min(r["started_at"] for r in records if r.get("started_at", 0) > 0)
print(f"Total records: {len(records)}\n")


# ── Finding 1: Orchestrator killed P1 while molecules still running ──────────
print("=" * 70)
print("FINDING 1 — Report polls for P1 agent vs. XTB completion times")
print("=" * 70)

p1_reports = [
    r for r in records
    if r.get("activity_id") == "report"
    and r.get("workflow_id", "").startswith("9371")
]
print("\nReport polls for agent 93717575 (field: academy_action/report):")
print(f"  {'t (relative)':>14}  {'molecules in generated':>22}")
for r in sorted(p1_reports, key=lambda x: x["started_at"]):
    t = round(r["started_at"] - t0, 1)
    n = len(r.get("generated", []))
    flag = "  ← stable → marked finished" if t == 143.4 else ""
    print(f"  {t:>13}s  {n:>22}{flag}")

xtb_p1 = [
    r for r in records
    if r.get("activity_id") == "compute_ionization_energy"
    and r.get("workflow_id", "").startswith("ff624cdb")
]
print("\nXTB completion times (ended_at field) for ff624cdb:")
print(f"  {'done at':>10}  {'IE (eV)':>10}  SMILES")
for r in sorted(xtb_p1, key=lambda x: x["ended_at"]):
    smiles = ast.literal_eval(r["used"]["input"])["smiles"]
    energy = r["generated"]["output"]
    t_end = round(r["ended_at"] - t0, 1)
    flag = "  ← GLOBAL BEST" if smiles == "NC(=O)NC(F)(F)F" else ""
    print(f"  t={t_end:>7}s  {energy:>10.4f}  {smiles}{flag}")

print(f"\nOrchestrator stopped monitoring P1 at: t=143.4s")
print(f"Global best molecule finished at:      t=210.8s")
print(f"Gap:                                   67.4s")


# ── Finding 2: Plan recommended seed; tool_calling ignored it ─────────────────
print()
print("=" * 70)
print("FINDING 2 — Plan vs. tool_calling: seed CNC(N)=O")
print("=" * 70)

llm = [r for r in records if r.get("activity_id", "").startswith("argo_llm")]
p1_plan = next(
    r for r in llm
    if r.get("workflow_id", "").startswith("ff624cdb")
    and "tasked" in r.get("used", {}).get("system_prompt", "")
)
print("\nRelevant excerpts from plan LLM generated.response:")
for line in p1_plan["generated"]["response"].splitlines():
    if any(kw in line for kw in ("CNC(N)=O", "Baseline", "best first batch", "1. `CNC")):
        print(f"  {line.strip()}")

p1_tool = next(
    r for r in llm
    if r.get("workflow_id", "").startswith("ff624cdb")
    and r.get("used", {}).get("has_tools")
)
dispatched = [c["args"]["smiles"] for c in p1_tool["generated"]["tool_calls"]]
print(f"\nSMILES dispatched by tool_calling LLM (generated.tool_calls):")
for s in dispatched:
    print(f"  {s}")
print(f"\n'CNC(N)=O' present in dispatched list: {('CNC(N)=O' in dispatched)}")


# ── Finding 3: Allophanamide/acetylurea hypothesis → refutation loop ──────────
print()
print("=" * 70)
print("FINDING 3 — Full hypothesis-simulation-refutation cycle")
print("=" * 70)

print("\nPlan predictions (generated.response excerpts):")
for line in p1_plan["generated"]["response"].splitlines():
    if any(kw in line for kw in ("llophan", "cetylurea", "Strong candidate",
                                  "trifluoromethylurea / N-cyanourea")):
        print(f"  PLAN: {line.strip()}")

target_smiles = {"NC(=O)NC(N)=O", "NC(=O)NC(C)=O", "NC(N)=O"}
print("\nXTB results (generated.output):")
print(f"  {'SMILES':<30}  {'IE (eV)':>10}  {'Duration (s)':>13}")
for r in xtb_p1:
    smiles = ast.literal_eval(r["used"]["input"])["smiles"]
    if smiles in target_smiles:
        energy = r["generated"]["output"]
        dur = round(r["ended_at"] - r["started_at"], 2)
        print(f"  {smiles:<30}  {energy:>10.4f}  {dur:>13.2f}")

p1_conclude = next(
    r for r in llm
    if r.get("workflow_id", "").startswith("ff624cdb")
    and "Synthesize" in r.get("used", {}).get("system_prompt", "")
)
print("\nConclude LLM correction (generated.response excerpts):")
for line in p1_conclude["generated"]["response"].splitlines():
    if any(kw in line for kw in ("llophan", "cetylurea", "incorrect", "lower than N-methyl")):
        print(f"  CONCLUDE: {line.strip()}")
