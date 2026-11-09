"""Read-only probes for the review. These expose limitations; they do not fix the engine."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from simverse.cost import expected_unique
from simverse.contracts import ScenarioSpec, AcceptanceSpec
from simverse.engine import run
from simverse.acceptance import resolve_alpha

OUT = Path(__file__).parent
scenario = ScenarioSpec(name="audit-baseline", model_id="tiny_dense_oracle",
                        draft_id="custom_chain", k=2, tp=1, batch=1, ctx_tokens=100,
                        deployment={"mode": "baseline"})
result = run(scenario)
residual = [op for op in result.ops if op.stage == "baseline_decode" and op.op.endswith("_add")]
findings = {
    "moe_union": {"experts":384,"top_k":6,"tokens":5,
        "draws_upper_bound":30,"current_iid_draws":expected_unique(384,30),
        "independent_uniform_topk":384*(1-(1-6/384)**5),"report_claim":38.9},
    "old_document_coverage": {"experts":512,"top_k":10,"tokens":100,
        "current_formula_fraction":1-(1-1/512)**1000,"document_claim":0.18},
    "same_mean_different_full_acceptance": {"K":2,"distribution_A":{"2":1.0},
        "distribution_B":{"1":0.5,"3":0.5},"mean_A":2,"mean_B":2,
        "p_full_A":0,"p_full_B":0.5},
    "mean_endpoint": {"K":2,"mean":3,"resolved_alpha":resolve_alpha(
        AcceptanceSpec(kind="committed_mean",committed_mean=3),2),"required_alpha":1.0},
    "baseline_timeline": {"requested_mode":"baseline",
        "events":[{"label":e.label,"stage":e.stage,"start_ms":e.start_ms,"end_ms":e.end_ms} for e in result.timeline],
        "contains_draft":any(e.lane=="draft" for e in result.timeline)},
    "residual_count": {"rows":len(residual),"current_flops":sum(x.flops for x in residual),
        "logical_add_operations":4*256},
    "snapshot_sha256":{},
}
for path in [ROOT/'docs/verification_report.md',ROOT/'simverse/engine.py',ROOT/'simverse/ledger.py',
             ROOT/'assets/config/deepseek_v41_config.json',
             OUT/'references/model-verse-principles-pinned.md',
             Path('C:/Users/41693/Downloads/simverse_v3_0.html')]:
    findings['snapshot_sha256'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
assert findings['moe_union']['current_iid_draws'] <= 30
assert findings['baseline_timeline']['contains_draft']
assert findings['residual_count']['current_flops']==0
(OUT/'audit-results.json').write_text(json.dumps(findings,ensure_ascii=False,indent=2),encoding='utf-8')
print('Saved read-only audit: six counterexamples and source hashes.')
