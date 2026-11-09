from pathlib import Path
import re,json,hashlib
r=Path(__file__).resolve().parent.parent
s=(r/"ui-data-contract.md").read_text(encoding="utf-8")
result={"kind":"document-consistency-only"}
for prefix,total in [("P",8),("M",23),("V",12)]:
    found=set(re.findall(r"^\| ("+prefix+r"\d{2}) ",s,re.M))
    expected={f"{prefix}{i:02}" for i in range(1,total+1)}
    assert found==expected,(prefix,found,expected)
    result[prefix+"_ids"]=len(found)
section=s.split("### 7.1",1)[1].split("### 7.2",1)[0]
covered=[]
for line in section.splitlines():
    m=re.match(r"\| M(\d{2})(?:–M(\d{2}))? \| P\d{2} / ([^|]+)",line)
    if m: covered.extend(range(int(m[1]),int(m[2] or m[1])+1))
assert sorted(covered)==list(range(1,24)),covered
result["metrics_with_region"]=len(covered)
q=(r/"questions.md").read_text(encoding="utf-8")
ids=re.findall(r"^### (Q\d+) ",q,re.M)
assert ids==[f"Q{i}" for i in range(1,13)]
assert q.count("我的回答：")==12
result["question_ids"]=ids
paths=[r/n for n in ["plan.md","principles.md","ui-data-contract.md","questions.md","README.md","review-resolution.md","validation.md","review-ui-round2/resolution.md"]]
missing=[]
# doc-checks.json is the current output, materialized before link checks.
out=Path(__file__).with_suffix(".json")
out.write_text("{}",encoding="utf-8")
for p in paths:
    for link in re.findall(r"\]\(([^)]+)\)",p.read_text(encoding="utf-8")):
        if "://" in link or link.startswith("#"):continue
        if not (p.parent/link.split("#")[0]).exists():missing.append([str(p.relative_to(r)),link])
assert not missing,missing
result["broken_local_links"]=missing
for engine in ["dsh","zcode"]:
    receipt=json.loads((r/f"review-ui-round2/{engine}-receipt.json").read_text(encoding="utf-8"))
    assert receipt["success"] is True
    review=(r/f"review-ui-round2/{engine}-review.md").read_text(encoding="utf-8")
    prefix="D2" if engine=="dsh" else "Z2"
    result[engine+"_finding_count"]=len(set(re.findall(r"^### ("+prefix+r"-\d{2})",review,re.M)))
result["counterexample_count"]=len(json.loads((r/"review-ui-round2/counterexamples.json").read_text(encoding="utf-8"))["cases"])
result["checked_document_hashes"]={str(p.relative_to(r)).replace("\\","/"):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps({k:v for k,v in result.items() if k!="checked_document_hashes"},ensure_ascii=False))
