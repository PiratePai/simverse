"""Independent arithmetic witnesses for review; does not exercise production code."""
from pathlib import Path
import json
out = []
# Same output shape, different mask semantics; uniform attention scores.
values = [1.0, 9.0]
causal = [sum(values[:i+1])/(i+1) for i in range(2)]
unmasked = [sum(values)/2 for _ in range(2)]
assert len(causal) == len(unmasked) and causal != unmasked
out.append(dict(id="C01",claim="shape equality does not prove mask semantics",causal=causal,unmasked=unmasked))
# Same server commits; client buffering changes TTFT/ITL. Not a hardware trace.
server_commit = [1, 2, 3]
client_immediate = [1, 2, 3]
client_buffered = [3, 3, 3]
def intervals(t): return [b-a for a,b in zip(t,t[1:])]
assert intervals(client_immediate) != intervals(client_buffered)
out.append(dict(id="C02",claim="commit timestamps do not determine client-visible latency",server_commit=server_commit,immediate=client_immediate,buffered=client_buffered,immediate_itl=intervals(client_immediate),buffered_itl=intervals(client_buffered)))
# Identical global resident bytes, different placement feasibility.
balanced, skewed, capacity = [80,80], [100,60], 80
assert sum(balanced)==sum(skewed) and max(balanced)<=capacity<max(skewed)
out.append(dict(id="C03",claim="global capacity conservation does not prove per-rank feasibility",balanced=balanced,skewed=skewed,per_rank_capacity=capacity))
# Window token-event throughput != completed-request token throughput.
start,end=0,10
requests={"a":[-2,1,2],"b":[3,8,12]}
window_tokens=sum(start<=t<end for times in requests.values() for t in times)
completion_tokens=sum(len(times) for times in requests.values() if start<=times[-1]<end)
assert window_tokens==4 and completion_tokens==3
out.append(dict(id="C04",claim="window policy changes token throughput",window=[start,end],requests=requests,event_token_rate=window_tokens/(end-start),completion_token_rate=completion_tokens/(end-start)))
# Same marginal acceptance and K probabilities, distinct joint laws.
en_a = 0.5*(1+0.9+0.9*0.8)+0.5*(1+0.5)
en_b = 0.5*(1+0.7+0.7*0.8)+0.5*(1+0.7)
assert abs(en_a-2.06)<1e-12 and abs(en_b-1.98)<1e-12
out.append(dict(id="C05",claim="dynamic K needs joint acceptance law",mean_time=8,en_a=en_a,en_b=en_b,tpot_a=8/en_a,tpot_b=8/en_b))
# A too-small bandwidth constant still satisfies a correct optimistic lower bound.
f,p,q,bw = 1000,100,2000,50
lower=max(f/p,q/bw)
wrong=max(f/p,q/(bw/10))
assert lower==40 and wrong==400 and wrong>=lower
out.append(dict(id="C06",claim="dimension and lower-bound checks cannot catch every transcription error",correct_time=lower,wrong_time=wrong,same_dimensions=True,wrong_satisfies_lower_bound=True))
# Legitimate overlap invalidates unconditional sum(duration)==elapsed.
events=[(0,10),(0,10)]
elapsed=max(e for _,e in events)-min(s for s,_ in events)
service=sum(e-s for s,e in events)
assert elapsed==10 and service==20
out.append(dict(id="C07",claim="parallel elapsed differs from sum of resource service times",events=events,elapsed=elapsed,service_sum=service))
Path(__file__).with_name("counterexamples.json").write_text(json.dumps({"scope":"arithmetic witnesses only; not production/model/GPU validation","cases":out},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(f"{len(out)} independent witnesses passed")
