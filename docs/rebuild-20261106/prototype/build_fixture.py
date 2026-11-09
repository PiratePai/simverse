"""Offline teaching fixtures, not a production simulator or a real-model prediction.

CPU only. Deliberately simple unfused FP16 kernels; illustrative hardware budgets.
No dependency on simverse production formulas. The browser only selects results.
"""
from pathlib import Path
import hashlib
import itertools
import json

OUT=Path(__file__).parent
def ledger(h,layers,heads,kv,ffn,q,ctx=1024):
    rows=[]
    def add(layer,name,f,w,ar,aw,kr=0,kw=0,shape='',formula='',tensor=True):
        # Hypothetical 2 TF/s matrix / 0.2 TF/s scalar, 20 GB/s, 4 us launch.
        peak=2e12 if tensor else .2e12
        compute=f/peak*1000;memory=(w+ar+aw+kr+kw)/20e9*1000
        rows.append(dict(id=f'l{layer}-{name}',layer=layer,name=name,flops=f,weight=w,
            act_read=ar,act_write=aw,kv_read=kr,kv_write=kw,shape=shape,formula=formula,
            comp=compute,mem=memory,ms=max(compute,memory)+.004,
            bound='计算' if compute>memory else '访存',peak=peak))
    def gemm(li,name,ic,oc):
        add(li,name,2*q*ic*oc,ic*oc*2,q*ic*2,q*oc*2,
            shape=f'[{q}, {ic}] × [{ic}, {oc}]',formula=f'2 × {q} × {ic} × {oc}')
    for li in range(layers):
        gemm(li,'Q 投影',h,heads*32);gemm(li,'K 投影',h,kv*32);gemm(li,'V 投影',h,kv*32)
        pairs=ctx*q+q*(q+1)//2
        add(li,'Attention',heads*pairs*(4*32+5),0,q*heads*32*2,q*heads*32*2,
            ctx*2*kv*32*2,q*2*kv*32*2,shape=f'{q} queries × {ctx} history',
            formula=f'{heads} × ({ctx}×{q}+{q}×{q+1}/2) × (4×32+5)',tensor=True)
        gemm(li,'输出投影',heads*32,h);gemm(li,'Gate / Up',h,2*ffn)
        add(li,'SwiGLU',q*ffn*5,0,q*ffn*4,q*ffn*2,shape=f'[{q}, {ffn}]',formula=f'5 × {q} × {ffn}（教学OP约定）',tensor=False)
        gemm(li,'Down',ffn,h)
        for name in ['Attention Norm','FFN Norm']:
            add(li,name,q*(4*h+2),h*2,q*h*2,q*h*2,shape=f'[{q}, {h}]',formula=f'{q} × (4×{h}+2)',tensor=False)
        for name in ['Attention Add','FFN Add']:
            add(li,name,q*h,0,q*h*4,q*h*2,shape=f'[{q}, {h}]',formula=f'{q} × {h}',tensor=False)
    gemm(-1,'LM Head',h,1000)
    add(-1,'Embedding',0,q*h*2,0,q*h*2,shape=f'{q} × {h}',formula='查表读取被选中的行',tensor=False)
    return rows

def total(rows):return sum(r['ms'] for r in rows)
target_base=ledger(256,2,8,2,512,1)
cases={}
for k,alpha in itertools.product([2,4,6],[.5,.75,.9]):
    ops=ledger(256,2,8,2,512,k+1)
    dm1=ledger(128,1,4,1,256,1);dm2=ledger(128,1,4,1,256,2)
    n=sum(alpha**i for i in range(k+1));pfull=alpha**k
    draft=(1-pfull)*total(dm1)+pfull*total(dm2)+(k-1)*total(dm1)
    verify=total(ops);base=total(target_base);link=.02
    # A protocol teaching budget with explicit fixed cost, not stochastic exact sampling.
    variants=[dict(id='base',name='普通解码',round=base,tpot=base,speedup=1,kind='estimate'),
      dict(id='co',name='同卡串行',round=draft+verify,tpot=(draft+verify)/n,speedup=base*n/(draft+verify),kind='estimate'),
      dict(id='dt',name='DT 串行',round=draft+verify+2*link,tpot=(draft+verify+2*link)/n,speedup=base*n/(draft+verify+2*link),kind='estimate'),
      dict(id='ideal',name='两微批理想周期',round=max(draft+link,verify)+link,tpot=None,speedup=None,kind='bound')]
    cases[f'{k}-{alpha}']=dict(k=k,alpha=alpha,expected=n,pfull=pfull,draft=draft,verify=verify,base=base,link=link,ops=ops,baseOps=target_base,variants=variants)
assert next(r for r in target_base if r['name']=='Q 投影')['flops']==131072
assert sum(r['flops'] for r in target_base if r['name'].endswith('Add'))==1024
assert abs(cases['2-0.5']['expected']-1.75)<1e-12
assert abs(cases['4-0.75']['expected']-3.05078125)<1e-12
assert all(c['variants'][2]['round']>=c['variants'][1]['round'] for c in cases.values())
bundle=dict(schema='ui-teaching-fixture/1',generated_by='build_fixture.py',
    source='教学预算；虚拟设备；独立离线预计算；未连接生产执行核；非实测',
    assumptions=['未融合FP16算子','矩阵2 TF/s、逐元素0.2 TF/s、带宽20 GB/s、每kernel启动4 µs，均为教学假设',
        'KV前缀理想读一次','链式逐位接受率相等；仅该教学状态机按全接受轮加一次catch-up',
        'DT每方向20 µs是人工预算；未实现真实随机采样的跨设备协议',
        '单请求固定上下文；不计有限输出末轮；无性能可信区间'],cases=cases)
payload=json.dumps(bundle,ensure_ascii=False,separators=(',',':'),allow_nan=False)
(OUT/'fixture.json').write_text(payload,encoding='utf-8')
(OUT/'fixture.js').write_text('window.SIM_FIXTURE='+payload+';',encoding='utf-8')
print('9 teaching scenarios; 5 independent arithmetic/invariant checks passed; sha256='+hashlib.sha256(payload.encode()).hexdigest())
