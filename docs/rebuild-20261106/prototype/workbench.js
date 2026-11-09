'use strict';
const $=id=>document.getElementById(id);
const fmt=(n,d=3)=>Number(n).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const compact=n=>n>=1e9?fmt(n/1e9,2)+' G':n>=1e6?fmt(n/1e6,2)+' M':n>=1e3?fmt(n/1e3,1)+' K':fmt(n,0);
const bytes=n=>n>=1048576?fmt(n/1048576,2)+' MiB':fmt(n/1024,2)+' KiB';
let loadedK=4,loadedAlpha=.75,pendingK=4,selected='dt',baseline='base',selectedOp='l0-Gate / Up';
let current=SIM_FIXTURE.cases['4-0.75'];
let toastTimer;
function toast(message){$('toast').textContent=message;$('toast').style.display='block';clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').style.display='none',2800);}
function dirty(){const changed=pendingK!==loadedK||Number($('alpha').value)!==loadedAlpha;$('stale').textContent=changed?'参数已改变，显示的仍是上一次结果':'当前结果与所选参数一致';$('stale').className=changed?'dirty':'';$('runid').textContent=changed?'结果待更新':`示例 #${loadedK}-${loadedAlpha}`;}
document.querySelectorAll('[data-k]').forEach(b=>b.onclick=()=>{pendingK=Number(b.dataset.k);document.querySelectorAll('[data-k]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));dirty();});
$('alpha').onchange=dirty;
$('apply').onclick=()=>{loadedK=pendingK;loadedAlpha=Number($('alpha').value);current=SIM_FIXTURE.cases[`${loadedK}-${loadedAlpha}`];dirty();renderToy();toast('方案、算子与时间预算已同步更新');};
function renderCompare(){const ref=current.variants.find(v=>v.id===baseline);$('compareRows').innerHTML=current.variants.map(v=>{
 const ratio=v.tpot==null?null:ref.tpot/v.tpot;
 return `<tr class="${v.id===selected?'selected':''}" data-variant="${v.id}"><td><button class="row-button" data-select="${v.id}" aria-pressed="${v.id===selected}">${v.name}</button></td><td>${fmt(v.round)} ms</td><td>${v.tpot==null?'不定义请求 TPOT':fmt(v.tpot)+' ms'}</td><td class="${ratio>1?'positive':''}">${ratio===null?'不参与比较':fmt(ratio,2)+'×'}</td><td><span class="tag ${v.kind==='bound'?'warn':'gray'}">${v.kind==='bound'?'乐观周期界':'教学预算'}</span></td><td>${v.kind==='bound'?'<span class="muted-inline">—</span>':`<button class="pin ${v.id===baseline?'active':''}" data-pin="${v.id}" aria-pressed="${v.id===baseline}">${v.id===baseline?'已固定':'设为基线'}</button>`}</td></tr>`;
 }).join('');
 $('expected').textContent=`E[N] = ${fmt(current.expected,4)} token/轮，包含纠正或 bonus。容量未建模，所有行均不作为真实部署推荐；微批周期与请求延迟分开。`;
 document.querySelectorAll('[data-select]').forEach(b=>b.onclick=()=>{selected=b.dataset.select;renderToy();});
 document.querySelectorAll('[data-pin]').forEach(b=>b.onclick=()=>{baseline=b.dataset.pin;renderCompare();toast('对比基线已固定');});
}
function ops(){return selected==='base'?current.baseOps:current.ops;}
function renderGraph(){const layer=Number($('layer').value);let list=ops().filter(x=>x.layer===layer);if(!list.some(x=>x.id===selectedOp))selectedOp=list[0].id;
 $('graphTitle').textContent=layer<0?'目标模型 · 边界算子':`目标模型 · 第 ${layer} 层`;
 $('graph').innerHTML=list.map(r=>`<button class="op ${r.id===selectedOp?'selected':''}" data-op="${r.id}" aria-pressed="${r.id===selectedOp}"><strong>${r.name}</strong><span>${fmt(r.ms*1000,1)} µs · ${r.bound}</span></button>`).join('');
 document.querySelectorAll('[data-op]').forEach(b=>b.onclick=()=>{selectedOp=b.dataset.op;renderGraph();renderInspector();});renderInspector();}
function renderInspector(){const r=ops().find(x=>x.id===selectedOp);$('opTitle').textContent=r.name;$('bound').textContent=r.bound+'受限';$('bound').className='tag';
 const accesses=[r.weight,r.act_read+r.act_write,r.kv_read+r.kv_write],total=accesses.reduce((a,b)=>a+b,0);
 const intensity=r.flops/Math.max(total,1),p=r.peak/1e9;
 const x=n=>42+Math.max(0,Math.min(250,(Math.log10(Math.max(.1,n))+1)/5*250));
 const y=n=>94-Math.max(0,Math.min(74,(Math.log10(Math.max(1,n)))/4*74));
 const knee=p/20,attainable=Math.min(p,intensity*20),point=`${x(intensity)},${y(attainable)}`;
 $('opDetails').innerHTML=`<div class="op-stats"><dl class="stat"><dt>理论预算 / 算子</dt><dd>${fmt(r.ms*1000,2)} µs</dd></dl><dl class="stat"><dt>逻辑工作量</dt><dd>${compact(r.flops)} FLOPs</dd></dl><dl class="stat"><dt>逻辑读写量</dt><dd>${bytes(total)}</dd></dl><dl class="stat"><dt>显存峰值</dt><dd>尚未建模</dd></dl></div><div class="byte-row">${accesses.map((v,i)=>`<span style="width:${total?v/total*100:0}%;background:${['#3568a6','#0b8179','#b28234'][i]}"></span>`).join('')}</div><div class="byte-labels"><span>权重 ${bytes(r.weight)}</span><span>激活 ${bytes(accesses[1])}</span><span>KV ${bytes(accesses[2])}</span></div><div class="formula"><div>${r.shape}</div><code>F = ${r.formula}</code><br><code>t = max(F / P, bytes / BW) + 4 µs</code></div><svg class="roof" viewBox="0 0 330 136" role="img" aria-label="教学设备 Roofline 示意，横轴为每字节FLOPs，纵轴为GFLOP每秒"><path d="M42 14V94H310" fill="none" stroke="#b9c9db"/><path d="M${x(.1)} ${y(2)}L${x(knee)} ${y(p)}H307" fill="none" stroke="#9ab3d2" stroke-width="2"/><circle cx="${point.split(',')[0]}" cy="${point.split(',')[1]}" r="4.5" fill="#00786f"/><text x="2" y="13">GFLOP/s</text><text x="220" y="131">FLOPs / Byte</text><text x="45" y="108">0.1</text><text x="93" y="108">1</text><text x="140" y="108">10</text><text x="191" y="108">100</text><text x="239" y="108">1k</text><text x="70" y="22">理想 roofline；启动开销另计</text></svg><p class="op-source">每卡 / 单次前向 · 工作量可复算，物理流量采用理想读假设。时延为教学估算；FLOPs公式与成本公式分别保留。选中节点对应下方目标前向。</p>`;
}
function renderTimeline(){const v=current.variants.find(x=>x.id===selected);$('timelineType').textContent='解析期望示意 / 非事件实录';
 if(selected==='ideal'){$('timelineTitle').textContent='理想周期 · 适用域';$('timeline').innerHTML=`<div class="teaching-bound">周期界 ${fmt(v.round)} ms，假设两个独立微批、资源充足、无竞争且可持续交错。本例 B=1 不具备该调度条件，因此不画一条虚假的并行时间线，也不计算单请求加速比。真实流水线需明确 cohort、请求和轮次。</div>`;$('timelineNote').textContent='解析下界作为研究参照保留，不能设为部署对比基线。';return;}
 $('timelineTitle').textContent='执行预算 · 单请求的一轮';
 let events=[],t=0;if(selected==='base'){events.push({lane:'target',name:'普通前向',start:0,duration:current.base});t=current.base;}
 else{events.push({lane:'draft',name:`${current.k} 步草稿`,start:0,duration:current.draft});t=current.draft;if(selected==='dt'){events.push({lane:'link',name:'D→T',start:t,duration:current.link});t+=current.link;}
 events.push({lane:'target',name:`Verify · ${current.k+1} token`,start:t,duration:current.verify});t+=current.verify;if(selected==='dt'){events.push({lane:'link',name:'T→D',start:t,duration:current.link});t+=current.link;}}
 $('timeline').innerHTML=['target','draft','link'].map(lane=>`<div class="track"><span class="track-label">${(selected==='co'?{target:'共享卡·目标',draft:'共享卡·草稿',link:'角色消息'}:{target:'目标设备',draft:'草稿设备',link:'角色消息'})[lane]}</span><div class="track-content">${events.filter(e=>e.lane===lane).map(e=>`<button class="bar ${lane}" data-lane="${lane}" style="left:${e.start/t*100}%;width:${e.duration/t*100}%" title="request=demo-1, round=sample；${e.name} ${fmt(e.duration)} ms">${e.name}</button>`).join('')}</div></div>`).join('')+`<div class="axis"><span>0 ms</span><span>${fmt(t/2)} ms</span><span>${fmt(t)} ms</span></div>`;
 document.querySelectorAll('[data-lane]').forEach(b=>b.onclick=()=>{if(b.dataset.lane==='target'){$('layer').value='0';selectedOp='l0-Attention';renderGraph();}else toast(b.dataset.lane==='draft'?'草稿成本来自1层小模型，catch-up按本例状态策略加权':'消息时延为每方向20 µs的教学预算，非真实协议测量');});
 $('timelineNote').textContent=`request=demo-1 / round=sample。${selected==='co'?'目标与草稿共享设备，按依赖串行。':''} 条长使用解析期望，不表示某个真实采样轮次。点击目标前向可定位 Attention。`;
}
$('layer').onchange=renderGraph;
function renderToy(){if($('model').value!=='toy')return;$('comparison').hidden=false;$('reviewPanel').hidden=true;$('graph').hidden=false;$('layerMap').hidden=true;$('layer').hidden=false;$('toyConfig').hidden=false;
 $('title').textContent='把收益拆到每一次前向';$('subtitle').textContent='微型 GQA 与结构展开的小草稿，比较三种串行执行预算。';$('notice').innerHTML='<span class="notice-dot"></span><div><strong>这是可复算的教学场景</strong><span>所有时间由虚拟设备预算离线生成；生产模型的证据状态另列。</span></div><button class="text-button" id="assumptionsBtn">查看假设</button>';bindEvidence();$('graphCaption').textContent='本层算子清单（不表示执行顺序）；选择一项查看字节与公式。';renderCompare();renderGraph();renderTimeline();dirty();}
function renderReview(){const model=$('model').value;if(model==='toy'){renderToy();return;}$('toyConfig').hidden=true;$('comparison').hidden=true;$('reviewPanel').hidden=false;$('graph').hidden=true;$('layer').hidden=true;$('layerMap').hidden=false;
 const meta={v41:{name:'DeepSeek V4.1',layers:40,full:[0,1],labels:['SWA','CSA2 / MoE'],state:'结构覆盖未闭合',text:'Config 身份已核对；Engram、hc_mult、候选块及草稿语义仍需完整审计。',issues:['Engram / hc_mult 尚未进入当前账本，容量结论需重新核对。','DSpark 当前链式代理不代表原生块算法，收益结果暂停发布。','TP通信与草稿缓存/临时内存缺失，不据旧报告判定可行性。']},next:{name:'Qwen3-Next',layers:48,full:Array.from({length:12},(_,i)=>3+i*4),labels:['全注意力','Gated DeltaNet'],state:'固定参考层序',text:'按已接入固定参考配置展示，0起始全注意力层为3、7、…、47。此轮未重新执行forward验证。',issues:['12层全注意力 + 36层Gated DeltaNet，需要逐层state与cache区分。','结构已展示不等于kernel成本已校准；本样例不输出性能预测。']},v3:{name:'DeepSeek-V3',layers:61,full:[0,1,2],labels:['MLA + Dense','MLA + MoE'],state:'固定参考层序',text:'61层MLA，前3层Dense，后58层MoE。此轮展示历史已接入结构，未重跑跨仓对账。',issues:['MLA缓存容量与prefill/decode实现分开记录。','路由专家、共享专家、EP通信和各rank容量必须分别核对。']}}[model];
 $('title').textContent=meta.name+' · 结构与证据';$('subtitle').textContent=meta.text;$('runid').textContent='结构样例 / 无性能结果';$('notice').innerHTML=`<span class="notice-dot"></span><div><strong>${meta.state}</strong><span>只展示已知结构；缺少依据的时间、容量和加速比不填数值。</span></div>`;
 $('reviewPanel').innerHTML=`<h3>影响结论的缺口</h3><ul>${meta.issues.map(x=>'<li>'+x+'</li>').join('')}</ul><a href="../evidence-audit.md">查看证据复核与源码版本</a>`;$('graphTitle').textContent='逐层结构';$('graphCaption').textContent='点击层号查看类型；颜色表示层类型，不表示性能可信度。';
 $('layerMap').innerHTML=`<div class="layer-grid">${Array.from({length:meta.layers},(_,i)=>`<button class="layer-cell ${meta.full.includes(i)?'full':''}" data-layer-index="${i}" title="Layer ${i}">${i}</button>`).join('')}</div><p class="layer-note">蓝色：${meta.labels[0]}。绿色：${meta.labels[1]}。层号从0开始。</p>`;
 const selectLayer=i=>{$('opTitle').textContent='Layer '+i;$('bound').textContent='结构证据';$('bound').className='tag gray';$('opDetails').innerHTML=`<h3>${meta.labels[meta.full.includes(i)?0:1]}</h3><p class="layer-note">${meta.text}</p><div class="formula">时延：未提供<br>FLOPs：本样例未展开<br>容量：尚未完成可行性检查<br>依据：固定配置与对应源码核对计划</div><a href="../principles.md">查看该架构的计数与状态契约</a>`;document.querySelectorAll('[data-layer-index]').forEach(b=>b.classList.toggle('active',Number(b.dataset.layerIndex)===i));};
 document.querySelectorAll('[data-layer-index]').forEach(b=>b.onclick=()=>selectLayer(Number(b.dataset.layerIndex)));selectLayer(0);$('timelineTitle').textContent='执行时间线';$('timelineType').textContent='等待合法执行计划';$('timeline').innerHTML='<div class="teaching-bound">当前没有带执行配方、资源依赖和完整成本的结果。结构图仍可检查；时间线将在执行计划通过验证后生成。</div>';$('timelineNote').textContent='未生成事件，不用示意条带冒充事件仿真结果。';
}
$('model').onchange=renderReview;
function bindEvidence(){const b=$('assumptionsBtn');if(b)b.onclick=()=>{$('assumptionsList').innerHTML=SIM_FIXTURE.assumptions.map(x=>'<li>'+x+'</li>').join('');$('evidence').showModal();};}
$('closeEvidence').onclick=()=>$('evidence').close();
$('export').onclick=()=>{const data=$('model').value==='toy'?{...current,source:SIM_FIXTURE.source,assumptions:SIM_FIXTURE.assumptions,selected,baseline,dirty:pendingK!==loadedK||Number($('alpha').value)!==loadedAlpha}:{model:$('model').value,kind:'structure-demo',performance:null,reason:'No validated execution result in this UI example'};const blob=new Blob([JSON.stringify(data,null,2)],{type:'application/json'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='simverse-ui-example.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);toast('已导出当前显示的结果与假设');};
renderToy();
