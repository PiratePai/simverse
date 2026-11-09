
(function() {
  // 1. Top-level Tab Switching
  var tabButtons = document.querySelectorAll('.main-tab-btn');
  var viewPanels = document.querySelectorAll('.view-panel');
  tabButtons.forEach(function(btn) {
    btn.addEventListener('click', function() {
      var targetId = btn.getAttribute('data-tab');
      tabButtons.forEach(function(b) { b.classList.remove('active'); });
      viewPanels.forEach(function(p) { p.classList.remove('active'); });
      btn.classList.add('active');
      var panel = document.getElementById(targetId);
      if (panel) panel.classList.add('active');
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
  });

  // 2. Interactive Workbench Simulation State & Data (Tab 4)
  var state = {
    target: 'dsv41',
    method: 'dflash2',
    deploy: 'dt_hetero',
    isl: 8192,
    alpha: 0.78,
    feat: 'cg_on',
    selectedOpId: 'op_target_moe',
    phaseFilter: 'all'
  };

  var methodSpecs = {
    mtp:     { name: 'MTP (1L 0.09B)',    s_draft: 4, m_draft: 1, k: 4, baseDraftMs: 0.92, commKb: 48,  isBlock: false },
    dspark:  { name: '链式 MoE 代理', s_draft: 4, m_draft: 1, k: 4, baseDraftMs: 2.45, commKb: 12,  isBlock: false },
    dflash:  { name: 'DFlash (B=8, it=2)', s_draft: 2, m_draft: 8, k: 8, baseDraftMs: 2.60, commKb: 16,  isBlock: true  },
    dflash2: { name: 'DFlash2 (B=8, it=1)',s_draft: 1, m_draft: 9, k: 8, baseDraftMs: 1.55, commKb: 16,  isBlock: true  },
    custom:  { name: 'Custom-1.8B (γ=4)',  s_draft: 4, m_draft: 1, k: 4, baseDraftMs: 2.10, commKb: 12,  isBlock: false }
  };

  var deployRows = [
    { key: 'colocated',    label: '② 同卡混部串行' },
    { key: 'dt_serial',    label: '③ 同构 DT 串行' },
    { key: 'dt_hetero',    label: '④ 异构 DT 串行 (H200+L40S)' },
    { key: 'dt_pipe_mb',   label: '⑤ 异构 DT 微批流水 (m=2)' },
    { key: 'dt_pipe_spec', label: '⑥ 异构 DT 超前流水 (β=0.82)' }
  ];

  function calcExpectedTokens(alpha, k, isBlock) {
    var effAlpha = isBlock ? Math.max(0.45, alpha - 0.04) : alpha;
    if (effAlpha >= 0.999) return k + 1;
    return (1 - Math.pow(effAlpha, k + 1)) / (1 - effAlpha);
  }

  function calcCellMetrics(deployKey, methodKey) {
    var m = methodSpecs[methodKey];
    var baseTargetDecodeMs = state.target === 'dsv41' ? 4.86 : 5.40;
    var islScale = state.isl / 8192;
    var featFactor = state.feat === 'eager' ? 1.16 : (state.feat === 'prefix50' ? 0.96 : 1.0);
    var baselineTpot = baseTargetDecodeMs * (0.88 + 0.12 * islScale) * featFactor;

    if (deployKey === 'no_spec') {
      return {
        baselineTpot: baselineTpot,
        tpot: baselineTpot,
        speedup: 1.00,
        en: 1.00,
        tDraft: 0,
        tCatchup: 0,
        tDraftRem: 0,
        tCommD2T: 0,
        tVerify: baselineTpot,
        tCommT2D: 0,
        tRound: baselineTpot,
        moeU: 1.00
      };
    }

    var en = calcExpectedTokens(state.alpha, m.k, m.isBlock);
    var mVerify = m.k + 1;
    var moeU = mVerify === 5 ? 2.68 : 4.12;
    var verifyScale = mVerify === 5 ? 1.32 : 1.54;
    var tVerify = baselineTpot * verifyScale;

    var draftHwFactor = (deployKey === 'colocated' || deployKey === 'dt_serial') ? 0.82 : (state.feat === 'ascend' ? 1.25 : 1.12);
    var tDraftTotal = m.baseDraftMs * draftHwFactor * featFactor;
    var tCatchup = m.isBlock ? +(tDraftTotal * 0.22).toFixed(2) : +(tDraftTotal / m.s_draft * 1.18).toFixed(2);
    var tDraftRem = +(tDraftTotal - tCatchup).toFixed(2);

    var tCommD2T = deployKey === 'colocated' ? 0.02 : (methodKey === 'mtp' ? 0.38 : 0.14);
    var tCommT2D = deployKey === 'colocated' ? 0.01 : (methodKey === 'mtp' ? 0.42 : 0.10);

    var tRound;
    if (deployKey === 'dt_pipe_mb') {
      tRound = Math.max(tDraftTotal + tCommD2T + tCommT2D, tVerify);
    } else if (deployKey === 'dt_pipe_spec') {
      var beta = 0.82;
      tRound = beta * Math.max(tDraftTotal + tCommD2T, tVerify) + (1 - beta) * (tDraftTotal + tVerify + tCommD2T + tCommT2D);
    } else {
      tRound = tDraftTotal + tCommD2T + tVerify + tCommT2D;
    }

    var tpot = tRound / en;
    var speedup = baselineTpot / tpot;
    return {
      baselineTpot: baselineTpot,
      tpot: tpot,
      speedup: speedup,
      en: en,
      tCatchup: tCatchup,
      tDraftRem: tDraftRem,
      tDraft: tDraftTotal,
      tCommD2T: tCommD2T,
      tVerify: tVerify,
      tCommT2D: tCommT2D,
      tRound: tRound,
      moeU: moeU
    };
  }

  function buildOpList(metrics) {
    var m = methodSpecs[state.method];
    var mVerify = state.deploy === 'no_spec' ? 1 : (m.k + 1);
    return [
      {
        id: 'op_draft_catchup',
        name: 'Draft Step 0 · Bonus Catch-up & Embed',
        role: 'Draft (' + (state.deploy === 'colocated' ? 'H200 TP8' : (state.feat === 'ascend' ? 'Ascend 910B TP1' : 'L40S TP1')) + ')',
        phase: 'draft_catchup',
        layerSpan: 'Draft L0..L_end (Step 0)',
        t_ms: metrics.tCatchup,
        flops_g: +(m.isBlock ? 36.8 : 7.4).toFixed(1),
        bytes_mb: { w: m.isBlock ? 1950 : 620, act_r: 4.2, act_w: 4.2, kv_r: 18.5, kv_w: 3.6 },
        mem_gb: { w: 3.8, kv: 1.6, act: 0.4 },
        act_shape: m.isBlock ? '[B=1, M_draft=9, d=2048]' : '[B=1, M_catchup=2, d=2048]',
        formula_id: 'F_DRAFT_CATCHUP_01',
        formula_expr: 't = max(2*M_0*N*K / Peak_draft, (W_draft + KV_draft) / BW_draft) + t_launch',
        ai: m.isBlock ? 18.4 : 11.2,
        bound: 'HBM 带宽受限'
      },
      {
        id: 'op_draft_propose',
        name: m.isBlock ? 'Draft Block Parallel Refine (M=' + m.m_draft + ')' : 'Draft Autoregressive Steps 1..' + (m.s_draft - 1) + ' (M=1)',
        role: 'Draft (' + (state.deploy === 'colocated' ? 'H200 TP8' : (state.feat === 'ascend' ? 'Ascend 910B TP1' : 'L40S TP1')) + ')',
        phase: 'draft_propose',
        layerSpan: m.isBlock ? 'Block Head & Parallel FFN' : 'Draft Steps 1..' + (m.s_draft - 1),
        t_ms: metrics.tDraftRem,
        flops_g: +(m.isBlock ? 28.4 : 18.6).toFixed(1),
        bytes_mb: { w: m.isBlock ? 1420 : 1840, act_r: 6.1, act_w: 6.1, kv_r: 42.0, kv_w: 5.4 },
        mem_gb: { w: 3.8, kv: 1.6, act: 0.4 },
        act_shape: '[B=1, M=' + m.m_draft + ', d=2048]',
        formula_id: 'F_SPEC_PROPOSE_02',
        formula_expr: 'T_propose = (S_draft - 1) * sum_op max(FLOPs_op/Peak, Bytes_op/BW)',
        ai: m.isBlock ? 19.8 : 9.8,
        bound: 'HBM 带宽受限'
      },
      {
        id: 'op_comm_d2t',
        name: 'DT Link Transfer (Draft→Target + Verify Feedback)',
        role: 'Inter-Role Link (RDMA / PCIe)',
        phase: 'dt_comm',
        layerSpan: 'Cross-Role Message + KV Rewind',
        t_ms: +(metrics.tCommD2T + metrics.tCommT2D).toFixed(2),
        flops_g: 0.1,
        bytes_mb: { w: 0, act_r: +(m.commKb / 1024).toFixed(3), act_w: +(m.commKb / 1024).toFixed(3), kv_r: 0, kv_w: 0 },
        mem_gb: { w: 0.0, kv: 0.0, act: 0.05 },
        act_shape: 'Msg(tokens=' + m.k + ', bonus=1, hidden=' + (state.method === 'mtp' ? 'n*7168' : '0') + ')',
        formula_id: 'F_COMM_DT_BIDIR_03',
        formula_expr: 't_comm = 2 * t_link_lat + (Bytes_D2T + Bytes_T2D) / BW_link + t_kv_rewind',
        ai: 0.5,
        bound: '通信延迟受限'
      },
      {
        id: 'op_target_mla',
        name: 'Target MLA Absorbed QKV + CSA2 Sparse Attn',
        role: 'Target (8×H200 TP8 · vllm-cuda)',
        phase: 'target_verify',
        layerSpan: 'Layers 0..60 (CSA2 topk=512+swa=128)',
        t_ms: +(metrics.tVerify * 0.34).toFixed(2),
        flops_g: +(42.5 * mVerify).toFixed(1),
        bytes_mb: { w: 980, act_r: 18.2, act_w: 18.2, kv_r: +(34.8 * (1 + 0.15 * (mVerify - 1))).toFixed(1), kv_w: +(0.89 * mVerify).toFixed(2) },
        mem_gb: { w: 14.2, kv: 18.6, act: 1.2 },
        act_shape: '[B=1, M_verify=' + mVerify + ', H_q=128, d_kv=512+64]',
        formula_id: 'F_MLA_CSA2_ABSORB_04',
        formula_expr: 'KV bytes: index / value / state separately; layout, reuse and TP replication require evidence',
        ai: +(14.0 * Math.pow(mVerify, 0.65)).toFixed(1),
        bound: mVerify >= 8 ? '算力与带宽均衡' : 'HBM 带宽受限'
      },
      {
        id: 'op_target_moe',
        name: 'Target MoE Routed + Shared Experts (FP8/FP4)',
        role: 'Target (8×H200 TP8/EP8 · vllm-cuda)',
        phase: 'target_verify',
        layerSpan: 'Layers 3..60 (CED Half-Layer Aware)',
        t_ms: +(metrics.tVerify * 0.54).toFixed(2),
        flops_g: +(86.0 * mVerify).toFixed(1),
        bytes_mb: { w: +(1120 * metrics.moeU).toFixed(0), act_r: 24.0, act_w: 24.0, kv_r: 0, kv_w: 0 },
        mem_gb: { w: 46.8, kv: 0.0, act: 1.6 },
        act_shape: '[B=1, M_verify=' + mVerify + ', d=7168, U(M)=' + metrics.moeU + 'x]',
        formula_id: 'F_MOE_UNION_ROOFLINE_05',
        formula_expr: 'E[U] = E * (1 - (1-k/E)^M), independent uniform top-k; W_read is recipe-dependent',
        ai: +(11.5 * mVerify / metrics.moeU).toFixed(1),
        bound: 'HBM 带宽受限'
      },
      {
        id: 'op_target_head',
        name: 'LM Head / Rejection Sampler 分组',
        role: 'Target (8×H200 TP8 · vllm-cuda)',
        phase: 'target_verify',
        layerSpan: 'Output Layer + Paged KV Rewind',
        t_ms: +(metrics.tVerify * 0.12).toFixed(2),
        flops_g: +(14.2 * mVerify).toFixed(1),
        bytes_mb: { w: 410, act_r: 4.5, act_w: 6.2, kv_r: 0, kv_w: 0 },
        mem_gb: { w: 3.2, kv: 0.0, act: 0.4 },
        act_shape: '[B=1, M_verify=' + mVerify + ', vocab=129280 ×2]',
        formula_id: 'F_DUAL_LMHEAD_VERIFY_06',
        formula_expr: 't = max(2 * (K+1) * d * V * 2 / Peak, W_head * 2 / BW) + t_allgather',
        ai: +(16.2 * mVerify).toFixed(1),
        bound: mVerify >= 5 ? 'Compute 算力受限' : 'HBM 带宽受限'
      }
    ];
  }

  function renderRooflineSvg(op) {
    var svg = document.getElementById('roofline-svg');
    if (!svg) return;
    var isTarget = op.role.indexOf('Target') === 0;
    var peakTflops = isTarget ? 1979 : (state.feat === 'ascend' ? 376 : 362);
    var bwTbs = isTarget ? 4.8 : (state.feat === 'ascend' ? 1.2 : 0.86);
    var ridgeAi = +(peakTflops / bwTbs).toFixed(1);

    function mapX(ai) {
      var clamped = Math.max(0.2, Math.min(800, ai));
      var norm = (Math.log10(clamped) - Math.log10(0.2)) / (Math.log10(800) - Math.log10(0.2));
      return 45 + norm * 310;
    }
    function mapY(tflops) {
      var clamped = Math.max(1, Math.min(3000, tflops));
      var norm = (Math.log10(clamped) - 0) / Math.log10(3000);
      return 130 - norm * 108;
    }

    var x0 = mapX(0.2);
    var y0 = mapY(0.2 * bwTbs);
    var xRidge = mapX(ridgeAi);
    var yPeak = mapY(peakTflops);
    var xMax = mapX(800);

    var opAttained = Math.min(peakTflops, op.ai * bwTbs) * 0.86;
    var cx = mapX(op.ai);
    var cy = mapY(Math.max(1.2, opAttained));

    document.getElementById('roofline-title').textContent =
      (isTarget ? 'H200 (FP8/BF16)' : (state.feat === 'ascend' ? 'Ascend 910B' : 'L40S')) +
      ' Roofline · 拐点 AI=' + ridgeAi + ' FLOP/B';
    document.getElementById('roofline-coord').textContent =
      'AI = ' + op.ai + ' FLOP/B · 实达 ≈ ' + opAttained.toFixed(0) + ' TFLOPS';

    svg.innerHTML =
      '<rect x="45" y="15" width="310" height="115" fill="#f8fafc" stroke="#e2e8f0" />' +
      '<line x1="' + xRidge + '" y1="15" x2="' + xRidge + '" y2="130" stroke="#cbd5e1" stroke-dasharray="3,3" />' +
      '<polyline fill="none" stroke="#2563eb" stroke-width="2.5" points="' + x0 + ',' + y0 + ' ' + xRidge + ',' + yPeak + ' ' + xMax + ',' + yPeak + '" />' +
      '<line x1="' + cx + '" y1="130" x2="' + cx + '" y2="' + cy + '" stroke="#0f766e" stroke-dasharray="2,2" />' +
      '<circle cx="' + cx + '" cy="' + cy + '" r="6" fill="#e11d48" stroke="#fff" stroke-width="2" />' +
      '<text x="' + Math.min(265, Math.max(50, cx - 25)) + '" y="' + Math.max(28, cy - 10) + '" font-size="10" font-weight="bold" fill="#be123c">' + op.name.split(' ')[0] + ' (' + op.t_ms + 'ms)</text>' +
      '<text x="50" y="27" font-size="9.5" fill="#475569">Peak ' + peakTflops + ' TFLOPS · BW ' + bwTbs + ' TB/s</text>' +
      '<text x="45" y="145" font-size="9.5" fill="#64748b">0.2</text>' +
      '<text x="' + (xRidge - 14) + '" y="145" font-size="9.5" fill="#1d4ed8">拐点 ' + ridgeAi + '</text>' +
      '<text x="330" y="145" font-size="9.5" fill="#64748b">800 FLOP/B</text>';
  }

  function renderWorkbench() {
    var curMetrics = calcCellMetrics(state.deploy, state.method);
    var m = methodSpecs[state.method];

    document.getElementById('wb-draft-recipe-badge').textContent =
      'Draft: ' + m.name + ' (' + (state.deploy === 'colocated' ? 'H200 同卡' : (state.feat === 'ascend' ? '910B · vllm-ascend' : 'L40S · 异构 DT')) + ')';

    document.getElementById('kpi-tpot').textContent = curMetrics.tpot.toFixed(2) + ' ms';
    var pct = (((curMetrics.tpot - curMetrics.baselineTpot) / curMetrics.baselineTpot) * 100).toFixed(1);
    document.getElementById('kpi-tpot-delta').textContent = '基线 ' + curMetrics.baselineTpot.toFixed(2) + ' ms (' + (pct > 0 ? '+' : '') + pct + '%)';
    document.getElementById('kpi-speedup').textContent = curMetrics.speedup.toFixed(2) + 'x';
    document.getElementById('kpi-throughput').textContent = (1000 / curMetrics.tpot).toFixed(1) + ' tokens/s';
    document.getElementById('kpi-en').textContent = curMetrics.en.toFixed(2) + ' tok';
    document.getElementById('kpi-k-info').textContent =
      state.deploy === 'no_spec' ? '无投机 (M=1)' : ('S_draft=' + m.s_draft + ', M_draft=' + m.m_draft + ', K=' + m.k);
    document.getElementById('kpi-tround').textContent = curMetrics.tRound.toFixed(2) + ' ms';
    document.getElementById('kpi-tround-break').textContent =
      'Draft ' + curMetrics.tDraft.toFixed(2) + ' + Comm ' + (curMetrics.tCommD2T + curMetrics.tCommT2D).toFixed(2) + ' + Verify ' + curMetrics.tVerify.toFixed(2);
    document.getElementById('kpi-moe-u').textContent = curMetrics.moeU.toFixed(2) + 'x';
    document.getElementById('kpi-phase-filter').textContent =
      '阶段过滤: ' + (state.phaseFilter === 'all' ? '全部阶段 (All)' : state.phaseFilter);

    var tbody = document.getElementById('matrix-tbody');
    var methodKeys = ['mtp', 'dspark', 'dflash', 'dflash2', 'custom'];
    var htmlRows = '';
    deployRows.forEach(function(row) {
      htmlRows += '<tr><td style="text-align:left;font-weight:600;">' + row.label + '</td>';
      methodKeys.forEach(function(mk) {
        var cell = calcCellMetrics(row.key, mk);
        var isSel = (state.deploy === row.key && state.method === mk);
        var bg = cell.speedup >= 2.0 ? '#dcfce7' : (cell.speedup >= 1.4 ? '#f0fdf4' : '#fef3c7');
        var color = cell.speedup >= 1.4 ? '#0f766e' : '#b45309';
        htmlRows += '<td class="mx-cell' + (isSel ? ' selected' : '') + '" data-deploy="' + row.key + '" data-method="' + mk + '" style="background:' + bg + ';color:' + color + ';">' +
          cell.speedup.toFixed(2) + 'x<div style="font-size:9.5px;font-weight:400;color:#475569;">' + cell.tpot.toFixed(2) + 'ms</div></td>';
      });
      htmlRows += '</tr>';
    });
    tbody.innerHTML = htmlRows;

    tbody.querySelectorAll('.mx-cell').forEach(function(td) {
      td.addEventListener('click', function() {
        state.deploy = td.getAttribute('data-deploy');
        state.method = td.getAttribute('data-method');
        document.getElementById('sel-deploy').value = state.deploy;
        document.getElementById('sel-method').value = state.method;
        renderWorkbench();
      });
    });

    document.getElementById('matrix-insight-box').innerHTML =
      '<strong>当前选中方案诊断：</strong> <code>' + state.deploy + ' × ' + m.name + '</code> — ' +
      (m.isBlock
        ? '并行块草稿仅需 <b>S_draft=' + m.s_draft + '</b> 次前向即可提议 8 个候选 token，避免了串行 γ 步的重复权重读取；'
        : '链式草稿需串行执行 <b>S_draft=' + m.s_draft + '</b> 步（其中 Step 0 采用原版 M=2 占位；真实输入由缓存状态决定）；') +
      '展示设定：验证步 M_verify=' + (m.k + 1) + ' 触发 MoE 专家并集放大 <b>' + curMetrics.moeU.toFixed(2) + 'x</b>。';

    var ops = buildOpList(curMetrics);
    if(state.deploy === 'no_spec') ops = ops.filter(function(o){return o.phase === 'target_verify';});
    ops.forEach(function(o){
      if(o.role.indexOf('Target')===0){
        o.role = state.target === 'qwen3' ? 'Target: Qwen3-Next · 配方待绑定' : 'Target: V4.1 Flash · 配方待绑定';
        o.layerSpan = '示意分组，不代表完整逐层图';
        if(o.id==='op_target_mla') {o.name=state.target==='qwen3'?'Gated DeltaNet + Full Attention 分组':'MLA / CSA2 / Indexer 分组';o.act_shape='模型相关形状待绑定；原数字仅作版式展示';}
        if(o.id==='op_comm_d2t') o.act_shape='候选 / 反馈 / hidden：实际通信协议待绑定';
        if(o.id==='op_target_moe') o.act_shape='路由/共享专家 · 实际维度与路由分布待绑定';
      }
    });
    var visibleOps=ops.filter(function(o){return state.phaseFilter==='all'||o.phase===state.phaseFilter;});
    if(!visibleOps.length){state.phaseFilter='all';visibleOps=ops;}
    if(!visibleOps.some(function(o){return o.id===state.selectedOpId;})) state.selectedOpId=visibleOps[0].id;

    var dagContainer = document.getElementById('dag-node-list');
    var dagHtml = '';
    ops.forEach(function(op) {
      if (state.phaseFilter !== 'all' && op.phase !== state.phaseFilter) return;
      var isAct = op.id === state.selectedOpId;
      var badgeClass = op.bound.indexOf('Compute') >= 0 ? 'cmp' : (op.bound.indexOf('通信') >= 0 ? 'com' : 'mem');
      dagHtml += '<div class="dag-node' + (isAct ? ' active' : '') + '" data-opid="' + op.id + '">' +
        '<div><div style="font-weight:700;">' + op.name + '</div>' +
        '<div style="font-size:11px;color:var(--ink3);">' + op.role + ' · ' + op.layerSpan + '</div></div>' +
        '<div style="text-align:right;"><div style="font-family:var(--mono);font-weight:800;color:var(--blue);">' + op.t_ms.toFixed(2) + ' ms</div>' +
        '<span class="dag-badge ' + badgeClass + '">' + op.bound + '</span></div></div>';
    });
    dagContainer.innerHTML = dagHtml;

    dagContainer.querySelectorAll('.dag-node').forEach(function(node) {
      node.addEventListener('click', function() {
        state.selectedOpId = node.getAttribute('data-opid');
        renderWorkbench();
      });
    });

    var selectedOp = ops.find(function(o) { return o.id === state.selectedOpId; }) || ops[0];
    document.getElementById('op-card-title').textContent = '③ 7 维透视：' + selectedOp.name;
    document.getElementById('op-bound-tag').textContent = selectedOp.bound;
    var totalBytesMb = (selectedOp.bytes_mb.w + selectedOp.bytes_mb.act_r + selectedOp.bytes_mb.act_w + selectedOp.bytes_mb.kv_r + selectedOp.bytes_mb.kv_w).toFixed(1);
    document.getElementById('op-detail-content').innerHTML =
      '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:6px;font-size:11.8px;margin-bottom:8px;">' +
        '<div style="background:#f8fafc;padding:6px 8px;border-radius:6px;border:1px solid var(--line);"><b>1. 示意时延</b><div style="font-family:var(--mono);font-size:14px;font-weight:800;color:var(--blue);">' + selectedOp.t_ms.toFixed(2) + ' ms</div></div>' +
        '<div style="background:#f8fafc;padding:6px 8px;border-radius:6px;border:1px solid var(--line);"><b>2. 计算量 FLOPs</b><div style="font-family:var(--mono);font-size:14px;font-weight:800;">' + selectedOp.flops_g + ' GFLOPs</div></div>' +
        '<div style="background:#f8fafc;padding:6px 8px;border-radius:6px;border:1px solid var(--line);"><b>3. 显存占用 (每卡)</b><div style="font-family:var(--mono);font-size:13px;font-weight:800;">W:' + selectedOp.mem_gb.w + ' / KV:' + selectedOp.mem_gb.kv + ' GB</div></div>' +
      '</div>' +
      '<div style="font-size:11.8px;background:#f8fafc;padding:7px 9px;border-radius:6px;border:1px solid var(--line);margin-bottom:6px;">' +
        '<b>4. 五类访存字节账本 (总计 ' + totalBytesMb + ' MB/step)：</b> ' +
        '<code>读权重 ' + selectedOp.bytes_mb.w + 'MB</code> · ' +
        '<code>读KV ' + selectedOp.bytes_mb.kv_r + 'MB</code> · ' +
        '<code>写KV ' + selectedOp.bytes_mb.kv_w + 'MB</code> · ' +
        '<code>读写激活 ' + (selectedOp.bytes_mb.act_r + selectedOp.bytes_mb.act_w).toFixed(1) + 'MB</code>' +
      '</div>' +
      '<div style="font-size:11.8px;margin-bottom:6px;"><b>5. 激活张量形状：</b> <code>' + selectedOp.act_shape + '</code></div>' +
      '<div style="font-size:11.5px;background:#0f172a;color:#e2e8f0;padding:6px 9px;border-radius:6px;font-family:var(--mono);">' +
        '<span style="color:#38bdf8;">[' + selectedOp.formula_id + ']</span> ' + selectedOp.formula_expr +
      '</div>';

    renderRooflineSvg(selectedOp);
    document.getElementById('op-detail-content').insertAdjacentHTML('afterbegin','<p class="sample-label">七维字段展示样本 · 数值与公式未作一致性验证</p>');

    var totalT = Math.max(0.1, curMetrics.tCatchup + curMetrics.tDraftRem + curMetrics.tCommD2T + curMetrics.tVerify + curMetrics.tCommT2D);
    var pCatch = (curMetrics.tCatchup / totalT * 100).toFixed(1);
    var pDraft = (curMetrics.tDraftRem / totalT * 100).toFixed(1);
    var pComm1 = (curMetrics.tCommD2T / totalT * 100).toFixed(1);
    var pVer   = (curMetrics.tVerify / totalT * 100).toFixed(1);
    var pComm2 = (curMetrics.tCommT2D / totalT * 100).toFixed(1);

    var isPipe = state.deploy.indexOf('dt_pipe') === 0;
    document.getElementById('tl-summary-label').textContent =
      (isPipe ? '流水线重叠模式 · 有效周期 = ' : '串行模式 · 单轮总耗时 = ') + curMetrics.tRound.toFixed(2) +
      ' ms · 期望提交 E[n] = ' + curMetrics.en.toFixed(2) + ' tokens (含1个Bonus) · 点击色块可过滤上方算子';

    var tlHtml =
      '<div class="tl-track">' +
        '<div class="tl-label">Draft 角色轨</div>' +
        '<div class="tl-bars">' +
          '<div class="tl-seg" data-phase="draft_catchup" style="width:' + pCatch + '%;background:#0d9488;" title="Catch-up 展示占位，实际输入由缓存状态确定">Catch-up ' + curMetrics.tCatchup.toFixed(2) + 'ms</div>' +
          '<div class="tl-seg" data-phase="draft_propose" style="width:' + pDraft + '%;background:#14b8a6;" title="Draft Propose">' + (m.isBlock ? 'Block Propose ' : 'Steps 1..' + (m.s_draft - 1) + ' ') + curMetrics.tDraftRem.toFixed(2) + 'ms</div>' +
          (isPipe ? '<div class="tl-seg" data-phase="draft_propose" style="width:' + pVer + '%;background:repeating-linear-gradient(45deg,#0d9488,#0d9488 6px,#115e59 6px,#115e59 12px);opacity:0.85;">下一微批/超前轮 Draft 重叠执行 (Overlapped)</div>' : '') +
        '</div>' +
      '</div>' +
      '<div class="tl-track">' +
        '<div class="tl-label">DT 链路通信轨</div>' +
        '<div class="tl-bars">' +
          '<div style="width:' + (parseFloat(pCatch) + parseFloat(pDraft)) + '%;"></div>' +
          '<div class="tl-seg" data-phase="dt_comm" style="width:' + Math.max(4, parseFloat(pComm1)) + '%;background:#7c3aed;">D→T</div>' +
          '<div style="width:' + Math.max(5, parseFloat(pVer) - 4) + '%;"></div>' +
          '<div class="tl-seg" data-phase="dt_comm" style="width:' + Math.max(5, parseFloat(pComm2)) + '%;background:#9333ea;">T→D+回退</div>' +
        '</div>' +
      '</div>' +
      '<div class="tl-track">' +
        '<div class="tl-label">Target 主模型轨</div>' +
        '<div class="tl-bars">' +
          (isPipe ? '' : '<div style="width:' + (parseFloat(pCatch) + parseFloat(pDraft) + parseFloat(pComm1)) + '%;"></div>') +
          '<div class="tl-seg" data-phase="target_verify" style="width:' + (isPipe ? 88 : pVer) + '%;background:#2563eb;">Target Verify (M=' + (m.k + 1) + ' tokens · MLA+CSA2+MoE · 产出 E[n]=' + curMetrics.en.toFixed(2) + ' tok) ' + curMetrics.tVerify.toFixed(2) + 'ms</div>' +
        '</div>' +
      '</div>';

    var tlContainer = document.getElementById('timeline-tracks-container');
    tlContainer.innerHTML = tlHtml;
    tlContainer.querySelectorAll('.tl-seg').forEach(function(seg) {
      seg.addEventListener('click', function() {
        var ph = seg.getAttribute('data-phase');
        state.phaseFilter = (state.phaseFilter === ph) ? 'all' : ph;
        renderWorkbench();
      });
    });
    reviewState(curMetrics);
  }

  function reviewState(metrics){
    var pipe=state.deploy.indexOf('dt_pipe')===0;
    document.getElementById('kpi-moe-u').nextElementSibling.textContent='验证输入与路由分布待绑定 · 当前为展示值';
    document.querySelectorAll('.mx-cell[data-deploy^="dt_pipe"]').forEach(function(c){c.textContent='待验证';c.style.background='#f1f5f9';c.style.color='#64748b';});
    document.querySelectorAll('.tl-seg[data-phase="target_verify"]').forEach(function(s){s.textContent='Target '+(state.deploy==='no_spec'?'Decode (M=1)':'Verify')+' · 模型算子分组示意';});
    document.getElementById('wb-active-recipe-badge').textContent='Target: '+(state.target==='qwen3'?'Qwen3-Next':'DeepSeek V4.1 Flash')+' · 角色配方示意';
    document.getElementById('review-status').textContent=(state.target==='dsv41'?'V4.1：Engram / hc / CSA2 字段覆盖未闭合。':'Qwen3-Next：固定参考层序48层，12层全注意力；本页算子分组为占位。')+' '+(state.method==='dspark'?'当前是链式MoE代理；原生DSpark块前向/Markov/confidence尚未接入。':'方法参数为原版展示预设，尚未验证。');
    document.getElementById('kpi-mem').textContent='待核实';
    if(pipe){
      document.getElementById('kpi-tpot').textContent='未提供';
      document.getElementById('kpi-speedup').textContent='不参与排名';
      document.getElementById('kpi-throughput').textContent='周期界 ≠ 单请求延迟';
      document.getElementById('kpi-tpot-delta').textContent='需 cohort / request / round 依赖';
      document.getElementById('tl-summary-label').textContent='流水方案待绑定合法事件图；下方按钮仅用于阶段筛选';
      document.getElementById('timeline-tracks-container').innerHTML='<p class="sample-label">不绘制未经因果验证的并行时间条。可继续检查阶段字段：</p>'+['draft_propose','dt_comm','target_verify'].map(function(ph){return '<button class="phase-demo" data-phase="'+ph+'">'+ph+'</button>';}).join('');
      document.querySelectorAll('.phase-demo').forEach(function(b){b.onclick=function(){state.phaseFilter=b.dataset.phase;renderWorkbench();};});
    } else if(state.deploy==='no_spec') {
      document.querySelectorAll('.tl-track').forEach(function(t){if(!t.textContent.includes('Target 主模型轨'))t.remove();});
    }
    document.querySelectorAll('.mx-cell,.dag-node,.tl-seg').forEach(function(e){e.setAttribute('role','button');e.tabIndex=0;e.onkeydown=function(k){if(k.key==='Enter'||k.key===' '){k.preventDefault();e.click();}};});
    window.exportUiScenario=function(){var b=new Blob([JSON.stringify({kind:'ui-display-only',scenario:state,metrics_are_verified:false,source:'simverse_v3_0.html',missing:['model lowering','method semantics','memory feasibility','cost calibration']},null,2)],{type:'application/json'});var a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='simverse-ui-scenario.json';a.click();setTimeout(function(){URL.revokeObjectURL(a.href);},1000);};
  }

  ['sel-target', 'sel-method', 'sel-deploy', 'inp-isl', 'inp-alpha', 'sel-feat'].forEach(function(id) {
    var el = document.getElementById(id);
    if (!el) return;
    el.addEventListener('change', function() {
      state.target = document.getElementById('sel-target').value;
      state.method = document.getElementById('sel-method').value;
      state.deploy = document.getElementById('sel-deploy').value;
      state.isl = parseInt(document.getElementById('inp-isl').value, 10);
      state.alpha = parseFloat(document.getElementById('inp-alpha').value);
      state.feat = document.getElementById('sel-feat').value;
      renderWorkbench();
    });
  });

  var resetBtn = document.getElementById('btn-reset-phase');
  if (resetBtn) {
    resetBtn.addEventListener('click', function() {
      state.phaseFilter = 'all';
      renderWorkbench();
    });
  }

  renderWorkbench();
})();
