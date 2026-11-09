/* Simverse 生产工作台 v0.2 —— 前端零物理计算。
 * 所有数字来自 /api/v2/run ResultBundle；本文件只做：
 * 格式化、单位/坐标投影、筛选、ID 联动、导出拼装。失败时显示原因，不回退演示数字。*/
(function () {
  'use strict';
  var state = {
    catalog: null, bundle: null, selectedRowKey: null,
    matrixRuns: {}, scenarioOf: null
  };
  var $ = function (id) { return document.getElementById(id); };

  function fmt(v, unit, digits) {
    if (v === null || v === undefined) return '–';
    if (typeof v === 'number') {
      var d = digits === undefined ? 3 : digits;
      return v.toFixed(d) + (unit ? ' ' + unit : '');
    }
    return String(v);
  }
  function fmtBytes(b) {
    if (b === null || b === undefined) return '–';
    if (b >= 1e9) return (b / 1e9).toFixed(2) + ' GB';
    if (b >= 1e6) return (b / 1e6).toFixed(1) + ' MB';
    if (b >= 1e3) return (b / 1e3).toFixed(1) + ' KB';
    return b + ' B';
  }
  function restoreBig(o) {  // $bigint 还原
    if (o && typeof o === 'object' && !Array.isArray(o)) {
      if (o.$bigint !== undefined) return parseInt(o.$bigint, 10);
      var out = {};
      Object.keys(o).forEach(function (k) { out[k] = restoreBig(o[k]); });
      return out;
    }
    if (Array.isArray(o)) return o.map(restoreBig);
    return o;
  }

  // ---------- catalog ----------
  function loadCatalog() {
    fetch('/api/v2/catalog').then(function (r) { return r.json(); }).then(function (cat) {
      state.catalog = cat;
      var models = Object.keys(cat.catalog.models).filter(function (k) {
        return !k.startsWith('tiny_');
      });
      var sel = $('sel-model');
      sel.innerHTML = '';
      models.forEach(function (k) {
        var m = cat.catalog.models[k];
        var o = document.createElement('option');
        o.value = k;
        o.textContent = m.label + (m.structure_complete ? '' : ' [结构缺口]');
        sel.appendChild(o);
      });
      sel.value = models.indexOf('qwen36_35b_a3b') >= 0 ? 'qwen36_35b_a3b' : models[0];
      var ms = $('sel-method');
      ms.innerHTML = '';
      cat.methods.supported.forEach(function (mid) {
        var o = document.createElement('option');
        o.value = mid; o.textContent = mid;
        ms.appendChild(o);
      });
      ms.value = 'dflash_v1';
      var dv = $('sel-target-dev'), dd = $('sel-draft-dev');
      dv.innerHTML = dd.innerHTML = '';
      Object.keys(cat.devices).forEach(function (k) {
        var label = cat.devices[k].label;
        [dv, dd].forEach(function (s) {
          var o = document.createElement('option');
          o.value = k; o.textContent = label;
          s.appendChild(o);
        });
      });
      dv.value = 'h100_sxm'; dd.value = 'l40';
      renderCatalogDrawer();
      renderMatrixHead();
    }).catch(function (e) { setStatus('能力目录加载失败：' + e, true); });
  }

  function currentScenario(over) {
    var accSel = $('sel-acc').value;
    var acceptance;
    if (accSel === 'mean2') acceptance = { kind: 'committed_mean', mean: 2.0 };
    else if (accSel === 'dynk') acceptance = { kind: 'dynamic_k' };
    else acceptance = { kind: 'geometric', alpha: parseFloat(accSel) };
    var sc = {
      schema_version: '0.2',
      name: 'workbench-' + $('sel-model').value + '-' + $('sel-method').value,
      model_id: $('sel-model').value,
      method_id: $('sel-method').value,
      draft_id: $('sel-method').value === 'dflash_v1'
        ? 'dflash_qwen36_35b_a3b'
        : ($('sel-method').value === 'baseline' ? null : 'dflash_qwen36_35b_a3b'),
      k: $('sel-method').value === 'dflash_v1' ? 15 : 4,
      block_size: parseInt($('inp-blk').value, 10),
      deployment: {
        mode: $('sel-method').value === 'baseline' ? 'baseline' : $('sel-deploy').value,
        target_device: $('sel-target-dev').value,
        target_tp: parseInt($('sel-tp').value, 10),
        draft_device: $('sel-draft-dev').value, draft_tp: 1
      },
      workload: {
        batch: parseInt($('inp-batch').value, 10),
        ctx_tokens: parseInt($('inp-ctx').value, 10)
      },
      acceptance: acceptance,
      quant: { dense_w: 8, expert_w: 4, act: 8, kv: 8, state: 32 },
      draft_quant: { dense_w: 8, expert_w: 8, act: 8, kv: 8, state: 32 },
      seed: 0
    };
    if (over) {
      Object.keys(over).forEach(function (k) {
        if (typeof over[k] === 'object' && typeof sc[k] === 'object')
          Object.assign(sc[k], over[k]);
        else sc[k] = over[k];
      });
    }
    return sc;
  }

  function setStatus(text, isErr, mono) {
    var el = $('run-status');
    el.className = 'status-strip' + (isErr ? ' err' : '');
    el.innerHTML = (mono ? '<span class="mono">' + text + '</span>' : text);
  }

  // ---------- run ----------
  function runScenario(sc) {
    return fetch('/api/v2/run', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(sc)
    }).then(function (r) {
      return r.json().then(function (j) {
        if (!r.ok) throw new Error(j.error || r.status);
        return restoreBig(j);
      });
    });
  }

  function doRun() {
    var sc = currentScenario();
    state.scenarioOf = sc;
    $('btn-run').disabled = true;
    setStatus('运行中… ' + sc.model_id + ' × ' + sc.method_id);
    runScenario(sc).then(function (b) {
      state.bundle = b;
      renderBundle();
      setStatus('run_id <span class="mono">' + b.manifest.run_id + '</span> · ' +
        '代码 ' + b.manifest.code_revision.slice(0, 10) + ' · 全部数字 T0 解析估算（Q2）·' +
        ' 接受率为假设输入（Q8）');
    }).catch(function (e) {
      setStatus('运行失败：' + e.message, true);
    }).finally(function () { $('btn-run').disabled = false; });
  }

  // ---------- render bundle ----------
  function metric(b, id) { return (b.metrics[id]) || null; }
  function val(b, id) { var m = metric(b, id); return m ? m.value : null; }
  function reason(b, id) { var m = metric(b, id); return m && m.reason ? m.reason : null; }

  function renderBundle() {
    var b = state.bundle;
    var tpot = val(b, 'M01_tpot_analytical');
    var tpotReason = reason(b, 'M01_tpot_analytical');
    $('kpi-tpot').textContent = tpot !== null ? fmt(tpot, 'ms', 4) : '未知';
    $('kpi-tpot-sub').textContent = tpotReason
      ? '原因：' + tpotReason.slice(0, 46)
      : 'E[T_round]/E[N] · committed · 解析';
    var base = val(b, 'M01_tpot_baseline');
    var sp = metric(b, 'M02_speedup_vs_baseline');
    $('kpi-speedup').textContent = sp && sp.value !== null
      ? fmt(sp.value, '×', 3) : '未知';
    $('kpi-speedup-sub').innerHTML = sp && sp.reason
      ? '⚠ ' + sp.reason : ('基线 ' + fmt(base, 'ms', 3) + '（同 B/ctx）');
    var en = val(b, 'M03_E_N');
    $('kpi-en').textContent = en !== null ? fmt(en, 'tok', 3)
      + ' / K=' + (val(b, 'M03_K') || 0) : '未知';
    $('kpi-en-sub').textContent = reason(b, 'M03_E_N')
      ? '原因：' + reason(b, 'M03_E_N').slice(0, 40) : '假设输入（Q8）';
    var round = val(b, 'M04_round_ms');
    $('kpi-round').textContent = round !== null ? fmt(round, 'ms', 3) : '未知';
    var br = b.metrics['M04_stage_breakdown_ms'] || {};
    $('kpi-round-sub').textContent = round !== null
      ? Object.keys(br).map(function (k) {
        return k.replace('_block', '').replace('draft_', 'd:') +
          ' ' + fmt(br[k].value, '', 2);
      }).join(' + ').slice(0, 60) : '—';
    var moe = metric(b, 'M05_moe_union');
    $('kpi-moe').textContent = moe ? fmtBytes(moe.value) + '/层' : '–';
    var memKey = Object.keys(b.metrics).filter(function (k) {
      return k.indexOf('M06_memory') === 0;
    })[0];
    var mem = memKey ? metric(b, memKey) : null;
    $('kpi-mem').textContent = mem ? fmtBytes(mem.value) : '–';
    $('kpi-mem-sub').textContent = mem
      ? (mem.feasibility === 'oom' ? 'OOM（超预算）'
        : mem.feasibility === 'unknown' ? 'feasible 判定 unknown（仅标称容量）'
        : '可分配预算内')
      : '–';
    var checks = (b.capability && b.capability.checks) || {};
    var bad = Object.keys(checks).filter(function (k) {
      return checks[k].status !== 'supported';
    });
    $('kpi-ev').textContent = (b.model_ref.structure_complete ? '结构完备' : '结构缺口')
      + ' · ' + (bad.length ? bad.length + ' 项注意' : '无阻断');
    $('kpi-ev-sub').textContent = '解析估算 · 假设 ' +
      Object.keys(b.assumption_dict || {}).length + ' 条';
    renderOps(b);
    renderTimeline(b);
    renderEvidence(b);
  }

  // ---------- op ledger ----------
  function targetRows(b) {
    var w = (b.work || {})['target_decode'] || { rows: [] };
    return w.rows || [];
  }

  function renderOps(b) {
    var rows = targetRows(b);
    var list = $('dag-list');
    list.innerHTML = '';
    // 聚合展示：按 op 名汇总（保留 layer 范围），点击展开行级
    var agg = {};
    rows.forEach(function (r, i) {
      var k = r.op;
      if (!agg[k]) agg[k] = { op: r.op, n: 0, flops: 0, t: 0, kv: 0,
        layers: [r.layer, r.layer], first: i };
      agg[k].n++; agg[k].flops += r.flops; agg[k].t += r.t_ms || 0;
      agg[k].kv += r.kv_read;
      agg[k].layers[1] = r.layer;
    });
    var keys = Object.keys(agg).sort(function (a, c) {
      return agg[c].t - agg[a].t;
    }).slice(0, 40);
    $('op-count-tag').textContent = rows.length + ' 行';
    keys.forEach(function (k) {
      var a = agg[k];
      var node = document.createElement('div');
      node.className = 'dag-node';
      node.setAttribute('role', 'button'); node.tabIndex = 0;
      node.innerHTML = '<div><div style="font-weight:700">' + a.op + '</div>' +
        '<div style="font-size:11px;color:var(--ink3)">×' + a.n + ' · 层 ' +
        (a.layers[0] === a.layers[1] ? a.layers[0] : a.layers[0] + '…' + a.layers[1]) +
        '</div></div><div style="text-align:right">' +
        '<div style="font-family:var(--mono);font-weight:800;color:var(--blue)">' +
        fmt(a.t, 'ms', 4) + '</div>' +
        '<span class="dag-badge ' + (a.kv > 0 ? 'mem' : 'cmp') + '">' +
        (a.flops / 1e6).toFixed(1) + ' MF</span></div>';
      var activate = function () {
        state.selectedRowKey = k;
        renderInspector(agg[k], rows.filter(function (r) { return r.op === k; })[0]);
        list.querySelectorAll('.dag-node').forEach(function (n2) {
          n2.classList.remove('active');
        });
        node.classList.add('active');
      };
      node.addEventListener('click', activate);
      node.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); activate(); }
      });
      list.appendChild(node);
    });
    if (keys.length) {
      state.selectedRowKey = keys[0];
      renderInspector(agg[keys[0]],
        rows.filter(function (r) { return r.op === keys[0]; })[0]);
      list.firstChild.classList.add('active');
    }
  }

  function renderInspector(agg, row) {
    if (!agg) { $('inspector').innerHTML = '<p>无账本行。</p>'; return; }
    var r = row || {};
    var wrex = r.weight_read_expected;
    $('inspector-title').textContent = '③ ' + agg.op + ' 七维透视';
    $('op-bound-tag').textContent = r.bound || '–';
    $('op-bound-tag').className = 'tag ' +
      (r.bound === 'memory' ? 'warn' : r.bound === 'compute' ? 'ok' : 'na');
    var bytes = (r.weight_read || 0) + (r.act_read || 0) + (r.act_write || 0) +
      (r.kv_read || 0) + (r.kv_write || 0) + (r.state_read || 0);
    var html = '<div class="op-grid">' +
      '<div class="op-cell"><b>1 时延（T0 解析）</b><span class="v">' +
      fmt(agg.t, 'ms', 4) + '</span></div>' +
      '<div class="op-cell"><b>2 逻辑算量</b><span class="v">' +
      (agg.flops / 1e6).toFixed(2) + ' MFLOPs</span></div>' +
      '<div class="op-cell"><b>3 驻留容量</b><span class="v">' +
      fmtBytes(r.weight_capacity || 0) + '</span></div></div>' +
      '<div class="op-cell" style="margin-bottom:6px"><b>4 五类字节（聚合，精确 int' +
      (wrex ? ' + 期望 float' : '') + '）</b>' +
      'W ' + fmtBytes(agg.w || (r.weight_read || 0)) +
      (wrex ? '（期望另计 ' + fmtBytes(wrex) + '）' : '') +
      ' · A ' + fmtBytes(r.act_read || 0) + '/' + fmtBytes(r.act_write || 0) +
      ' · KV 读 ' + fmtBytes(r.kv_read || 0) + ' 写 ' + fmtBytes(r.kv_write || 0) +
      (r.state_read ? ' · 状态 ' + fmtBytes(r.state_read) : '') + '</div>' +
      '<div class="kv"><b>5 形状：</b><code>' + (r.shape || '') + '</code></div>' +
      '<div class="kv"><b>6 精度/统计身份：</b><span class="tag ' +
      (r.statistic === 'expectation' ? 'warn' : 'ok') + '">' +
      (r.statistic || 'exact') + '</span> 权重 8b/4b · 激活 8b · KV 8b</div>' +
      '<div class="formula-box"><span class="fid">[' + (r.formula || '') + ']</span> ' +
      formulaExpr(r.formula) + '</div>';
    $('inspector').innerHTML = html;
    drawRoofline(r);
  }

  function formulaExpr(fid) {
    var f = state.catalog && null;  // 公式文本来自 bundle.formula_dict
    var fd = state.bundle && state.bundle.formula_dict;
    if (fd && fd[fid]) return fd[fid].expr + ' —— ' + fd[fid].source;
    return '';
  }

  // ---------- roofline（坐标全部由后端 cost.roofline_points 提供） ----------
  function drawRoofline(row) {
    var svg = $('roofline-svg');
    var b = state.bundle;
    var pts = (b.cost && b.cost.roofline_points) || [];
    var dev = b.scenario_snapshot.deployment.target_device;
    var dspec = state.catalog.devices[dev];
    var peak = dspec ? dspec.peak_by_dtype.fp8 / 1e12 : 1979;
    var bw = dspec ? dspec.hbm_bw / 1e12 : 3.35;
    var pt = pts.filter(function (p) { return p.op === row.op; })[0] || pts[0];
    function mapX(ai) {
      var c = Math.max(0.2, Math.min(800, ai));
      return 45 + (Math.log10(c) - Math.log10(0.2)) /
        (Math.log10(800) - Math.log10(0.2)) * 310;
    }
    function mapY(tf) {
      var c = Math.max(1, Math.min(3000, tf));
      return 128 - (Math.log10(c) / Math.log10(3000)) * 108;
    }
    var ridge = peak / bw;
    var x0 = mapX(0.2), y0 = mapY(0.2 * bw);
    var xr = mapX(ridge), yp = mapY(peak), xm = mapX(800);
    var cx = pt ? mapX(pt.ai_flop_per_byte) : mapX(1);
    var cy = pt ? mapY(Math.max(1, pt.attained_tflops)) : mapY(1);
    svg.innerHTML =
      '<rect x="45" y="12" width="310" height="116" fill="#f8fafc" stroke="#e2e8f0"/>' +
      '<line x1="' + xr + '" y1="12" x2="' + xr + '" y2="128" stroke="#cbd5e1" stroke-dasharray="3,3"/>' +
      '<polyline fill="none" stroke="#2563eb" stroke-width="2.5" points="' +
      x0 + ',' + y0 + ' ' + xr + ',' + yp + ' ' + xm + ',' + yp + '"/>' +
      (pt ? '<line x1="' + cx + '" y1="128" x2="' + cx + '" y2="' + cy +
        '" stroke="#0f766e" stroke-dasharray="2,2"/>' +
        '<circle cx="' + cx + '" cy="' + cy + '" r="6" fill="#e11d48" stroke="#fff" stroke-width="2"/>'
        : '') +
      '<text x="50" y="24" font-size="9.5" fill="#475569">峰 ' + peak.toFixed(0) +
      ' TF/s · BW ' + bw.toFixed(2) + ' TB/s（规格口径）</text>' +
      '<text x="' + (xr - 16) + '" y="142" font-size="9.5" fill="#1d4ed8">拐点 ' +
      ridge.toFixed(0) + '</text>' +
      '<text x="45" y="142" font-size="9.5" fill="#64748b">0.2</text>' +
      '<text x="330" y="142" font-size="9.5" fill="#64748b">800 F/B</text>';
    $('roofline-coord').textContent = pt
      ? 'AI=' + pt.ai_flop_per_byte.toFixed(1) + ' F/B · 实达 ' +
        pt.attained_tflops.toFixed(0) + ' TF/s（T0 估算）' : '无有效点（F=0 或 Q=0）';
  }

  // ---------- timeline（事件来自 execution.rounds，含依赖） ----------
  function renderTimeline(b) {
    var rounds = (b.execution && b.execution.rounds) || [];
    var evs = (rounds[0] && rounds[0].events) || [];
    var el = $('timeline');
    if (!evs.length) {
      el.innerHTML = '<div class="unknown-note">无事件（baseline 或接受分布未知：'
        + (b.execution && b.execution.note || '') + '）</div>';
      $('tl-summary').textContent = '–';
      return;
    }
    var total = evs[evs.length - 1].start_ms + evs[evs.length - 1].t_ms;
    var lanes = { draft: [], link: [], target: [], commit: [] };
    evs.forEach(function (e) { (lanes[e.role] || lanes.target).push(e); });
    var colors = { draft: '#0d9488', link: '#7c3aed', target: '#2563eb',
      commit: '#334155' };
    var html = '';
    [['draft', 'Draft 角色轨'], ['link', 'DT 链路轨'], ['target', 'Target 角色轨']]
      .forEach(function (pair) {
        var lane = lanes[pair[0]].concat(pair[0] === 'target' ? lanes.commit : []);
        html += '<div class="tl-track"><div class="tl-label">' + pair[1] +
          '</div><div class="tl-bars">';
        if (!lane.length) {
          html += '<div style="width:100%;display:flex;align-items:center;' +
            'justify-content:center;color:#94a3b8;font-size:11px">（本部署无此轨事件）</div>';
        }
        lane.forEach(function (e) {
          var l = Math.max(1.2, e.t_ms / total * 100);
          var off = e.start_ms / total * 100;
          html += '<div class="tl-seg" title="' + e.event_id + ' ← ' +
            (e.depends_on || []).join(',') + '" style="position:absolute;left:' +
            off + '%;width:' + l + '%;background:' + colors[e.role] + '">' +
            e.stage + ' ' + fmt(e.t_ms, '', 2) + '</div>';
        });
        html += '</div></div>';
      });
    el.innerHTML = html;
    var en = val(b, 'M03_E_N');
    $('tl-summary').textContent = 'T_round=' + fmt(total, 'ms', 3) +
      '（串行：阶段和=墙钟）· E[N]=' + fmt(en, '', 2) + ' · 事件依赖可追溯（悬停查看）';
  }

  // ---------- evidence drawer ----------
  function renderEvidence(b) {
    var checks = (b.capability && b.capability.checks) || {};
    var html = '<table class="plain"><tr><th>检查</th><th>状态</th><th>原因/受影响</th></tr>';
    Object.keys(checks).forEach(function (k) {
      var c = checks[k];
      var cls = c.status === 'supported' ? 'ok' : c.status === 'unknown' ? 'warn' : 'bad';
      html += '<tr><td>' + k + '</td><td><span class="tag ' + cls + '">' + c.status +
        '</span></td><td>' + c.reason +
        (c.affected_metrics && c.affected_metrics.length
          ? '<br><i>受影响：' + c.affected_metrics.join(', ') + '</i>' : '') +
        '</td></tr>';
    });
    html += '</table><h4>假设（全部随 run 导出）</h4><ul>';
    Object.keys(b.assumption_dict || {}).forEach(function (k) {
      html += '<li><code>' + k + '</code>：' + b.assumption_dict[k] + '</li>';
    });
    html += '</ul><h4>证据引用</h4><p><code>' +
      (b.evidence_refs || []).join('</code> · <code>') + '</code></p>';
    $('evidence-body').innerHTML = html;
  }

  function renderCatalogDrawer() {
    var cat = state.catalog.catalog;
    var html = '<table class="plain"><tr><th>模型</th><th>家族/层序</th><th>身份</th>' +
      '<th>覆盖/缺口</th></tr>';
    Object.keys(cat.models).forEach(function (k) {
      var m = cat.models[k];
      var cov = { mapped: 0, 'metadata-only': 0, 'ignored-with-reason': 0,
        'unsupported-semantic': 0 };
      Object.keys(m.coverage || {}).forEach(function (ck) {
        var s = m.coverage[ck].status;
        cov[s] = (cov[s] || 0) + 1;
      });
      html += '<tr><td><b>' + k + '</b></td><td>' + m.family + ' · ' +
        JSON.stringify(m.kind_counts) + '</td><td style="font-size:11.5px">' +
        (m.source.repo_id || '') + '@' + (m.source.revision || '').slice(0, 10) +
        '<br><code style="font-size:10.5px">' + (m.source.config_sha256 || '').slice(0, 16) +
        '</code></td><td>' + (m.structure_complete
          ? '<span class="tag ok">结构完备</span>'
          : '<span class="tag bad">' + m.gaps.length + ' 缺口</span> ' +
            (m.gaps[0] || '').slice(0, 40)) +
        '<br><span style="font-size:11px;color:var(--ink3)">mapped ' + (cov.mapped || 0) +
        ' · meta ' + (cov['metadata-only'] || 0) + ' · ignored ' +
        (cov['ignored-with-reason'] || 0) + ' · unsupported ' +
        (cov['unsupported-semantic'] || 0) + '</span></td></tr>';
    });
    html += '</table><h4>草稿</h4><table class="plain"><tr><th>id</th><th>结构</th>' +
      '<th>绑定</th></tr>';
    Object.keys(cat.drafts).forEach(function (k) {
      var d = cat.drafts[k];
      html += '<tr><td>' + k + '</td><td>' + d.n_layers + ' 层 [' +
        d.layer_kinds.join(',') + '] blk=' + d.block_size +
        '</td><td style="font-size:11.5px">' + d.source.repo_id + '@' +
        d.source.revision.slice(0, 10) + '</td></tr>';
    });
    html += '</table><h4>待核实资产（如实标注）</h4><ul>';
    Object.keys(cat.pending_assets).forEach(function (k) {
      var p = cat.pending_assets[k];
      html += '<li><b>' + k + '</b>：' + p.reason + '</li>';
    });
    $('catalog-body').innerHTML = html + '</ul>';
  }

  // ---------- matrix（每格独立 run） ----------
  var METHODS = ['dflash_v1', 'chain_draft', 'mtp', 'chain_moe_proxy'];
  var DEPLOYS = [
    { key: 'colocated', label: '同卡混部' },
    { key: 'dt_serial', label: 'DT 分离串行' }
  ];

  function renderMatrixHead() {
    var head = '<tr><th style="text-align:left">部署 \\ 方法</th>' +
      METHODS.map(function (m) { return '<th>' + m + '</th>'; }).join('') + '</tr>';
    $('matrix-head').innerHTML = head;
    var body = '';
    DEPLOYS.forEach(function (d) {
      body += '<tr><td style="text-align:left;font-weight:600">' + d.label + '</td>' +
        METHODS.map(function (m) {
          return '<td class="mx-cell pending" id="mx-' + d.key + '-' + m +
            '" data-deploy="' + d.key + '" data-method="' + m +
            '">待运行</td>';
        }).join('') + '</tr>';
    });
    body += '<tr><td style="text-align:left;font-weight:600">无投机基线</td>' +
      '<td class="mx-cell pending" id="mx-baseline" data-deploy="baseline" ' +
      'data-method="baseline" colspan="' + METHODS.length +
      '">待运行</td></tr>';
    $('matrix-body').innerHTML = body;
    $('matrix-body').querySelectorAll('.mx-cell').forEach(function (td) {
      td.setAttribute('role', 'button'); td.tabIndex = 0;
      var act = function () { runCell(td); };
      td.addEventListener('click', act);
      td.addEventListener('keydown', function (e) {
        if (e.key === 'Enter') { e.preventDefault(); act(); }
      });
    });
  }

  function cellScenario(deploy, method) {
    return currentScenario({
      method_id: method,
      draft_id: method === 'dflash_v1' ? 'dflash_qwen36_35b_a3b'
        : (method === 'baseline' ? null : 'dflash_qwen36_35b_a3b'),
      deployment: { mode: deploy },
      k: method === 'dflash_v1' ? 15 : (method === 'mtp' ? 1 : 4),
      name: 'matrix-' + deploy + '-' + method
    });
  }

  function runCell(td) {
    var deploy = td.getAttribute('data-deploy');
    var method = td.getAttribute('data-method');
    td.className = 'mx-cell pending'; td.textContent = '运行中…';
    runScenario(cellScenario(deploy, method)).then(function (b) {
      state.matrixRuns[deploy + '-' + method] = b;
      var tpot = val(b, 'M01_tpot_analytical');
      var sp = metric(b, 'M02_speedup_vs_baseline');
      if (tpot === null) {
        td.className = 'mx-cell na';
        td.innerHTML = '未知<div style="font-size:9.5px;font-weight:400">' +
          (reason(b, 'M01_tpot_analytical') || '').slice(0, 24) + '</div>';
        return;
      }
      var ineligible = sp && sp.reason;
      td.className = 'mx-cell' + (ineligible ? ' na' : '');
      td.innerHTML = fmt(sp ? sp.value : 0, '×', 2) +
        '<div style="font-size:9.5px;font-weight:400;color:#475569">' +
        fmt(tpot, 'ms', 2) + (ineligible ? ' ·不排名' : '') + '</div>';
      td.title = 'run_id ' + b.manifest.run_id.slice(7, 19) +
        (ineligible ? '\n' + sp.reason : '');
    }).catch(function (e) {
      td.className = 'mx-cell bad';
      td.textContent = '失败：' + String(e.message).slice(0, 22);
    });
  }

  function runMatrix() {
    var cells = $('matrix-body').querySelectorAll('.mx-cell');
    var chain = Promise.resolve();
    cells.forEach(function (td) {
      chain = chain.then(function () { return runCell(td); });
    });
  }

  // ---------- sweep ----------
  function runSweep() {
    var btn = $('btn-sweep'); btn.disabled = true; btn.textContent = '扫描中…';
    var sc = currentScenario();
    fetch('/api/v2/sweep', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(sc)
    }).then(function (r) { return r.json(); }).then(function (j) {
      j = restoreBig(j);
      var html = '<table class="plain"><tr><th>维度</th><th>点值 → TPOT(ms)/加速比</th>' +
        '<th>盈亏平衡</th></tr>';
      Object.keys(j).forEach(function (k) {
        if (j[k] && j[k].points) {
          html += '<tr><td>' + k + '</td><td style="font-size:11.5px">' +
            j[k].points.map(function (p) {
              return p.value + ': ' + (p.speedup === null ? '未知'
                : p.speedup.toFixed(2) + '×');
            }).join(' · ') + '</td><td>' +
            (j.breakevens[k] === null ? '区间内无交点' : j.breakevens[k]) + '</td></tr>';
        }
      });
      $('sweep-result').innerHTML = html + '</table>' +
        '<p style="font-size:11.5px;color:var(--ink3)">每点独立 run_id；' +
        '值域为演示默认（Q8 假设，非用户确认）。</p>';
    }).catch(function (e) {
      $('sweep-result').innerHTML = '<div class="unknown-note">扫描失败：' +
        e.message + '</div>';
    }).finally(function () { btn.disabled = false; btn.textContent = '运行扫描（当前场景）'; });
  }

  // ---------- export（同一 bundle 派生；前端只做格式化拼装） ----------
  function download(name, text, mime) {
    var b = new Blob([text], { type: mime || 'application/json' });
    var a = document.createElement('a');
    a.href = URL.createObjectURL(b); a.download = name; a.click();
    setTimeout(function () { URL.revokeObjectURL(a.href); }, 1000);
  }
  function exportJson() {
    if (!state.bundle) { setStatus('先运行一次再导出。', true); return; }
    download('simverse-run-' + state.bundle.manifest.run_id.slice(7, 19) + '.json',
      JSON.stringify(state.bundle, null, 1));
  }
  function exportCsv() {
    if (!state.bundle) { setStatus('先运行一次再导出。', true); return; }
    var rows = targetRows(state.bundle);
    var cols = ['stage', 'layer', 'op', 'flops', 'weight_read',
      'weight_read_expected', 'act_read', 'act_write', 'kv_read', 'kv_write',
      'state_read', 'state_write', 'weight_capacity', 'statistic', 'formula',
      't_ms', 'bound', 'shape'];
    var lines = [cols.join(',')];
    rows.forEach(function (r) {
      lines.push(cols.map(function (c) {
        var v = r[c];
        return String(v === null || v === undefined ? '' : v)
          .replace(/,/g, ';').replace(/\n/g, ' ');
      }).join(','));
    });
    download('simverse-ops-' + state.bundle.manifest.run_id.slice(7, 19) + '.csv',
      lines.join('\n'), 'text/csv');
  }

  // ---------- wire ----------
  ['sel-model', 'sel-method', 'sel-deploy', 'sel-target-dev', 'sel-tp',
    'sel-draft-dev', 'inp-ctx', 'inp-batch', 'sel-acc', 'inp-blk'
  ].forEach(function (id) {
    $(id).addEventListener('change', function () {
      var m = $('sel-method').value;
      $('sel-deploy').disabled = m === 'baseline';
      $('inp-blk').disabled = m !== 'dflash_v1';
    });
  });
  $('btn-run').addEventListener('click', doRun);
  $('btn-matrix').addEventListener('click', runMatrix);
  $('btn-sweep').addEventListener('click', runSweep);
  $('btn-export-json').addEventListener('click', exportJson);
  $('btn-export-csv').addEventListener('click', exportCsv);
  loadCatalog();
})();
