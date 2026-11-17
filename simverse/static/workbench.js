/* Simverse 生产工作台 v0.2 —— 前端零物理计算。
 * 所有数字来自 /api/v2/run ResultBundle；本文件只做：
 * 格式化、单位/坐标投影、筛选、ID 联动、导出拼装。失败时显示原因，不回退演示数字。
 *
 * 2026-11-17 工作台审查修复：
 *  - 矩阵串行执行（runCell 返回真 Promise；单格失败继续，不冒充成功数值）；
 *  - Sweep 检查 HTTP 状态/非 JSON/网络错误，{error} 不再渲染为成功空表；
 *  - 控件联动反映后端硬事实（V4.1 冻结量化/仅 baseline、MTP 拒 DT、DFlash DT
 *    为解析研究假设）；chain_draft/chain_moe_proxy 不进产品选择器；
 *  - 七维透视按算子组聚合（精确/期望分列；容量=代表行，不按行求和；多 shape
 *    如实标注；bound 取组内计价行集合，不沿用首行）；
 *  - Roofline 优先消费后端峰值/带宽元数据（有则画规格上界与有效成本上界），
 *    无则明确“计算上界元数据未提供”，不硬编码 FP8/坐标范围；
 *  - M06 显存逐端（target/draft/各 rank）展示，四态区分，detail 非 Metric；
 *  - CSV 走 RFC 4180 标准转义（static/csv.js），与 CLI 同语义。*/
(function () {
  'use strict';
  var state = {
    catalog: null, bundle: null, selectedRowKey: null,
    matrixRuns: {}, scenarioOf: null, matrixRunning: false
  };
  var $ = function (id) { return document.getElementById(id); };
  // 精度档（后端 scenario.validate 契约：dense_w/expert_w/act/kv ∈ {4,8,16}，state=32）
  var QUANT_PRESETS = {
    bf16: { dense_w: 16, expert_w: 16, act: 16, kv: 16, state: 32 },
    w8a8: { dense_w: 8, expert_w: 8, act: 8, kv: 8, state: 32 },
    w8a8_fp4e: { dense_w: 8, expert_w: 4, act: 8, kv: 8, state: 32 }
  };
  // R22：V4.1 checkpoint 量化格式 identity 冻结（MXFP8 线形/MXFP4 专家）+
  // 注册设备（≤SM90 Marlin W8A16/W4A16）执行域 → 激活 BF16。后端
  // scenario.validate 同值校验（V41_QUANT）；界面冻结选择器并显示配方。
  var V41_QUANT = { dense_w: 8, expert_w: 4, act: 16, kv: 8, state: 32 };
  // 组合支持表：逐条镜像 pipeline.run_v2 的硬拒绝/能力判定（后端为权威；
  // 这里只做运行前提示与控件禁用，不新建与后端冲突的规则）。
  var COMBO = {
    v41OnlyBaseline: 'V4.1 仅 baseline 可运行：DFlash checkpoint 只核对 '
      + 'Qwen3.6-35B-A3B 配对；MTP 原生微头仅 Qwen3.6（后端均显式拒绝）',
    v41QuantFrozen: '量化冻结：MXFP8 线性 / MXFP4 专家 / BF16 激活'
      + '（R22 quant-pin；选择器对 V4.1 不适用）',
    mtpDtRejected: 'MTP 无 DT 分离实现：草稿经 load_eagle_model 在 target '
      + '进程内加载并物理共享 embed/lm_head（后端拒绝 dt 模式）',
    dflashDtResearch: 'DFlash DT 分离 = 解析研究假设：两栈均无 draft/target '
      + '分离部署（vLLM in-process；SGLang disagg 为 target 侧 PD）。可计算、'
      + '不参与排名，不等于已验证服务栈',
    dflashColocated: 'DFlash 同卡混部：双栈源码核对支持（source-checked，'
      + '非已部署实测；served 标记待 R07b 真实 trace）',
    mtpColocated: 'MTP 同卡混部：双栈源码核对（R23；结构/加载/共享 mapped）。'
      + '接受率为显式假设输入；kernel/CPU/GPU 未执行未实测',
    draftDevShared: '同卡混部/baseline：草稿与 target 同 TP 组，使用目标设备'
      + '（草稿设备选择器不生效）'
  };
  var METHOD_LABELS = {
    baseline: 'baseline（无投机对照）',
    dflash_v1: 'dflash_v1（DFlash 并行块草稿）',
    mtp: 'mtp（Qwen3.6 原生微头 · 双栈源码核对；接受率假设，CPU/GPU 未执行）'
  };

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
  function esc(s) {  // 后端原因文本进 HTML 前转义
    return String(s === null || s === undefined ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }
  function rowKey(r) { return r.stage + ':' + r.layer + ':' + r.op; }

  // ---------- catalog ----------
  function loadCatalog() {
    fetch('/api/v2/catalog').then(function (r) {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.json();
    }).then(function (cat) {
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
        o.textContent = m.label
          + (m.config_coverage_complete ? '' : ' [配置缺口]')
          + (m.forward_evidence_complete ? '' : ' [forward未完备]');
        sel.appendChild(o);
      });
      sel.value = models.indexOf('qwen36_35b_a3b') >= 0 ? 'qwen36_35b_a3b' : models[0];
      var ms = $('sel-method');
      ms.innerHTML = '';
      // 只放产品方法（supported + research）；chain_draft/chain_moe_proxy 为
      // 内部研究代理，不重新开放到产品入口
      cat.methods.supported.concat(cat.methods.research || []).forEach(function (mid) {
        var o = document.createElement('option');
        o.value = mid;
        o.textContent = METHOD_LABELS[mid] || mid;
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
      var pf = $('sel-profile');
      var rc = $('sel-record-cost');
      if (rc) {
        rc.innerHTML = '<option value="">不使用，保留纯解析</option>';
        (cat.cost_record_sets || []).forEach(function (item) {
          var o = document.createElement('option'); o.value = item.id; o.textContent = item.label;
          rc.appendChild(o);
        });
      }
      if (pf && cat.profile_sets) {
        pf.innerHTML = '<option value="">（无 · 纯解析 T0）</option>';
        cat.profile_sets.forEach(function (id) {
          var o = document.createElement('option');
          o.value = id; o.textContent = id;
          pf.appendChild(o);
        });
      }
      renderCatalogDrawer();
      renderMatrixHead();
      syncControls();
    }).catch(function (e) { setStatus('能力目录加载失败：' + e, true); });
  }

  // ---------- 控件联动（镜像后端硬事实；不臆测无法提前判定的组合） ----------
  function syncControls() {
    if (!state.catalog) return;
    var model = $('sel-model').value;
    var method = $('sel-method').value;
    var deploy = $('sel-deploy');
    var mode = method === 'baseline' ? 'baseline' : deploy.value;
    var hints = [];
    // 方法：V4.1 只有 baseline（后端对 dflash/mtp 显式拒绝）
    var ms = $('sel-method');
    Array.prototype.forEach.call(ms.options, function (o) {
      var rejected = model === 'deepseek_v41_flash' && o.value !== 'baseline';
      o.disabled = rejected;
      o.title = rejected ? COMBO.v41OnlyBaseline : '';
    });
    if (model === 'deepseek_v41_flash' && method !== 'baseline') {
      ms.value = 'baseline';
      method = 'baseline';
      hints.push('已切回 baseline：' + COMBO.v41OnlyBaseline);
    }
    // 部署：MTP 拒 DT（后端硬拒绝）；baseline 用专用模式
    var dtOpts = ['dt_serial', 'dt_pipeline_sim'];
    Array.prototype.forEach.call(deploy.options, function (o) {
      var dis = false, why = '';
      if (method === 'mtp' && dtOpts.indexOf(o.value) >= 0) {
        dis = true; why = COMBO.mtpDtRejected;
      }
      o.disabled = dis;
      o.title = why;
    });
    deploy.disabled = method === 'baseline';
    if (method === 'mtp' && dtOpts.indexOf(deploy.value) >= 0) {
      deploy.value = 'colocated';
      mode = 'colocated';
      hints.push('部署已切回同卡混部：' + COMBO.mtpDtRejected);
    }
    // 精度：V4.1 冻结（选择器不适用，显示配方）
    var prec = $('sel-prec');
    prec.disabled = model === 'deepseek_v41_flash';
    if (model === 'deepseek_v41_flash') hints.push(COMBO.v41QuantFrozen);
    var targetSpec = state.catalog && state.catalog.devices[$('sel-target-dev').value];
    var targetAscend = targetSpec && targetSpec.backend === 'ascend';
    var recordSelect = $('sel-record-cost');
    if (targetAscend && state.recordTarget !== $('sel-target-dev').value
        && recordSelect && !recordSelect.value) recordSelect.value = 'ascend-r29-hccl';
    state.recordTarget = $('sel-target-dev').value;
    if (recordSelect && recordSelect.value) {
      $('sel-profile').value = '';
      $('sel-profile').disabled = true;
      hints.push('记录档仅作同类算子/通信的条件成本对照；不同软件栈不混用，不等于实测校准。');
      if (recordSelect.value === 'ascend-r29-hccl' && $('sel-tp').value !== '2')
        hints.push('R29 通信只有双卡记录；当前 TP 不匹配时保留解析值，选择 TP2 可检查命中。');
    } else $('sel-profile').disabled = false;
    if (targetAscend) {
      if (model === 'deepseek_v41_flash') {
        hints.push('910B3 尚无 V4.1 后端配方；请选择 Qwen3.6，或切回支持该配方的设备。');
      } else {
        prec.value = 'bf16';
        prec.disabled = true;
        hints.push('910B3：沿用 model-verse 的 BF16 解析档，未校准；容量与部署资格仍需验证。');
      }
    }
    // 草稿设备：colocated/baseline 与 target 同组（不生效）
    var draftDev = $('sel-draft-dev');
    var dtMode = mode.indexOf('dt_') === 0;
    var draftSpec = state.catalog && state.catalog.devices[draftDev.value];
    if (dtMode && draftSpec && draftSpec.backend === 'ascend') {
      hints.push('910B3 草稿端使用 BF16 解析档；跨后端服务集成与通信未验证。');
    }
    draftDev.disabled = !dtMode;
    if (!dtMode) hints.push(COMBO.draftDevShared);
    // DT 两端预算：仅 DT 生效
    $('inp-draft-budget').disabled = !dtMode;
    // 块长：仅 DFlash
    $('inp-blk').disabled = method !== 'dflash_v1';
    // 当前组合支持状态
    if (targetAscend || (dtMode && draftSpec && draftSpec.backend === 'ascend')) {
      hints.push('当前仅做解析研究；Ascend 的投机方法、分片、通信和服务集成尚未联合验证。');
    } else if (method === 'dflash_v1') {
      hints.push(dtMode ? COMBO.dflashDtResearch : COMBO.dflashColocated);
    } else if (method === 'mtp') {
      hints.push(COMBO.mtpColocated);
    } else if (model === 'deepseek_v41_flash') {
      hints.push('V4.1 baseline：structure_complete=False（剩余执行变体），'
        + '不参与部署排名；CPU/GPU 未执行未实测');
    }
    renderComboHint(hints);
  }
  function renderComboHint(hints) {
    var el = $('combo-hint');
    if (!el) return;
    el.innerHTML = hints.length
      ? hints.map(function (h) { return '· ' + esc(h); }).join('<br>')
      : '· 当前组合无已知阻断（后端运行为权威判定）';
  }

  function currentScenario(over) {
    var accSel = $('sel-acc').value;
    var acceptance;
    if (accSel === 'mean2') acceptance = { kind: 'committed_mean', mean: 2.0 };
    else if (accSel === 'dynk') acceptance = { kind: 'dynamic_k' };
    else acceptance = { kind: 'geometric', alpha: parseFloat(accSel) };
    var method = $('sel-method').value;
    var mode = method === 'baseline' ? 'baseline' : $('sel-deploy').value;
    var sc = {
      schema_version: '0.2',
      name: 'workbench-' + $('sel-model').value + '-' + method,
      model_id: $('sel-model').value,
      method_id: method,
      draft_id: method === 'dflash_v1' ? 'dflash_qwen36_35b_a3b' : null,
      k: method === 'dflash_v1' ? 15 : (method === 'mtp' ? 1 : 4),
      block_size: parseInt($('inp-blk').value, 10),
      deployment: {
        mode: mode,
        target_device: $('sel-target-dev').value,
        target_tp: parseInt($('sel-tp').value, 10),
        // EP：关闭=inter/tp 切分；开启=整专家分置（必须等于 TP，vLLM 配方）
        target_ep: ($('sel-ep').value === 'tp')
          ? parseInt($('sel-tp').value, 10) : 1,
        draft_device: $('sel-draft-dev').value, draft_tp: 1
      },
      workload: {
        batch: parseInt($('inp-batch').value, 10),
        ctx_tokens: parseInt($('inp-ctx').value, 10)
      },
      acceptance: acceptance,
      // 精度档 → 后端量化契约（scenario.validate 允许 4/8/16；state 固定 32bit）。
      // V4.1 为 R22 冻结配方（界面同步冻结选择器，非静默覆盖）。
      quant: ($('sel-model').value === 'deepseek_v41_flash')
        ? V41_QUANT
        : QUANT_PRESETS[$('sel-prec') ? $('sel-prec').value : 'w8a8_fp4e'],
      draft_quant: (state.catalog && state.catalog.devices[
        mode === 'colocated' ? $('sel-target-dev').value : $('sel-draft-dev').value
      ].backend === 'ascend') ? QUANT_PRESETS.bf16
        : { dense_w: 8, expert_w: 8, act: 8, kv: 8, state: 32 },
      // 显式实现路径（活跃草稿无 quantization_config → raw 融合副本）；
      // dense_w 只是计价位宽，不决定该分支。
      draft_ctx_kv_path: 'fused_raw',
      cost_profile: ($('sel-profile') && $('sel-profile').value) || null,
      cost_records: ($('sel-record-cost') && $('sel-record-cost').value) || null,
      seed: 0
    };
    // 预算输入（R03a 契约）：数值输入 → hypothesis 假设预算（只产生条件结论，
    // 不冒充 measured）；留空 = unknown（不自动套标称容量）。开销 >0 必带来源。
    var budgetGB = parseFloat($('inp-budget') && $('inp-budget').value);
    if (isFinite(budgetGB) && budgetGB > 0) {
      sc.deployment.budget = {
        allocatable_bytes: Math.round(budgetGB * 1024 * 1024 * 1024),
        allocatable_kind: 'hypothesis',
        allocatable_source: '工作台用户输入（假设预算；非实测）',
        extra_overhead_bytes: 0, overhead_source: ''
      };
    }
    var ohGB = parseFloat($('inp-overhead') && $('inp-overhead').value);
    if (sc.deployment.budget && isFinite(ohGB) && ohGB > 0) {
      sc.deployment.budget.extra_overhead_bytes =
        Math.round(ohGB * 1024 * 1024 * 1024);
      sc.deployment.budget.overhead_source =
        '工作台用户输入（假设开销，未证实；留空=0 为输入缺失而非已证实为零）';
    }
    // DT 草稿端独立预算（仅 dt 模式；同契约）
    if (mode.indexOf('dt_') === 0) {
      var dGB = parseFloat($('inp-draft-budget') && $('inp-draft-budget').value);
      if (isFinite(dGB) && dGB > 0) {
        sc.deployment.draft_budget = {
          allocatable_bytes: Math.round(dGB * 1024 * 1024 * 1024),
          allocatable_kind: 'hypothesis',
          allocatable_source: '工作台用户输入（草稿端假设预算；非实测）',
          extra_overhead_bytes: 0, overhead_source: ''
        };
      }
    }
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
  function bundleError(b) {  // 后端 failed bundle（如 unsupported_combo）
    if (b && b.manifest && b.manifest.status === 'failed') {
      var m = b.metrics && b.metrics.error;
      return (m && (m.reason || m.value)) ||
        ('运行被后端拒绝（' + (b.assumption_dict || {}).error + '）');
    }
    return null;
  }

  // ---------- run ----------
  function runScenario(sc) {
    return fetch('/api/v2/run', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(sc)
    }).then(function (r) {
      return r.json().then(function (j) {
        if (!r.ok) throw new Error(j && j.error ? j.error : ('HTTP ' + r.status));
        return restoreBig(j);
      }, function () {
        throw new Error('HTTP ' + r.status + '（响应非 JSON）');
      });
    });
  }

  function doRun() {
    var sc = currentScenario();
    state.scenarioOf = sc;
    $('btn-run').disabled = true;
    setStatus('运行中… ' + sc.model_id + ' × ' + sc.method_id);
    runScenario(sc).then(function (b) {
      var err = bundleError(b);
      if (err) {
        setStatus('运行被拒绝（后端判定）：' + esc(err), true);
        return;
      }
      state.bundle = b;
      renderBundle();
      setStatus('run_id <span class="mono">' + b.manifest.run_id + '</span> · ' +
        '代码 ' + b.manifest.code_revision.slice(0, 10) + (b.cost && b.cost.record_cost
          ? ' · 独立记录条件对照 + 解析回退（非生产校准）·' : ' · 全部数字 T0 解析估算（Q2）·') +
        ' 接受率为假设输入（Q8）');
    }).catch(function (e) {
      setStatus('运行失败：' + esc(e.message), true);
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
    var tpotC = val(b, 'M01_tpot_profile_calibrated');
    if (tpotC !== null) {
      $('kpi-tpot').textContent = fmt(tpotC, 'ms', 4);
      $('kpi-tpot-sub').textContent = '实测校准（算子级组合；解析值 '
        + fmt(tpot, 'ms', 4) + ' 保留在指标区）';
    }
    var recordTpot = val(b, 'M01_record_tpot');
    $('kpi-tpot-label').textContent = recordTpot !== null
      ? 'M01 等效 TPOT（记录条件对照）'
      : (tpotC !== null ? 'M01 等效 TPOT（算子校准组合）' : 'M01 等效 TPOT（解析）');
    if (recordTpot !== null) {
      $('kpi-tpot').textContent = fmt(recordTpot, 'ms', 4);
      $('kpi-tpot-sub').textContent = '记录条件对照；纯解析 ' + fmt(tpot, 'ms', 4) + '（原值保留）';
    }
    var base = val(b, 'M01_tpot_baseline');
    var sp = metric(b, 'M02_record_speedup') || metric(b, 'M02_speedup_vs_baseline');
    $('kpi-speedup').textContent = sp && sp.value !== null
      ? fmt(sp.value, '×', 3) : '未知';
    $('kpi-speedup-sub').textContent = sp && sp.reason
      ? '不参与排名：见下方「证据 / 能力检查」中的容量与方法原因'
      : ('基线 ' + fmt(base, 'ms', 3) + '（同 B/ctx）');
    $('kpi-speedup-sub').title = sp && sp.reason ? sp.reason : '';
    var en = val(b, 'M03_E_N');
    $('kpi-en').textContent = en !== null ? fmt(en, 'tok', 3)
      + ' / K=' + (val(b, 'M03_K') || 0) : '未知';
    $('kpi-en-sub').textContent = reason(b, 'M03_E_N')
      ? '原因：' + reason(b, 'M03_E_N').slice(0, 40) : '假设输入（Q8）';
    var round = val(b, 'M04_record_round_ms');
    if (round === null) round = val(b, 'M04_round_ms');
    $('kpi-round').textContent = round !== null ? fmt(round, 'ms', 3) : '未知';
    var br = b.metrics['M04_stage_breakdown_ms'] || {};
    $('kpi-round-sub').textContent = round !== null
      ? Object.keys(br).map(function (k) {
        return k.replace('_block', '').replace('draft_', 'd:') +
          ' ' + fmt(br[k].value, '', 2);
      }).join(' + ').slice(0, 60) : '—';
    if (val(b, 'M04_record_round_ms') !== null)
      $('kpi-round-sub').textContent = '条件记录成本；纯解析 ' + fmt(val(b, 'M04_round_ms'), 'ms', 3);
    var moe = metric(b, 'M05_moe_union');
    $('kpi-moe').textContent = moe ? fmtBytes(moe.value) + '/层' : '–';
    renderMemory(b);
    var checks = (b.capability && b.capability.checks) || {};
    var bad = Object.keys(checks).filter(function (k) {
      return checks[k].status !== 'supported';
    });
    var mr = b.model_ref || {};
    var ev = mr.evidence || {};
    $('kpi-ev').textContent =
      '配置' + (mr.config_coverage_complete ? '完备' : '缺口')
      + ' · forward ' + (mr.forward_evidence_complete ? '源码完备'
        : (ev.forward_source || 'config-only'))
      + ' · CPU ' + (ev.cpu_forward === 'cpu-executed' ? '已执行' : '未执行')
      + ' · GPU ' + (ev.gpu_measurement === 'none' ? '无实测' : ev.gpu_measurement);
    $('kpi-ev-sub').textContent = (bad.length ? bad.length + ' 项注意' : '无阻断')
      + ' · 假设 ' + Object.keys(b.assumption_dict || {}).length + ' 条';
    renderOps(b);
    renderTimeline(b);
    renderEvidence(b);
    renderRecordCosts(b);
  }

  // ---------- M06 显存（逐端；target/draft 不合并；detail 非 Metric） ----------
  function classifyMem(m) {
    if (!m) return { cls: 'na', label: '–' };
    if (m.feasibility === 'oom') return { cls: 'bad', label: 'OOM（超预算）' };
    if (m.feasibility === 'unknown')
      return { cls: 'warn', label: 'unknown（缺预算或峰值证据）' };
    if (m.support === 'approximated')
      return { cls: 'warn', label: '条件可行（假设预算，非实测）' };
    return { cls: 'ok', label: '可行（判定链路内）' };
  }
  function renderMemory(b) {
    var keys = Object.keys(b.metrics).filter(function (k) {
      return k.indexOf('M06_memory') === 0 && k.indexOf('_detail') < 0;
    }).sort();
    var rows = keys.map(function (k) {
      return { key: k, m: b.metrics[k], d: b.metrics[k + '_detail'] || {} };
    });
    var kpi = $('kpi-mem'), sub = $('kpi-mem-sub');
    if (!rows.length) { kpi.textContent = '–'; sub.textContent = '无 M06 指标'; return; }
    // KPI 概览：分端展示（colocated 取最坏 rank；dt 分 target/draft）
    var parts = [], states = [];
    rows.forEach(function (r) {
      var st = classifyMem(r.m);
      states.push(st.label);
      var rank = r.m.scope && r.m.scope.rank ? r.m.scope.rank : r.key.slice(11);
      if (rank.indexOf('worst') >= 0) return;  // 概览行不重复列 worst
      parts.push(rank + ' ' + fmtBytes(r.m.value));
    });
    kpi.textContent = parts.join(' · ') || fmtBytes(rows[0].m.value);
    var uniq = states.filter(function (v, i, a) { return a.indexOf(v) === i; });
    sub.textContent = uniq.join(' / ').slice(0, 90);
    // 明细表：容量、预算、可行性、原因、detail（非 Metric，仅账本字段）
    var dep = (b.scenario_snapshot && b.scenario_snapshot.deployment) || {};
    var html = '<table class="plain"><tr><th>端 / rank</th><th>已知峰值</th>'
      + '<th>判定</th><th>预算（场景输入）</th><th>原因 / 缺口组件</th></tr>';
    rows.forEach(function (r) {
      var st = classifyMem(r.m);
      var isDraft = (r.m.scope && r.m.scope.role) === 'draft';
      var bud = isDraft ? dep.draft_budget : dep.budget;
      var devId = isDraft ? dep.draft_device : dep.target_device;
      var dspec = state.catalog && state.catalog.devices[devId];
      var budgetTxt = bud && bud.allocatable_bytes
        ? fmtBytes(bud.allocatable_bytes) + '（' + bud.allocatable_kind
          + '；' + esc(bud.allocatable_source || '无来源') + '）'
          + (bud.extra_overhead_bytes
            ? ' · 开销 ' + fmtBytes(bud.extra_overhead_bytes) : '')
        : '未提供（unknown；不套标称）';
      html += '<tr><td><b>' + esc(r.key.slice(11)) + '</b><br><span style="font-size:'
        + '11px;color:var(--ink3)">' + esc(devId || '') + ' 标称 '
        + (dspec ? fmtBytes(dspec.mem_capacity) : '–') + '</span></td>'
        + '<td class="mono" style="font-family:var(--mono)">' + fmtBytes(r.m.value) + '</td>'
        + '<td><span class="tag ' + st.cls + '">' + st.label + '</span></td>'
        + '<td style="font-size:11.5px">' + budgetTxt + '</td>'
        + '<td style="font-size:11.5px">' + (r.m.reason ? esc(r.m.reason) : '—')
        + (r.d.unknown_components && r.d.unknown_components.length
          ? '<br><i>峰值缺口组件：' + r.d.unknown_components.length + ' 项（'
            + esc(r.d.unknown_components[0]) + '…）</i>' : '')
        + '<br><span style="font-size:11px;color:var(--ink3)">steady '
        + fmtBytes(r.d.steady_bytes) + ' · transient '
        + fmtBytes(r.d.transient_peak_bytes) + ' · 峰值'
        + (r.d.peak_complete ? '完整' : '不完整') + '</span></td></tr>';
    });
    html += '</table><p style="font-size:11.5px;color:var(--ink3)">'
      + 'colocated 各 rank 各自判定（EP 余数可不同；worst=最坏 rank）；'
      + 'DT 分离两端为不同设备，容量与判定不相加、不合并出单一部署结论。'
      + '_detail 为账本字段（非 Metric）；假设预算只产生条件结论。</p>';
    $('memory-body').innerHTML = html;
  }

  // ---------- op ledger（按 op 名聚合；数字全部来自 bundle 行/计价行） ----------
  function targetRows(b) {
    var w = (b.work || {})['target_decode'] || { rows: [] };
    return w.rows || [];
  }
  function rooflineIndex(b) {
    var idx = {};
    var rc = (b.cost || {}).record_cost;
    ((rc && rc.matched ? rc.selected_row_prices : null)
      || ((b.cost || {}).row_prices) || ((b.cost || {}).roofline_points) || []).forEach(function (p) {
      idx[p.row_id] = p;
    });
    return idx;
  }
  function aggOp(rowsOfOp, ptIdx) {
    var a = { op: rowsOfOp[0].op, n: rowsOfOp.length,
      layers: [rowsOfOp[0].layer, rowsOfOp[0].layer],
      flops: 0, weight_read: 0, act_read: 0, act_write: 0,
      kv_read: 0, kv_write: 0, state_read: 0, state_write: 0,
      weight_read_expected: 0, expectedRows: 0,
      t: 0, priced: 0, bounds: {}, caps: {}, shapes: {}, stats: {},
      rep: rowsOfOp[0], rows: rowsOfOp };
    rowsOfOp.forEach(function (r) {
      a.flops += r.flops || 0;
      a.weight_read += r.weight_read || 0;
      a.act_read += r.act_read || 0;
      a.act_write += r.act_write || 0;
      a.kv_read += r.kv_read || 0;
      a.kv_write += r.kv_write || 0;
      a.state_read += r.state_read || 0;
      a.state_write += r.state_write || 0;
      if (r.weight_read_expected) {
        a.weight_read_expected += r.weight_read_expected;
        a.expectedRows++;
      }
      if (r.layer < a.layers[0]) a.layers[0] = r.layer;
      if (r.layer > a.layers[1]) a.layers[1] = r.layer;
      var p = ptIdx[rowKey(r)];
      if (p) {
        a.t += p.t_ms || 0;
        a.priced++;
        a.bounds[p.bound] = (a.bounds[p.bound] || 0) + 1;
      }
      a.caps[r.weight_capacity] = (a.caps[r.weight_capacity] || 0) + 1;
      a.shapes[r.shape] = (a.shapes[r.shape] || 0) + 1;
      a.stats[r.statistic || 'exact'] = true;
    });
    return a;
  }

  function renderOps(b) {
    var rows = targetRows(b);
    var ptIdx = rooflineIndex(b);
    var list = $('dag-list');
    list.innerHTML = '';
    var byOp = {};
    rows.forEach(function (r, i) {
      if (!byOp[r.op]) byOp[r.op] = [];
      byOp[r.op].push(r);
    });
    var aggs = Object.keys(byOp).map(function (k) {
      return aggOp(byOp[k], ptIdx);
    }).sort(function (a, c) { return c.t - a.t; });
    $('op-count-tag').textContent = rows.length + ' 行';
    aggs.forEach(function (a) {
      var node = document.createElement('div');
      node.className = 'dag-node';
      node.setAttribute('role', 'button'); node.tabIndex = 0;
      node.innerHTML = '<div><div style="font-weight:700">' + esc(a.op) + '</div>' +
        '<div style="font-size:11px;color:var(--ink3)">×' + a.n + ' · 层 ' +
        (a.layers[0] === a.layers[1] ? a.layers[0] : a.layers[0] + '…' + a.layers[1]) +
        (Object.keys(a.shapes).length > 1
          ? ' · ' + Object.keys(a.shapes).length + ' 种 shape' : '') +
        '</div></div><div style="text-align:right">' +
        '<div style="font-family:var(--mono);font-weight:800;color:var(--blue)">' +
        (a.priced ? fmt(a.t, 'ms', 4) : '–') + '</div>' +
        '<span class="dag-badge ' + (a.kv_read + a.state_read > 0 ? 'mem' : 'cmp') +
        '">' + (a.flops / 1e6).toFixed(1) + ' MF</span></div>';
      var activate = function () {
        state.selectedRowKey = a.op;
        renderInspector(a);
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
    if (aggs.length) {
      state.selectedRowKey = aggs[0].op;
      renderInspector(aggs[0]);
      list.firstChild.classList.add('active');
    } else {
      state.selectedRowKey = null;
      $('inspector').innerHTML = '<p>无账本行。</p>';
      $('roofline-coord').textContent = '–';
    }
  }

  function renderInspector(a) {
    if (!a) { $('inspector').innerHTML = '<p>无账本行。</p>'; return; }
    var ptIdx = rooflineIndex(state.bundle);
    var rep = a.rep || {};
    var boundKeys = Object.keys(a.bounds);
    var boundLabel = boundKeys.length === 1 ? boundKeys[0]
      : boundKeys.length ? '混合(' + boundKeys.map(function (k) {
          return k.slice(0, 3) + '×' + a.bounds[k];
        }).join('/') + ')' : null;
    $('op-bound-tag').textContent = boundLabel || '–';
    $('op-bound-tag').className = 'tag ' + (!boundLabel ? 'na'
      : boundKeys.length === 1 && boundKeys[0] === 'memory' ? 'warn'
      : boundKeys.length === 1 && boundKeys[0] === 'compute' ? 'ok' : 'warn');
    $('op-bound-tag').title = boundKeys.length
      ? '组内计价行 bound 分布（非首行结论）' : '该组无后端导出计价行';
    var capVals = Object.keys(a.caps);
    var shapeVals = Object.keys(a.shapes);
    var statTags = Object.keys(a.stats).map(function (s) {
      return '<span class="tag ' + (s === 'expectation' ? 'warn' : 'ok') + '">'
        + esc(s) + '</span>';
    }).join(' ');
    var html = '<div class="op-grid">' +
      '<div class="op-cell"><b>1 时延（T0 解析）</b><span class="v">' +
      (a.priced ? fmt(a.t, 'ms', 4) : '–') + '</span>'
      + (a.priced && a.priced < a.n
        ? '<span style="font-size:10.5px;color:var(--ink3)">计价行 ' + a.priced
          + '/' + a.n + '（旧结果仅导出 top-40）</span>'
        : (a.priced ? '' : '<span style="font-size:10.5px;color:var(--ink3)">'
          + '组内行未进后端导出计价行</span>')) +
      '</div>' +
      '<div class="op-cell"><b>2 逻辑算量（组内 ' + a.n + ' 行求和，精确）</b>'
      + '<span class="v">' + (a.flops / 1e6).toFixed(2) + ' MFLOPs</span></div>' +
      '<div class="op-cell"><b>3 驻留容量（代表行 L' + rep.layer + '）</b>'
      + '<span class="v">' + fmtBytes(rep.weight_capacity || 0) + '</span>'
      + (capVals.length > 1
        ? '<span style="font-size:10.5px;color:var(--amber)">组内 '
          + capVals.length + ' 种驻留值——物理对象跨行共享，不按行求和；'
          + '总容量以 M06 内存账本为准</span>'
        : '<span style="font-size:10.5px;color:var(--ink3)">物理对象共享，'
          + '不按行求和；总容量见 M06</span>') + '</div></div>' +
      '<div class="op-cell" style="margin-bottom:6px"><b>4 五类字节（组内求和：'
      + '精确 int' + (a.expectedRows ? ' + 期望 float 分列' : '') + '）</b>' +
      'W ' + fmtBytes(a.weight_read) +
      (a.expectedRows ? '（期望另计 ' + fmtBytes(a.weight_read_expected)
        + '，' + a.expectedRows + ' 行）' : '') +
      ' · A ' + fmtBytes(a.act_read) + '/' + fmtBytes(a.act_write) +
      ' · KV 读 ' + fmtBytes(a.kv_read) + ' 写 ' + fmtBytes(a.kv_write) +
      (a.state_read || a.state_write
        ? ' · 状态读 ' + fmtBytes(a.state_read) + ' 写 ' + fmtBytes(a.state_write)
        : '') + '</div>' +
      '<div class="kv"><b>5 形状：</b>' + (shapeVals.length > 1
        ? '组内 ' + shapeVals.length + ' 种 shape（逐行见下）——不代表全组相同。'
          + '代表行 L' + rep.layer + '：<code>' + esc(rep.shape || '') + '</code>'
        : '<code>' + esc(rep.shape || '') + '</code>') + '</div>' +
      '<div class="kv"><b>6 精度/统计身份：</b>' + statTags + ' ' + quantLabel() + '</div>' +
      '<div class="formula-box"><span class="fid">[' + esc(rep.formula || '') + ']</span> '
      + esc(formulaExpr(rep.formula)) + '</div>' +
      '<details style="margin-top:6px"><summary style="cursor:pointer;font-size:11.5px;'
      + 'color:var(--ink2)">行级查看（' + a.n + ' 行，数字逐行可追溯）</summary>'
      + '<div style="max-height:180px;overflow:auto;margin-top:4px">'
      + '<table class="plain" style="font-size:10.8px"><tr><th>row_id</th><th>层</th>'
      + '<th>shape</th><th>FLOPs</th><th>W 读</th><th>期望 W</th><th>KV r/w</th>'
      + '<th>t_ms</th><th>bound</th></tr>'
      + a.rows.slice(0, 30).map(function (r) {
        var p = ptIdx[rowKey(r)];
        return '<tr><td class="mono" style="font-family:var(--mono);font-size:10px">'
          + esc(rowKey(r)) + '</td><td>' + r.layer + '</td><td>' + esc(r.shape || '')
          + '</td><td>' + (r.flops || 0) + '</td><td>' + fmtBytes(r.weight_read)
          + '</td><td>' + (r.weight_read_expected ? fmtBytes(r.weight_read_expected)
            : '–') + '</td><td>' + fmtBytes(r.kv_read) + '/' + fmtBytes(r.kv_write)
          + '</td><td>' + (p ? fmt(p.t_ms, '', 4) : '–') + '</td><td>'
          + (p ? esc(p.bound) : '–') + '</td></tr>';
      }).join('')
      + (a.n > 30 ? '<tr><td colspan="9">…仅列前 30 行（共 ' + a.n
        + ' 行；导出 CSV 含全量）</td></tr>' : '')
      + '</table></div></details>';
    $('inspector').innerHTML = html;
    drawRoofline(rep);
  }

  function quantLabel() {  // 从运行快照读实际精度（不写死；UI=CLI=JSON 同源）
    var q = state.bundle && state.bundle.scenario_snapshot &&
            state.bundle.scenario_snapshot.quant;
    if (!q) return '';
    return '权重 ' + q.dense_w + 'b/' + q.expert_w + 'b · 激活 ' + q.act +
           'b · KV ' + q.kv + 'b';
  }

  function formulaExpr(fid) {
    var fd = state.bundle && state.bundle.formula_dict;
    if (fd && fd[fid]) return fd[fid].expr + ' —— ' + fd[fid].source;
    return '';
  }

  // ---------- roofline ----------
  /* 坐标点来自 bundle.cost.roofline_points（后端算好 AI/attained/bound）。
   * 新响应可选元数据（能力检测，不假设存在）：compute_dtype / peak_tflops
   * （设备规格峰值，dense）/ effective_peak_tflops（实际计价有效峰值）/
   * bandwidth_tb_s（实际计价带宽，十进制 TB/s）/ cost_basis。有则按元数据画
   * 规格上界（虚线）与有效成本上界（实线）；无则只画点并明示
   * “计算上界元数据未提供”——不硬编码 FP8 峰值、不推断 kernel。
   * 选择点按稳定 row_id 匹配代表行；无对应点时明示，不用其他算子冒充。 */
  function drawRoofline(row) {
    var svg = $('roofline-svg');
    var b = state.bundle;
    var rc = (b.cost || {}).record_cost;
    var pts = (rc && rc.matched ? rc.selected_roofline_points : null)
      || (b.cost && b.cost.roofline_points) || [];
    var ptIdx = rooflineIndex(b);
    var repId = row ? rowKey(row) : null;
    var pt = repId ? ptIdx[repId] : null;
    if (pt && pts.indexOf(pt) < 0) pts = pts.concat([pt]);
    // 元数据能力检测（设备级常量，取任一携带者）
    var meta = pt && pt.peak_tflops != null ? pt : null;
    for (var i = 0; !meta && i < pts.length; i++) {
      if (pts[i].peak_tflops != null || pts[i].effective_peak_tflops != null
          || pts[i].bandwidth_tb_s != null) { meta = pts[i]; break; }
    }
    var hasEff = !!(meta && meta.effective_peak_tflops && meta.bandwidth_tb_s);
    var hasSpec = !!(meta && meta.peak_tflops);
    // 动态坐标范围：只取有效正值，log 轴 + 1.6× 边距；空/零/极端由过滤兜底
    var xs = pts.map(function (p) { return p.ai_flop_per_byte; })
      .filter(function (v) { return v > 0; });
    var ys = pts.map(function (p) { return p.attained_tflops; })
      .filter(function (v) { return v > 0; });
    if (hasEff) {
      xs.push(meta.effective_peak_tflops / meta.bandwidth_tb_s);
      ys.push(meta.effective_peak_tflops);
    }
    if (hasSpec) ys.push(meta.peak_tflops);
    if (!xs.length || !ys.length) {
      svg.innerHTML = '<rect x="45" y="12" width="310" height="116" fill="#f8fafc" '
        + 'stroke="#e2e8f0"/><text x="55" y="70" font-size="11" fill="#64748b">'
        + '无有效坐标点（AI 或吞吐为 0/空）</text>';
      $('roofline-coord').textContent = '无有效点';
      return;
    }
    var xmin = Math.min.apply(null, xs), xmax = Math.max.apply(null, xs);
    var ymin = Math.min.apply(null, ys), ymax = Math.max.apply(null, ys);
    xmax *= 1.6; xmin /= 1.6; ymax *= 1.6; if (ymin > 0) ymin /= 1.6;
    var X0, Y0, W = 310, H = 116, LX = 45, TY = 12;
    function mapX(v) {
      return LX + (Math.log10(v) - Math.log10(xmin))
        / (Math.log10(xmax) - Math.log10(xmin)) * W;
    }
    function mapY(v) {
      return TY + H - (Math.log10(v) - Math.log10(ymin))
        / (Math.log10(ymax) - Math.log10(ymin)) * H;
    }
    var s = '<rect x="' + LX + '" y="' + TY + '" width="' + W + '" height="' + H
      + '" fill="#f8fafc" stroke="#e2e8f0"/>';
    if (hasEff) {
      var ridge = meta.effective_peak_tflops / meta.bandwidth_tb_s;
      var xr = Math.min(Math.max(ridge, xmin), xmax);
      var yTop = mapY(Math.min(meta.effective_peak_tflops, ymax));
      // 有效成本上界（实线）：带宽斜线 → 平台
      var xStart = Math.max(xmin, ymin / meta.bandwidth_tb_s);
      if (xStart < xr) {
        s += '<polyline fill="none" stroke="#2563eb" stroke-width="2.5" points="'
          + mapX(xStart) + ',' + mapY(xStart * meta.bandwidth_tb_s) + ' '
          + mapX(xr) + ',' + yTop + ' '
          + mapX(xmax) + ',' + yTop + '"/>';
      }
      s += '<line x1="' + mapX(xr) + '" y1="' + TY + '" x2="' + mapX(xr)
        + '" y2="' + (TY + H) + '" stroke="#cbd5e1" stroke-dasharray="3,3"/>'
        + '<text x="' + Math.max(mapX(xr) - 30, LX + 2) + '" y="' + (TY + 10)
        + '" font-size="9" fill="#1d4ed8">拐点 ' + ridge.toFixed(0) + ' F/B</text>';
      if (hasSpec) {
        var ys2 = mapY(Math.min(meta.peak_tflops, ymax));
        var specRidge = meta.peak_tflops / meta.bandwidth_tb_s;
        var specStart = Math.max(xmin, ymin / meta.bandwidth_tb_s);
        s += '<polyline fill="none" stroke="#64748b" stroke-width="1.5" '
          + 'stroke-dasharray="5,4" points="'
          + mapX(specStart) + ',' + mapY(specStart * meta.bandwidth_tb_s) + ' '
          + mapX(Math.min(specRidge, xmax)) + ','
          + mapY(Math.min(meta.peak_tflops, ymax)) + ' '
          + mapX(xmax) + ',' + ys2 + '"/>'
          + '<text x="' + (LX + 4) + '" y="' + (ys2 - 3)
          + '" font-size="9" fill="#64748b">规格上界 '
          + meta.peak_tflops.toFixed(0) + ' TF/s（dense）</text>';
      }
      s += '<text x="' + (LX + 5) + '" y="' + (TY + H - 5)
        + '" font-size="9" fill="#475569">有效成本上界 '
        + meta.effective_peak_tflops.toFixed(0) + ' TF/s · BW '
        + meta.bandwidth_tb_s.toFixed(2) + ' TB/s'
        + (meta.cost_basis ? '（' + esc(meta.cost_basis) + '）' : '')
        + (meta.compute_dtype ? ' · ' + esc(meta.compute_dtype) : '') + '</text>';
    } else {
      s += '<text x="' + (LX + 5) + '" y="' + (TY + 12)
        + '" font-size="9.5" fill="#b45309">计算上界元数据未提供：后端未随点输出'
        + '峰值/带宽档位，本图不推断（不硬编码 FP8 峰值）</text>';
    }
    // 全部计价行（上下文淡点）+ 代表行高亮（row_id 稳定匹配）
    pts.forEach(function (p) {
      if (!(p.ai_flop_per_byte > 0) || !(p.attained_tflops > 0)) return;
      s += '<circle cx="' + mapX(p.ai_flop_per_byte) + '" cy="'
        + mapY(p.attained_tflops) + '" r="2.4" fill="#93c5fd"/>';
    });
    if (pt && pt.ai_flop_per_byte > 0 && pt.attained_tflops > 0) {
      var cx = mapX(pt.ai_flop_per_byte), cy = mapY(Math.max(pt.attained_tflops, ymin));
      s += '<line x1="' + cx + '" y1="' + (TY + H) + '" x2="' + cx + '" y2="' + cy
        + '" stroke="#0f766e" stroke-dasharray="2,2"/>'
        + '<circle cx="' + cx + '" cy="' + cy
        + '" r="6" fill="#e11d48" stroke="#fff" stroke-width="2"/>';
    }
    s += '<text x="' + LX + '" y="' + (TY + H + 14) + '" font-size="9.5" '
      + 'fill="#64748b">' + fmt(xmin, '', 1) + '</text>'
      + '<text x="' + (LX + W - 58) + '" y="' + (TY + H + 14)
      + '" font-size="9.5" fill="#64748b">F/B ↑' + fmt(xmax, '', 0) + '</text>';
    svg.innerHTML = s;
    $('roofline-coord').textContent = !row
      ? '未选择行'
      : (pt
        ? '代表行 L' + row.layer + '：AI=' + pt.ai_flop_per_byte.toFixed(1)
          + ' F/B · ' + (pt.record_refs && pt.record_refs.length
            ? '记录条件对照（含host/sync）' : '解析估算（T0，非实测）')
          + pt.attained_tflops.toFixed(1) + ' TF/s'
        : '代表行 L' + row.layer + ' 无对应点（未进后端导出计价行）');
  }

  // ---------- timeline（事件来自 execution.rounds，含依赖） ----------
  function renderTimeline(b) {
    var rc = (b.cost || {}).record_cost;
    var execution = rc && rc.matched ? rc.selected_execution : b.execution;
    var rounds = (execution && execution.rounds) || [];
    var evs = (rounds[0] && rounds[0].events) || [];
    var el = $('timeline');
    if (!evs.length) {
      el.innerHTML = '<div class="unknown-note">无事件（baseline 或接受分布未知：'
        + (execution && execution.note || '') + '）</div>';
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
          html += '<div class="tl-seg" title="' + esc(e.event_id) + ' ← ' +
            esc((e.depends_on || []).join(',')) + '" style="position:absolute;left:' +
            off + '%;width:' + l + '%;background:' + colors[e.role] + '">' +
            esc(e.stage) + ' ' + fmt(e.t_ms, '', 2) + '</div>';
        });
        html += '</div></div>';
      });
    el.innerHTML = html;
    var en = val(b, 'M03_E_N');
    $('tl-summary').textContent = (rc && rc.matched ? '记录条件对照 · ' : '') + 'T_round=' + fmt(total, 'ms', 3) +
      '（串行：阶段和=墙钟）· E[N]=' + fmt(en, '', 2) + ' · 事件依赖可追溯（悬停查看）';
  }

  // ---------- evidence drawer ----------
  function renderRecordCosts(b) {
    var rc = (b.cost || {}).record_cost, panel = $('cost-records-panel');
    if (!panel) return;
    panel.hidden = !rc;
    if (!rc) return;
    var html = '<p><b>' + esc(rc.label) + '</b> · 命中 ' + rc.matched + '/' + rc.total
      + ' 个成本实例。' + (rc.matched ? 'KPI/逐行成本/时间轴显示条件对照，原解析值保留。'
        : '本场景无匹配记录，全部保留原解析值。') + '</p>'
      + '<p>覆盖统计按独立基线/验证/草稿的成本评估实例计算，不是真机 E2E 时间覆盖；不参与部署排名。</p>'
      + '<table class="plain"><tr><th>阶段</th><th>命中/总数</th><th>记录成本 ms</th><th>解析回退 ms</th><th>记录成本占比</th></tr>';
    Object.keys(rc.stages || {}).forEach(function (stage) {
      var s = rc.stages[stage];
      html += '<tr><td>' + esc(stage) + '</td><td>' + s.matched + '/' + s.total + '</td><td>'
        + fmt(s.record_ms, '', 4) + '</td><td>' + fmt(s.fallback_ms, '', 4) + '</td><td>'
        + fmt(s.record_cost_fraction * 100, '%', 1) + '</td></tr>';
    });
    html += '</table><details><summary>记录身份、范围与未准入原因</summary><pre>'
      + esc(JSON.stringify(rc.domain, null, 2)) + '</pre><ul>'
      + (rc.limits || []).map(function (s) { return '<li>' + esc(s) + '</li>'; }).join('')
      + '</ul><p>完整 SHA256：<code>' + esc(rc.content_digest) + '</code></p>';
    if ((rc.unpriced_components || []).length)
      html += '<p>另有未计价方法阶段，时长 unknown，不计入上述覆盖分母，不能认作 0ms。</p><pre>'
        + esc(JSON.stringify(rc.unpriced_components, null, 2)) + '</pre>';
    (rc.available_curves || []).forEach(function (c) {
      html += '<p>' + esc(c.op) + ' · ' + c.points + ' 点 · 每 rank 逻辑输入 '
        + fmtBytes(c.input_bytes[0]) + '–' + fmtBytes(c.input_bytes[1]) + '</p>';
    });
    if (rc.available_operator_shapes && rc.available_operator_shapes.length)
      html += '<pre>' + esc(JSON.stringify(rc.available_operator_shapes, null, 2)) + '</pre>';
    html += '</details><h4>按本次回退调整的补采顺序</h4><ol>'
      + (rc.prerequisites || []).map(function (s) { return '<li>' + esc(s) + '</li>'; }).join('') + '</ol>'
      + '<table class="plain"><tr><th>回退原因</th><th>实例/不同形状</th><th>回退解析成本 ms</th><th>下一步</th></tr>';
    (rc.measurement_plan || []).forEach(function (p) {
      html += '<tr><td>' + esc(p.reason) + '</td><td>' + p.instances + '/' + p.unique_cases
        + '</td><td>' + fmt(p.fallback_analytic_ms, '', 3) + '</td><td>' + esc(p.action)
        + '<details><summary>待补形状示例</summary><pre>' + esc(JSON.stringify(p.examples, null, 2))
        + '</pre></details></td></tr>';
    });
    html += '</table><details><summary>逐项命中与回退（前 200 项；完整清单随 JSON 导出）</summary>'
      + '<table class="plain"><tr><th>阶段/算子</th><th>范围</th><th>解析→选用 ms</th><th>记录或回退原因</th></tr>';
    (rc.entries || []).slice().sort(function (a, z) { return z.record_refs.length - a.record_refs.length; })
      .slice(0, 200).forEach(function (e) {
        html += '<tr><td>' + esc(e.stage + '/' + e.name) + '</td><td>' + esc(JSON.stringify(e.shape))
          + '</td><td>' + fmt(e.analytic_ms, '', 4) + ' → ' + fmt(e.selected_ms, '', 4) + '</td><td>'
          + esc(e.record_refs.length ? e.lookup + ': ' + e.record_refs.join(';') : e.fallback_reason) + '</td></tr>';
      });
    html += '</table></details><details><summary>已隔离的代理/缺来源记录</summary><pre>'
      + esc(JSON.stringify(rc.excluded_records || [], null, 2)) + '</pre></details>';
    $('cost-records-body').innerHTML = html;
  }

  function renderEvidence(b) {
    var checks = (b.capability && b.capability.checks) || {};
    var html = '<table class="plain"><tr><th>检查</th><th>状态</th><th>原因/受影响</th></tr>';
    Object.keys(checks).forEach(function (k) {
      var c = checks[k];
      var cls = c.status === 'supported' ? 'ok' : c.status === 'unknown' ? 'warn' : 'bad';
      html += '<tr><td>' + esc(k) + '</td><td><span class="tag ' + cls + '">' + esc(c.status) +
        '</span></td><td>' + esc(c.reason) +
        (c.affected_metrics && c.affected_metrics.length
          ? '<br><i>受影响：' + esc(c.affected_metrics.join(', ')) + '</i>' : '') +
        '</td></tr>';
    });
    html += '</table><h4>假设（全部随 run 导出）</h4><ul>';
    Object.keys(b.assumption_dict || {}).forEach(function (k) {
      html += '<li><code>' + esc(k) + '</code>：' + esc(b.assumption_dict[k]) + '</li>';
    });
    html += '</ul><h4>证据引用</h4><p><code>' +
      (b.evidence_refs || []).map(esc).join('</code> · <code>') + '</code></p>';
    $('evidence-body').innerHTML = html;
  }

  function renderCatalogDrawer() {
    var cat = state.catalog.catalog;
    var methods = state.catalog.methods || {};
    var html = '<p style="font-size:12px;color:var(--ink2)">方法目录（后端事实）：'
      + 'supported <code>' + esc((methods.supported || []).join(', ')) + '</code> · '
      + 'research <code>' + esc((methods.research || []).join(', ')) + '</code>；'
      + 'chain_draft/chain_moe_proxy/tiny 为内部研究资产，不进产品选择器。</p>'
      + '<table class="plain"><tr><th>模型</th><th>家族/层序</th><th>身份</th>'
      + '<th>config 覆盖</th><th>结构/forward 源码</th><th>执行/实测</th></tr>';
    Object.keys(cat.models).forEach(function (k) {
      var m = cat.models[k];
      var cov = { mapped: 0, 'metadata-only': 0, 'ignored-with-reason': 0,
        'unsupported-semantic': 0 };
      Object.keys(m.coverage || {}).forEach(function (ck) {
        var s = m.coverage[ck].status;
        cov[s] = (cov[s] || 0) + 1;
      });
      html += '<tr><td><b>' + esc(k) + '</b></td><td>' + esc(m.family) + ' · ' +
        JSON.stringify(m.kind_counts) + '</td><td style="font-size:11.5px">' +
        esc(m.source.repo_id || '') + '@' + esc((m.source.revision || '').slice(0, 10)) +
        '<br><code style="font-size:10.5px">' + esc((m.source.config_sha256 || '').slice(0, 16)) +
        '</code></td><td>' + (m.config_coverage_complete
          ? '<span class="tag ok">完整</span>'
          : '<span class="tag bad">' + m.gaps.length + ' 缺口</span> ' +
            esc((m.gaps[0] || '').slice(0, 36)))
        + '<br><span style="font-size:11px;color:var(--ink3)">mapped ' + (cov.mapped || 0) +
        ' · meta ' + (cov['metadata-only'] || 0) + ' · ignored ' +
        (cov['ignored-with-reason'] || 0) + ' · unsupported ' +
        (cov['unsupported-semantic'] || 0) + '</span></td><td>'
        + (m.structure_complete
          ? '<span class="tag ok">structure_complete</span>' : '<span class="tag warn">'
            + 'structure 未完备</span><br>')
        + ((m.evidence && m.evidence.forward_source) || 'config-only')
        + '<br><span style="font-size:11px;color:var(--ink3)">' +
        (!m.forward_evidence_complete && (m.evidence.unverified_paths || []).length
          ? esc((m.evidence.unverified_paths[0] || '').slice(0, 44)) : '') +
        '</span></td><td>CPU ' + (((m.evidence || {}).cpu_forward) || 'not-run') +
        '<br>GPU ' + (((m.evidence || {}).gpu_measurement) || 'none') +
        '</td></tr>';
    });
    html += '</table><h4>草稿</h4><table class="plain"><tr><th>id</th><th>结构</th>' +
      '<th>绑定</th><th>证据</th></tr>';
    Object.keys(cat.drafts).forEach(function (k) {
      var d = cat.drafts[k];
      var de = d.evidence || {};
      html += '<tr><td>' + esc(k) + '</td><td>' + d.n_layers + ' 层 [' +
        d.layer_kinds.join(',') + '] blk=' + d.block_size +
        '</td><td style="font-size:11.5px">' + esc(d.source.repo_id) + '@' +
        esc(d.source.revision.slice(0, 10)) +
        '</td><td style="font-size:11.5px">' + esc(de.forward_source || 'config-only') +
        ' / CPU ' + (de.cpu_forward || 'not-run') +
        ' / GPU ' + (de.gpu_measurement || 'none') +
        '<br>' + esc(((de.unverified_paths || [])[0] || '').slice(0, 52)) +
        '</td></tr>';
    });
    html += '</table><h4>待核实资产（如实标注）</h4><ul>';
    Object.keys(cat.pending_assets).forEach(function (k) {
      var p = cat.pending_assets[k];
      html += '<li><b>' + esc(k) + '</b>：' + esc(p.reason) + '</li>';
    });
    $('catalog-body').innerHTML = html + '</ul>';
  }

  // ---------- matrix（每格独立 run；串行：最大在途请求 = 1） ----------
  var METHODS = ['dflash_v1', 'mtp'];
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
      draft_id: method === 'dflash_v1' ? 'dflash_qwen36_35b_a3b' : null,
      deployment: { mode: deploy },
      k: method === 'dflash_v1' ? 15 : (method === 'mtp' ? 1 : 4),
      name: 'matrix-' + deploy + '-' + method
    });
  }

  // 返回 Promise：请求处理完成（成功/失败/unsupported）后才 resolve——
  // 供 runMatrix 严格串行（最大在途 = 1）。内部不 reject。
  function runCell(td) {
    var deploy = td.getAttribute('data-deploy');
    var method = td.getAttribute('data-method');
    td.className = 'mx-cell pending'; td.textContent = '运行中…';
    return runScenario(cellScenario(deploy, method)).then(function (b) {
      var err = bundleError(b);
      if (err) {
        state.matrixRuns[deploy + '-' + method] = b;
        td.className = 'mx-cell bad';
        td.textContent = '不支持/失败';
        td.title = '后端拒绝：' + err;
        return;
      }
      state.matrixRuns[deploy + '-' + method] = b;
      var tpot = val(b, 'M01_record_tpot');
      if (tpot === null) tpot = val(b, 'M01_tpot_analytical');
      var sp = metric(b, 'M02_record_speedup') || metric(b, 'M02_speedup_vs_baseline');
      if (tpot === null) {
        td.className = 'mx-cell na';
        td.innerHTML = '未知<div style="font-size:9.5px;font-weight:400">' +
          esc((reason(b, 'M01_tpot_analytical') || '').slice(0, 24)) + '</div>';
        td.title = reason(b, 'M01_tpot_analytical') || '';
        return;
      }
      var ineligible = sp && sp.reason;
      td.className = 'mx-cell' + (ineligible ? ' na' : '');
      td.innerHTML = fmt(sp && sp.value !== null ? sp.value : null, '×', 2) +
        '<div style="font-size:9.5px;font-weight:400;color:#475569">' +
        fmt(tpot, 'ms', 2) + (ineligible ? ' ·不排名' : '') + '</div>';
      td.title = 'run_id ' + b.manifest.run_id.slice(7, 19) +
        (ineligible ? '\n' + sp.reason : '');
    }).catch(function (e) {
      td.className = 'mx-cell bad';
      td.textContent = '失败：' + String(e.message).slice(0, 22);
      td.title = String(e.message);
    });
  }

  function runMatrix() {
    if (state.matrixRunning) return;  // 防重复点击的重叠批次
    state.matrixRunning = true;
    var btn = $('btn-matrix');
    btn.disabled = true; btn.textContent = '串行运行中…';
    var cells = Array.prototype.slice.call(
      $('matrix-body').querySelectorAll('.mx-cell'));
    // 严格串行：await 每格完成再发下一格（runCell 永不 reject →
    // 单格失败/unsupported 后继续运行后续格子）
    return cells.reduce(function (chain, td) {
      return chain.then(function () { return runCell(td); });
    }, Promise.resolve()).finally(function () {
      state.matrixRunning = false;
      btn.disabled = false; btn.textContent = '批量运行矩阵（串行）';
    });
  }

  // ---------- sweep ----------
  function runSweep() {
    var btn = $('btn-sweep'); btn.disabled = true; btn.textContent = '扫描中…';
    var sc = currentScenario();
    fetch('/api/v2/sweep', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(sc)
    }).then(function (r) {
      return r.json().then(function (j) {
        if (!r.ok) throw new Error(j && j.error ? j.error : ('HTTP ' + r.status));
        return j;
      }, function () {
        throw new Error('HTTP ' + r.status + '：扫描响应非 JSON');
      });
    }).then(function (j) {
      j = restoreBig(j);
      var html = '<table class="plain"><tr><th>维度</th><th>点值 → TPOT(ms)/加速比</th>' +
        '<th>盈亏平衡</th></tr>';
      var dims = 0;
      Object.keys(j).forEach(function (k) {
        if (j[k] && j[k].points) {
          dims++;
          html += '<tr><td>' + esc(k) + '</td><td style="font-size:11.5px">' +
            j[k].points.map(function (p) {
              // 每点独立 run：失败/未知点如实标注原因，不渲染成成功数值
              var ok = p.status === 'completed' && p.speedup !== null
                && p.speedup !== undefined;
              var cell = ok ? (p.value + ': ' + p.speedup.toFixed(2) + '×'
                + '（' + fmt(p.tpot_ms, '', 2) + 'ms）')
                : (p.value + ': <span style="color:var(--amber)">未知/失败'
                  + (p.reason ? '—' + esc(String(p.reason).slice(0, 40)) : '')
                  + '</span>');
              return cell;
            }).join(' · ') + '</td><td>' +
            (j.breakevens && j.breakevens[k] === null ? '区间内无交点'
              : esc(j.breakevens ? j.breakevens[k] : '')) + '</td></tr>';
        }
      });
      if (!dims) {
        $('sweep-result').innerHTML = '<div class="unknown-note">扫描返回无数据维度'
          + '（请联系后端排查；不渲染为空成功表）</div>';
        return;
      }
      $('sweep-result').innerHTML = html + '</table>' +
        '<p style="font-size:11.5px;color:var(--ink3)">每点独立 run_id；'
        + '值域为演示默认（Q8 假设，非用户确认）。</p>';
    }).catch(function (e) {
      $('sweep-result').innerHTML = '<div class="unknown-note">扫描失败：'
        + esc(e.message) + '（网络错误/后端拒绝/响应异常均显示原因，'
        + '不渲染为成功空表）</div>';
    }).finally(function () {
      btn.disabled = false; btn.textContent = '运行扫描（当前场景）';
    });
  }

  // ---------- export（同一 bundle 派生；CSV 走 RFC 4180 标准转义） ----------
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
    // t_ms/bound 从完整逐行计价按 row_id 关联；旧结果兼容 top-40。
    var ptIdx = rooflineIndex(state.bundle);
    var recordCosts = (state.bundle.cost || {}).record_cost;
    var useRecords = !!recordCosts;
    if (useRecords) {
      ptIdx = {};
      (recordCosts.selected_row_prices || []).forEach(function (p) { ptIdx[p.row_id] = p; });
    }
    if (useRecords) cols = cols.concat(['analytic_t_ms', 'record_t_ms', 'cost_source', 'record_refs', 'fallback_reason']);
    var records = rows.map(function (r) {
      var p = ptIdx[rowKey(r)] || {};
      var o = {}; cols.forEach(function (c) { o[c] = r[c]; });
      o.t_ms = p.t_ms !== undefined ? p.t_ms : '';
      o.bound = p.bound !== undefined ? p.bound : '';
      if (useRecords) {
        o.analytic_t_ms = p.analytic_ms !== undefined ? p.analytic_ms : '';
        o.record_t_ms = p.t_ms !== undefined ? p.t_ms : '';
        o.cost_source = p.cost_source || '';
        o.record_refs = (p.record_refs || []).join(';');
        o.fallback_reason = p.fallback_reason || '';
      }
      return o;
    });
    download('simverse-ops-' + state.bundle.manifest.run_id.slice(7, 19) + '.csv',
      window.simverseCsv.toCsv(cols, records), 'text/csv');
  }

  // ---------- wire ----------
  ['sel-model', 'sel-method', 'sel-deploy', 'sel-target-dev', 'sel-tp',
    'sel-ep', 'sel-draft-dev', 'inp-ctx', 'inp-batch', 'sel-acc', 'inp-blk',
    'sel-prec', 'inp-budget', 'inp-draft-budget', 'inp-overhead', 'sel-record-cost'
  ].forEach(function (id) {
    $(id).addEventListener('change', syncControls);
  });
  $('btn-run').addEventListener('click', doRun);
  $('btn-matrix').addEventListener('click', runMatrix);
  $('btn-sweep').addEventListener('click', runSweep);
  $('btn-export-json').addEventListener('click', exportJson);
  $('btn-export-csv').addEventListener('click', exportCsv);
  loadCatalog();

  // 浏览器验收钩子（仅 ?testhook=1 时暴露；不影响正常运行）
  if (/[?&]testhook=1/.test(location.search)) {
    window.__wb = {
      state: state, currentScenario: currentScenario, syncControls: syncControls,
      runCell: runCell, runMatrix: runMatrix, runSweep: runSweep,
      renderBundle: renderBundle, renderInspector: renderInspector,
      renderMemory: renderMemory, drawRoofline: drawRoofline,
      classifyMem: classifyMem, aggOp: aggOp, bundleError: bundleError
    };
  }
})();
