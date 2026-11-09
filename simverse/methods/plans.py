"""MethodPlan：方法的程序图与协议绑定（M2）。

注册方法（身份诚实原则）：
  baseline        无投机普通 decode
  chain_draft     链式自回归草稿（custom；S=K 步、M=1、catch-up 1+P_full）
  mtp             config 原生 MTP 微头链式（mtp_layers≤K；target hidden 依赖）
  dflash_v1       DFlash（z-lab/dflash pin）：并行块草稿、单次前向提案、
                  逐位拒绝+残差 bonus、KV crop 回退、target_hidden 8 层拼接
  chain_moe_proxy 「链式 MoE 代理」（原 v0.1 假想方法，改名保留；不得称 DSpark）
  dspark          原生 DSpark：unsupported（块 backbone/Markov/confidence 程序
                  待按 DeepSpec/vLLM pin 实现；V4.1 原生 dspark_* 字段在册）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .. import formula_registry as FR
from .acceptance import (AcceptanceLaw, expected_committed, law_from_scenario,
                         p_full, position_prefix_probs)

DFLASH_PIN = "z-lab/dflash@07ebd93db9f472af339b644bb70221ad8428328a"


@dataclass
class MethodStep:
    step_id: str
    role: str                    # target | draft | link
    desc: str
    tokens: str = ""             # 形状描述（M/Q）
    lane: str = "serial"


@dataclass
class MessageSpec:
    direction: str               # "D->T" | "T->D"
    content: str
    bytes_per_round_expr: str    # 公式串（数值在 pipeline 按场景计算）
    formula: str = ""


@dataclass
class MethodPlan:
    method_id: str
    version: str
    label: str
    source_pin: str
    steps: List[MethodStep] = field(default_factory=list)
    messages: List[MessageSpec] = field(default_factory=list)
    en_formula: str = ""
    cache_policy: Dict[str, Any] = field(default_factory=dict)
    transitions: Dict[str, Any] = field(default_factory=dict)
    status: str = "supported"    # supported | approximated | unsupported
    approximations: List[str] = field(default_factory=list)
    acceptance: Optional[AcceptanceLaw] = None
    k_effective: Optional[int] = None     # 每轮候选数（块方法 = block−1）
    block_size: Optional[int] = None

    def to_json(self) -> dict:
        return {
            "method_id": self.method_id, "version": self.version,
            "label": self.label, "source_pin": self.source_pin,
            "status": self.status, "approximations": list(self.approximations),
            "steps": [s.__dict__ for s in self.steps],
            "messages": [m.__dict__ for m in self.messages],
            "en_formula": self.en_formula, "cache_policy": self.cache_policy,
            "transitions": self.transitions,
            "k_effective": self.k_effective, "block_size": self.block_size,
            "acceptance": ({"kind": self.acceptance.kind,
                            "k_limit": self.acceptance.k_limit}
                           if self.acceptance else None),
        }


# ---------------------------------------------------------------------------
# baseline
# ---------------------------------------------------------------------------


def plan_baseline(acceptance: dict | None = None) -> MethodPlan:
    return MethodPlan(
        method_id="baseline", version="0.2", label="普通 decode（无投机）",
        source_pin="internal",
        steps=[MethodStep("decode", "target", "单 token decode 前向", "M=1")],
        en_formula="N=1",
        cache_policy={"target_kv": "append-only"},
        status="supported")


# ---------------------------------------------------------------------------
# 链式草稿（custom / MTP）
# ---------------------------------------------------------------------------


def plan_chain(method_id: str, label: str, draft_note: str, k: int,
               mtp_layers: int = 0, t2d_hidden: bool = False) -> MethodPlan:
    """链式草稿程序图：S=K 步、步 M=1；首步 catch-up E[M0]=1+P_full。"""
    steps = [MethodStep("draft_step_0", "draft",
                        f"草稿步 0（含 catch-up：全接受轮喂 2 token）",
                        "M∈{1,2} 加权")]
    for j in range(1, k):
        steps.append(MethodStep(f"draft_step_{j}", "draft", f"草稿步 {j}", "M=1"))
    steps.append(MethodStep("d2t", "link", "候选 token(+概率) D→T",
                            "K+1 token"))
    steps.append(MethodStep("verify", "target", "目标验证前向", f"M=K+1={k + 1}"))
    steps.append(MethodStep("t2d", "link", "验收结果(+hidden) T→D", "E[N] token"))
    msgs = [MessageSpec("D->T", "token ids + 采样概率",
                        "B·(K+1)·(4+2) B", FR.F_DFLASH_MSG)]
    if t2d_hidden:
        msgs.append(MessageSpec("T->D", "accepted hidden（MTP 分离）",
                                "B·E[N]·h·B_act", FR.F_DFLASH_MSG))
    approx = [draft_note]
    if method_id == "mtp":
        if k > mtp_layers:
            raise ValueError(
                f"K={k} 超出 MTP 原生层数 {mtp_layers}（config num_nextn/"
                "mtp_num_hidden_layers）")
        approx.append(f"微头按 config {mtp_layers} 层结构与主干共享驻留")
    return MethodPlan(
        method_id=method_id, version="0.2", label=label,
        source_pin="internal:chain" if method_id == "chain_draft"
        else "config:num_nextn_predict_layers/mtp_num_hidden_layers",
        steps=steps, messages=msgs,
        en_formula=FR.F_EN_CHAIN, k_effective=k,
        cache_policy={
            "target_kv": "全接受追加 K+1，否则回退至接受前缀（bonus 1 行）",
            "draft_kv": "链式自回归追加",
        },
        transitions={
            "catch_up": "E[M_draft,0] = 1 + P_full（仅简单链式实现有效；"
                        "非通则 principles §7）",
            "bonus": "每轮 +1 目标采样 token",
        },
        status="approximated", approximations=approx)


# ---------------------------------------------------------------------------
# DFlash v1（真实语义；pin z-lab/dflash）
# ---------------------------------------------------------------------------


def plan_dflash(draft_ir, k_limit: Optional[int] = None,
                block_size: Optional[int] = None) -> MethodPlan:
    """DFlash v1 MethodPlan（源码绑定 dflash_generate L233-308）。

    每轮：
      1. draft 一次前向：q_len=verify_size（首位=已提交 token 的 embedding，
         其余 mask 噪声）；K/V = 历史缓存（target_hidden 投影的 ctx + 噪声）
         + 本块；并行读出 verify_size−1 个候选（用目标 lm_head）。
      2. D→T：候选 token（+随机采样时 q 概率）。
      3. target 一次前向验证 verify_size 个位置（块内因果）。
      4. 逐位拒绝（贪心=argmax 前缀匹配；随机=残差 max(p−q,0) 重采样）；
         bonus 从残差/目标分布。
      5. 提交 A+1；target KV crop 回退未接受行；draft KV crop；
         target_hidden 取 produced 前缀（8 层拼接）供下轮 fc。
    """
    blk = block_size or draft_ir.block_size
    k_eff = blk - 1
    if k_limit is not None:
        k_eff = min(k_eff, k_limit)
        blk = k_eff + 1
    h = draft_ir.hidden
    n_ctx_layers = len(draft_ir.target_layer_ids)
    msgs = [
        MessageSpec("D->T", "块候选 token ids（贪心）/+q 概率（随机）",
                    f"(blk−1)·4B（贪edy）或 ·(4+2)B（随机）", FR.F_DFLASH_MSG),
        MessageSpec("T->D", "produced 个 token 的 8 层 target hidden 拼接（fc 输入）",
                    f"E[N]·{n_ctx_layers}·{h}·B_act（fc 可在任一侧，拼接不省字节）",
                    FR.F_DFLASH_MSG),
    ]
    return MethodPlan(
        method_id="dflash_v1", version="0.2",
        label=f"DFlash v1（并行块草稿 blk={blk}，单次前向提案）",
        source_pin=DFLASH_PIN,
        steps=[
            MethodStep("draft_block", "draft",
                       "草稿单次前向（ctx K/V + 块噪声 → 并行候选；用目标 lm_head）",
                       f"q_len={blk}"),
            MethodStep("d2t", "link", "候选 ids(+q) D→T", f"{blk - 1} 候选"),
            MethodStep("verify_block", "target", "目标块验证前向（块内因果）",
                       f"M={blk}"),
            MethodStep("commit_rewind", "target",
                       "逐位拒绝→提交 A+1；KV crop 回退（浪费 verify_size−produced 行）",
                       "E[N] 提交"),
            MethodStep("hidden_extract", "target",
                       f"target_hidden 取 produced 前缀（{n_ctx_layers} 层拼接，offset+1）",
                       "produced×8 层"),
        ],
        messages=msgs,
        en_formula=FR.F_EN_BLOCK, k_effective=k_eff, block_size=blk,
        cache_policy={
            "target_kv": "块验证写 verify_size 行，crop 保留 produced（回退浪费）",
            "draft_kv": "ctx(target_hidden 投影)+噪声双源缓存；crop 至 start",
            "target_hidden": "每轮重取 produced 前缀（非全历史）",
        },
        transitions={
            "bonus": "A=blk−1 时从目标分布采 bonus；否则残差 max(p−q,0) 归一重采",
            "eos_tail": "verify_size=min(blk, 剩余)；尾轮截断",
        },
        status="approximated",
        approximations=[
            "块内逐位接受率 β_i 假设输入（Q8 hypothesis）；真实接受分布未采",
            "SGLang/vLLM 服务端集成路径未逐行核对（部署能力另核）",
        ])


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------

UNSUPPORTED = {
    "dspark": "原生 DSpark（块 backbone/顺序 Markov/confidence 截断）需按 "
              "DeepSpec/vLLM pin 实现程序图；当前不可用链式代理冒名",
    "dflash2": "DFlash2（候选选择器+动态卷积迭代）未接入；v1 已可用",
    "empirical_hist_acceptance": "A/K 直方图输入属 trace 扩展",
    # M4 服务特性（Q5 顺序；契约保留，不空实现掩盖缺口）
    "chunked_prefill": "M4：需分块边界/插队/队列事件（依赖 M3 资源事件核）",
    "pd_disaggregation": "M4：Prefill/Decode 角色分离的状态转移/传输/双端生命周期"
                         "（与目标/草稿 DT 分离不同域，不混名）",
    "prefix_cache": "M4：命中 token/block 分母、冷/暖初态、驱逐与实际节省"
                    "（不是单一 h 系数）",
}


def build_method_plan(method_id: str, scenario_acceptance: dict, k_limit: int,
                      draft_ir=None, mtp_layers: int = 0,
                      block_size: Optional[int] = None) -> MethodPlan:
    if method_id == "baseline":
        plan = plan_baseline()
        plan.acceptance = None
        return plan
    if method_id in UNSUPPORTED:
        raise ValueError(f"unsupported_method: {method_id} — {UNSUPPORTED[method_id]}")
    if method_id in ("chain_draft", "custom_chain"):
        plan = plan_chain("chain_draft",
                          "链式草稿（演示：tiny_gqa 合成草稿结构）",
                          "v0.2 目录中的链式草稿用 tiny_gqa 合成结构演示闭环；"
                          "真实自定义草稿需注册 DraftIR/ModelIR", k_limit)
    elif method_id == "chain_moe_proxy":
        plan = plan_chain("chain_moe_proxy",
                          "链式 MoE 代理（假想方法，非 DSpark；演示结构同 tiny_gqa）",
                          "假想方法保留作对照；原生 DSpark 另列 unsupported", k_limit)
    elif method_id == "mtp":
        plan = plan_chain("mtp", "MTP（config 原生微头链式）", "",
                          k_limit, mtp_layers=mtp_layers, t2d_hidden=True)
    elif method_id == "dflash_v1":
        if draft_ir is None:
            raise ValueError("dflash_v1 需要 DraftIR（catalog.drafts）")
        plan = plan_dflash(draft_ir, k_limit=k_limit, block_size=block_size)
    else:
        raise ValueError(f"未知方法 {method_id}")
    plan.acceptance = law_from_scenario(scenario_acceptance, plan.k_effective or k_limit)
    return plan


def acceptance_summary(plan: MethodPlan) -> Dict[str, Any]:
    """接受摘要：E[N]/P_full/逐位前缀概率；unknown 给 reason。"""
    if plan.acceptance is None:
        return {"E_N": 1.0, "P_full": None, "prefix": [], "reason": None}
    en, reason = expected_committed(plan.acceptance)
    return {"E_N": en, "P_full": p_full(plan.acceptance),
            "prefix": position_prefix_probs(plan.acceptance),
            "reason": reason,
            "statistic": "expectation" if en is not None else None}
