"""M3/V07/V08：指标归约器人工 oracle + 事件验证器负例。

期望值全部来自 golden/oracles_v02.py §5 的人工轨迹（独立手算）。
每条负例注明“哪个错误由哪个断言发现”。
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simverse.execute.reduce import (busy_integral, goodput_tokens_per_ms,  # noqa: E402
                                     quantile_nearest_rank,
                                     request_mean_itl,
                                     throughput_tokens_per_ms,
                                     window_token_count)
from simverse.execute.validate import TraceEvent, validate_trace  # noqa: E402

_spec = importlib.util.spec_from_file_location("golden_v02", ROOT / "golden" / "oracles_v02.py")
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)

EM = {"r1": G.TRACE_TOKENS["r1"], "r2": G.TRACE_TOKENS["r2"]}
COMPLETED = ["r1", "r2"]


def test_request_mean_itl_manual():
    """r1 = (40−10)/3 = 10；r2 同刻双 token → 0（Q12：同批同刻合法，不均摊）。"""
    assert request_mean_itl(EM["r1"]) == G.R1_MEAN_ITL == 10.0
    assert request_mean_itl(EM["r2"]) == 0.0
    assert request_mean_itl([42.0]) is None      # N≤1 不适用（None，不造 0）


def test_window_token_count_vs_completed_requests():
    """窗口 token 事件 6 ≠ 窗口完成请求 2（§7.6 两种口径不同名）。"""
    assert window_token_count(EM, 0, 50) == G.WINDOW_TOKEN_COUNT == 6
    t = throughput_tokens_per_ms(EM, COMPLETED, 0, 50, "completed_requests")
    assert t == G.THROUGHPUT_COMPLETED == 0.12
    t_events = throughput_tokens_per_ms(EM, COMPLETED, 0, 50, "token_events")
    assert t_events == 6 / 50.0          # 同批同刻 token 全计（不均摊 Q12）


def test_goodput_slo():
    """SLO ITL≤15：r1（10）与 r2（N≤1 视为满足）→ 6/50。"""
    g = goodput_tokens_per_ms(EM, COMPLETED, 0, 50, 15.0)
    assert g == G.GOODPUT_SLO15 == 0.12
    # SLO 收紧到 5ms：r1 不满足 → 只剩 r2 的 2 token
    g2 = goodput_tokens_per_ms(EM, COMPLETED, 0, 50, 5.0)
    assert g2 == 2 / 50.0
    # 取消请求不计入
    g3 = goodput_tokens_per_ms(EM, ["r1"], 0, 50, 15.0)
    assert g3 == 4 / 50.0


def test_busy_integral_no_double_count():
    """(0,12)+(20,33) 无重叠 → 25；构造重叠负例不得双计。"""
    assert busy_integral(G.BUSY_SEGMENTS) == G.BUSY_INTEGRAL == 25
    assert busy_integral([(0, 30), (20, 33)]) == 33        # 重叠并集
    assert busy_integral([(10, 20), (0, 5), (30, 40)]) == 25


def test_quantile_nearest_rank():
    """最近秩：[10,20,30,40] p50=20、p75=30（人工 oracle）。"""
    assert quantile_nearest_rank([10, 20, 30, 40], 0.5) == G.QUANT_NN_P50
    assert quantile_nearest_rank([10, 20, 30, 40], 0.75) == G.QUANT_NN_P75
    assert quantile_nearest_rank([], 0.5) is None


# ---------------------------------------------------------------------------
# V07 负例：每个已知错误被对应检查抓住
# ---------------------------------------------------------------------------


def _ok_trace():
    """合法串行轮：draft→(send)→verify→commit，各资源不重叠。"""
    return [
        TraceEvent("m1", "draft", "gpu0", 0.0, 1.0, stage="draft_block"),
        TraceEvent("m2", "link", "nic0", 1.0, 1.1, depends_on=["m1"],
                   is_message_send=True, stage="d2t"),
        TraceEvent("m3", "target", "gpu1", 1.1, 3.0, depends_on=["m2"],
                   consumes_messages=["m2"], stage="verify_block"),
        TraceEvent("m4", "target", "gpu1", 3.0, 3.1, depends_on=["m3"],
                   stage="commit_rewind"),
    ]


def test_valid_trace_passes():
    assert validate_trace(_ok_trace()) == []


def test_negative_missing_dependency_edge():
    """引用不存在的依赖（删除 arrival→batch_start 类边）：missing_dependency。"""
    evs = _ok_trace()
    evs[2] = TraceEvent("m3", "target", "gpu1", 1.1, 3.0,
                        depends_on=["m_ghost"], consumes_messages=["m2"],
                        stage="verify")
    v = validate_trace(evs)
    assert any(x.check == "missing_dependency" for x in v)


def test_negative_consume_before_ready():
    """消费尚未到达的消息（E19 类：先验状态未就绪）：consume_before_send_end。"""
    evs = _ok_trace()
    evs[2] = TraceEvent("m3", "target", "gpu1", 0.5, 3.0, depends_on=["m2"],
                        consumes_messages=["m2"], stage="verify")
    v = validate_trace(evs)
    assert any(x.check == "starts_before_dependency_end" for x in v)


def test_negative_resource_overlap():
    """同资源非法重叠（两事件同时占 gpu1）：resource_overlap。"""
    evs = _ok_trace()
    evs.append(TraceEvent("m5", "target", "gpu1", 2.0, 2.5, depends_on=["m2"],
                          stage="another_verify"))
    v = validate_trace(evs)
    assert any(x.check == "resource_overlap" for x in v)


def test_negative_consume_unknown_message():
    """消费不存在的消息 id：consume_unknown_message。"""
    evs = _ok_trace()
    evs[2] = TraceEvent("m3", "target", "gpu1", 1.1, 3.0, depends_on=["m2"],
                        consumes_messages=["mX"], stage="verify")
    v = validate_trace(evs)
    assert any(x.check == "consume_unknown_message" for x in v)


def test_negative_message_edge_missing():
    """消费已到达消息但未声明依赖边：missing_message_edge。"""
    evs = _ok_trace()
    evs[2] = TraceEvent("m3", "target", "gpu1", 1.2, 3.0, depends_on=[],
                        consumes_messages=["m2"], stage="verify")
    v = validate_trace(evs)
    assert any(x.check == "missing_message_edge" for x in v)
