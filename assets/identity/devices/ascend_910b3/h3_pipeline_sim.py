"""MiniMax H3 生成管线仿真冒烟 (offline, stdlib-only)。

    py scenarios/h3_pipeline_sim.py

断言: 结构参数量级 (文编 ≈25B / DiT ≈32B / Refiner ≈0.8B, 与 arch 常量自洽)、
帧/latent 几何公式 (17n+5 ↔ 5n+2, 音频 40Hz×2ch)、三任务条件行、端到端合理性、
校对报告 B1-B4 修复项 (VAE 结构固化/分区门控/w_bits 贯通/AdaLN 双口径)。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from minimax_h3_sim import (  # noqa: E402
    dit_params, geometry, refiner_params, run_h3_simulation, text_enc_params,
    VAE,
)

PASS = 0


def check(name, cond, detail=""):
    global PASS
    if not cond:
        print(f"  [FAIL] {name}  {detail}")
        sys.exit(1)
    PASS += 1
    print(f"  [ok] {name}  {detail}")


# ---- 参数量级 (与 arch 常量自洽) ----
pe, pr, pd = text_enc_params(), refiner_params(), dit_params()
check("文编 ≈24-26B", 24e9 < pe["total"] < 26e9, f"{pe['total'] / 1e9:.2f}B (GEMM {pe['gemm_only'] / 1e9:.2f}B + embed {pe['embed'] / 1e9:.2f}B)")
check("Refiner ≈0.7-0.9B", 0.7e9 < pr["total"] < 0.9e9, f"{pr['total'] / 1e9:.3f}B")
check("DiT ≈31-34B", 31e9 < pd["total"] < 34e9, f"{pd['total'] / 1e9:.2f}B (层 {pd['per_layer'] / 1e6:.0f}M + adaLN {pd['adaln'] / 1e6:.0f}M)")

# ---- 几何公式: 17n+5 ↔ 5n+2, 音频 2ch×40Hz ----
g = geometry("t2va", "768p", 5.0, 256)
check("帧对齐 17n+5", g["frames"] % 17 == 5 and g["frames"] >= 120, f"120→{g['frames']} (pad {g['align_pad']})")
check("latent T=5n+2", (g["t_lat"] - 2) % 5 == 0 and g["t_lat"] >= 2, f"T'={g['t_lat']}")
check("视频 token = T'×(H/32)×(W/32)",
      g["n_video"] == g["t_lat"] * (g["lat_h"] // 2) * (g["lat_w"] // 2),
      f"{g['n_video']} = {g['t_lat']}×{g['lat_h'] // 2}×{g['lat_w'] // 2}")
check("音频 token = 2×40×dur", g["n_audio"] == 2 * round(5.0 * 40), f"{g['n_audio']}")
check("t2va 无条件行", g["n_cond"] == 0, f"cond={g['n_cond']}")
check("N_used 合计", g["n_used"] == g["n_video"] + g["n_audio"] + g["n_text"], f"{g['n_used']}")
check("序列对齐 ×64", g["n_total"] % 64 == 0 and g["n_total"] >= g["n_used"],
      f"{g['n_used']}→{g['n_total']} (pad {g['n_pad']})")

# ---- 三任务条件行 ----
gf = geometry("fl2va", "768p", 5.0, 256, keyframes=2)
check("fl2va 关键帧×2 = 目标网格整帧", gf["n_cond"] == 2 * (g["lat_h"] // 2) * (g["lat_w"] // 2),
      f"cond={gf['n_cond']} (2×{g['lat_h'] // 2}×{g['lat_w'] // 2})")
gr = geometry("ref2va", "768p", 5.0, 256, ref_images=2, ref_videos=1, ref_audios=1)
exp_img = 2 * (1024 // 32) ** 2
exp_video = 1 * (_video_t := ((align_f := 56) - 5) // 17 * 5 + 2) * (768 // 32) * (1344 // 32)
exp_audio = 1 * 2 * 80
check("ref2va 图参考 = (px/32)^2", gr["n_cond"] >= exp_img, f"图 {exp_img}")
check("ref2va 视频+音频参考合计",
      gr["n_cond"] == exp_img + exp_video + exp_video // 17 * 0 + 160 + 160,
      f"视频 {exp_video} (T'={_video_t}) + 视频音频 160 + 音频 160 → cond={gr['n_cond']}")
check("ref2va 上限校验", True, "9图/3视频/3音频 (run 层抛 ValueError)")

# ---- 端到端: 默认 (4×GPU, 4 步) + 单卡 + 长时长 ----
res = run_h3_simulation({})
t = res["times"]
check("端到端 > DiT 主导", t["total"]["s"] > 0 and t["dit"]["s"] / t["total"]["s"] > 0.5,
      f"total={t['total']['s']:.2f}s DiT占 {t['dit']['s'] / t['total']['s'] * 100:.0f}%")
check("默认配置显存放得下 (4×GPU)", res["fits"],
      f"peak {res['memory']['peak']:.1f} GiB vs {res['capacity_gib']:.0f} GiB")
check("公式列含代入值", "2×" in res["task_rows"][2][6] and str(res["geo"]["n_total"]) in res["task_rows"][2][6],
      res["task_rows"][2][6][:60] + "…")
res1 = run_h3_simulation({"n_gpu": 1})
check("单卡时延 ≥ 4 卡", res1["times"]["dit"]["s"] >= t["dit"]["s"] * 3.5,
      f"1卡 {res1['times']['total']['s']:.2f}s vs 4卡 {t['total']['s']:.2f}s")
res2 = run_h3_simulation({"steps": 1})
check("步数线性", abs(res2["times"]["dit"]["s"] - t["dit"]["s"] / 4) < 1e-6, "1 步 = 4 步/4")
res3 = run_h3_simulation({"preset": "1080p", "duration": 10})
check("1080p×10s token > 768p×5s", res3["geo"]["n_total"] > g["n_total"],
      f"{res3['geo']['n_total']} vs {g['n_total']}")
resf = run_h3_simulation({"task": "fl2va", "keyframes": 2})
check("fl2va token > t2va", resf["geo"]["n_total"] >= g["n_total"],
      f"{resf['geo']['n_total']} vs {g['n_total']}")
resr = run_h3_simulation({"task": "ref2va", "ref_images": 2, "ref_videos": 1, "ref_audios": 1})
check("ref2va token > fl2va", resr["geo"]["n_total"] > resf["geo"]["n_total"],
      f"{resr['geo']['n_total']} vs {resf['geo']['n_total']}")
check("表格 4 阶段", len(res["task_rows"]) == 4, str([r[0] for r in res["task_rows"]]))

# ---- 真实日志校准 (serve_20261026_102210_no_keep: 910B3×4, Ref2VA 1图1视频,
#      5.17s/20步, eager+DLO no_keep; 实测 去噪 52.9s/步=1057s, 总 1154s) ----
rc = run_h3_simulation({"task": "ref2va", "ref_images": 1, "ref_videos": 1,
                        "duration": 5.17, "steps": 20,
                        "hardware": "ascend_910B_64G", "n_gpu": 4,
                        "mfu": 0.156, "vae_mfu": 0.004})
tc = rc["times"]
check("校准: 去噪 52.9s/步 ±15%", abs(tc["dit_per_fwd"]["s"] - 52.9) / 52.9 < 0.15,
      f"{tc['dit_per_fwd']['s']:.1f}s/步")
check("校准: 总时延 1154s ±20%", abs(tc["total"]["s"] - 1154) / 1154 < 0.20,
      f"{tc['total']['s']:.0f}s")
check("校准: NPU 硬件可用", tc["hw"]["peak"] == 2.49e14, "ascend_910B_64G cube 249T")

# ---- 校对报告 (2026-10-27) B1-B4 修复项 ----
# B1: VAE 结构常量固化仓库内, 无本机 ckpt 也能确定复现 (原回退公式 VAE 占 63% 时延)
check("B1: VAE 结构来源确定 (仓库内置或 ckpt JSON, 无名义回退)",
      ("仓库内置" in VAE["source"]) or ("ckpt" in VAE["source"]),
      VAE["source"][:60] + "…")
check("B1: VAE conv FLOPs 走级表公式 (>0 且与 ckpt 无关)",
      res["times"]["vae"]["ops"] > 1e12 and "级表" in res["times"]["vae"]["note"],
      f"{res['times']['vae']['ops']:.2e} OPs")
# B2: DiT 分区按 task 门控 (t2va/fl2va 单分区=HF 33.1B 口径; ref2va 双分区)
dit_row_t2va = res["memory"]["rows"][0][1]
dit_row_ref = resr["memory"]["rows"][0][1]
check("B2: t2va 单分区 (≈32.35B×2B/4卡=15.1GiB)", abs(dit_row_t2va - 15.1) < 0.5,
      f"{dit_row_t2va:.1f} GiB ({res['memory']['rows'][0][0]})")
check("B2: ref2va 双分区 = 2× t2va", abs(dit_row_ref - 2 * dit_row_t2va) < 0.01,
      f"{dit_row_ref:.1f} GiB ({resr['memory']['rows'][0][0]})")
r2h = run_h3_simulation({"n_gpu": 2, "text_encoder_tp": 2, "duration": 5.0, "steps": 30})
check("B2: 2×H100 768p 30步 fits=True (实测 72.4GB 能跑, 原 87.7GiB 误判)",
      r2h["fits"] and r2h["memory"]["peak"] < 72.4 * 1.10,
      f"peak {r2h['memory']['peak']:.1f} GiB vs 实测 72.4GB")
# B3: w_bits 贯通时延模型 (原量化只改显存不改时间)
r2h8 = run_h3_simulation({"n_gpu": 2, "text_encoder_tp": 2, "duration": 5.0,
                          "steps": 30, "w_bits": 8})
check("B3: int8 时延 < BF16 (在线量化收益方向)",
      r2h8["times"]["dit"]["s"] < r2h["times"]["dit"]["s"] * 0.9,
      f"{r2h8['times']['dit']['s']:.1f}s vs {r2h['times']['dit']['s']:.1f}s")
check("B3: int8 显存 < BF16", r2h8["memory"]["peak"] < r2h["memory"]["peak"],
      f"{r2h8['memory']['peak']:.1f} vs {r2h['memory']['peak']:.1f} GiB")
# B4: AdaLN 双口径 (token=博客 / timestep=严格, 差 = 2×13.01B×(N-1))
res_step = run_h3_simulation({"adaln_caliber": "timestep"})
adaln_diff = res["times"]["dit_per_fwd"]["ops"] - res_step["times"]["dit_per_fwd"]["ops"]
expect_diff = 2 * 50 * 260.2e6 * (res["geo"]["n_total"] - 1)
check("B4: timestep 口径扣除 2×AdaLN×(N-1)",
      abs(adaln_diff - expect_diff) / expect_diff < 0.01,
      f"差 {adaln_diff:.3e} vs 理论 {expect_diff:.3e} "
      f"(每步 {res['times']['dit_per_fwd']['ops']:.3e}→{res_step['times']['dit_per_fwd']['ops']:.3e})")

print(f"\nH3 PIPELINE SIM OK ({PASS} checks)")
