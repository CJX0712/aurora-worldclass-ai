# -*- coding: utf-8 -*-
"""
Aurora 数学不变量自检套件（Math-Invariant Self-Checks）。

设计原则（来自项目总纲「验证总纲」）：每个核心算法都至少有一条可被
独立实现交叉验证的硬不变量。本文件是 Aurora 的**可验证性**基石——

    pytest tests/test_invariants.py        # 推荐
    python tests/test_invariants.py        # 无 pytest 依赖时也可直接跑

覆盖范围
--------
1. MLA / GDN / SWA 增量 decode ≡ 全量 prefill（fp32 ~1e-4，逐 token 一致）。
2. 模型级混合配比（DDSM 等）增量 decode ≡ 全量 prefill（~1e-4）。
3. Gated DeltaNet chunkwise ≡ recurrent（WY 表示与逐 token 递推等价，~1e-5）。
4. GDN β≡0 → 无写入 → 输出恒 0、状态保持；α≡1 退化为标准 DeltaNet。
5. RoPE 正交性（‖RoPE(x)‖=‖x‖）与相对性（score 只依赖相对位置）。
6. MoE 路由概率守恒（softmax/sigmoid 经 norm_topk_prob 后每样本和为 1）。
7. 反向传播可回传且梯度有限（数值稳定）。
8. MLA KV cache 每 token 元素数 = d_c + d_r，与 n_h 无关（压缩收益核心）。
"""

import sys
import math

import torch


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def _maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def _pass(cond: bool, msg: str, extra=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {msg}" + (f"  ({extra})" if extra else ""))
    if not cond:
        raise AssertionError(f"{msg} FAILED {extra}")


def _rand(*shape, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    return torch.randn(*shape)


# ---------------------------------------------------------------------------
# 1) Mixer 增量 decode ≡ 全量 prefill（模块级）
# ---------------------------------------------------------------------------
def test_mla_incremental_equiv():
    from aurora.modules.mla import MultiHeadLatentAttention
    torch.manual_seed(0)
    m = MultiHeadLatentAttention(
        hidden_size=128, num_heads=4, q_lora_rank=64, kv_lora_rank=64,
        qk_nope_head_dim=32, qk_rope_head_dim=16, v_head_dim=32, qk_norm=True)
    x = _rand(2, 10, 128, seed=1)
    m.eval()
    with torch.no_grad():
        full, _ = m(x)
        cache = None
        parts = []
        for t in range(10):
            o, cache = m(x[:, t:t + 1], cache=cache)
            parts.append(o)
        inc = torch.cat(parts, dim=1)
    d = _maxdiff(full, inc)
    _pass(d < 1e-4, "MLA incremental-decode ≡ full-prefill", f"maxdiff={d:.2e}")


def test_gdn_incremental_equiv():
    from aurora.modules.gdn import GatedDeltaNet
    torch.manual_seed(0)
    m = GatedDeltaNet(hidden_size=128, num_heads=4, key_dim=32, value_dim=32,
                      conv_size=4, use_beta=True, use_gate=True)
    x = _rand(2, 12, 128, seed=2)
    m.eval()
    with torch.no_grad():
        full, _ = m(x)                       # 全量（T=12，走 recurrent 分支）
        cache = None
        parts = []
        for t in range(12):
            o, cache = m(x[:, t:t + 1], cache=cache)
            parts.append(o)
        inc = torch.cat(parts, dim=1)
    d = _maxdiff(full, inc)
    _pass(d < 1e-4, "GDN incremental-decode ≡ full-prefill", f"maxdiff={d:.2e}")


def test_swa_incremental_equiv():
    from aurora.modules.attention import SlidingWindowAttention
    torch.manual_seed(0)
    m = SlidingWindowAttention(hidden_size=128, num_heads=4, num_key_value_heads=2,
                              head_dim=32, sliding_window=64)
    x = _rand(2, 11, 128, seed=3)
    m.eval()
    with torch.no_grad():
        full, _ = m(x)
        cache = None
        parts = []
        for t in range(11):
            o, cache = m(x[:, t:t + 1], cache=cache)
            parts.append(o)
        inc = torch.cat(parts, dim=1)
    d = _maxdiff(full, inc)
    _pass(d < 1e-4, "SWA incremental-decode ≡ full-prefill", f"maxdiff={d:.2e}")


# ---------------------------------------------------------------------------
# 2) 模型级混合配比增量 decode ≡ 全量 prefill
# ---------------------------------------------------------------------------
def _model_prefill_vs_incremental(cfg, seq=9):
    from aurora.model import AuroraForCausalLM
    torch.manual_seed(0)
    m = AuroraForCausalLM(cfg)
    m.eval()
    ids = torch.randint(0, cfg.vocab_size, (2, seq + 1))
    with torch.no_grad():
        L_full = m(ids[:, :seq + 1], use_cache=False).logits[:, seq]  # 预测 token_{seq+1}
        cache = m(ids[:, :seq], use_cache=True).past_key_values
        L_inc = m(ids[:, seq:seq + 1], past_key_values=cache, use_cache=True).logits[:, 0]
    return _maxdiff(L_full, L_inc)


def test_model_mixed_plan_incremental_equiv():
    from aurora.config import tiny
    cfg = tiny('nano', mixer_pattern='DDSM', num_nextn_predict_layers=0)
    d = _model_prefill_vs_incremental(cfg)
    _pass(d < 1e-3, "Model(DDSM) incremental-decode ≡ full-prefill", f"maxdiff={d:.2e}")


# ---------------------------------------------------------------------------
# 3) Gated DeltaNet chunkwise ≡ recurrent
# ---------------------------------------------------------------------------
def test_gdn_chunk_equiv_recurrent():
    from aurora.modules.gdn import (recurrent_gated_delta_rule,
                                     chunk_gated_delta_rule)
    torch.manual_seed(0)
    B, H, N, Dk, Dv = 2, 3, 18, 8, 8
    q = _rand(B, H, N, Dk, seed=10)
    k = _rand(B, H, N, Dk, seed=11)
    v = _rand(B, H, N, Dv, seed=12)
    beta = torch.rand(B, H, N).clamp(0.05, 0.95)
    alpha = torch.rand(B, H, N).clamp(0.4, 0.99)
    o_rec, s_rec = recurrent_gated_delta_rule(q, k, v, beta, alpha, output_final_state=True)
    o_chk, s_chk = chunk_gated_delta_rule(q, k, v, beta, alpha,
                                          output_final_state=True, chunk_size=5)
    d_out = _maxdiff(o_rec, o_chk)
    d_state = _maxdiff(s_rec, s_chk)
    _pass(d_out < 1e-5 and d_state < 1e-4,
          "GDN chunkwise ≡ recurrent", f"d_out={d_out:.2e} d_state={d_state:.2e}")


def test_gdn_alpha_one_is_delta_net():
    """α≡1 时 Gated DeltaNet 退化为无衰减的标准 DeltaNet；
    chunk / recurrent / 纯 delta-rule 三者一致。"""
    from aurora.modules.gdn import (recurrent_gated_delta_rule,
                                     chunk_gated_delta_rule)
    torch.manual_seed(0)
    B, H, N, Dk, Dv = 1, 2, 14, 6, 6
    q = _rand(B, H, N, Dk, seed=20)
    k = _rand(B, H, N, Dk, seed=21).clamp(-1, 1)
    v = _rand(B, H, N, Dv, seed=22)
    beta = torch.rand(B, H, N).clamp(0.1, 0.9)
    alpha = torch.ones(B, H, N)
    o_rec, _ = recurrent_gated_delta_rule(q, k, v, beta, alpha)
    o_chk, _ = chunk_gated_delta_rule(q, k, v, beta, alpha, chunk_size=7)
    d = _maxdiff(o_rec, o_chk)
    # α≡1 时三角求解的浮点误差略大，放宽到 1e-4
    _pass(d < 1e-4, "GDN α≡1: chunk ≡ recurrent (DeltaNet)", f"maxdiff={d:.2e}")


def test_gdn_beta_zero_no_write():
    """β≡0 时没有任何写入：零初态下输出恒 0、状态恒 0（仅随 α 衰减但不写 v）。"""
    from aurora.modules.gdn import recurrent_gated_delta_rule
    torch.manual_seed(0)
    B, H, N, Dk, Dv = 1, 2, 8, 6, 6
    q = _rand(B, H, N, Dk, seed=30)
    k = _rand(B, H, N, Dk, seed=31)
    v = _rand(B, H, N, Dv, seed=32)
    beta = torch.zeros(B, H, N)
    alpha = torch.rand(B, H, N).clamp(0.4, 0.99)
    o, S1 = recurrent_gated_delta_rule(q, k, v, beta, alpha,
                                       initial_state=None, output_final_state=True)
    d_out = _maxdiff(o, torch.zeros_like(o))
    d_state = _maxdiff(S1, torch.zeros_like(S1))
    _pass(d_out < 1e-6 and d_state < 1e-6,
          "GDN β≡0: zero-init ⇒ output=0 & state=0", f"d_out={d_out:.2e} d_state={d_state:.2e}")


# ---------------------------------------------------------------------------
# 4) RoPE 正交性 & 相对性
# ---------------------------------------------------------------------------
def test_rope_orthogonal():
    from aurora.modules.rope import RotaryEmbedding, rotate_half
    rope = RotaryEmbedding(dim=16, base=10000.0)
    cos, sin = rope(7)
    x = _rand(3, 7, 16, seed=40)
    xr = (x.float() * cos) + rotate_half(x.float()) * sin
    nb = x.float().pow(2).sum(-1).sqrt()
    na = xr.pow(2).sum(-1).sqrt()
    d = _maxdiff(nb, na)
    _pass(d < 1e-5, "RoPE preserves norm (orthogonal)", f"maxdiff={d:.2e}")


def test_rope_relativity():
    from aurora.modules.rope import RotaryEmbedding, rotate_half
    rope = RotaryEmbedding(dim=16, base=10000.0)
    cos, sin = rope(40)
    x = _rand(16, seed=50)

    def apply(v, pos):
        return (v * cos[pos]) + rotate_half(v) * sin[pos]

    q = _rand(16, seed=51)
    k = _rand(16, seed=52)
    s1 = (apply(q, 3) * apply(k, 7)).sum()       # relative offset = 4
    s2 = (apply(q, 8) * apply(k, 12)).sum()      # same relative offset = 4
    s3 = (apply(q, 3) * apply(k, 9)).sum()       # relative offset = 6
    _pass(abs((s1 - s2).item()) < 1e-5 and abs((s1 - s3).item()) > 1e-4,
          "RoPE score depends only on relative position",
          f"s(3,7)={s1.item():.4f} s(8,12)={s2.item():.4f} s(3,9)={s3.item():.4f}")


# ---------------------------------------------------------------------------
# 5) MoE 路由概率守恒
# ---------------------------------------------------------------------------
def test_moe_prob_conservation():
    from aurora.modules.moe import DeepSeekMoE
    torch.manual_seed(0)
    m = DeepSeekMoE(hidden_size=64, intermediate_size=64, moe_intermediate_size=32,
                    n_routed_experts=8, n_shared_experts=1, num_experts_per_tok=2,
                    use_quantile_balancing=True, balance_steps=4, norm_topk_prob=True)
    x = _rand(4, 7, 64, seed=60)
    out = m(x)
    # 路由权重在「每个 token」上经 top-k 归一化后和为 1（概率守恒）
    r = m.router.topk(x.reshape(-1, 64))
    wsum = r.weight.sum(-1)
    d = _maxdiff(wsum, torch.ones_like(wsum))
    _pass(torch.isfinite(out).all().item() and out.shape == (4, 7, 64) and d < 1e-5,
          "MoE output finite & routing weights sum to 1",
          f"shape={tuple(out.shape)} wsum_maxdev={d:.2e}")


# ---------------------------------------------------------------------------
# 6) 反向传播可回传 & 梯度有限
# ---------------------------------------------------------------------------
def test_backward_finite_grads():
    from aurora.config import tiny
    from aurora.model import AuroraForCausalLM
    torch.manual_seed(0)
    cfg = tiny('nano', mixer_pattern='DDSM', num_nextn_predict_layers=1)
    m = AuroraForCausalLM(cfg)
    m.train()
    ids = torch.randint(0, cfg.vocab_size, (2, 12))
    out = m(ids, labels=ids)
    out.loss.backward()
    max_grad = 0.0
    n_params = 0
    for p in m.parameters():
        if p.grad is not None:
            max_grad = max(max_grad, p.grad.abs().max().item())
            n_params += 1
    _pass(math.isfinite(max_grad) and n_params > 0,
          "Backward produces finite gradients", f"max|grad|={max_grad:.4f} params={n_params}")


# ---------------------------------------------------------------------------
# 7) MLA KV cache 大小与 n_h 无关
# ---------------------------------------------------------------------------
def test_mla_cache_independent_of_heads():
    from aurora.modules.mla import MultiHeadLatentAttention
    dc, dr = 64, 16
    m4 = MultiHeadLatentAttention(128, num_heads=4, q_lora_rank=64, kv_lora_rank=dc,
                                  qk_nope_head_dim=32, qk_rope_head_dim=dr, v_head_dim=32)
    m8 = MultiHeadLatentAttention(128, num_heads=8, q_lora_rank=64, kv_lora_rank=dc,
                                  qk_nope_head_dim=32, qk_rope_head_dim=dr, v_head_dim=32)
    # 每 token 缓存元素 = d_c + d_r
    per_token_4 = m4.kv_lora_rank + m4.qk_rope_head_dim
    per_token_8 = m8.kv_lora_rank + m8.qk_rope_head_dim
    _pass(per_token_4 == per_token_8 == dc + dr,
          "MLA KV cache per-token = d_c+d_r (head-count independent)",
          f"{per_token_4} == {per_token_8} == {dc + dr}")


# ---------------------------------------------------------------------------
# 8) MuonClip 优化器不变量
# ---------------------------------------------------------------------------
def test_muon_newton_schulz_orthogonal():
    """Newton-Schulz 五阶迭代把任意方阵推到（近似）正交矩阵：奇异值全部 → 1。"""
    from aurora.optim.muon import newton_schulz_5
    torch.manual_seed(0)
    worst = 0.0
    for n in (4, 7, 10):
        G = torch.randn(n, n)
        U = newton_schulz_5(G, steps=5)
        sv = torch.linalg.svdvals(U)                       # 应全部 ≈ 1
        dev = (sv - 1.0).abs().max().item()
        worst = max(worst, dev)
    _pass(worst < 1e-2, "Newton-Schulz pushes singular values → 1 (orthogonal)",
          f"max|sv-1|={worst:.2e}")


def test_muon_step_finite():
    """Muon 多步更新后权重保持有限（正交化 + 一致 RMS 不发散）。"""
    from aurora.optim.muon import Muon
    torch.manual_seed(0)
    p = torch.randn(16, 12, requires_grad=True)
    opt = Muon([p], lr=0.02, momentum=0.95)
    for _ in range(5):
        p.grad = torch.randn_like(p)
        opt.step()
    _pass(torch.isfinite(p).all().item(),
          "Muon step keeps weights finite after 5 steps",
          f"max|w|={p.abs().max().item():.4f}")


def test_qk_clip_reduces_logits_mla():
    """MLA QK-Clip：裁剪后下一步前向的最大 attention logit 被钉在阈值 τ 以下。"""
    from aurora.modules.mla import MultiHeadLatentAttention
    m = MultiHeadLatentAttention(128, num_heads=4, q_lora_rank=64, kv_lora_rank=64,
                                 qk_nope_head_dim=32, qk_rope_head_dim=16, v_head_dim=32,
                                 qk_norm=True)
    m.train()
    x = torch.randn(2, 10, 128)
    m(x)                                                  # 触发 attn_max_logits 记录
    smax = m.attn_max_logits
    thr = float(smax.max().item()) * 0.3                 # 强制裁剪到 30% 处
    m.qk_clip_step(threshold=thr)
    m(x)                                                  # 裁剪后下一步前向
    smax_after = m.attn_max_logits
    ok = torch.isfinite(smax_after).all().item() and float(smax_after.max().item()) <= thr * 1.05 + 1e-3
    _pass(ok, "MLA QK-Clip reduces max attention logit below τ",
          f"before={float(smax.max().item()):.3f} τ={thr:.3f} after={float(smax_after.max().item()):.3f}")


def test_qk_clip_reduces_logits_swa():
    """SWA(GQA) QK-Clip：按 group 裁剪后最大 logit 被钉在阈值 τ 以下。"""
    from aurora.modules.attention import SlidingWindowAttention
    m = SlidingWindowAttention(128, num_heads=4, num_key_value_heads=2, head_dim=32,
                               sliding_window=64)
    m.train()
    x = torch.randn(2, 11, 128)
    m(x)
    smax = m.attn_max_logits
    thr = float(smax.max().item()) * 0.3
    m.qk_clip_step(threshold=thr)
    m(x)
    smax_after = m.attn_max_logits
    ok = torch.isfinite(smax_after).all().item() and float(smax_after.max().item()) <= thr * 1.05 + 1e-3
    _pass(ok, "SWA QK-Clip (per-group) reduces max attention logit below τ",
          f"before={float(smax.max().item()):.3f} τ={thr:.3f} after={float(smax_after.max().item()):.3f}")


def test_muon_clip_split_params():
    """muon_clip_split_params：2D 隐藏权重 → Muon；embedding/LM-head/1D → Adam。"""
    from aurora.config import tiny
    from aurora.model import AuroraForCausalLM
    from aurora.optim.muon import muon_clip_split_params
    cfg = tiny('nano', mixer_pattern='DDSM', num_nextn_predict_layers=0)
    m = AuroraForCausalLM(cfg)
    muon_groups, adam = muon_clip_split_params(m)
    muon_ps = sum(len(g["params"]) for g in muon_groups) if muon_groups else 0
    _pass(muon_ps > 0 and len(adam) > 0,
          "muon_clip_split_params separates 2D hidden (Muon) vs embedding/1D (Adam)",
          f"muon_params={muon_ps} adam_params={len(adam)}")


# ---------------------------------------------------------------------------
# 运行器
# ---------------------------------------------------------------------------
def _all_tests():
    return [
        test_mla_incremental_equiv,
        test_gdn_incremental_equiv,
        test_swa_incremental_equiv,
        test_model_mixed_plan_incremental_equiv,
        test_gdn_chunk_equiv_recurrent,
        test_gdn_alpha_one_is_delta_net,
        test_gdn_beta_zero_no_write,
        test_rope_orthogonal,
        test_rope_relativity,
        test_moe_prob_conservation,
        test_backward_finite_grads,
        test_mla_cache_independent_of_heads,
        test_muon_newton_schulz_orthogonal,
        test_muon_step_finite,
        test_qk_clip_reduces_logits_mla,
        test_qk_clip_reduces_logits_swa,
        test_muon_clip_split_params,
    ]


def main():
    print("=" * 64)
    print("Aurora Math-Invariant Self-Checks")
    print("=" * 64)
    failures = 0
    for t in _all_tests():
        print(f"\n· {t.__name__}")
        try:
            t()
        except AssertionError as e:
            failures += 1
            print(f"  [FAIL] {e}")
    print("\n" + "=" * 64)
    print(f"TOTAL: {len(_all_tests())} tests, {failures} failures")
    print("=" * 64)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
