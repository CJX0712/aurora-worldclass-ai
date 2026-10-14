# -*- coding: utf-8 -*-
"""
Aurora 训练栈 / 推理引擎 不变量自检。

覆盖
----
1. Byte-level BPE 分词器：编码→解码严格无损（round-trip 恒等）。
2. 数据集：滑窗批次 (x, y) 形状正确且 y = x 右移一位。
3. Trainer（MuonClip+QK-Clip）：训练若干步 loss 有限且 QK-Clip 把超额 logit 钉在阈值下。
4. 推理引擎 == 朴素贪婪解码（逐 token 一致）。
5. MTP 自投机解码（greedy）== 非投机 greedy（输出逐 token 一致）。
6. GRPO：单次 train_step loss 有限、反向可回传。

运行
----
    python tests/test_stack.py
"""

import sys

import torch


def _pass(cond, msg, extra=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {msg}" + (f"  ({extra})" if extra else ""))
    if not cond:
        raise AssertionError(f"{msg} FAILED {extra}")


# ---------------------------------------------------------------------------
# 1) 分词器 round-trip
# ---------------------------------------------------------------------------
def test_tokenizer_roundtrip():
    from aurora.data.tokenizer import ByteLevelBPETokenizer
    corpus = [
        "Aurora 是晨星锻造的世界顶级 AI。",
        "Multi-head Latent Attention compresses the KV cache by 100x.",
        "The quick brown fox jumps over the lazy dog 1234567890.",
        "混合注意力：MLA + GDN + SWA = frontier-class language model.",
    ] * 20
    tok = ByteLevelBPETokenizer(vocab_size=2000,
                                special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"])
    tok.train(corpus, verbose=False)
    texts = [
        "Aurora 是晨星锻造的世界顶级 AI。",
        "The quick brown fox jumps over the lazy dog 1234567890.",
        "混合注意力：MLA + GDN + SWA = frontier-class language model.",
    ]
    for t in texts:
        ids = tok.encode(t)
        _pass(all(0 <= i < tok.vocab_size for i in ids),
              "token ids within vocab range", f"n={len(ids)}")
        dec = tok.decode(ids)
        _pass(dec == t, "tokenizer round-trip is lossless", f"{dec!r} vs {t!r}")


# ---------------------------------------------------------------------------
# 2) 数据集滑窗批次
# ---------------------------------------------------------------------------
def test_dataset_batch():
    from aurora.data.dataset import build_pretrain_dataloader
    ids = list(range(500))
    loader = build_pretrain_dataloader(ids, batch_size=4, seq_len=32)
    x, y = next(loader)
    _pass(tuple(x.shape) == (4, 32) and tuple(y.shape) == (4, 32),
          "dataloader batch shape (B, T)", f"x={tuple(x.shape)} y={tuple(y.shape)}")
    _pass(x[:, 1:].tolist() == y[:, :-1].tolist(),
          "labels are input shifted right by 1", "")


# ---------------------------------------------------------------------------
# 3) Trainer 运行（MuonClip + QK-Clip）
# ---------------------------------------------------------------------------
def test_trainer_runs():
    from aurora.config import tiny
    from aurora.model import AuroraForCausalLM
    from aurora.train.trainer import Trainer
    from aurora.data.dataset import build_pretrain_dataloader

    torch.manual_seed(0)
    cfg = tiny("nano", mixer_pattern="DDSM")
    model = AuroraForCausalLM(cfg)
    # 构造可记忆的重复模式语料：因果 LM 能学会 → loss 应下降
    pattern = [10, 20, 30, 40, 50, 60, 70, 80]
    ids = (pattern * 400)[:3200]
    loader = build_pretrain_dataloader(ids, batch_size=4, seq_len=32, seed=123)

    trainer = Trainer(model, cfg, lr=1e-3, muon_lr=0.02, grad_clip=1.0,
                      qk_clip_threshold=100.0)
    losses = [trainer.train_step(*next(loader)) for _ in range(60)]
    _pass(all(torch.isfinite(torch.tensor(losses)).tolist()),
          "trainer loss finite over 60 steps", f"last={losses[-1]:.3f}")
    _pass(losses[-1] < losses[0],
          "trainer loss decreases on memorizable data",
          f"first={losses[0]:.3f} last={losses[-1]:.3f}")
    # QK-Clip 生效：最大 attention logit 不会无限膨胀
    amax = 0.0
    for m in trainer.attn_modules:
        a = getattr(m, "attn_max_logits", None)
        if a is not None:
            amax = max(amax, float(a.max().item()))
    _pass(amax < 120.0, "QK-Clip keeps max attention logit bounded", f"amax={amax:.2f}")


# ---------------------------------------------------------------------------
# 4) 推理引擎 == 朴素贪婪解码
# ---------------------------------------------------------------------------
def _greedy_reference(model, prompt_ids, max_new=24, eos=None):
    model.eval()
    with torch.no_grad():
        ids = list(prompt_ids)
        past = None
        for _ in range(max_new):
            cur = torch.tensor([ids[-1:]] if past is not None else [ids],
                               dtype=torch.long)
            out = model(cur, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = int(out.logits[0, -1].argmax(-1).item())
            if eos is not None and nxt == eos:
                break
            ids.append(nxt)
    return ids


def test_engine_matches_naive():
    from aurora.config import tiny
    from aurora.model import AuroraForCausalLM
    from aurora.inference.engine import AuroraEngine

    torch.manual_seed(0)
    cfg = tiny("nano", mixer_pattern="DDSM", num_nextn_predict_layers=1)
    model = AuroraForCausalLM(cfg)
    engine = AuroraEngine(model, tokenizer=None, device="cpu",
                          eos_token_id=None, bos_token_id=None)
    prompt = [5, 11, 23, 7, 42]
    ref = _greedy_reference(model, prompt, max_new=20)
    out = engine.generate([prompt], max_new_tokens=20, return_ids=True)
    _pass(out[0] == ref, "engine greedy == reference greedy decode",
          f"len engine={len(out[0])} ref={len(ref)}")


# ---------------------------------------------------------------------------
# 5) MTP 自投机 == 非投机（greedy）
# ---------------------------------------------------------------------------
def test_speculative_eq_greedy():
    from aurora.config import tiny
    from aurora.model import AuroraForCausalLM
    from aurora.inference.engine import AuroraEngine

    torch.manual_seed(0)
    cfg = tiny("nano", mixer_pattern="DDSM", num_nextn_predict_layers=2)
    model = AuroraForCausalLM(cfg)
    engine = AuroraEngine(model, tokenizer=None, device="cpu",
                          eos_token_id=None, bos_token_id=None)
    prompt = [5, 11, 23, 7, 42]
    base = engine.generate([prompt], max_new_tokens=20, speculative=1, return_ids=True)[0]
    spec = engine.generate([prompt], max_new_tokens=20, speculative=3, return_ids=True)[0]
    _pass(spec == base, "MTP speculative == non-speculative (greedy)",
          f"len base={len(base)} spec={len(spec)}")


# ---------------------------------------------------------------------------
# 6) GRPO 单步可训练
# ---------------------------------------------------------------------------
def test_grpo_runs():
    from aurora.config import tiny
    from aurora.model import AuroraForCausalLM
    from aurora.train.grpo import GRPOTrainer

    torch.manual_seed(0)
    cfg = tiny("nano", mixer_pattern="DDSM")
    model = AuroraForCausalLM(cfg)
    ref = AuroraForCausalLM(cfg)
    trainer = GRPOTrainer(model, ref, lr=1e-5, device="cpu")

    B = 4
    T = 16
    ids = torch.randint(0, cfg.vocab_size, (B, T), dtype=torch.long)
    attn = torch.ones(B, T, dtype=torch.long)
    rewards = torch.randn(B)
    groups = torch.arange(B, dtype=torch.long)
    loss, info = trainer.train_step(ids, attn, rewards, groups)
    _pass(torch.isfinite(torch.tensor(loss)).item(),
          "GRPO train_step loss finite", f"loss={loss:.4f} kl={info['kl']:.4f}")


# ---------------------------------------------------------------------------
def _all_tests():
    return [
        test_tokenizer_roundtrip,
        test_dataset_batch,
        test_trainer_runs,
        test_engine_matches_naive,
        test_speculative_eq_greedy,
        test_grpo_runs,
    ]


def main():
    print("=" * 64)
    print("Aurora Training / Inference Stack Self-Checks")
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
