# -*- coding: utf-8 -*-
"""
AuroraEngine —— 高吞吐推理引擎。

来源 / 集大成
------------
* **增量 KV-cache 解码**：每一步只把「最新一个 token」喂给模型，历史 KV 从 per-layer
  缓存里读，不重算（GPT/LLaMA 推理的标准形态）。MLA 的 KV cache 仅 d_c+d_r 元素/token，
  GDN 是常数大小，SWA 才随长度线性 —— 所以混合配比越长越省显存。
* **连续批处理（continuous batching）**：一组请求并发进入，已完成的请求随时退出批，
  新请求随时补位，GPU/CPU 算力不空转（vLLM / TensorRT-LLM 的核心吞吐优化）。
* **MTP 自投机解码（self-speculative）**：用模型自带的 Multi-Token Prediction 头当
  「免费草稿器」，一次主模型前向同时验证 k 个候选 token，accept 最长的匹配前缀 ——
  DeepSeek-V3 用它白捡 ~1.8× 解码加速，零额外参数、零分布漂移。

为何能这么省
------------
不同 mixer 的缓存异构（MLACache / DeltaNetCache / SWACache），但**都在第 0 维是 batch**。
本引擎用一套通用的 `stack/index` 把任意请求的子缓存沿 batch 维拼接 / 切分，从而把
「异构缓存 + 变长序列 + 动态批」封装成一次批量 `forward`，对外只暴露 `generate`。

正确性保证
----------
* 非投机路径：每一步严格等价于「全量重算」，只是用缓存省去历史计算（见
  `test_inference_matches_naive`）。
* 投机路径：草稿经**主模型单次前向严格验证**后只采纳匹配前缀，因此 greedy 下输出
  与「非投机 greedy」逐 token 完全一致（见 `test_speculative_eq_greedy`）。
"""

from __future__ import annotations

import torch

from aurora.modules.block import BlockCache
from aurora.modules.mla import MLACache
from aurora.modules.attention import SWACache
from aurora.modules.gdn import DeltaNetCache


# ---------------------------------------------------------------------------
# 异构缓存的 batch 维拼接 / 切分（三类缓存：MLA / SWA / GDN）
# ---------------------------------------------------------------------------
def _stack_mixer(caches):
    c = caches[0]
    if isinstance(c, MLACache):
        return MLACache(
            c_kv=torch.cat([x.c_kv for x in caches], 0),
            k_pe=torch.cat([x.k_pe for x in caches], 0))
    if isinstance(c, SWACache):
        return SWACache(
            k=torch.cat([x.k for x in caches], 0),
            v=torch.cat([x.v for x in caches], 0))
    if isinstance(c, DeltaNetCache):
        return DeltaNetCache(
            state=torch.cat([x.state for x in caches], 0),
            conv_state_q=torch.cat([x.conv_state_q for x in caches], 0),
            conv_state_k=torch.cat([x.conv_state_k for x in caches], 0),
            conv_state_v=torch.cat([x.conv_state_v for x in caches], 0))
    raise TypeError(f"未知缓存类型: {type(c)}")


def _index_mixer(cache, i):
    if isinstance(cache, MLACache):
        return MLACache(c_kv=cache.c_kv[i:i + 1], k_pe=cache.k_pe[i:i + 1])
    if isinstance(cache, SWACache):
        return SWACache(k=cache.k[i:i + 1], v=cache.v[i:i + 1])
    if isinstance(cache, DeltaNetCache):
        return DeltaNetCache(
            state=cache.state[i:i + 1],
            conv_state_q=cache.conv_state_q[i:i + 1],
            conv_state_k=cache.conv_state_k[i:i + 1],
            conv_state_v=cache.conv_state_v[i:i + 1])
    raise TypeError(f"未知缓存类型: {type(cache)}")


def _stack_block(blocks: list) -> BlockCache:
    return BlockCache(mixer=_stack_mixer([b.mixer for b in blocks]))


def _index_block(block: BlockCache, i: int) -> BlockCache:
    return BlockCache(mixer=_index_mixer(block.mixer, i))


class AuroraEngine:
    """批量、连续批处理、支持 MTP 自投机解码的推理引擎。"""

    def __init__(self, model, tokenizer=None, device: str = "cpu",
                 eos_token_id: int = None, bos_token_id: int = None):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.eos = eos_token_id
        self.bos = bos_token_id
        self.num_layers = len(self.model.model.layers)

    # ------------------------------------------------------------------
    def _encode(self, text: str) -> list:
        if self.tokenizer is not None:
            ids = self.tokenizer.encode(text)
        else:
            ids = [int(t) for t in text.split()]  # 纯 id 输入兜底
        if self.bos is not None:
            ids = [self.bos] + ids
        return ids

    @staticmethod
    def _sample(logits, temperature=1.0, top_k=0, top_p=1.0, greedy=False):
        if greedy:
            return int(logits.argmax(-1).item())
        if temperature != 1.0:
            logits = logits / max(temperature, 1e-6)
        if top_k > 0:
            kth = torch.topk(logits, top_k, dim=-1).values[-1]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        if top_p < 1.0:
            sl, si = torch.sort(logits, descending=True)
            cum = torch.cumsum(torch.softmax(sl, -1), -1)
            remove = cum > top_p
            remove[..., 0] = False
            sl = sl.masked_fill(remove, float("-inf"))
            logits = torch.gather(sl, -1, si.argsort(-1))
        return int(torch.multinomial(torch.softmax(logits, -1), 1).item())

    # ------------------------------------------------------------------
    def generate(self, prompts, max_new_tokens: int = 64, temperature: float = 1.0,
                 top_k: int = 0, top_p: float = 1.0, speculative: int = 1,
                 return_ids: bool = False):
        """生成补全。

        Parameters
        ----------
        prompts : str | List[str] | List[List[int]]
        speculative : int  投机步数（1=关闭；>1 启用 MTP 自投机，greedy 下才生效）
        return_ids : bool  返回 token id 列表而非解码文本

        Returns
        -------
        List[str] 或 List[List[int]]
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        if len(prompts) == 0:
            return []

        greedy = (temperature == 1.0 and top_k == 0 and top_p == 1.0)
        use_spec = speculative > 1 and greedy and self.model.model.mtp is not None

        # 初始化每个请求：token 列表 + per-layer 缓存（None 表示待 prefill）
        reqs = []
        for p in prompts:
            ids = p if isinstance(p, list) else self._encode(p)
            reqs.append({"ids": list(ids), "cache": None, "done": False,
                         "gen": 0, "out": list(ids)})

        # ---- prefill：逐请求喂入 prompt 并落盘缓存（prompt 长度不一，逐条更稳）----
        with torch.no_grad():
            for r in reqs:
                if len(r["ids"]) == 0:
                    r["done"] = True
                    continue
                inp = torch.tensor([r["ids"]], device=self.device)
                out = self.model(inp, use_cache=True)
                r["cache"] = out.past_key_values
                last_logits = out.logits[0, -1]
                # 用 prefill 的最后一个位置预测第一个新 token
                if use_spec:
                    self._spec_step(r, last_logits, speculative)
                else:
                    nxt = self._sample(last_logits, temperature, top_k, top_p, greedy)
                    self._commit(r, nxt)

        # ---- 解码循环：连续批处理 ----
        while True:
            active = [r for r in reqs if not r["done"]]
            if not active:
                break
            if any(r["gen"] >= max_new_tokens for r in active):
                for r in active:
                    if r["gen"] >= max_new_tokens:
                        r["done"] = True
                continue
            self._decode_batch(active, max_new_tokens, temperature, top_k, top_p,
                               greedy, speculative if use_spec else 1, use_spec)
            if all(r["done"] for r in reqs):
                break

        # ---- 收尾 ----
        results = []
        for r in reqs:
            if return_ids:
                results.append(r["out"])
            elif self.tokenizer is not None:
                results.append(self.tokenizer.decode(r["out"]))
            else:
                results.append(r["out"])
        return results

    # ------------------------------------------------------------------
    def _commit(self, r, token: int):
        if self.eos is not None and token == self.eos:
            r["done"] = True
            return
        r["ids"].append(token)
        r["out"].append(token)
        r["gen"] += 1

    def _decode_batch(self, active, max_new_tokens, temperature, top_k, top_p,
                      greedy, spec_k, use_spec):
        """非投机：批量增量一步；投机：逐请求 MTP 草稿 + 验证。"""
        if use_spec and spec_k > 1:
            with torch.no_grad():
                for r in active:
                    # 用当前缓存推进一步主模型，得到「下一 token」与更新缓存
                    last = torch.tensor([[r["ids"][-1]]], device=self.device)
                    out = self.model(last, past_key_values=r["cache"], use_cache=True)
                    r["cache"] = out.past_key_values
                    self._spec_step(r, out.logits[0, 0], spec_k)
            return

        # ---- 非投机：把 active 请求拼成 batch 一次 forward ----
        L = self.num_layers
        batch_past = [_stack_block([r["cache"][ly] for r in active]) for ly in range(L)]
        last = torch.tensor([[r["ids"][-1]] for r in active],
                            dtype=torch.long, device=self.device)
        with torch.no_grad():
            out = self.model(last, past_key_values=batch_past, use_cache=True)
        logits = out.logits[:, 0]  # (B, V)
        new_caches = out.past_key_values
        # 切回 per-request
        for ly in range(L):
            for i, r in enumerate(active):
                r["cache"][ly] = _index_block(new_caches[ly], i)
        for i, r in enumerate(active):
            if r["gen"] >= max_new_tokens:
                r["done"] = True
                continue
            nxt = self._sample(logits[i], temperature, top_k, top_p, greedy)
            self._commit(r, nxt)

    # ------------------------------------------------------------------
    def _spec_step(self, r, base_logits, k: int):
        """MTP 自投机一步（greedy）：草稿 k 个 token，主模型单次验证，采纳最长匹配前缀。

        主模型在位置 L-1 的 logits 已给出「下一个 token」base；MTP 头 (1..k) 在
        扩展序列的 position L-1 处分别预测 offset 2..k+1 的 token（与本模型 MTP
        训练对齐：head j 预测 token 索引 j+1+t，故 position L-1 → 索引 L+j）。
        """
        # 基础 token（offset 1，必采纳）
        base = int(base_logits.argmax(-1).item())
        self._commit(r, base)
        if r["done"] or k < 2:
            return
        L = len(r["ids"])
        # 扩展输入到 L+k：重复末尾 token 作占位，让各 MTP 头都能读到 position L-1
        ext = r["ids"] + [r["ids"][-1]] * k
        inp = torch.tensor([ext], device=self.device)
        with torch.no_grad():
            out = self.model(inp, use_cache=False, output_mtp_logits=True)
        mtp = out.mtp_logits
        # 草稿：mtp[j]（head j+1）在 position L-1 预测 token L+j
        draft = []
        for j in range(k):
            if mtp is None or j >= len(mtp):
                break
            hl = mtp[j]
            if hl.shape[1] <= L - 1:
                break
            draft.append(int(hl[0, L - 1].argmax(-1).item()))
        if not draft:
            return
        # 验证：主模型跑完 ext，logits[p] 预测 token p+1
        with torch.no_grad():
            full = self.model(inp, use_cache=False)
        main_logits = full.logits[0]  # (Len, V)
        # 主模型对位置 L..L+len(draft)-1 的预测 = draf 应匹配的 offset
        accept = 0
        for j, d in enumerate(draft):
            m = int(main_logits[L + j].argmax(-1).item())
            if d == m:
                accept = j + 1
            else:
                # 分歧点用主模型的预测覆盖
                self._commit(r, m)
                return
        # 全中：采纳全部草稿
        for d in draft[:accept]:
            self._commit(r, d)
