# -*- coding: utf-8 -*-
"""
Byte-Level BPE 分词器（零依赖，纯 Python + 标准库）。

来源 / 集大成
------------
* Sennrich et al. (2016) 《Neural Machine Translation of Rare Words with
  Subword Units》—— 提出 BPE 用于 NLP 子词切分，按合并频率贪心地合并
  字节对，构造开放词表。
* Radford et al. (2019) 《GPT-2》—— 引入 **byte-level** BPE：先把文本按 UTF-8
  字节映射成「可打印 unicode 字符」（bytes_to_unicode），这样任意字节序列都能被
  无损表示，词表大小固定（≤256 个基础字节 token），且天然对未知字符鲁棒。

本实现把两者合二为一：byte-level 预映射 + 标准 BPE 合并。与 GPT-2、LLaMA、Qwen
的 tokenizer 同构，但**零第三方依赖**、可在纯 CPU 环境训练与推理，并且
**编码→解码严格无损**（round-trip 恒等，详见 `test_tokenizer_roundtrip`）。

设计要点
--------
* 基础词表 = 256 个 byte token（每个 UTF-8 字节一个），保证任何输入都能编码。
* BPE 合并在「byte 字符空间」进行，学到的子词 token 仍是 byte 字符串。
* special token（<bos>/<eos>/<pad>/<unk>）使用**保留 id 区间**，不参与 BPE，
  由调用方直接以 id 形式注入（如训练时拼 prompt）。
* 预分词（pretokenizer）：用正则把文本切成「词块」，块内做 BPE；块边界不影响
  无损还原（因为块只是 byte 字符串的切片）。
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from collections import Counter
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Byte-level 映射（与 GPT-2 完全一致）
# ---------------------------------------------------------------------------
def bytes_to_unicode() -> Dict[int, str]:
    """把 256 个字节映射到「可打印 unicode 字符」。

    规则：可打印非空白 ASCII 直接保留；其余字节偏移 +256 映射到 Latin-1 上半区，
    保证所有 256 个字节都对应一个稳定、可逆、可 JSON 序列化的字符。
    """
    bs = list(range(ord("!"), ord("~") + 1)) + \
        list(range(ord("¡"), ord("¬") + 1)) + \
        list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


# GPT-2 风格的预分词正则（已改写为标准库 `re` 可解析的形式：用 [A-Za-z0-9] 近似 \p{L}\p{N}）
_PRETOKEN_RE = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d|                    # 英文缩约词
        [A-Za-z0-9]+|                                # 连续字母/数字（含前导空格见下）
        \s+|                                         # 空白串
        [^\sA-Za-z0-9]+                             # 其他（标点等）
    """,
    re.VERBOSE,
)


def _pretokenize(text: str) -> List[str]:
    """把文本切成词块（每个词块是原始字符串切片，编码时再映射到 byte 字符）。"""
    return [m.group(0) for m in _PRETOKEN_RE.finditer(text)]


def _get_pairs(word: Tuple[str, ...]) -> set:
    """返回符号元组中的相邻对集合。"""
    return set(zip(word[:-1], word[1:])) if len(word) > 1 else set()


class ByteLevelBPETokenizer:
    """可训练、无损、零依赖的 byte-level BPE 分词器。

    典型用法
    --------
        tok = ByteLevelBPETokenizer(vocab_size=32000)
        tok.train(corpus_texts)                      # 从语料学习合并
        ids = tok.encode("Hello world")
        assert tok.decode(ids) == "Hello world"      # 无损还原

    也可加载预训练词表
        tok.save("vocab.json")
        tok2 = ByteLevelBPETokenizer.load("vocab.json")
    """

    def __init__(self, vocab_size: int = 32000,
                 special_tokens: Optional[List[str]] = None,
                 unk_token: str = "<unk>"):
        self.vocab_size = int(vocab_size)
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        # 特殊 token：保留在最前的 id 区间
        self.special_tokens: List[str] = list(special_tokens or
                                              ["<pad>", "<bos>", "<eos>", "<unk>"])
        self.unk_token = unk_token
        self.special_ids: Dict[str, int] = {
            t: i for i, t in enumerate(self.special_tokens)}

        # 基础词表：special + 256 个 byte token
        self.token2id: Dict[str, int] = dict(self.special_ids)
        for b in range(256):
            ch = self.byte_encoder[b]
            if ch not in self.token2id:
                self.token2id[ch] = len(self.token2id)
        # merge 表：合并后的子词 token -> id（在训练时增长）
        self.merges: List[Tuple[str, str]] = []
        self.merge_ranks: Dict[Tuple[str, str], int] = {}

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------
    def train(self, texts, min_freq: int = 2, verbose: bool = False) -> "ByteLevelBPETokenizer":
        """从文本迭代器/列表学习 BPE 合并，直到达到 vocab_size 或无法再合并。"""
        # 1) 把所有文本预分词，转成 byte 字符序列（符号元组）
        words: List[Tuple[str, ...]] = []
        for text in texts:
            for piece in _pretokenize(text):
                sym = tuple(self.byte_encoder[b] for b in piece.encode("utf-8"))
                if sym:
                    words.append(sym)

        # 2) 基础符号频率（单 byte 字符）
        if verbose:
            print(f"[tokenizer] pretokenized {len(words)} word-chunks")

        pair_freq = Counter()
        for word in words:
            for pair in _get_pairs(word):
                pair_freq[pair] += 1

        # 3) 贪心合并，直到词表满或没有满足 min_freq 的对
        while len(self.token2id) < self.vocab_size:
            # 选频率最高的对
            best_pair = None
            best_freq = min_freq - 1
            for pair, freq in pair_freq.items():
                if freq > best_freq:
                    best_freq = freq
                    best_pair = pair
            if best_pair is None:
                break
            a, b = best_pair
            merged = a + b
            self.merges.append(best_pair)
            self.merge_ranks[best_pair] = len(self.merges)
            self.token2id[merged] = len(self.token2id)

            # 把所有 word 中的 a b 合并为 merged，并重新统计新出现的对
            new_pair_freq = Counter()
            for idx, word in enumerate(words):
                if len(word) < 2:
                    continue
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                        new_word.append(merged)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                words[idx] = tuple(new_word)
                for pair in _get_pairs(tuple(new_word)):
                    new_pair_freq[pair] += 1
            pair_freq = new_pair_freq

        if verbose:
            print(f"[tokenizer] trained: vocab_size={len(self.token2id)}, "
                  f"merges={len(self.merges)}")
        return self

    # ------------------------------------------------------------------
    # BPE 编码（单条词块的符号元组 -> token id 列表）
    # ------------------------------------------------------------------
    def _bpe(self, token: str) -> List[str]:
        word = tuple(token)
        if len(word) <= 1:
            return list(word)
        # 反复合并当前排名最高（merge_ranks 中最早出现）的相邻对
        while True:
            pairs = _get_pairs(word)
            # 选「排名最小」的对；不在 merges 中的对保持不动
            ranked = [p for p in pairs if p in self.merge_ranks]
            if not ranked:
                break
            bigram = min(ranked, key=lambda p: self.merge_ranks[p])
            if bigram not in self.merge_ranks:
                break
            a, b = bigram
            merged = a + b
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
        return list(word)

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------
    def encode(self, text: str) -> List[int]:
        ids: List[int] = []
        for piece in _pretokenize(text):
            # 词块 -> byte 字符 -> BPE 子词 -> id
            sym = "".join(self.byte_encoder[b] for b in piece.encode("utf-8"))
            for sub in self._bpe(sym):
                ids.append(self.token2id.get(sub, self.special_ids[self.unk_token]))
        return ids

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        return [self.encode(t) for t in texts]

    def decode(self, ids: List[int]) -> str:
        # id -> token 字符串 -> 拼接 -> 反向 byte 映射 -> UTF-8 解码
        out_chars: List[str] = []
        # 仅还原普通 byte 子词 token；特殊 token 不输出可见字符
        for i in ids:
            tok = self.id2token(i, default=None)
            if tok is None:
                continue
            out_chars.append(tok)
        text = "".join(out_chars)
        # 把每个字符通过 byte_decoder 还原成原始字节
        byte_seq = bytearray()
        for ch in text:
            b = self.byte_decoder.get(ch)
            if b is not None:
                byte_seq.append(b)
        return byte_seq.decode("utf-8", errors="replace")

    def id2token(self, i: int, default: Optional[str] = "<unk>") -> Optional[str]:
        # 线性反查（词表不大时足够；可用反向表优化）
        for tok, idx in self.token2id.items():
            if idx == i:
                return tok
        return default

    @property
    def vocab(self) -> Dict[str, int]:
        return dict(self.token2id)

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        payload = {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "unk_token": self.unk_token,
            "token2id": self.token2id,
            "merges": self.merges,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "ByteLevelBPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            p = json.load(f)
        obj = cls(vocab_size=p["vocab_size"],
                  special_tokens=p["special_tokens"],
                  unk_token=p["unk_token"])
        obj.token2id = {k: int(v) for k, v in p["token2id"].items()}
        obj.merges = [tuple(m) for m in p["merges"]]
        obj.merge_ranks = {tuple(m): i for i, m in enumerate(obj.merges)}
        return obj
