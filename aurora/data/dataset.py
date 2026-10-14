# -*- coding: utf-8 -*-
"""
预训练数据集（零依赖，CPU 友好）。

来源 / 集大成
------------
* 语言建模标准做法：把整本语料 tokenize 后拼成一个扁平的 int 序列，训练时以
  **滑动窗口** 连续采样 (x, x_shift) 对，最大化样本利用率（见 GPT / Llama 训练）。
* **内存映射（mmap）**：超大语料不进内存，靠 `numpy.memmap` 按需分页读取，
  单机 CPU 也能训练任意规模语料（HuggingFace `tokenizers` + `datasets` 的
  memmap 思路同源，这里零依赖复现）。

两个入口
--------
* `TokenizedDataset`   : 从 .bin（uint16/int32）memmap 加载，滑窗采样。
* `TextCorpusDataset`  : 直接在内存里放若干条已编码文本，演示 / 小规模训练用。
* `build_pretrain_dataloader` : 随机打乱起点的批量采样器，返回 (ids, labels)。
"""

from __future__ import annotations

import numpy as np
from typing import Iterable, Iterator, List, Optional, Tuple

from aurora.data.tokenizer import ByteLevelBPETokenizer


class TokenizedDataset:
    """内存映射的扁平 token 序列，滑窗返回 (chunk, chunk_shifted)。"""

    def __init__(self, path: str, dtype: str = "int32"):
        self.path = path
        self.arr = np.memmap(path, dtype=dtype, mode="r")
        self.n = int(self.arr.shape[0])

    def __len__(self) -> int:
        return self.n

    def chunk(self, start: int, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
        """取 [start, start+seq_len+1) 的连续窗口，返回 (x, y=x[1:])。"""
        end = min(start + seq_len + 1, self.n)
        block = self.arr[start:end].astype(np.int64)
        if block.shape[0] < seq_len + 1:
            # 尾部不足则从头回卷补齐（保证形状恒定）
            need = seq_len + 1 - block.shape[0]
            head = self.arr[:need].astype(np.int64)
            block = np.concatenate([block, head])
        x = block[:-1]
        y = block[1:]
        return x, y

    def random_batch(self, batch_size: int, seq_len: int,
                     rng: Optional[np.random.Generator] = None):
        rng = rng or np.random.default_rng()
        xs, ys = [], []
        for _ in range(batch_size):
            s = int(rng.integers(0, max(1, self.n - seq_len)))
            x, y = self.chunk(s, seq_len)
            xs.append(x)
            ys.append(y)
        return np.stack(xs), np.stack(ys)


class TextCorpusDataset:
    """内存态语料：多条已编码文本拼接 + 滑窗采样。演示 / 小模型训练用。"""

    def __init__(self, token_ids: List[int]):
        self.ids = np.asarray(token_ids, dtype=np.int64)
        self.n = int(self.ids.shape[0])

    @classmethod
    def from_texts(cls, texts: Iterable[str], tokenizer: ByteLevelBPETokenizer,
                   bos: str = "<bos>", eos: str = "<eos>") -> "TextCorpusDataset":
        out: List[int] = []
        for t in texts:
            if bos in tokenizer.special_ids:
                out.append(tokenizer.special_ids[bos])
            out.extend(tokenizer.encode(t))
            if eos in tokenizer.special_ids:
                out.append(tokenizer.special_ids[eos])
        return cls(out)

    def __len__(self) -> int:
        return self.n


def build_pretrain_dataloader(
    source: "TokenizedDataset | TextCorpusDataset | List[int]",
    batch_size: int,
    seq_len: int,
    shuffle: bool = True,
    seed: int = 0,
):
    """返回一个无限迭代器，每次产出 (input_ids (B,T), labels (B,T)) 的 numpy 数组。

    支持三种 source：
      - TokenizedDataset        （memmap）
      - TextCorpusDataset / np.ndarray / list（内存态）
    全部转为「扁平 int 序列 + 滑窗」语义，统一接口。
    """
    if isinstance(source, TokenizedDataset):
        arr = source.arr
    elif isinstance(source, TextCorpusDataset):
        arr = source.ids
    elif isinstance(source, (list, tuple)):
        arr = np.asarray(source, dtype=np.int64)
    else:
        arr = np.asarray(source, dtype=np.int64)
    n = int(arr.shape[0])
    rng = np.random.default_rng(seed)

    def _gen() -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        while True:
            xs, ys = [], []
            for _ in range(batch_size):
                if shuffle:
                    s = int(rng.integers(0, max(1, n - seq_len)))
                else:
                    s = (getattr(_gen, "_i", 0) * seq_len) % max(1, n - seq_len)
                    _gen._i = getattr(_gen, "_i", 0) + 1
                end = min(s + seq_len + 1, n)
                block = np.asarray(arr[s:end], dtype=np.int64)
                if block.shape[0] < seq_len + 1:
                    need = seq_len + 1 - block.shape[0]
                    block = np.concatenate([block, np.asarray(arr[:need], dtype=np.int64)])
                xs.append(block[:-1])
                ys.append(block[1:])
            yield np.stack(xs), np.stack(ys)

    return _gen()
