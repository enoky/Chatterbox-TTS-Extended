"""CUDA-graph replay of the T3 decode step.

Decoding one speech token runs the 30-layer Llama backbone on a single position, which
is hundreds of small kernels launched one by one from Python. On most GPUs the CPU can't
issue them as fast as the GPU runs them, so the GPU idles for much of each step. Recording
one decode step as a CUDA graph and replaying it removes that overhead. Graphs need fixed
tensor addresses, so the KV cache is a preallocated `StaticCache`.
"""
import torch
from transformers import StaticCache


class CudaGraphDecoder:
    """A static KV cache plus a captured decode step, for one batch size and cache length.

    Reused across sequences: `prefill()` restarts the cache for a new context. Not
    thread-safe; the caller must hold a lock from `prefill()` until its last `step()`.
    """

    def __init__(self, backend, batch_size: int, max_cache_len: int):
        self.backend = backend
        self.batch_size = batch_size
        self.max_cache_len = max_cache_len
        llama = backend.model
        device = backend.speech_head.weight.device
        self.cache = StaticCache(config=llama.config, max_cache_len=max_cache_len)
        self.static_in = torch.zeros(batch_size, 1, llama.config.hidden_size, device=device, dtype=llama.dtype)
        self.static_pos = torch.zeros(1, dtype=torch.long, device=device)
        self.graph = None
        self.static_logits = None

    def _set_length(self, n: int):
        # StaticCache writes each new token at a per-layer counter; rewind it to `n`.
        for layer in self.cache.layers:
            layer.cumulative_length.fill_(n)

    def _decode(self):
        out = self.backend(
            inputs_embeds=self.static_in,
            past_key_values=self.cache,
            cache_position=self.static_pos,
            return_dict=True,
        )
        return out.logits

    def prefill(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """Run the context (B, L, D) eagerly into a fresh cache. Returns its logits (B, L, V)."""
        L = inputs_embeds.size(1)
        if L >= self.max_cache_len:
            raise ValueError(f"context of {L} tokens does not fit a cache of {self.max_cache_len}")
        self._set_length(0)
        out = self.backend(
            inputs_embeds=inputs_embeds,
            past_key_values=self.cache,
            cache_position=torch.arange(L, device=inputs_embeds.device),
            return_dict=True,
        )
        if self.graph is None:
            self._capture(L)
        return out.logits

    def _capture(self, L: int):
        # Warm up on a side stream (required before capture). These steps write junk at
        # position L, which the first real step overwrites.
        self.static_pos.fill_(L)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._set_length(L)
                self._decode()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        # thread_local: other worker threads may keep using the GPU during capture.
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            self.static_logits = self._decode()
        self.graph = graph
        self._set_length(L)

    def step(self, embed: torch.Tensor, position: int) -> torch.Tensor:
        """Decode one token: `embed` (B, 1, D) at absolute `position` (the current cache
        length). Returns logits (B, 1, V), which the next call overwrites."""
        if position >= self.max_cache_len:
            raise ValueError(f"position {position} is past the cache length {self.max_cache_len}")
        self.static_in.copy_(embed)
        self.static_pos.fill_(position)
        self.graph.replay()
        return self.static_logits
