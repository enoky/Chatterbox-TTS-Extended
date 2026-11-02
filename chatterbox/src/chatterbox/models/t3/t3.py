# Copyright (c) 2025 Resemble AI
# MIT License
import logging
import os
import threading
from typing import Optional, Tuple

from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from transformers import LlamaModel, LlamaConfig
from transformers.generation.logits_process import TopPLogitsWarper, RepetitionPenaltyLogitsProcessor

from .modules.learned_pos_emb import LearnedPositionEmbeddings
from .modules.cond_enc import T3CondEnc, T3Cond
from .modules.t3_config import T3Config
from .llama_configs import LLAMA_CONFIGS
from .inference.t3_hf_backend import T3HuggingfaceBackend
from .inference.cuda_graph_decoder import CudaGraphDecoder

logger = logging.getLogger(__name__)


class GenerationInterrupted(Exception):
    """Raised from the sampling loop when `stop_check()` returns True."""


# Optional callable polled once per sampled token; the UI sets it so a Stop button
# can abort a chunk mid-generation instead of waiting for it to finish.
stop_check = None

# Checking for end-of-speech forces a GPU->CPU sync, which stops the CPU from queuing
# the next step while the GPU works. Check every N steps instead; rows that already
# finished keep emitting stop tokens, which callers drop.
EOS_CHECK_INTERVAL = 8

# Replay the decode step as a CUDA graph (see CudaGraphDecoder). Set
# CHATTERBOX_CUDA_GRAPHS=0 to use the plain eager loop instead.
USE_CUDA_GRAPHS = os.environ.get("CHATTERBOX_CUDA_GRAPHS", "1").strip() != "0"
# Static cache sizes are rounded up to this so one captured graph serves most chunks.
GRAPH_CACHE_BUCKET = 256


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def _ensure_BOT_EOT(text_tokens: Tensor, hp):
    B = text_tokens.size(0)
    assert (text_tokens == hp.start_text_token).int().sum() >= B, "missing start_text_token"
    assert (text_tokens == hp.stop_text_token).int().sum() >= B, "missing stop_text_token"


class T3(nn.Module):
    """
    Token-To-Token (T3) TTS model using huggingface transformer models as backbones.
    """
    def __init__(self, hp=T3Config()):
        super().__init__()
        self.hp = hp
        self.cfg = LlamaConfig(**LLAMA_CONFIGS[hp.llama_config_name])
        self.tfmr = LlamaModel(self.cfg)
        self.dim = self.cfg.hidden_size
        self.deepspeed_patch_applied = False

        # conditioning / embedding
        self.cond_enc = T3CondEnc(hp)
        self.text_emb = nn.Embedding(hp.text_tokens_dict_size, self.dim)
        self.speech_emb = nn.Embedding(hp.speech_tokens_dict_size, self.dim)

        # custom position embedding
        if hp.input_pos_emb == "learned":
            max_text_seq_len = hp.max_text_tokens + 2
            self.text_pos_emb = LearnedPositionEmbeddings(max_text_seq_len, self.dim)

            max_mel_seq_len = hp.max_speech_tokens + 2 + 2
            self.speech_pos_emb = LearnedPositionEmbeddings(max_mel_seq_len, self.dim)

        # logit projection
        self.text_head = nn.Linear(self.cfg.hidden_size, hp.text_tokens_dict_size, bias=False)
        self.speech_head = nn.Linear(self.cfg.hidden_size, hp.speech_tokens_dict_size, bias=False)
        self.compiled = False

        # CUDA-graph decoding state, shared by all threads using this model.
        self._graph_decoder = None
        self._graph_lock = threading.Lock()
        self._graphs_failed = False

    @property
    def device(self):
        return self.speech_head.weight.device

    def set_inference_dtype(self, dtype: torch.dtype):
        """Run the Llama backbone in `dtype` (fp16/bf16 roughly doubles decode speed on GPUs
        with tensor cores). Embeddings, conditioning and logit heads stay in fp32."""
        self.tfmr.to(dtype)
        self._graph_decoder = None  # captured for the old dtype

    def prepare_conditioning(self, t3_cond: T3Cond):
        """
        Token cond data needs to be embedded, so that needs to be here instead of in `T3CondEnc`.
        """
        if t3_cond.cond_prompt_speech_tokens is not None and t3_cond.cond_prompt_speech_emb is None:
            t3_cond.cond_prompt_speech_emb = self.speech_emb(t3_cond.cond_prompt_speech_tokens) + \
                self.speech_pos_emb(t3_cond.cond_prompt_speech_tokens)
        return self.cond_enc(t3_cond)  # (B, len_cond, dim)

    def _pad_stack_embeds(
        self,
        cond_emb: Tensor,     # (B, Lc, D) or (1, Lc, D)
        text_emb: Tensor,     # (B, Lt, D)
        speech_emb: Tensor,   # (B, Ls, D)
    ) -> Tuple[Tensor, int]:
        """
        Concatenate cond/text/speech per item, pad to a common length across the batch,
        and stack into a single tensor. Works with variable text/speech lengths.
        """
        B = text_emb.size(0)
        device, dtype = text_emb.device, text_emb.dtype
        len_cond = cond_emb.size(1)

        # If cond_emb is for a single item but text_emb has B items, repeat it.
        if cond_emb.size(0) == 1 and B > 1:
            cond_emb = cond_emb.expand(B, -1, -1)
        elif cond_emb.size(0) * 2 == text_emb.size(0):
            # When CFG duplicated later (2B), cond_emb may be size B; repeat rows.
            pass

        pieces = []
        max_len = 0
        for b in range(B):
            e = torch.cat((cond_emb[b], text_emb[b], speech_emb[b]), dim=0)  # (Lc+Lt+Ls, D)
            pieces.append(e)
            max_len = max(max_len, e.size(0))

        # Pad to common length and stack.
        padded = []
        for e in pieces:
            if e.size(0) < max_len:
                pad = torch.zeros(max_len - e.size(0), e.size(1), device=device, dtype=dtype)
                e = torch.cat([e, pad], dim=0)
            padded.append(e)
        stacked = torch.stack(padded, dim=0)  # (B, Lmax, D)
        return stacked, len_cond

    def prepare_input_embeds(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: torch.LongTensor,     # (B, Lt)
        speech_tokens: torch.LongTensor,   # (B, Ls)
    ):
        # Embed
        cond_emb = self.prepare_conditioning(t3_cond)    # (B or 1, Lc, D)
        text_emb = self.text_emb(text_tokens)            # (B, Lt, D)
        speech_emb = self.speech_emb(speech_tokens)      # (B, Ls, D)

        # Positional
        if self.hp.input_pos_emb == "learned":
            text_emb = text_emb + self.text_pos_emb(text_tokens)
            speech_emb = speech_emb + self.speech_pos_emb(speech_tokens)

        # If CFG duplication has been applied to the batch (every pair is cond/uncond),
        # zero out text embeddings on the uncond rows (1,3,5,...).
        if text_emb.size(0) >= 2:
            text_emb[1::2] = 0.0

        embeds, len_cond = self._pad_stack_embeds(cond_emb, text_emb, speech_emb)  # (B, Lmax, D)
        return embeds, len_cond

    def forward(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: torch.LongTensor,
        text_token_lens: torch.LongTensor,
        speech_tokens: torch.LongTensor,
        speech_token_lens: torch.LongTensor,
        training=False,
    ):
        _ensure_BOT_EOT(text_tokens, self.hp)

        embeds, len_cond = self.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=speech_tokens,
        )

        tfmr_out = self.tfmr.forward(
            input_ids=None,
            inputs_embeds=embeds,
            output_hidden_states=True,
            return_dict=True,
            use_cache=(not training),
        )
        hidden_states = tfmr_out.hidden_states[-1]  # (B, seq, dim)

        # Slice text/speech regions back out (for training only)
        len_text = text_tokens.size(1)
        len_speech = speech_tokens.size(1)
        B, _, dim = hidden_states.shape
        device, dtype = hidden_states.device, hidden_states.dtype
        text_latents = torch.zeros(B, len_text, dim, dtype=dtype, device=device)
        speech_latents = torch.zeros(B, len_speech, dim, dtype=dtype, device=device)
        ttl, stl = text_token_lens, speech_token_lens
        for i in range(B):
            text_end = len_cond + ttl[i].item()
            speech_start = len_cond + text_tokens.size(1)
            speech_end = speech_start + stl[i].item()
            text_latents[i, :ttl[i]] = hidden_states[i, len_cond:text_end]
            speech_latents[i, :stl[i]] = hidden_states[i, speech_start:speech_end]

        text_logits = self.text_head(text_latents)
        speech_logits = self.speech_head(speech_latents)

        return AttrDict(
            text_logits=text_logits,
            text_latents=text_latents,
            speech_logits=speech_logits,
            speech_latents=speech_latents,
            hidden_states=hidden_states,
        )

    @torch.inference_mode()
    def _sample_loop(
        self,
        *,
        inputs_embeds: Tensor,            # (2B or B, Lctx, D)
        cfg_weight: float,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        generator: Optional[torch.Generator],
        cond_batch: int,                  # B if CFG>0 else total B
        stop_id: int,
    ) -> Tensor:
        """
        Shared sampling loop, vectorized. Returns predicted cond tokens: (B, T).
        """
        use_graphs = USE_CUDA_GRAPHS and inputs_embeds.is_cuda and not self._graphs_failed
        if use_graphs:
            # One captured graph per model: threads take turns (the GPU is saturated
            # during graph decoding anyway, so running them side by side gains little).
            with self._graph_lock:
                try:
                    decoder = self._get_graph_decoder(
                        batch_size=inputs_embeds.size(0),
                        needed_len=inputs_embeds.size(1) + max_new_tokens + 1,
                    )
                    return self._sample_tokens(decoder, inputs_embeds, cfg_weight, max_new_tokens, temperature,
                                               top_p, repetition_penalty, generator, cond_batch, stop_id)
                except GenerationInterrupted:
                    raise
                except Exception as e:
                    logger.warning("CUDA graph decoding failed (%s: %s); using the eager loop from now on.",
                                   type(e).__name__, e)
                    self._graphs_failed = True
                    self._graph_decoder = None
        return self._sample_tokens(None, inputs_embeds, cfg_weight, max_new_tokens, temperature,
                                   top_p, repetition_penalty, generator, cond_batch, stop_id)

    def _get_graph_decoder(self, batch_size: int, needed_len: int) -> CudaGraphDecoder:
        decoder = self._graph_decoder
        if decoder is None or decoder.batch_size != batch_size or decoder.max_cache_len < needed_len:
            self._graph_decoder = None  # release the old cache before allocating a new one
            max_len = -(-needed_len // GRAPH_CACHE_BUCKET) * GRAPH_CACHE_BUCKET
            decoder = CudaGraphDecoder(self.patched_model, batch_size, max_len)
            self._graph_decoder = decoder
        return decoder

    def _sample_tokens(self, decoder, inputs_embeds, cfg_weight, max_new_tokens, temperature,
                       top_p, repetition_penalty, generator, cond_batch, stop_id) -> Tensor:
        """The sampling loop. Uses `decoder` (a CudaGraphDecoder) when given, else the
        eager backend with a dynamic KV cache."""
        # Logit processors
        top_p_warper = TopPLogitsWarper(top_p=top_p)
        repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty)

        # Initial forward. Attentions and per-layer hidden states are not requested:
        # nothing reads them, and output_attentions=True forces the slow eager attention path.
        ctx_len = inputs_embeds.size(1)
        if decoder is not None:
            step_logits = decoder.prefill(inputs_embeds)[:, -1, :]

            def advance(embed, i):
                return decoder.step(embed, ctx_len + i)[:, -1, :]
        else:
            output = self.patched_model(
                inputs_embeds=inputs_embeds,
                past_key_values=None,
                use_cache=True,
                return_dict=True,
            )
            past = output.past_key_values
            step_logits = output.logits[:, -1, :]

            def advance(embed, i):
                nonlocal past
                output = self.patched_model(
                    inputs_embeds=embed,
                    past_key_values=past,
                    return_dict=True,
                )
                past = output.past_key_values
                return output.logits[:, -1, :]

        # Track generated tokens for the COND rows only
        device = inputs_embeds.device
        bos_token = torch.tensor([[self.hp.start_speech_token]], dtype=torch.long, device=device)
        generated_ids = bos_token.repeat(cond_batch, 1)  # (B, 1)

        predicted = []
        alive = torch.ones(cond_batch, dtype=torch.bool, device=device)

        for i in tqdm(range(max_new_tokens), desc="Sampling", dynamic_ncols=True):
            if stop_check is not None and stop_check():
                raise GenerationInterrupted()

            if cfg_weight > 0.0:
                # cond rows are 0,2,4,...; uncond rows are 1,3,5,...
                logits_cond = step_logits[0::2]           # (B, V)
                logits_uncond = step_logits[1::2]         # (B, V)
                logits = logits_cond + cfg_weight * (logits_cond - logits_uncond)
            else:
                logits = step_logits                       # (B, V)

            # Temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Repetition penalty + top-p work on the cond batch
            logits = repetition_penalty_processor(generated_ids, logits)
            logits = top_p_warper(None, logits)

            probs = torch.softmax(logits, dim=-1)  # (B, V)
            next_token = torch.multinomial(probs, num_samples=1, generator=generator)  # (B, 1)

            # Respect EOS per sequence
            next_token[~alive] = stop_id
            predicted.append(next_token)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            # Update alive mask
            alive &= (next_token.view(-1) != stop_id)
            if (i + 1) % EOS_CHECK_INTERVAL == 0 and not alive.any():
                break

            # Advance the model one token (duplicate for CFG if used)
            next_tok_embed = self.speech_emb(next_token) + self.speech_pos_emb.get_fixed_embedding(i + 1)  # (B,1,D)
            if cfg_weight > 0.0:
                next_tok_embed = torch.repeat_interleave(next_tok_embed, repeats=2, dim=0)                 # (2B,1,D)

            step_logits = advance(next_tok_embed, i)  # (2B or B, V)

        if not predicted:
            return torch.zeros(cond_batch, 0, dtype=torch.long, device=device)
        out = torch.cat(predicted, dim=1)
        # Drop the extra stop tokens emitted between EOS checks: keep each row up to its
        # first stop token, so the output matches checking every step.
        is_stop = out == stop_id
        ends = torch.where(is_stop.any(dim=1), is_stop.int().argmax(dim=1) + 1, out.size(1))
        return out[:, :int(ends.max())]

    @torch.inference_mode()
    def inference(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: Tensor,                       # (2, Lt) if CFG>0 via caller
        initial_speech_tokens: Optional[Tensor]=None,
        prepend_prompt_speech_tokens: Optional[Tensor]=None,
        num_return_sequences=1,
        max_new_tokens=1000,
        stop_on_eos=True,
        do_sample=True,
        temperature=0.8,
        top_p=0.8,
        length_penalty=1.0,
        repetition_penalty=2.0,
        cfg_weight=0.0,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """
        Single-item path (kept for compatibility). Returns (1, T).
        Caller passes text_tokens duplicated for CFG if cfg_weight>0.0.
        """
        assert prepend_prompt_speech_tokens is None, "not implemented"
        _ensure_BOT_EOT(text_tokens, self.hp)
        text_tokens = torch.atleast_2d(text_tokens).to(dtype=torch.long, device=self.device)

        B = text_tokens.size(0)
        use_cfg = cfg_weight > 0.0
        assert (not use_cfg) or (B % 2 == 0), "CFG enabled but batch not duplicated"

        if initial_speech_tokens is None:
            initial_speech_tokens = self.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])

        # Prepare context embeds for the (already duplicated) batch
        embeds, _ = self.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=initial_speech_tokens,
        )

        # Compile backend once
        if not self.compiled:
            self.patched_model = T3HuggingfaceBackend(
                config=self.cfg,
                llama=self.tfmr,
                speech_enc=self.speech_emb,
                speech_head=self.speech_head,
            )
            self.compiled = True

        # BOS embed, duplicated for CFG if used, then concat to context
        device = embeds.device
        bos_token = torch.tensor([[self.hp.start_speech_token]], dtype=torch.long, device=device)
        bos_embed = self.speech_emb(bos_token) + self.speech_pos_emb.get_fixed_embedding(0)  # (1,1,D)
        if use_cfg:
            bos_embed = torch.cat([bos_embed, bos_embed], dim=0)  # (2,1,D)
        inputs_embeds = torch.cat([embeds, bos_embed.expand(embeds.size(0), -1, -1)], dim=1)

        cond_batch = (B // 2) if use_cfg else B
        pred = self._sample_loop(
            inputs_embeds=inputs_embeds,
            cfg_weight=cfg_weight,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            generator=generator,
            cond_batch=cond_batch,
            stop_id=self.hp.stop_speech_token,
        )
        return pred  # (1, T) when single item

    @torch.inference_mode()
    def inference_batch(
        self,
        *,
        t3_cond: T3Cond,                         # batched conds (B, ...)
        text_tokens: Tensor,                     # (B, Lt) BEFORE CFG duplication
        initial_speech_tokens: Optional[Tensor]=None,
        max_new_tokens=1000,
        temperature=0.8,
        top_p=0.8,
        repetition_penalty=2.0,
        cfg_weight=0.0,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """
        Batched path. Returns predicted cond tokens (B, T).
        """
        _ensure_BOT_EOT(text_tokens, self.hp)
        text_tokens = torch.atleast_2d(text_tokens).to(dtype=torch.long, device=self.device)
        B = text_tokens.size(0)

        if initial_speech_tokens is None:
            initial_speech_tokens = self.hp.start_speech_token * torch.ones((B, 1), dtype=torch.long, device=self.device)

        # Duplicate batch for CFG if used: (2B, Lt) / (2B, Ls)
        if cfg_weight > 0.0:
            text_tokens_cfg = torch.cat([text_tokens, text_tokens], dim=0)
            speech_tokens_cfg = torch.cat([initial_speech_tokens, initial_speech_tokens], dim=0)
        else:
            text_tokens_cfg = text_tokens
            speech_tokens_cfg = initial_speech_tokens

        # Prepare context embeddings
        embeds, _ = self.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens_cfg,
            speech_tokens=speech_tokens_cfg,
        )

        # Compile backend once
        if not self.compiled:
            self.patched_model = T3HuggingfaceBackend(
                config=self.cfg,
                llama=self.tfmr,
                speech_enc=self.speech_emb,
                speech_head=self.speech_head,
            )
            self.compiled = True

        # BOS embed for each row (already matched size via expand)
        device = embeds.device
        bos_token = torch.tensor([[self.hp.start_speech_token]], dtype=torch.long, device=device)
        bos_embed = self.speech_emb(bos_token) + self.speech_pos_emb.get_fixed_embedding(0)  # (1,1,D)
        inputs_embeds = torch.cat([embeds, bos_embed.expand(embeds.size(0), -1, -1)], dim=1)

        # Vectorized sampling; cond_batch is B when CFG used, else B
        cond_batch = B
        pred = self._sample_loop(
            inputs_embeds=inputs_embeds,
            cfg_weight=cfg_weight,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            generator=generator,
            cond_batch=cond_batch,
            stop_id=self.hp.stop_speech_token,
        )  # (B, T)
        return pred
