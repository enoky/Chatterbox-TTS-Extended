from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import os

import librosa
import torch
import torch.nn.functional as F
import threading
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import numpy as np

# Watermark uses Python/NumPy RNG; serialize just that tiny section (kept for safety).
_WATERMARK_LOCK = threading.Lock()

from .models.t3 import T3
from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.tokenizers import EnTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond


def _pick_t3_dtype(device) -> torch.dtype:
    """Precision for the T3 Llama backbone, set with CHATTERBOX_T3_DTYPE=auto|fp32|fp16|bf16.

    auto (default): fp16 on CUDA GPUs, fp32 on CPU/MPS. fp16 tracks fp32 more closely
    than bf16 here and runs on older GPUs (e.g. T4) too.
    """
    choice = os.environ.get("CHATTERBOX_T3_DTYPE", "auto").strip().lower()
    named = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    if choice in named:
        return named[choice]
    if str(device).startswith("cuda") and torch.cuda.is_available():
        return torch.float16
    return torch.float32

REPO_ID = "ResembleAI/chatterbox"

def punc_norm(text: str) -> str:
    if not text:
        return "You need to add some text for me to talk."
    if text[0].islower():
        text = text[0].upper() + text[1:]
    text = " ".join(text.split())
    for old, new in [
        ("...", ", "), ("…", ", "), (":", ","), (" - ", ", "),
        (";", ", "), ("—", "-"), ("–", "-"), (" ,", ","),
        ("“", "\""), ("”", "\""), ("‘", "'"), ("’", "'"),
    ]:
        text = text.replace(old, new)
    text = text.rstrip(" ")
    if not any(text.endswith(p) for p in {".", "!", "?", "-", ","}):
        text += "."
    return text

@dataclass
class Conditionals:
    t3: T3Cond
    gen: dict
    def to(self, device):
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self
    def save(self, fpath: Path):
        torch.save(dict(t3=self.t3.__dict__, gen=self.gen), fpath)
    @classmethod
    def load(cls, fpath, map_location="cpu"):
        if isinstance(map_location, str):
            map_location = torch.device(map_location)
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])

class ChatterboxTTS:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(self, t3: T3, s3gen: S3Gen, ve: VoiceEncoder, tokenizer: EnTokenizer, device: str, conds: Conditionals=None):
        self.sr = S3GEN_SR
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.conds = conds
        # self.watermarker = perth.PerthImplicitWatermarker()
        self.default_conds = conds

    @classmethod
    def from_local(cls, ckpt_dir, device) -> 'ChatterboxTTS':
        ckpt_dir = Path(ckpt_dir)
        map_location = torch.device('cpu') if device in ["cpu", "mps"] else None

        ve = VoiceEncoder()
        ve.load_state_dict(load_file(ckpt_dir / "ve.safetensors"))
        ve.to(device).eval()

        t3 = T3()
        t3_state = load_file(ckpt_dir / "t3_cfg.safetensors")
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        t3.to(device).eval()
        t3_dtype = _pick_t3_dtype(device)
        if t3_dtype != torch.float32:
            t3.set_inference_dtype(t3_dtype)
        print(f"T3 backbone dtype: {t3_dtype}")

        s3gen = S3Gen()
        s3gen.load_state_dict(load_file(ckpt_dir / "s3gen.safetensors"), strict=False)
        s3gen.to(device).eval()

        tokenizer = EnTokenizer(str(ckpt_dir / "tokenizer.json"))

        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice, map_location=map_location).to(device)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    @classmethod
    def from_pretrained(cls, device) -> 'ChatterboxTTS':
        if device == "mps" and not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available (PyTorch not built with MPS).")
            else:
                print("MPS not available (macOS < 12.3 or no MPS device).")
            device = "cpu"
        for fpath in ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]:
            local_path = hf_hub_download(repo_id=REPO_ID, filename=fpath)
        return cls.from_local(Path(local_path).parent, device)

    # ---------- Conditionals helpers ----------
    def _build_conditionals(self, wav_fpath, exaggeration=0.5) -> Conditionals:
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)
        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

        t3_cond_prompt_tokens = None
        if (plen := self.t3.hp.speech_cond_prompt_len):
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)

        ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=torch.ones(1, 1, 1, device=self.device) * exaggeration,
        ).to(device=self.device)
        return Conditionals(t3_cond, s3gen_ref_dict)

    def _clone_conditionals(self, conds: Conditionals, exaggeration=None) -> Conditionals:
        _cond: T3Cond = conds.t3
        emo = _cond.emotion_adv if exaggeration is None else (exaggeration * torch.ones(1,1,1, device=_cond.emotion_adv.device))
        t3_cond = T3Cond(
            speaker_emb=_cond.speaker_emb.clone(),
            cond_prompt_speech_tokens=None if _cond.cond_prompt_speech_tokens is None else _cond.cond_prompt_speech_tokens.clone(),
            emotion_adv=emo.clone()
        ).to(device=self.device)
        gen = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in conds.gen.items()}
        return Conditionals(t3_cond, gen)

    # ---------- Single item ----------
    def generate(
        self,
        text: str,
        audio_prompt_path: Optional[str]=None,
        exaggeration: float=0.5,
        cfg_weight: float=0.5,
        temperature: float=0.8,
        apply_watermark: bool=False,
        generator: Optional[torch.Generator]=None,
        **kwargs,
    ) -> torch.Tensor:
        # text → tokens
        text = punc_norm(text)
        text_tokens = self.tokenizer.text_to_tokens(text).to(self.device)
        if cfg_weight > 0.0:
            text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # (2, Lt)
        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        # conditionals
        if audio_prompt_path:
            conds_local = self._build_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            if self.default_conds is None:
                raise RuntimeError("No default speaker embedding available; cannot reset to base voice.")
            conds_local = self._clone_conditionals(self.default_conds, exaggeration=exaggeration)

        # RNG sandbox (thread/process local)
        devices = [torch.cuda.current_device()] if self.device == "cuda" else []
        with torch.inference_mode(), torch.random.fork_rng(devices=devices, enabled=True):
            if generator is not None:
                seed_int = int(generator.initial_seed())
                torch.manual_seed(seed_int)
                if self.device == "cuda":
                    torch.cuda.manual_seed_all(seed_int)

            speech_tokens = self.t3.inference(
                t3_cond=conds_local.t3,
                text_tokens=text_tokens,
                max_new_tokens=1000,
                temperature=temperature,
                cfg_weight=cfg_weight,
                generator=generator,
            )
            speech_tokens = speech_tokens[0]
            speech_tokens = drop_invalid_tokens(speech_tokens)
            speech_tokens = speech_tokens[speech_tokens < 6561]
            speech_tokens = speech_tokens.to(self.device)

            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=conds_local.gen,
            )
            wav_np = wav.squeeze(0).detach().cpu().numpy()
            if apply_watermark and hasattr(self, "watermarker"):
                with _WATERMARK_LOCK:
                    wav_np = self.watermarker.apply_watermark(wav_np, sample_rate=self.sr)
            return torch.from_numpy(wav_np).unsqueeze(0)

    # ---------- Batched (vectorized sampler) ----------
    def generate_batch(
        self,
        texts: List[str],
        audio_prompt_paths: Optional[List[Optional[str]]]=None,
        exaggeration: float=0.5,
        cfg_weight: float=0.5,
        temperature: float=0.8,
        apply_watermark: bool=False,
        generator: Optional[torch.Generator]=None,
    ) -> List[torch.Tensor]:
        """
        Vectorized sampler: generate N utterances in one synchronized pass.
        Returns a list of (1, num_samples) tensors, one per input.
        """
        N = len(texts)
        assert N >= 1

        # Normalize + tokenize, then pad to common length
        token_list = []
        for txt in texts:
            t = punc_norm(txt)
            tt = self.tokenizer.text_to_tokens(t).to(self.device)
            sot = self.t3.hp.start_text_token
            eot = self.t3.hp.stop_text_token
            tt = F.pad(tt, (1, 0), value=sot)
            tt = F.pad(tt, (0, 1), value=eot)
            token_list.append(tt)

        max_len = max(x.size(1) for x in token_list)
        text_tokens = torch.stack([F.pad(x, (0, max_len - x.size(1)), value=self.t3.hp.stop_text_token) for x in token_list], dim=0)  # (B, Lt)

        # Conditionals per item
        conds_batch: List[Conditionals] = []
        if audio_prompt_paths is None:
            audio_prompt_paths = [None] * N

        for ap in audio_prompt_paths:
            if ap:
                conds_batch.append(self._build_conditionals(ap, exaggeration=exaggeration))
            else:
                if self.default_conds is None:
                    raise RuntimeError("No default speaker embedding available; cannot reset to base voice.")
                conds_batch.append(self._clone_conditionals(self.default_conds, exaggeration=exaggeration))

        # Merge T3 conds (stack speaker_emb / emotion_adv; drop prompt tokens if lengths mismatch)
        speaker_emb = torch.cat([c.t3.speaker_emb for c in conds_batch], dim=0).to(self.device)        # (B, D)
        emotion_adv = torch.cat([c.t3.emotion_adv for c in conds_batch], dim=0).to(self.device)        # (B,1,1)
        # cond_prompt_speech_tokens may have variable lengths; only keep if all present & equal length
        cps_list = [c.t3.cond_prompt_speech_tokens for c in conds_batch]
        if all((x is not None) for x in cps_list) and len({x.size(1) for x in cps_list}) == 1:
            cond_prompt_speech_tokens = torch.cat(cps_list, dim=0).to(self.device)
        else:
            cond_prompt_speech_tokens = None

        t3_cond = T3Cond(
            speaker_emb=speaker_emb,
            cond_prompt_speech_tokens=cond_prompt_speech_tokens,
            emotion_adv=emotion_adv,
        ).to(device=self.device)

        # RNG sandbox (thread/process local)
        devices = [torch.cuda.current_device()] if self.device == "cuda" else []
        with torch.inference_mode(), torch.random.fork_rng(devices=devices, enabled=True):
            if generator is not None:
                seed_int = int(generator.initial_seed())
                torch.manual_seed(seed_int)
                if self.device == "cuda":
                    torch.cuda.manual_seed_all(seed_int)

            # Vectorized AR sampling (B, Ttok)
            speech_tokens_batch = self.t3.inference_batch(
                t3_cond=t3_cond,
                text_tokens=text_tokens,
                max_new_tokens=1000,
                temperature=temperature,
                cfg_weight=cfg_weight,
                generator=generator,
            )  # (B, Ts)

            # Decode each item (S3Gen is deterministic path)
            out_wavs: List[torch.Tensor] = []
            for b in range(N):
                st = speech_tokens_batch[b]
                st = drop_invalid_tokens(st)
                st = st[st < 6561].to(self.device)

                wav, _ = self.s3gen.inference(
                    speech_tokens=st,
                    ref_dict=conds_batch[b].gen,
                )
                wav_np = wav.squeeze(0).detach().cpu().numpy()
                if apply_watermark and hasattr(self, "watermarker"):
                    with _WATERMARK_LOCK:
                        wav_np = self.watermarker.apply_watermark(wav_np, sample_rate=self.sr)
                out_wavs.append(torch.from_numpy(wav_np).unsqueeze(0))
            return out_wavs
