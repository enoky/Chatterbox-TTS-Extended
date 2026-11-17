"""Tests for per-chunk candidate generation, attempts and retry seeding."""
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch


def _import_chatter():
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # Stub the TTS/VC model modules so importing Chatter doesn't load real models.
    if "Chatter" not in sys.modules:
        chatterbox_mod = types.ModuleType("chatterbox")
        chatterbox_src = types.ModuleType("chatterbox.src")
        inner_cb = types.ModuleType("chatterbox.src.chatterbox")
        tts_mod = types.ModuleType("chatterbox.src.chatterbox.tts")
        vc_mod = types.ModuleType("chatterbox.src.chatterbox.vc")
        models_mod = types.ModuleType("chatterbox.src.chatterbox.models")
        t3_pkg = types.ModuleType("chatterbox.src.chatterbox.models.t3")
        t3_mod = types.ModuleType("chatterbox.src.chatterbox.models.t3.t3")
        tts_mod.ChatterboxTTS = type("ChatterboxTTS", (), {})
        vc_mod.ChatterboxVC = type("ChatterboxVC", (), {})
        t3_mod.GenerationInterrupted = type("GenerationInterrupted", (Exception,), {})
        t3_mod.stop_check = None
        t3_pkg.t3 = t3_mod
        models_mod.t3 = t3_pkg
        inner_cb.models = models_mod
        chatterbox_mod.src = chatterbox_src
        chatterbox_src.chatterbox = inner_cb
        sys.modules.update({
            "chatterbox": chatterbox_mod,
            "chatterbox.src": chatterbox_src,
            "chatterbox.src.chatterbox": inner_cb,
            "chatterbox.src.chatterbox.tts": tts_mod,
            "chatterbox.src.chatterbox.vc": vc_mod,
            "chatterbox.src.chatterbox.models": models_mod,
            "chatterbox.src.chatterbox.models.t3": t3_pkg,
            "chatterbox.src.chatterbox.models.t3.t3": t3_mod,
        })
    import Chatter
    return Chatter


class RecordingModel:
    """Fake TTS model that records the seed of every generate() call.

    The output waveform is a pure function of the generator's seed, mirroring
    how the real model's sampling is fully determined by the seed.
    """
    sr = 16000
    device = "cpu"

    def __init__(self, fail_calls=()):
        self.seeds = []
        self.fail_calls = set(fail_calls)

    def generate(self, text, audio_prompt_path=None, exaggeration=0.5,
                 cfg_weight=0.5, temperature=0.8, apply_watermark=False,
                 generator=None):
        call_no = len(self.seeds)
        self.seeds.append(generator.initial_seed())
        if call_no in self.fail_calls:
            raise RuntimeError("simulated generation failure")
        return torch.rand((1, self.sr), generator=generator) * 0.1


@pytest.fixture
def chatter(tmp_path, monkeypatch):
    Chatter = _import_chatter()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "temp").mkdir()
    Chatter._stop_event.clear()
    return Chatter


def _run(Chatter, model, *, candidates=3, attempts=3, bypass=False, retry_number=1, seed=1234):
    return Chatter.process_one_chunk_deterministic(
        model, "Hello there, this is a test.", 0, 0, seed,
        None, 0.5, 0.8, 0.5,
        True, candidates, attempts,
        bypass,
        retry_number,
    )[1]


def test_derive_seed_round_zero_matches_original_formula(chatter):
    # Round 0 must keep producing the same seeds as before retry rounds existed,
    # so first-pass output for a given user seed is unchanged.
    for args in [(1234, 0, 0, 0), (99, 5, 2, 1), (2**32 - 1, 12, 9, 9)]:
        base, chunk, cand, attempt = args
        mix = (np.uint64(base) * np.uint64(1000003)
               + np.uint64(chunk) * np.uint64(10007)
               + np.uint64(cand) * np.uint64(10009)
               + np.uint64(attempt) * np.uint64(101))
        expected = int(mix & np.uint64(0xFFFFFFFF)) or 1
        assert chatter.derive_seed(*args) == expected
        assert chatter.derive_seed(*args, retry_round=0) == expected


def test_derive_seed_unique_across_retry_rounds(chatter):
    seeds = {
        chatter.derive_seed(1234, 3, cand, attempt, retry_round=rnd)
        for cand in range(10) for attempt in range(10) for rnd in range(10)
    }
    assert len(seeds) == 10 * 10 * 10


def test_successful_candidate_is_not_regenerated(chatter):
    model = RecordingModel()
    cands = _run(chatter, model, candidates=3, attempts=3)
    # One generation per candidate when nothing fails (not candidates * attempts).
    assert len(model.seeds) == 3
    assert [c["cand_idx"] for c in cands] == [0, 1, 2]
    assert all(c["attempt"] == 0 for c in cands)


def test_failed_generation_is_retried_with_next_attempt(chatter):
    model = RecordingModel(fail_calls={0})  # first attempt of candidate 0 errors
    cands = _run(chatter, model, candidates=2, attempts=3)
    assert len(model.seeds) == 3
    assert [(c["cand_idx"], c["attempt"]) for c in cands] == [(0, 1), (1, 0)]


def test_retry_round_uses_new_seeds_and_audio(chatter):
    model = RecordingModel()
    first = _run(chatter, model, candidates=2, attempts=1, retry_number=1)
    retry = _run(chatter, model, candidates=2, attempts=1, retry_number=2)
    assert not {c["seed"] for c in first} & {c["seed"] for c in retry}
    for a, b in zip(first, retry):
        wav_a, _ = chatter.load_wav(a["path"])
        wav_b, _ = chatter.load_wav(b["path"])
        assert not torch.equal(wav_a, wav_b)


def test_first_pass_is_reproducible(chatter):
    a = _run(chatter, RecordingModel(), candidates=2, attempts=1)
    b = _run(chatter, RecordingModel(), candidates=2, attempts=1)
    assert [c["seed"] for c in a] == [c["seed"] for c in b]


def test_bypass_stops_after_first_successful_candidate(chatter):
    model = RecordingModel()
    cands = _run(chatter, model, candidates=3, attempts=3, bypass=True)
    assert len(model.seeds) == 1
    assert [(c["cand_idx"], c["attempt"]) for c in cands] == [(0, 0)]


def test_bypass_falls_through_failed_attempts(chatter):
    model = RecordingModel(fail_calls={0, 1, 2})  # all attempts of candidate 0 fail
    cands = _run(chatter, model, candidates=3, attempts=3, bypass=True)
    assert len(model.seeds) == 4
    assert [(c["cand_idx"], c["attempt"]) for c in cands] == [(1, 0)]
