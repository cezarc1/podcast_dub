import re

import pytest

from podcast_dub import model_catalog
from podcast_dub.stages import asr, diarize, tts


def test_local_model_revisions_are_immutable_commit_hashes():
    revisions = (
        getattr(asr, "ASR_REVISION", None),
        getattr(asr, "ALIGNER_REVISION", None),
        getattr(diarize, "SORTFORMER_REVISION", None),
        getattr(tts, "TTS_REVISION", None),
    )

    assert all(isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision) for revision in revisions)


def test_model_snapshot_preflight_accepts_plain_local_configs(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type": "qwen3_tts", "architectures": ["Qwen3TTS"]}')
    nested = tmp_path / "speech_tokenizer"
    nested.mkdir()
    (nested / "config.json").write_text('{"model_type": "qwen3_tts_tokenizer"}')

    validate = getattr(model_catalog, "validate_model_snapshot", None)
    assert validate is not None
    validate(tmp_path)


@pytest.mark.parametrize(
    "unsafe_config",
    [
        '{"_attn_implementation_internal": "attacker/kernel"}',
        '{"nested": {"_experts_implementation_internal": "attacker/kernel"}}',
        '{"auto_map": {"AutoModel": "remote.Model"}}',
        '{"trust_remote_code": true}',
    ],
)
def test_model_snapshot_preflight_rejects_remote_code_hooks(tmp_path, unsafe_config):
    (tmp_path / "config.json").write_text(unsafe_config)

    validate = getattr(model_catalog, "validate_model_snapshot", None)
    assert validate is not None
    with pytest.raises(RuntimeError, match="unsafe model config field"):
        validate(tmp_path)
