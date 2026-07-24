import subprocess
import wave

import numpy as np

SR = 44100


def dur_of(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(out.stdout.strip())


def atempo_filters(tempo: float) -> list[str]:
    """Chain atempo factors <= 2.0 for safety with old ffmpeg builds."""
    filters, t = [], tempo
    while t > 2.0:
        filters.append("atempo=2.0")
        t /= 2.0
    filters.append(f"atempo={t:.5f}")
    return filters


def decode_f32(path: str, tempo: float | None = 1.0, *, sr: int = SR, duration_s: float | None = None) -> np.ndarray:
    """Decode audio to mono float32 at sr; optionally tempo-adjusted and input-duration-limited."""
    cmd = ["ffmpeg", "-v", "error"]
    if duration_s is not None:
        cmd += ["-t", str(duration_s)]
    cmd += ["-i", path]
    if tempo is not None:
        cmd += ["-af", ",".join(atempo_filters(tempo))]
    cmd += ["-f", "f32le", "-ac", "1", "-ar", str(sr), "pipe:1"]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"decode failed {path}: {r.stderr.decode()[:300]}")
    return np.frombuffer(r.stdout, dtype=np.float32).copy()


def write_wav_pcm16(path: str, pcm: np.ndarray, sr: int) -> None:
    """Write mono 16-bit PCM samples as a RIFF wav."""
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sr)
        wav_file.writeframes(pcm.astype("<i2", copy=False).tobytes())


def srt_ts(t: float) -> str:
    h = int(t // 3600)
    m = int(t % 3600 // 60)
    s = t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")
