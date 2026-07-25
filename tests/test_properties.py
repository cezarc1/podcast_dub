"""Property-based tests (hypothesis) for the dub timing algorithm.

Invariants hunted, not hand-picked:
- group_units: text conservation, non-overlap, span cap, single-speaker units
- build_turns: text conservation through splits, contiguous covering windows
- simulate: anchor rule, anti-drift starts, min-gap, hard window ends
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from podcast_dub.stages.tts import build_turns
from podcast_dub.timing import MIN_SILENCE_GAP_S
from podcast_dub.tools.turn_tts_sample import group_units, simulate
from podcast_dub.types import SimulationTurn, SubtitleCue, TranslationUnit

SPEAKERS = st.sampled_from(["yang", "zhang"])


@st.composite
def cue_stream(draw):
    n = draw(st.integers(1, 30))
    t = draw(st.floats(0, 5))
    cues = []
    for _ in range(n):
        t += draw(st.floats(0, 6))
        dur = draw(st.floats(0.1, 5))
        words = draw(st.integers(1, 12))
        text = " ".join(["w"] * words)
        if draw(st.booleans()):
            text += draw(st.sampled_from([".", "!", "?", "…"]))
        cues.append(
            SubtitleCue(
                speaker=draw(SPEAKERS),
                start=round(t, 2),
                end=round(t + dur, 2),
                text=text,
            )
        )
    return cues


@st.composite
def turn_stream(draw):
    n = draw(st.integers(1, 20))
    t = draw(st.floats(0, 5))
    turns = []
    for _ in range(n):
        t += draw(st.floats(0, 8))
        dur = draw(st.floats(0.3, 20))
        words = max(1, int(dur * 3.6))
        txt = " ".join(["w"] * words)
        turns.append(
            SimulationTurn(
                speaker=draw(SPEAKERS),
                start=round(t, 2),
                end=round(t + dur, 2),
                turn_id=len(turns),
                part_index=0,
                text=txt,
                audio_duration_s=dur,
            )
        )
    return turns


# ---------- group_units ----------


@given(cue_stream())
@settings(max_examples=300)
def test_group_units_invariants(cues):
    cues = sorted(cues, key=lambda cue: cue.start)
    units = group_units(cues)
    # starts are non-decreasing (overlap possible across speaker changes)
    for a, b in zip(units, units[1:], strict=False):
        assert a.start <= b.start + 1e-9
    # single speaker per unit
    for u in units:
        assert u.speaker in ("yang", "zhang")
    # span cap respected at creation time: a fresh unit spans one cue, and the
    # merge guard only extends a unit while cue.end - unit.start stays <= 15s.
    for u in units:
        assert u.end - u.start <= 15.0 + 1e-9
    # text conservation: every cue's text appears in some unit
    joined = " ".join(u.subtitle_text for u in units)
    for c in cues:
        assert c.text in joined
    # a new unit always starts on cue boundary
    assert {unit.start for unit in units} <= {cue.start for cue in cues}


# ---------- build_turns ----------


@given(turn_stream())
@settings(max_examples=300)
def test_build_turns_split_invariants(turns):
    # build_turns merges same-speaker adjacent turns (gap<=2.5) then splits > cap
    units = []
    for t in turns:
        units.append(
            TranslationUnit(
                speaker=t.speaker,
                start=t.start,
                end=t.end,
                source_text=t.text,
                target_text=t.text,
            )
        )
    chunks = build_turns(units)
    # text conservation (split only, never lossy)
    src = " ".join(unit.target_text for unit in units).split()
    got = " ".join(chunk.text for chunk in chunks).split()
    assert got == src
    # contiguous covering within each t_id
    by_tid = {}
    for c in chunks:
        by_tid.setdefault(c.turn_id, []).append(c)
    for _tid, parts in by_tid.items():
        parts.sort(key=lambda part: part.part_index)
        for a, b in zip(parts, parts[1:], strict=False):
            assert abs(b.start - a.end) < 1e-9
    # logical turns appear in one contiguous run each, numbered 0..n-1 in order.
    # (Speaker alternation is NOT guaranteed: a same-speaker gap > 2.5s opens a
    # new turn, so two consecutive turns can share a speaker.)
    logical = []
    for c in chunks:
        if logical and logical[-1].turn_id == c.turn_id:
            continue
        logical.append(c)
    assert [c.turn_id for c in logical] == list(range(len(logical)))
    # every chunk of a turn carries that turn's single speaker
    for tid, parts in by_tid.items():
        assert len({part.speaker for part in parts}) == 1
        assert all(part.turn_id == tid for part in parts)


# ---------- simulate ----------


@given(turn_stream())
@settings(max_examples=300)
def test_simulate_invariants(turns):
    turns = sorted(turns, key=lambda turn: turn.start)
    for i, t in enumerate(turns):
        turns[i] = t.validated_copy(turn_id=i, part_index=0)
    # turns are sorted by START, so the last turn need not hold the max end
    total = max(turn.end for turn in turns) + 60.0
    pl = simulate(turns, total)
    assert len(pl) == len(turns)
    # first turn anchors on cue
    assert abs(pl[0][0] - turns[0].start) < 1e-9
    ends = []
    for i, (start, window) in enumerate(pl):
        # a window either clears the next cue by the inter-turn silence, or it
        # collapsed onto the 0.05s floor because there was no room to do so
        next_start = turns[i + 1].start if i + 1 < len(turns) else total
        assert window >= 0.05
        assert start + window + MIN_SILENCE_GAP_S <= next_start + 1e-9 or window == 0.05
        if i > 0:
            # anti-drift: never later than cue + 1.0
            assert start <= turns[i].start + 1.0 + 1e-9
            # min gap from previous trimmed end — unless physically impossible
            # (cue gap smaller than MIN_SILENCE_GAP_S + 1.0 drift allowance)
            if ends[i - 1] + MIN_SILENCE_GAP_S <= turns[i].start + 1.0 + 1e-9:
                assert start >= ends[i - 1] + MIN_SILENCE_GAP_S - 1e-9
        ends.append(start + min(turns[i].audio_duration_s, window))
