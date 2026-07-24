"""Unit tests for the coalescing/chunking code (the class of bug that produced
the split-window timing disaster): group_units (cues -> sentence units) and
build_turns (units -> speaker turns with capped split chunks).

Invariants under test:
- text conservation (nothing lost, added, or reordered)
- window monotonicity (non-overlapping, contiguous within a turn, covering the span)
- split constraints (sentence boundaries, size cap)
"""

from podcast_dub.models import SubtitleCue, TranslationUnit
from podcast_dub.stages.tts import MAX_CHUNK_S, SENT_END, TURN_GAP, build_turns
from podcast_dub.tools.turn_tts_sample import group_units


def cue(speaker, start, end, text):
    return SubtitleCue(speaker=speaker, start=start, end=end, text=text)


def unit(speaker, start, end, text):
    return TranslationUnit(
        speaker=speaker,
        start=start,
        end=end,
        source_text=f"source:{text}",
        target_text=text,
    )


def words(s):
    return len(s.split())


# ---------------- group_units ----------------


class TestGroupUnits:
    def test_merges_same_speaker_unfinished_sentence(self):
        cues = [cue("yang", 0, 2, "I think that"), cue("yang", 2.2, 4, "this is one sentence.")]
        units = group_units(cues)
        assert len(units) == 1
        assert units[0].start == 0 and units[0].end == 4

    def test_breaks_on_sentence_end(self):
        cues = [cue("yang", 0, 2, "Done here."), cue("yang", 2.2, 4, "Next one.")]
        assert len(group_units(cues)) == 2

    def test_breaks_on_speaker_change(self):
        cues = [cue("yang", 0, 2, "unfinished"), cue("zhang", 2.2, 3, "reply")]
        assert len(group_units(cues)) == 2

    def test_breaks_on_large_gap(self):
        cues = [cue("yang", 0, 2, "unfinished"), cue("yang", 5.0, 6, "later")]
        assert len(group_units(cues)) == 2

    def test_breaks_on_span_cap(self):
        cues = [
            cue("yang", 0, 10, "a very long unfinished thought"),
            cue("yang", 10.2, 20, "continuing past fifteen seconds"),
        ]
        assert len(group_units(cues)) == 2

    def test_text_conservation(self):
        cues = [cue("yang", 0, 1, "alpha"), cue("yang", 1.1, 2, "beta"), cue("zhang", 2.1, 3, "gamma.")]
        units = group_units(cues)
        joined = " ".join(u.subtitle_text for u in units)
        for t in ("alpha", "beta", "gamma."):
            assert t in joined

    def test_windows_monotonic(self):
        cues = [cue("yang", 0, 1, "a"), cue("yang", 1.1, 2, "b"), cue("yang", 2.1, 3, "c."), cue("zhang", 3.1, 4, "d")]
        units = group_units(cues)
        for a, b in zip(units, units[1:], strict=False):
            assert a.end <= b.start + 1e-9


# ---------------- build_turns ----------------


class TestBuildTurns:
    def test_groups_into_turns_by_speaker_and_gap(self):
        units = [
            unit("yang", 0, 5, "one."),
            unit("yang", 6, 10, "two."),
            unit("zhang", 11, 12, "q."),
            unit("yang", 30, 35, "later."),
        ]
        turns = build_turns(units)
        assert len(turns) == 3
        assert turns[0].speaker == "yang"
        assert turns[1].speaker == "zhang"
        assert turns[2].start == 30

    def test_text_conservation_through_split(self):
        long_text = " ".join(f"Sentence number {i} is here." for i in range(60))
        units = [unit("yang", 0, 300, long_text)]
        turns = build_turns(units)
        assert len(turns) > 1  # must split
        joined = " ".join(t.text for t in turns)
        assert joined.split() == long_text.split()

    def test_split_windows_monotonic_and_covering(self):
        """The actual bug: split halves shared one window. Guard against it."""
        long_text = " ".join(f"Sentence number {i} is here." for i in range(60))
        units = [unit("yang", 100, 400, long_text)]
        turns = build_turns(units)
        assert len(turns) > 1
        assert turns[0].start == 100
        assert abs(turns[-1].end - 400) < 1e-9
        for a, b in zip(turns, turns[1:], strict=False):
            assert b.start >= a.end - 1e-9, f"overlap: {a.end} > {b.start}"
            assert abs(b.start - a.end) < 1e-9, "gap inside turn"

    def test_split_respects_sentence_boundaries(self):
        long_text = " ".join(f"Sentence number {i} is here." for i in range(60))
        turns = build_turns([unit("yang", 0, 300, long_text)])
        for t in turns[:-1]:  # all but the last chunk must end on sentence punctuation
            assert SENT_END.search(t.text), f"chunk ends mid-sentence: ...{t.text[-40:]}"

    def test_split_chunks_respect_size_cap(self):
        long_text = " ".join(f"Sentence number {i} is here." for i in range(60))
        turns = build_turns([unit("yang", 0, 300, long_text)])
        for t in turns:
            assert words(t.text) / 2.5 <= MAX_CHUNK_S + 8  # one sentence of slack

    def test_short_turn_not_split(self):
        turns = build_turns([unit("yang", 0, 10, "Short turn.")])
        assert len(turns) == 1

    def test_no_split_means_windows_unchanged(self):
        units = [unit("yang", 0, 10, "Short."), unit("zhang", 11, 20, "Q.")]
        turns = build_turns(units)
        assert [(t.start, t.end) for t in turns] == [(0, 10), (11, 20)]

    def test_gap_boundary_uses_turn_gap_constant(self):
        units = [unit("yang", 0, 5, "a."), unit("yang", 5 + TURN_GAP + 0.1, 10, "b.")]
        assert len(build_turns(units)) == 2
        units[1] = units[1].validated_copy(start=5 + TURN_GAP - 0.1)
        assert len(build_turns(units)) == 1

    def test_zero_length_unit_widened(self):
        """Aligner point spans (start == end) must not crash build_turns."""
        turns = build_turns([unit("yang", 22.0, 22.0, "Hello.")])
        assert len(turns) == 1
        assert turns[0].start == 22.0
        assert turns[0].end > turns[0].start
        assert turns[0].text == "Hello."

    def test_zero_length_unit_merges_with_neighbor(self):
        units = [unit("yang", 10, 20, "real."), unit("yang", 20.5, 20.5, "point.")]
        turns = build_turns(units)
        assert len(turns) == 1
        assert abs(turns[0].end - 20.8) < 1e-9
        assert "point." in turns[0].text

    def test_nested_unit_keeps_monotone_end(self):
        """A unit fully inside the previous turn must not shrink the merged window."""
        units = [unit("yang", 10, 20, "start."), unit("yang", 15, 16, "nested.")]
        turns = build_turns(units)
        assert len(turns) == 1
        assert turns[0].end == 20
        assert "nested." in turns[0].text

    def test_empty_input(self):
        assert build_turns([]) == ()
