"""Unit tests for podcast_dub.stages.diarize.label_phrases — the split-at-handoff logic.

Invariants under test:
- a phrase never spans two speakers when word timings allow splitting
- diarization blips (< MIN_SPLIT_S on one side) never cause splits
- text conservation across splits (nothing lost, added, or reordered)
- noise speakers (< MIN_SPEECH_FRAC of total) are dropped from labeling
- phrases without word timings fall back to whole-phrase max-overlap labeling
"""

from podcast_dub.stages.diarize import MIN_SPEECH_FRAC, label_phrases
from podcast_dub.types import DiarizationSegment, Phrase, PhraseWord, SpeakerPhrase


def seg(start, end, speaker):
    return DiarizationSegment(start=start, end=end, speaker=speaker)


def word(t, s, e):
    return PhraseWord(text=t, start=s, end=e)


def phrase(start, end, text, words=None):
    return Phrase(start=start, end=end, text=text, words=tuple(words or ()))


# two speakers, both well above the noise floor
SEGS = [seg(0, 10, "speaker_0"), seg(10.1, 20, "speaker_1")]


class TestLabelPhrases:
    def test_simple_overlap_labeling(self):
        phr = [phrase(1, 3, "ab"), phrase(12, 14, "cd")]
        result = label_phrases(phr, SEGS, ["host", "guest"])
        assert [p.speaker for p in result.phrases] == ["host", "guest"]
        assert result.assignment.mapping == {"speaker_0": "host", "speaker_1": "guest"}

    def test_split_at_real_handoff(self):
        # words 0-5s host, 5-9s guest; handoff segment boundary at 5.0/5.1
        # (speaker_0 has more total time -> maps to first name "host")
        segs = [seg(0, 5.0, "speaker_0"), seg(5.1, 9.0, "speaker_1")]
        words = [word("你", 1.0, 1.4), word("好", 1.5, 1.9), word("大", 5.3, 5.7), word("家", 5.8, 6.2)]
        phr = [phrase(1.0, 6.2, "你好大家", words)]
        out = label_phrases(phr, segs, ["host", "guest"]).phrases
        assert len(out) == 2
        assert out[0].speaker == "host" and out[0].text == "你好"
        assert out[1].speaker == "guest" and out[1].text == "大家"
        assert out[0].end <= out[1].start
        # text conservation
        assert "".join(p.text for p in out) == "你好大家"

    def test_blip_does_not_split(self):
        # host speaks throughout; diarizer flips to guest for 0.2s (backchannel)
        segs = [
            seg(0, 3.0, "speaker_0"),
            seg(3.0, 3.2, "speaker_1"),
            seg(3.2, 12, "speaker_0"),
            seg(20, 30, "speaker_1"),
        ]
        # keep speaker_1 above the noise floor: 10s of 30s total
        words = [word("a", 1.0, 1.4), word("b", 2.0, 2.4), word("c", 4.0, 4.4), word("d", 5.0, 5.4)]
        phr = [phrase(1.0, 5.4, "abcd", words)]
        out = label_phrases(phr, segs, ["host", "guest"]).phrases
        assert len(out) == 1
        assert out[0].speaker == "host"
        assert out[0].text == "abcd"

    def test_noise_speaker_dropped(self):
        # speaker_2 has 0.05s of 20s total (< 1%) -> dropped
        segs = [seg(0, 10, "speaker_0"), seg(10, 10.05, "speaker_2"), seg(10.05, 20, "speaker_1")]
        phr = [phrase(9.9, 10.2, "x")]
        result = label_phrases(phr, segs, [])
        assert "speaker_2" not in result.assignment.mapping
        # the 9.9-10.2s phrase overlaps spk_0 by 0.10s and spk_1 by 0.15s,
        # so max-overlap labelling must pick spk_1
        assert result.phrases[0].speaker == "spk_1"

    def test_no_words_fallback_no_split(self):
        # spanning a real handoff but no word timings -> whole-phrase label
        # (speaker_0 longer -> "host"; phrase overlaps host 4.0s, guest 2.9s)
        segs = [seg(0, 5.0, "speaker_0"), seg(5.1, 8.0, "speaker_1")]
        phr = [phrase(1.0, 8.0, "abcdef")]
        out = label_phrases(phr, segs, ["host", "guest"]).phrases
        assert len(out) == 1
        assert out[0].speaker == "host"

    def test_split_multiple_handoffs(self):
        # host 0-4, guest 4.1-8, host 8.1-12: two real handoffs in one phrase
        segs = [seg(0, 4.0, "speaker_0"), seg(4.1, 8.0, "speaker_1"), seg(8.1, 20, "speaker_0")]
        # need guest >= noise floor: 4s of 20s = 20% ok
        words = [word("a", 1.0, 1.4), word("b", 5.0, 5.4), word("c", 6.0, 6.4), word("d", 9.0, 9.4)]
        phr = [phrase(1.0, 9.4, "abcd", words)]
        out = label_phrases(phr, segs, ["host", "guest"]).phrases
        assert [p.speaker for p in out] == ["host", "guest", "host"]
        assert [p.text for p in out] == ["a", "bc", "d"]

    def test_noise_floor_constant_sane(self):
        assert 0 < MIN_SPEECH_FRAC < 0.2


class TestMergeOrphans:
    def test_merges_into_previous_same_speaker(self):
        from podcast_dub.stages.diarize import merge_orphans

        out = [
            SpeakerPhrase(
                start=10.0,
                end=12.0,
                text="是更重要",
                speaker="a",
                words=(word("是", 10.0, 10.4), word("更", 10.5, 11.0)),
            ),
            SpeakerPhrase(start=12.3, end=12.3, text="的", speaker="a", words=(word("的", 12.3, 12.3),)),
        ]
        m = merge_orphans(out)
        assert len(m) == 1
        assert m[0].text == "是更重要的"
        assert m[0].end == 12.3
        assert len(m[0].words) == 3

    def test_merges_into_next_when_no_previous_match(self):
        from podcast_dub.stages.diarize import merge_orphans

        out = [
            SpeakerPhrase(start=1.0, end=1.2, text="嗯", speaker="a", words=(word("嗯", 1.0, 1.2),)),
            SpeakerPhrase(start=1.5, end=3.0, text="对对对", speaker="a", words=(word("对", 1.5, 1.9),)),
        ]
        m = merge_orphans(out)
        assert len(m) == 1
        assert m[0].text == "嗯对对对"
        assert m[0].start == 1.0

    def test_isolated_fragment_between_speakers_stays(self):
        from podcast_dub.stages.diarize import merge_orphans

        out = [
            SpeakerPhrase(start=1.0, end=2.0, text="好的", speaker="a"),
            SpeakerPhrase(start=2.2, end=2.4, text="嗯", speaker="b"),
            SpeakerPhrase(start=2.6, end=4.0, text="再见", speaker="a"),
        ]
        m = merge_orphans(out)
        assert len(m) == 3
        assert m[1].text == "嗯"

    def test_text_conservation(self):
        from podcast_dub.stages.diarize import merge_orphans

        out = [
            SpeakerPhrase(start=1.0, end=2.0, text="完整句子", speaker="a"),
            SpeakerPhrase(start=2.1, end=2.1, text="的", speaker="a"),
            SpeakerPhrase(start=3.0, end=4.0, text="下一句", speaker="a"),
        ]
        m = merge_orphans(out)
        assert "".join(p.text for p in m) == "完整句子的下一句"
