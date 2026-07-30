from core.codec_video_reader import _select_best_effort_ip


def test_best_effort_sampling_uses_positive_stream_origin():
    records = [
        {
            "time_seconds": 0.04,
            "pict_type": "I",
            "source_index": 0,
            "selected": False,
        },
        {
            "time_seconds": 0.44,
            "pict_type": "P",
            "source_index": 11,
            "selected": False,
        },
        {
            "time_seconds": 0.92,
            "pict_type": "P",
            "source_index": 23,
            "selected": False,
        },
        {
            "time_seconds": 1.40,
            "pict_type": "P",
            "source_index": 35,
            "selected": False,
        },
    ]

    _select_best_effort_ip(records, target_fps=2.0, target_count=4)

    assert [
        record["source_index"] for record in records if record["selected"]
    ] == [0, 11, 23, 35]
