import pandas as pd

from fraud_detection.features import add_temporal_features


def test_weekend_flag_uses_transaction_timestamp_not_wall_clock():
    df = pd.DataFrame(
        {
            "event_created_at": [
                "2026-07-18T10:00:00",  # Saturday
                "2026-07-20T10:00:00",  # Monday
            ]
        }
    )
    out = add_temporal_features(df)
    assert out["is_weekend"].tolist() == [1, 0]


def test_night_flag_boundaries():
    df = pd.DataFrame(
        {
            "event_created_at": [
                "2026-07-20T05:59:00",  # night
                "2026-07-20T06:00:00",  # day
                "2026-07-20T17:59:00",  # day
                "2026-07-20T18:00:00",  # night
            ]
        }
    )
    out = add_temporal_features(df)
    assert out["is_night"].tolist() == [1, 0, 0, 1]


def test_does_not_mutate_input():
    df = pd.DataFrame({"event_created_at": ["2026-07-20T10:00:00"]})
    original_columns = list(df.columns)
    add_temporal_features(df)
    assert list(df.columns) == original_columns
