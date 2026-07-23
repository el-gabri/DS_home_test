"""Canonical schema for the raw PIX transaction dataset.

Derived from ``DS_II_home_test_features - Feature description.csv``. Keeping
this list in one place means training and serving can never silently drift
apart on which columns the model expects.
"""

# Columns that identify or label a transaction rather than describe it.
# Never fed to the model.
TARGET_COL = "infraction"
TIMESTAMP_COL = "event_created_at"
MERCHANT_ID_COL = "merchant_code"
NON_FEATURE_COLS = (TARGET_COL, TIMESTAMP_COL, MERCHANT_ID_COL)

# Merchant-level attributes present on every transaction (not aggregates).
MERCHANT_TYPE_COLS = ("is_legal_entity", "is_mei", "is_person")

# Aggregate ("mc_*"/"mcc_*") features materialized ahead of time by whatever
# upstream feature store computes them (lifetime/rolling-window counters,
# sums, decline rates, etc). These are exactly the columns the real-time
# serving path must obtain from a feature store keyed by merchant_code,
# because a caller submitting a PIX event cannot compute them itself.
AGGREGATE_FEATURE_COLS = (
    "mc_weeks_signup",
    "mc_atm_br_tx_out_count_lifetime",
    "mc_credit_transfer_out_br_sum_last_30d",
    "mc_billpay_br_sum_last_15d",
    "mc_billpay_br_count_last_1y",
    "mc_credit_transfer_out_br_sum_last_1y",
    "mc_billpay_br_count_sanction_last_30d",
    "mc_pur_br_count_last_1y",
    "mc_billpay_br_count_last_15d",
    "mc_atm_br_tx_out_count_last_1y",
    "mc_sanction_count_last_2d",
    "mc_credit_transfer_in_rounded_amounts_br_sum_last_30d",
    "mc_pur_br_sum_last_15d",
    "mc_credit_transfer_in_br_sum_last_180d",
    "mc_credit_transfer_out_br_count_lifetime",
    "mc_credit_transfer_out_br_count_sanction_last_30d",
    "mc_credit_transfer_out_br_count_last_15d",
    "mc_pur_br_count_last_15d",
    "mc_pep_count_last_2d",
    "mc_credit_transfer_in_br_sum_last_120d",
    "mc_pur_br_count_lifetime",
    "mc_credit_transfer_in_br_count_last_30d",
    "mc_credit_transfer_in_br_sum_last_90d",
    "mc_credit_transfer_out_br_count_last_1y",
    "mc_billpay_br_sum_last_1y",
    "mc_billpay_br_count_pep_br_last_30d",
    "mc_credit_transfer_in_br_count_pep_br_last_30d",
    "mc_credit_transfer_out_rounded_amounts_br_sum_last_30d",
    "mc_credit_transfer_out_br_sum_last_15d",
    "mc_credit_transfer_in_br_count_sanction_last_30d",
    "mc_credit_transfer_in_br_sum_last_15d",
    "mc_billpay_br_count_lifetime",
    "mc_atm_br_tx_out_count_last_15d",
    "mc_credit_transfer_out_rounded_amounts_br_count_last_30d",
    "mc_credit_transfer_in_br_sum_last_60d",
    "mc_pur_br_sum_last_1y",
    "mc_credit_transfer_out_br_count_pep_br_last_30d",
    "mc_chip_tx_succ_count_30d",
    "mc_chip_tx_decl_ext_count_7d",
    "mc_chip_tx_amount_local_decl_ext_sum_15d",
    "mc_chip_tx_amount_succ_sum_7d",
    "mc_chip_tx_amount_local_succ_sum_7d",
    "mc_chip_tx_amount_local_succ_sum_15d",
    "mc_chip_tx_amount_decl_ext_sum_30d",
    "mc_chip_tx_amount_succ_sum_30d",
    "mc_chip_tx_succ_count_7d",
    "decline_ext_rate_30d",
    "mc_tx_cp_mean_decl_ext_last_31d_mw",
    "mc_tx_amount_sum_last_60m",
    "mc_tx_amount_decl_ext_sum_lifetime",
    "mc_tx_amount_succ_sum_last_30d",
    "mc_tx_succ_count_last_24h",
    "mc_tx_amount_succ_sum_last_14d",
    "mc_tx_amount_fail_sum_last_60m",
    "mc_tx_amount_succ_sum_last_60m",
    "mc_tx_amount_succ_sum_last_2d",
    "mc_tx_amount_succ_sum_last_90d",
    "mc_tx_amount_fail_sum_last_14d",
    "mc_tx_amount_succ_sum_lifetime",
    "mc_tx_amount_pend_sum_last_2d",
    "mc_tx_amount_decl_int_sum_lifetime",
    "mc_tx_amount_sum_last_24h",
    "mc_tx_amount_sum_lifetime",
    "mc_tx_succ_count_last_14d",
    "mc_tx_succ_count_lifetime",
    "mc_tx_amount_succ_sum_last_24h",
    "mc_tx_amount_pend_sum_last_14d",
    "mc_tx_amount_decl_ext_sum_last_14d",
    "mc_tx_amount_decl_ext_sum_last_24h",
    "mc_tx_amount_decl_ext_sum_last_60m",
    "mc_tx_succ_count_last_2d",
    "mc_tx_amount_fail_sum_lifetime",
    "mc_tx_succ_count_last_30d",
    "mc_tx_amount_sum_last_2d",
    "mc_tx_succ_count_last_365d",
    "mc_tx_succ_count_last_60m",
    "mc_tx_amount_sum_last_14d",
    "mc_tx_cp_sum_decl_ext_last_24h",
    "mc_tx_cp_dbt_sum_sq_decl_ext_last_30d",
    "mc_tx_count_decl_ext_last_30d",
    "mc_tx_cp_sum_decl_ext_last_30d",
    "mc_tx_cp_count_decl_ext_last_24h",
    "mc_tx_cp_dbt_sum_sq_fail_last_24h",
    "mc_tx_cp_count_decl_ext_last_30d",
    "mc_tx_cp_dbt_sum_sq_fail_last_14d",
    "mc_merchants_ip_count_dist_shared_last_15d",
    "mcc_sum_succ_1y",
    "mcc_sum_sq_succ_30d",
    "mcc_count_succ_1y",
    "mcc_sum_sq_succ_1y",
    "mcc_sum_succ_30d",
)

# Present directly on the transaction event (known at the moment the PIX is
# received, no feature-store lookup required).
TRANSACTION_FEATURE_COLS = ("amount",)

# Merchant-level attributes (type + aggregates) that a real-time caller
# cannot compute itself; the serving API resolves these from a feature store
# keyed by merchant_code (see app/feature_store.py).
MERCHANT_PROFILE_COLS = MERCHANT_TYPE_COLS + AGGREGATE_FEATURE_COLS

# Derived purely from the transaction timestamp; computed identically in
# training and serving by fraud_detection.features.add_temporal_features.
TEMPORAL_FEATURE_COLS = ("is_weekend", "is_night", "month", "day", "hour")

# Full ordered feature set consumed by the model.
RAW_FEATURE_COLUMNS = TRANSACTION_FEATURE_COLS + MERCHANT_PROFILE_COLS
MODEL_FEATURE_COLUMNS = RAW_FEATURE_COLUMNS + TEMPORAL_FEATURE_COLS
