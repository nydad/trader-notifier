-- Migration 0001: Initial schema
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;
BEGIN;

CREATE TABLE IF NOT EXISTS fetch_sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_key TEXT NOT NULL UNIQUE,
    source_name TEXT NOT NULL,
    base_url TEXT,
    cadence TEXT NOT NULL DEFAULT 'daily',
    timezone TEXT NOT NULL DEFAULT 'Asia/Seoul'
);

CREATE TABLE IF NOT EXISTS watchlist_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    source_file TEXT NOT NULL,
    loaded_at TEXT NOT NULL DEFAULT (datetime('now')),
    payload_hash TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS watchlist_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    display_name TEXT NOT NULL,
    category TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (snapshot_id) REFERENCES watchlist_snapshots(id) ON DELETE CASCADE,
    UNIQUE (snapshot_id, symbol)
);

CREATE TABLE IF NOT EXISTS series_catalog (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    symbol TEXT NOT NULL,
    display_name TEXT NOT NULL,
    source_id INTEGER NOT NULL,
    source_symbol TEXT NOT NULL,
    frequency TEXT NOT NULL DEFAULT 'daily',
    tz TEXT NOT NULL DEFAULT 'Asia/Seoul',
    unit TEXT,
    lag_days INTEGER NOT NULL DEFAULT 0,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (source_id) REFERENCES fetch_sources(id),
    UNIQUE (kind, symbol, source_id)
);

CREATE TABLE IF NOT EXISTS etf_price_points (
    series_id INTEGER NOT NULL,
    as_of_date TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL NOT NULL,
    adj_close REAL, volume REAL,
    inserted_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (series_id, as_of_date),
    FOREIGN KEY (series_id) REFERENCES series_catalog(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS indicator_points (
    series_id INTEGER NOT NULL,
    as_of_date TEXT NOT NULL,
    value REAL NOT NULL,
    inserted_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (series_id, as_of_date),
    FOREIGN KEY (series_id) REFERENCES series_catalog(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS series_fingerprint (
    series_id INTEGER PRIMARY KEY,
    row_count INTEGER NOT NULL,
    earliest_date TEXT NOT NULL,
    latest_date TEXT NOT NULL,
    value_hash TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (series_id) REFERENCES series_catalog(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS correlation_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cache_key TEXT NOT NULL UNIQUE,
    start_date TEXT NOT NULL, end_date TEXT NOT NULL,
    methods_json TEXT NOT NULL, windows_json TEXT,
    max_lag INTEGER, partial_controls_json TEXT,
    min_periods INTEGER NOT NULL DEFAULT 20,
    status TEXT NOT NULL DEFAULT 'complete',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    expires_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS correlation_pairs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    metric TEXT NOT NULL,
    window INTEGER, lag INTEGER NOT NULL DEFAULT 0,
    series_a_id INTEGER NOT NULL, series_b_id INTEGER NOT NULL,
    correlation REAL NOT NULL, p_value REAL, n_obs INTEGER,
    computed_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (run_id) REFERENCES correlation_runs(id) ON DELETE CASCADE,
    FOREIGN KEY (series_a_id) REFERENCES series_catalog(id),
    FOREIGN KEY (series_b_id) REFERENCES series_catalog(id)
);

CREATE TABLE IF NOT EXISTS correlation_run_versions (
    run_id INTEGER NOT NULL, series_id INTEGER NOT NULL,
    row_count INTEGER NOT NULL, latest_date TEXT NOT NULL, value_hash TEXT NOT NULL,
    PRIMARY KEY (run_id, series_id),
    FOREIGN KEY (run_id) REFERENCES correlation_runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS correlation_rankings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL, metric TEXT NOT NULL,
    rank_type TEXT NOT NULL,
    ranking_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (run_id) REFERENCES correlation_runs(id) ON DELETE CASCADE,
    UNIQUE (run_id, metric, rank_type)
);

CREATE TABLE IF NOT EXISTS backtest_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    label TEXT NOT NULL,
    watchlist_snapshot_id INTEGER,
    start_date TEXT NOT NULL, end_date TEXT NOT NULL,
    initial_capital REAL NOT NULL, params_json TEXT NOT NULL,
    correlation_run_id INTEGER,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (watchlist_snapshot_id) REFERENCES watchlist_snapshots(id),
    FOREIGN KEY (correlation_run_id) REFERENCES correlation_runs(id)
);

CREATE TABLE IF NOT EXISTS signal_combinations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    backtest_run_id INTEGER NOT NULL,
    combo_key TEXT NOT NULL, combo_json TEXT NOT NULL,
    FOREIGN KEY (backtest_run_id) REFERENCES backtest_runs(id) ON DELETE CASCADE,
    UNIQUE (backtest_run_id, combo_key)
);

CREATE TABLE IF NOT EXISTS backtest_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    backtest_run_id INTEGER NOT NULL, combo_id INTEGER NOT NULL,
    series_symbol TEXT NOT NULL,
    direction INTEGER NOT NULL,
    entry_date TEXT NOT NULL, entry_price REAL NOT NULL,
    exit_date TEXT, exit_price REAL,
    qty REAL NOT NULL, pnl REAL, return_pct REAL,
    holding_days INTEGER, exit_reason TEXT,
    FOREIGN KEY (backtest_run_id) REFERENCES backtest_runs(id) ON DELETE CASCADE,
    FOREIGN KEY (combo_id) REFERENCES signal_combinations(id)
);

CREATE TABLE IF NOT EXISTS backtest_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    backtest_run_id INTEGER NOT NULL, combo_id INTEGER NOT NULL,
    metric_name TEXT NOT NULL, metric_value REAL NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (backtest_run_id) REFERENCES backtest_runs(id) ON DELETE CASCADE,
    FOREIGN KEY (combo_id) REFERENCES signal_combinations(id),
    UNIQUE (backtest_run_id, combo_id, metric_name)
);

CREATE TABLE IF NOT EXISTS fetch_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    series_id INTEGER NOT NULL,
    requested_at TEXT NOT NULL DEFAULT (datetime('now')),
    start_date TEXT NOT NULL, end_date TEXT NOT NULL,
    status TEXT NOT NULL,
    rows_written INTEGER NOT NULL DEFAULT 0,
    latency_ms INTEGER, error_text TEXT,
    FOREIGN KEY (series_id) REFERENCES series_catalog(id)
);

CREATE TABLE IF NOT EXISTS viz_artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    target_type TEXT NOT NULL, target_id INTEGER NOT NULL,
    artifact_type TEXT NOT NULL, file_path TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_etf_price_sd ON etf_price_points(series_id, as_of_date);
CREATE INDEX IF NOT EXISTS idx_indicator_sd ON indicator_points(series_id, as_of_date);
CREATE INDEX IF NOT EXISTS idx_catalog_kind ON series_catalog(kind, is_active);
CREATE INDEX IF NOT EXISTS idx_corr_pairs ON correlation_pairs(run_id, metric, series_a_id, series_b_id);
CREATE INDEX IF NOT EXISTS idx_bt_trades ON backtest_trades(backtest_run_id, combo_id);
CREATE INDEX IF NOT EXISTS idx_fetch_logs ON fetch_logs(series_id, status);

COMMIT;
