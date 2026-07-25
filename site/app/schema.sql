-- Horse-race site schema. Idempotent: safe to execute at every startup.

CREATE TABLE IF NOT EXISTS question_sets (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    topic        TEXT NOT NULL,
    created_at   TEXT NOT NULL,              -- UTC ISO-8601
    from_date    TEXT NOT NULL,              -- Guardian collection window
    to_date      TEXT NOT NULL,
    n_articles   INTEGER,
    n_generated  INTEGER,
    n_passing    INTEGER,
    seed         INTEGER,
    is_active    INTEGER NOT NULL DEFAULT 0
);
-- Exactly one active set at a time.
CREATE UNIQUE INDEX IF NOT EXISTS one_active_set
    ON question_sets (is_active) WHERE is_active = 1;

CREATE TABLE IF NOT EXISTS questions (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    set_id             INTEGER NOT NULL REFERENCES question_sets(id),
    position           INTEGER NOT NULL,     -- 1..N quiz order
    source_question_id TEXT NOT NULL,        -- toolkit question id
    article_id         TEXT NOT NULL,
    article_url        TEXT,
    headline           TEXT,
    published          TEXT,
    question           TEXT NOT NULL,
    options_json       TEXT NOT NULL,        -- JSON array of 4 strings
    correct_letter     TEXT NOT NULL CHECK (correct_letter IN ('A','B','C','D')),
    explanation        TEXT,
    generator_model    TEXT,
    n_models_passing   INTEGER,
    UNIQUE (set_id, position)
);

CREATE TABLE IF NOT EXISTS sessions (
    id           TEXT PRIMARY KEY,           -- uuid4
    set_id       INTEGER NOT NULL REFERENCES question_sets(id),
    started_at   TEXT NOT NULL,
    completed_at TEXT,
    user_agent   TEXT
);

-- One row per "question shown to a human" — the unit AI runs attach to.
-- The UNIQUE constraint is the double-spend guard on page refresh.
CREATE TABLE IF NOT EXISTS presentations (
    id           TEXT PRIMARY KEY,           -- uuid4
    session_id   TEXT NOT NULL REFERENCES sessions(id),
    question_id  INTEGER NOT NULL REFERENCES questions(id),
    presented_at TEXT NOT NULL,
    UNIQUE (session_id, question_id)
);

CREATE TABLE IF NOT EXISTS human_answers (
    presentation_id TEXT PRIMARY KEY REFERENCES presentations(id),
    answer_letter   TEXT NOT NULL CHECK (answer_letter IN ('A','B','C','D')),
    is_correct      INTEGER NOT NULL,
    confidence      INTEGER NOT NULL CHECK (confidence BETWEEN 0 AND 100),
    started_at      TEXT NOT NULL,           -- = presentations.presented_at
    ended_at        TEXT NOT NULL,
    elapsed_ms      INTEGER NOT NULL         -- server-computed
);

CREATE TABLE IF NOT EXISTS ai_answers (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    presentation_id TEXT NOT NULL REFERENCES presentations(id),
    model           TEXT NOT NULL,
    provider        TEXT NOT NULL,
    status          TEXT NOT NULL CHECK (status IN ('pending','ok','error')),
    answer_letter   TEXT CHECK (answer_letter IN ('A','B','C','D')),
    is_correct      INTEGER,
    confidence      INTEGER CHECK (confidence BETWEEN 0 AND 100),
    reasoning       TEXT,
    search_used     INTEGER,
    started_at      TEXT,
    ended_at        TEXT,
    elapsed_ms      INTEGER,
    error           TEXT,
    UNIQUE (presentation_id, model)
);
CREATE INDEX IF NOT EXISTS ai_answers_presentation
    ON ai_answers (presentation_id);
