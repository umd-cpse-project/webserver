CREATE TABLE IF NOT EXISTS categories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    context TEXT NOT NULL DEFAULT '',
    x SMALLINT NOT NULL DEFAULT 0,
    y SMALLINT NOT NULL DEFAULT 0,
    width SMALLINT NOT NULL DEFAULT 0,
    height SMALLINT NOT NULL DEFAULT 0,
    hue INTEGER NOT NULL DEFAULT 0
);
