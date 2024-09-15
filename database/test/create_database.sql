CREATE TABLE IF NOT EXISTS batches
(
    id              INT UNSIGNED NOT NULL PRIMARY KEY ASC,
    laser_settings  INT          NULL,
    specimen        INT          NULL,
    fixture         INT          NULL,
    Al_batch        INT          NULL,
    polymer_lower   VARCHAR(255) NULL,
    polymer_upper   VARCHAR(255) NULL,
    Al_mass_mg      FLOAT        NULL,
    production_date DATE         NULL
);

CREATE TABLE IF NOT EXISTS laser_settings
(
    id        INT UNSIGNED NOT NULL PRIMARY KEY ASC,
    pattern   INT          NULL,
    file_path TEXT         NULL,
    distance  FLOAT        NULL,
    speed     FLOAT        NULL,
    power     INT          NULL
);

CREATE TABLE IF NOT EXISTS patterns
(
    id          INT UNSIGNED NOT NULL PRIMARY KEY ASC,
    file_path   TEXT         NULL,
    description TEXT         NULL
);
CREATE TABLE IF NOT EXISTS tests
(
    specimen_id               VARCHAR(255) NOT NULL PRIMARY KEY,
    batch_id                  INT          NULL,
    file_path                 TEXT         NULL,
    ultimate_tensile_strength FLOAT        NULL,
    strain_at_break           FLOAT        NULL,
    test_date                 DATE         NULL
);
CREATE TABLE IF NOT EXISTS raw_Al_batches
(
    id            INT UNSIGNED NOT NULL PRIMARY KEY ASC,
    diameter_mean FLOAT        NULL,
    diameter_std  FLOAT        NULL,
    aspect_ratio  FLOAT        NULL,
    supplier      TEXT         NULL,
    description   TEXT         NULL,
    LM_images     TEXT         NULL,
    SEM_images    TEXT         NULL
);
CREATE TABLE IF NOT EXISTS specimen_designs
(
    id        INT UNSIGNED NOT NULL PRIMARY KEY ASC,
    file_path TEXT         NULL,
    version   INT          NULL
);
CREATE TABLE IF NOT EXISTS fixture_version
(
    id        INT UNSIGNED NOT NULL PRIMARY KEY ASC,
    version   TEXT         NOT NULL,
    file_path TEXT         NOT NULL
);
CREATE TABLE IF NOT EXISTS structured_Al_batches
(
    id              INT UNSIGNED NOT NULL PRIMARY KEY ASC,
    raw_batch       INT          NULL,
    process         INT          NULL,
    production_date DATE         NULL,
    diameter_mean   FLOAT        NULL,
    diameter_std    FLOAT        NULL,
    start_weight    FLOAT        NULL,
    end_weight      FLOAT        NULL,
    LM_images       TEXT         NULL,
    SEM_images      TEXT         NULL
);
CREATE TABLE IF NOT EXISTS uCT_images
(
    id            INT UNSIGNED NOT NULL PRIMARY KEY ASC,
    test          TEXT         NULL,
    path          TEXT         NULL,
    voxel_size_um FLOAT        NULL
);
CREATE TABLE IF NOT EXISTS etch_processes
(
    id          INT UNSIGNED NOT NULL PRIMARY KEY ASC,
    high_T      INT          NULL,
    high_T_time INT          NULL,
    low_T       INT          NULL,
    low_T_time  INT          NULL
);