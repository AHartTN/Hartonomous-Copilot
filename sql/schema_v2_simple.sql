-- Hartonomous Schema v2.0 - Simplified Working Version
-- Focus: Get basic geometric embedding storage and spatial queries working

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Atoms table: Store all atomic data with geometric coordinates
CREATE TABLE atoms (
    -- Primary key: Hilbert curve index for deterministic deduplication
    hilbert_id BIGINT PRIMARY KEY,

    -- 4D geometric coordinates for spatial queries
    -- Dimensions: (x, y, z, m) where:
    --   x: Primary value (embedding component, byte value, etc.)
    --   y: Secondary dimension (embedding dim index, token position, etc.)
    --   z: Tertiary dimension (derived features)
    --   m: Metadata dimension (frequency, layer, etc.)
    geom GEOMETRY(PointZM, 0) NOT NULL,

    -- Raw bytes (for primitive atoms)
    raw_value BYTEA,

    -- Atom type
    atom_type TEXT NOT NULL CHECK (atom_type IN ('byte', 'token', 'float', 'embedding_dim')),

    -- Optional metadata
    meta JSONB,

    -- Audit
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Spatial index for k-NN queries
CREATE INDEX idx_atoms_geom ON atoms USING GIST(geom);

-- Type index
CREATE INDEX idx_atoms_type ON atoms (atom_type);

-- Token embeddings: Map tokens to their embedding vectors (as atom arrays)
CREATE TABLE token_embeddings (
    token_id INTEGER PRIMARY KEY,
    token_text TEXT NOT NULL,

    -- Array of hilbert_ids pointing to embedding dimension atoms
    embedding_atom_ids BIGINT[],

    -- Original embedding vector (for validation)
    embedding_vector FLOAT[],

    -- Metadata
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Relations: Neural network structure
CREATE TABLE relations (
    id BIGSERIAL PRIMARY KEY,
    from_atom_id BIGINT REFERENCES atoms(hilbert_id),
    to_atom_id BIGINT REFERENCES atoms(hilbert_id),
    relation_type TEXT NOT NULL,
    weight_atom_id BIGINT REFERENCES atoms(hilbert_id),
    layer_name TEXT,
    position INTEGER,
    meta JSONB
);

CREATE INDEX idx_relations_from ON relations(from_atom_id);
CREATE INDEX idx_relations_to ON relations(to_atom_id);
CREATE INDEX idx_relations_type ON relations(relation_type);

-- Stats table for monitoring
CREATE TABLE ingestion_stats (
    id SERIAL PRIMARY KEY,
    batch_name TEXT NOT NULL,
    atoms_created INTEGER DEFAULT 0,
    embeddings_created INTEGER DEFAULT 0,
    started_at TIMESTAMPTZ DEFAULT now(),
    completed_at TIMESTAMPTZ,
    status TEXT CHECK (status IN ('running', 'completed', 'failed')),
    error_message TEXT
);

-- Helper function: Find k nearest atoms to a given point
CREATE OR REPLACE FUNCTION find_nearest_atoms(
    target_x FLOAT,
    target_y FLOAT,
    target_z FLOAT,
    target_m FLOAT,
    k INTEGER DEFAULT 10,
    filter_type TEXT DEFAULT NULL
)
RETURNS TABLE(
    hilbert_id BIGINT,
    distance FLOAT,
    atom_type TEXT,
    raw_value BYTEA
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        a.hilbert_id,
        ST_Distance(a.geom, ST_MakePointZM(target_x, target_y, target_z, target_m))::FLOAT as distance,
        a.atom_type,
        a.raw_value
    FROM atoms a
    WHERE (filter_type IS NULL OR a.atom_type = filter_type)
    ORDER BY a.geom <-> ST_MakePointZM(target_x, target_y, target_z, target_m)
    LIMIT k;
END;
$$ LANGUAGE plpgsql;

COMMENT ON TABLE atoms IS 'Atomic data store with geometric spatial indexing';
COMMENT ON COLUMN atoms.hilbert_id IS 'Hilbert curve index for deterministic ID generation';
COMMENT ON COLUMN atoms.geom IS '4D point (PointZM) for spatial k-NN queries';
COMMENT ON TABLE token_embeddings IS 'Token vocabulary with embedding vector references';
