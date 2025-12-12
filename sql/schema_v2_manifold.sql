-- Hartonomous Schema v2.0: Manifold-Based Architecture
-- Implements the complete vision:
-- - Embeddings ARE the manifold (semantic coordinates)
-- - 4-manifold Hilbert encoding (deterministic content-addressing)
-- - Meta-learning weight consolidation
-- - Zero-loss primitive storage

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Atoms: Everything decomposes to this
CREATE TABLE IF NOT EXISTS atoms (
    -- 4-manifold Hilbert PRIMARY KEY (168 bits total)
    -- This IS the content-derived deterministic ID
    h_xy BIGINT NOT NULL,  -- Spatial × Entropy manifold (42 bits)
    h_yz BIGINT NOT NULL,  -- Entropy × Compressibility manifold (42 bits)
    h_zm BIGINT NOT NULL,  -- Compressibility × Connectivity manifold (42 bits)
    h_my BIGINT NOT NULL,  -- Connectivity × Spatial manifold (42 bits, closes cycle)
    PRIMARY KEY (h_xy, h_yz, h_zm, h_my),

    -- Source of truth (zero loss)
    raw_value BYTEA,

    -- Spatial indexing geometry
    -- For tokens: coordinates FROM embeddings (semantic manifold from training)
    -- For floats: coordinates from value properties (entropy, compressibility)
    -- SRID 0 = arbitrary Cartesian 4D space
    geom GEOMETRY(PointZM, 0) NOT NULL,

    atom_type TEXT NOT NULL CHECK (atom_type IN ('byte', 'float', 'token', 'embedding_vector')),
    ref_count BIGINT DEFAULT 1,

    meta JSONB
) PARTITION BY RANGE (h_xy);

-- Create 256 partitions for horizontal scalability
DO $$
DECLARE
    partition_size CONSTANT BIGINT := 17179869184;  -- 2^42 / 256 (42-bit Hilbert values)
    i INT;
    start_val BIGINT;
    end_val BIGINT;
    partition_name TEXT;
BEGIN
    FOR i IN 0..255 LOOP
        start_val := i::BIGINT * partition_size;
        end_val := (i::BIGINT + 1) * partition_size;
        partition_name := 'atoms_p' || LPAD(i::TEXT, 3, '0');

        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS %I PARTITION OF atoms FOR VALUES FROM (%L) TO (%L)',
            partition_name, start_val, end_val
        );

        -- GIST spatial index per partition (for k-NN)
        EXECUTE format(
            'CREATE INDEX IF NOT EXISTS idx_%I_spatial ON %I USING GIST(geom)',
            partition_name, partition_name
        );

        -- Type index
        EXECUTE format(
            'CREATE INDEX IF NOT EXISTS idx_%I_type ON %I (atom_type)',
            partition_name, partition_name
        );
    END LOOP;
END $$;

-- Relations: Meta-learning weight consolidation
CREATE TABLE IF NOT EXISTS relations (
    -- Source atom (e.g., input neuron)
    parent_h_xy BIGINT,
    parent_h_yz BIGINT,
    parent_h_zm BIGINT,
    parent_h_my BIGINT,

    -- Destination atom (e.g., output neuron)
    child_h_xy BIGINT,
    child_h_yz BIGINT,
    child_h_zm BIGINT,
    child_h_my BIGINT,

    relation_type TEXT NOT NULL,  -- 'embedding_dim', 'q_proj_weight', 'k_proj_weight', etc.
    position INT NOT NULL DEFAULT 0,
    layer_name TEXT NOT NULL,

    -- Meta-learning statistics (Welford's algorithm)
    weight_mean_h BIGINT[4],  -- Hilbert ID of mean weight float atom
    weight_count INT DEFAULT 1,
    weight_m2 DOUBLE PRECISION DEFAULT 0,  -- Sum of squared deviations
    weight_min DOUBLE PRECISION,
    weight_max DOUBLE PRECISION,

    -- Model attribution
    model_sources TEXT[],

    PRIMARY KEY (parent_h_xy, parent_h_yz, parent_h_zm, parent_h_my,
                 child_h_xy, child_h_yz, child_h_zm, child_h_my,
                 relation_type, layer_name, position)
);

CREATE INDEX idx_relations_parent ON relations(parent_h_xy, parent_h_yz, parent_h_zm, parent_h_my);
CREATE INDEX idx_relations_child ON relations(child_h_xy, child_h_yz, child_h_zm, child_h_my);
CREATE INDEX idx_relations_layer ON relations(layer_name, relation_type);

-- Helper function: Get float value from Hilbert ID
CREATE OR REPLACE FUNCTION get_float_value(h BIGINT[4])
RETURNS DOUBLE PRECISION AS $$
BEGIN
    RETURN (
        SELECT ST_X(geom)  -- X coordinate = the float value itself
        FROM atoms
        WHERE h_xy = h[1] AND h_yz = h[2] AND h_zm = h[3] AND h_my = h[4]
        LIMIT 1
    );
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Helper function: Get mean from weight relation
CREATE OR REPLACE FUNCTION get_mean_from_hilbert(h BIGINT[4])
RETURNS DOUBLE PRECISION AS $$
BEGIN
    RETURN COALESCE(get_float_value(h), 0.0);
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Semantic similarity search function
CREATE OR REPLACE FUNCTION find_similar_tokens(
    query_token TEXT,
    k INTEGER DEFAULT 10
)
RETURNS TABLE(
    token TEXT,
    distance DOUBLE PRECISION,
    ref_count BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        encode(a2.raw_value, 'escape') as token,
        ST_Distance(a1.geom, a2.geom)::DOUBLE PRECISION as distance,
        a2.ref_count
    FROM atoms a1, atoms a2
    WHERE a1.raw_value = query_token::bytea
      AND a1.atom_type = 'token'
      AND a2.atom_type = 'token'
      AND (a2.h_xy, a2.h_yz, a2.h_zm, a2.h_my) != (a1.h_xy, a1.h_yz, a1.h_zm, a1.h_my)
    ORDER BY a1.geom <-> a2.geom
    LIMIT k;
END;
$$ LANGUAGE plpgsql;

-- Statistics view
CREATE OR REPLACE VIEW atom_stats AS
SELECT
    atom_type,
    COUNT(*) as count,
    SUM(ref_count) as total_refs,
    AVG(ref_count) as avg_refs,
    MIN(ref_count) as min_refs,
    MAX(ref_count) as max_refs
FROM atoms
GROUP BY atom_type;

-- Weight statistics view
CREATE OR REPLACE VIEW weight_stats AS
SELECT
    layer_name,
    relation_type,
    COUNT(*) as connection_count,
    AVG(weight_count) as avg_models_per_connection,
    AVG(SQRT(weight_m2 / GREATEST(weight_count - 1, 1))) as avg_std_dev,
    COUNT(DISTINCT model_sources[1]) as unique_models
FROM relations
GROUP BY layer_name, relation_type
ORDER BY layer_name, relation_type;

COMMENT ON TABLE atoms IS 'Universal atomic data store - embeddings define semantic manifold';
COMMENT ON COLUMN atoms.h_xy IS '42-bit Hilbert index: Spatial × Entropy';
COMMENT ON COLUMN atoms.h_yz IS '42-bit Hilbert index: Entropy × Compressibility';
COMMENT ON COLUMN atoms.h_zm IS '42-bit Hilbert index: Compressibility × Connectivity';
COMMENT ON COLUMN atoms.h_my IS '42-bit Hilbert index: Connectivity × Spatial (cycle)';
COMMENT ON COLUMN atoms.geom IS 'Semantic coordinates: for tokens, from Laplacian Eigenmaps of embeddings';
COMMENT ON TABLE relations IS 'Neural network connections with meta-learning consolidation';
COMMENT ON COLUMN relations.weight_mean_h IS 'Hilbert ID of consensus weight (averaged across models)';
COMMENT ON COLUMN relations.weight_m2 IS 'Welford M2: sum of squared deviations for variance';
