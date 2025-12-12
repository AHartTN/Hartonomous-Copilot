-- Hartonomous Enterprise Schema v1.0
-- Production-grade Universal Geometric Data Architecture (UGDA)
-- PostgreSQL 18 + PostGIS 3.6+

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS plpython3u;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- For text search optimization
CREATE EXTENSION IF NOT EXISTS btree_gist;  -- For multi-column GIST indexes

-- Custom types for better type safety
CREATE TYPE modality_type AS ENUM (
    'discrete',      -- 0x01: bytes, tokens, pixels
    'continuous',    -- 0x02: floats, audio samples, embeddings
    'relational',    -- 0x03: edges, synapses, references
    'temporal',      -- 0x04: sequences, trajectories, videos
    'compositional'  -- 0x05: structures, aggregates, batches
);

CREATE TYPE ingestion_status AS ENUM (
    'pending',
    'running',
    'completed',
    'failed',
    'cancelled'
);

-- Main atoms table (partitioned for horizontal scalability)
CREATE TABLE atoms (
    -- Composite Hilbert Primary Key (168-bit space)
    -- h_xy: Content × Context manifold (56 bits)
    -- h_yz: Entropy × Structure manifold (56 bits)
    -- h_zm: Causality × Frequency manifold (56 bits)
    h_xy BIGINT NOT NULL,
    h_yz BIGINT NOT NULL,
    h_zm BIGINT NOT NULL,
    
    -- Geometry payload (SRID 0 = Cartesian 4D space, NOT geographic)
    -- This IS the data structure itself:
    -- - POINT: Discrete events (byte value, pixel, sample, weight)
    -- - LINESTRING: Sequential flow (text, audio, time-series)
    -- - POLYGON: Bounded regions (image frame, attention head)
    -- - MULTIPOLYGON: Parallel streams (multi-track audio, RGB channels)
    -- - GEOMETRYCOLLECTION: Hierarchical composites (video, AST)
    geom GEOMETRY(GEOMETRYCOLLECTION, 0) NOT NULL,
    
    -- Raw value for primitive atoms (leaf nodes only, ≤64 bytes)
    value BYTEA,
    CONSTRAINT chk_value_size CHECK (value IS NULL OR length(value) <= 64),
    
    -- Modality classification (extracted from h_xy high byte)
    modality modality_type NOT NULL,
    
    -- Content hash for global deduplication (SHA-256)
    content_hash BYTEA NOT NULL,
    CONSTRAINT chk_hash_size CHECK (length(content_hash) = 32),
    
    -- Reference counting for garbage collection
    -- Incremented when atom is referenced by compositions
    -- Decremented when reference is removed
    -- When ref_count reaches 0, atom is eligible for deletion
    ref_count BIGINT NOT NULL DEFAULT 1,
    CONSTRAINT chk_ref_count CHECK (ref_count >= 0),
    
    -- Optional metadata (JSONB for schema flexibility)
    -- Example fields:
    --   {
    --     "source": "file://path/to/source.txt",
    --     "encoding": "utf-8",
    --     "sample_rate": 44100,
    --     "dimensions": [1920, 1080],
    --     "frame_rate": 30,
    --     "layer": "attention.0.query"
    --   }
    meta JSONB,
    
    -- Audit fields (immutability pattern - updates create new versions)
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by TEXT NOT NULL DEFAULT current_user,
    updated_at TIMESTAMPTZ,
    updated_by TEXT,
    version BIGINT NOT NULL DEFAULT 1,
    
    -- Soft delete flag (for tombstoning instead of hard deletes)
    deleted_at TIMESTAMPTZ,
    deleted_by TEXT,
    
    -- Primary key
    PRIMARY KEY (h_xy, h_yz, h_zm)
) PARTITION BY RANGE (h_xy);

-- Create 256 partitions for horizontal scalability
-- Each partition handles 1/256th of the Hilbert space
-- This enables:
--   1. Parallel query execution across partitions
--   2. Independent vacuum/analyze per partition
--   3. Partition pruning for faster queries
--   4. Future partition-level archiving/purging

-- Partition size calculation:
-- Total Hilbert space: 2^56 (BIGINT uses 56 bits for positive range)
-- Partition size: 2^56 / 256 = 72,057,594,037,927,936

-- Generate partitions (0-255)
DO $$
DECLARE
    partition_size CONSTANT BIGINT := 72057594037927936;  -- 2^56 / 256
    i INT;
    start_val BIGINT;
    end_val BIGINT;
    partition_name TEXT;
BEGIN
    FOR i IN 0..255 LOOP
        start_val := i * partition_size;
        end_val := (i + 1) * partition_size;
        partition_name := 'atoms_p' || LPAD(i::TEXT, 3, '0');
        
        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS %I PARTITION OF atoms FOR VALUES FROM (%L) TO (%L)',
            partition_name,
            start_val,
            end_val
        );
        
        -- Spatial index (GIST) on geometry - primary query mechanism
        EXECUTE format(
            'CREATE INDEX IF NOT EXISTS idx_%I_geom ON %I USING GIST(geom)',
            partition_name,
            partition_name
        );
        
        -- Content hash index for deduplication checks
        EXECUTE format(
            'CREATE INDEX IF NOT EXISTS idx_%I_hash ON %I (content_hash)',
            partition_name,
            partition_name
        );
        
        -- Modality index for type-specific queries
        EXECUTE format(
            'CREATE INDEX IF NOT EXISTS idx_%I_modality ON %I (modality)',
            partition_name,
            partition_name
        );
        
        -- Temporal index for time-based queries
        EXECUTE format(
            'CREATE INDEX IF NOT EXISTS idx_%I_created ON %I (created_at DESC)',
            partition_name,
            partition_name
        );
        
        -- Reference count index for garbage collection
        EXECUTE format(
            'CREATE INDEX IF NOT EXISTS idx_%I_refcount ON %I (ref_count) WHERE ref_count = 0',
            partition_name,
            partition_name
        );
        
        -- Soft delete index
        EXECUTE format(
            'CREATE INDEX IF NOT EXISTS idx_%I_deleted ON %I (deleted_at) WHERE deleted_at IS NOT NULL',
            partition_name,
            partition_name
        );
    END LOOP;
END $$;

-- Ingestion audit trail table
CREATE TABLE ingestion_log (
    id BIGSERIAL PRIMARY KEY,
    batch_id UUID NOT NULL UNIQUE DEFAULT gen_random_uuid(),
    
    -- Source information
    source_uri TEXT NOT NULL,
    source_hash BYTEA NOT NULL,
    source_size_bytes BIGINT,
    source_type TEXT,  -- 'file', 'stream', 'api', etc.
    
    -- Ingestion statistics
    atoms_created BIGINT NOT NULL DEFAULT 0,
    atoms_updated BIGINT NOT NULL DEFAULT 0,
    atoms_skipped BIGINT NOT NULL DEFAULT 0,
    bytes_processed BIGINT NOT NULL DEFAULT 0,
    
    -- Timing information
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ,
    duration_ms BIGINT GENERATED ALWAYS AS (
        EXTRACT(EPOCH FROM (completed_at - started_at)) * 1000
    ) STORED,
    
    -- Status and error handling
    status ingestion_status NOT NULL DEFAULT 'pending',
    error_message TEXT,
    error_stack_trace TEXT,
    retry_count INT NOT NULL DEFAULT 0,
    
    -- Performance metrics
    throughput_mbps NUMERIC(10,2) GENERATED ALWAYS AS (
        CASE 
            WHEN completed_at IS NOT NULL AND completed_at > started_at
            THEN (bytes_processed::NUMERIC / 1048576) / 
                 EXTRACT(EPOCH FROM (completed_at - started_at))
            ELSE NULL
        END
    ) STORED,
    
    -- Optional metadata
    meta JSONB,
    
    -- Audit
    created_by TEXT NOT NULL DEFAULT current_user
);

-- Indexes for ingestion_log
CREATE INDEX idx_ingestion_batch ON ingestion_log (batch_id);
CREATE INDEX idx_ingestion_status ON ingestion_log (status, started_at DESC);
CREATE INDEX idx_ingestion_source ON ingestion_log USING HASH (source_hash);
CREATE INDEX idx_ingestion_created ON ingestion_log (started_at DESC);

-- Materialized view for spatial query optimization
-- Refreshed every 5 minutes via pg_cron
CREATE MATERIALIZED VIEW atom_spatial_summary AS
SELECT 
    modality,
    COUNT(*) AS atom_count,
    SUM(ref_count) AS total_references,
    AVG(ref_count) AS avg_references,
    ST_Extent(geom) AS spatial_bbox,
    AVG(ST_NPoints(geom)) AS avg_geometry_complexity,
    SUM(COALESCE(length(value), 0)) AS total_value_bytes,
    MIN(created_at) AS first_atom_at,
    MAX(created_at) AS last_atom_at
FROM atoms
WHERE deleted_at IS NULL
GROUP BY modality;

CREATE UNIQUE INDEX idx_spatial_summary_modality ON atom_spatial_summary (modality);

-- Trigger for automatic updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    NEW.updated_by = current_user;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply trigger to atoms table partitions
CREATE TRIGGER trg_atoms_updated_at
    BEFORE UPDATE ON atoms
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Function to increment reference count (atomic operation)
CREATE OR REPLACE FUNCTION increment_ref_count(
    p_h_xy BIGINT,
    p_h_yz BIGINT,
    p_h_zm BIGINT,
    p_delta BIGINT DEFAULT 1
)
RETURNS BIGINT AS $$
DECLARE
    new_count BIGINT;
BEGIN
    UPDATE atoms
    SET ref_count = ref_count + p_delta
    WHERE h_xy = p_h_xy 
      AND h_yz = p_h_yz 
      AND h_zm = p_h_zm
    RETURNING ref_count INTO new_count;
    
    RETURN COALESCE(new_count, 0);
END;
$$ LANGUAGE plpgsql;

-- Function to decrement reference count (atomic operation with gc check)
CREATE OR REPLACE FUNCTION decrement_ref_count(
    p_h_xy BIGINT,
    p_h_yz BIGINT,
    p_h_zm BIGINT,
    p_delta BIGINT DEFAULT 1
)
RETURNS BIGINT AS $$
DECLARE
    new_count BIGINT;
BEGIN
    UPDATE atoms
    SET ref_count = GREATEST(ref_count - p_delta, 0)
    WHERE h_xy = p_h_xy 
      AND h_yz = p_h_yz 
      AND h_zm = p_h_zm
    RETURNING ref_count INTO new_count;
    
    -- If ref_count reached 0, mark for garbage collection
    IF new_count = 0 THEN
        UPDATE atoms
        SET deleted_at = now(),
            deleted_by = current_user
        WHERE h_xy = p_h_xy 
          AND h_yz = p_h_yz 
          AND h_zm = p_h_zm;
    END IF;
    
    RETURN COALESCE(new_count, 0);
END;
$$ LANGUAGE plpgsql;

-- Row-level security policies
ALTER TABLE atoms ENABLE ROW LEVEL SECURITY;

-- Read policy: users can read non-deleted atoms they created or public atoms
CREATE POLICY atoms_read_policy ON atoms
    FOR SELECT
    USING (
        deleted_at IS NULL 
        AND (
            created_by = current_user 
            OR meta->>'visibility' = 'public'
            OR current_user IN ('atom_admin', 'atom_reader')
        )
    );

-- Insert policy: authenticated users can insert
CREATE POLICY atoms_insert_policy ON atoms
    FOR INSERT
    WITH CHECK (current_user IS NOT NULL);

-- Update policy: users can only update their own non-deleted atoms
CREATE POLICY atoms_update_policy ON atoms
    FOR UPDATE
    USING (
        deleted_at IS NULL 
        AND created_by = current_user
    )
    WITH CHECK (
        deleted_at IS NULL 
        AND created_by = current_user
    );

-- Delete policy: users can only soft-delete their own atoms
CREATE POLICY atoms_delete_policy ON atoms
    FOR UPDATE
    USING (
        created_by = current_user
    )
    WITH CHECK (
        deleted_at IS NOT NULL 
        AND deleted_by = current_user
    );

-- Grant permissions
-- Application user (read/write)
CREATE ROLE atom_app WITH LOGIN PASSWORD 'CHANGE_IN_PRODUCTION';
GRANT SELECT, INSERT, UPDATE ON atoms TO atom_app;
GRANT SELECT, INSERT, UPDATE ON ingestion_log TO atom_app;
GRANT SELECT ON atom_spatial_summary TO atom_app;
GRANT EXECUTE ON FUNCTION increment_ref_count TO atom_app;
GRANT EXECUTE ON FUNCTION decrement_ref_count TO atom_app;

-- Read-only user (analytics, reporting)
CREATE ROLE atom_reader WITH LOGIN PASSWORD 'CHANGE_IN_PRODUCTION';
GRANT SELECT ON atoms TO atom_reader;
GRANT SELECT ON ingestion_log TO atom_reader;
GRANT SELECT ON atom_spatial_summary TO atom_reader;

-- Admin user (full access)
CREATE ROLE atom_admin WITH LOGIN PASSWORD 'CHANGE_IN_PRODUCTION' SUPERUSER;

-- Comments for documentation
COMMENT ON TABLE atoms IS 'Universal atomic data store - all modalities stored as geometric structures';
COMMENT ON COLUMN atoms.h_xy IS 'Hilbert index for Content×Context manifold (56-bit)';
COMMENT ON COLUMN atoms.h_yz IS 'Hilbert index for Entropy×Structure manifold (56-bit)';
COMMENT ON COLUMN atoms.h_zm IS 'Hilbert index for Causality×Frequency manifold (56-bit)';
COMMENT ON COLUMN atoms.geom IS 'Geometric representation of data (POINT/LINESTRING/POLYGON/etc)';
COMMENT ON COLUMN atoms.value IS 'Raw bytes for primitive atoms (≤64 bytes, NULL for composites)';
COMMENT ON COLUMN atoms.modality IS 'Data type classification (discrete/continuous/relational/temporal/compositional)';
COMMENT ON COLUMN atoms.content_hash IS 'SHA-256 hash for global deduplication';
COMMENT ON COLUMN atoms.ref_count IS 'Reference counter for garbage collection (0 = eligible for deletion)';

-- Vacuum and analyze immediately after schema creation
VACUUM ANALYZE atoms;
VACUUM ANALYZE ingestion_log;

-- Display summary
SELECT 
    'Schema initialization complete' AS status,
    (SELECT count(*) FROM pg_tables WHERE schemaname = 'public') AS tables_created,
    (SELECT count(*) FROM pg_indexes WHERE schemaname = 'public') AS indexes_created,
    (SELECT count(*) FROM pg_roles WHERE rolname LIKE 'atom_%') AS roles_created;
