# Hartonomous-Copilot Enterprise Architecture

## System Overview

A production-grade Universal Geometric Data Architecture (UGDA) implementation for storing, querying, and operating on multi-modal data (text, audio, images, video, neural networks, code) as geometric structures in PostgreSQL/PostGIS.

## Design Principles

1. **Single Source of Truth**: One atoms table, all modalities
2. **ACID Compliance**: Full transactional guarantees
3. **Zero Data Loss**: Immutable append-only with versioning
4. **Horizontal Scalability**: Partition by Hilbert space
5. **Observability First**: Metrics, logs, traces at every layer
6. **Security by Default**: Parameterized queries, row-level security
7. **Performance Critical**: Sub-millisecond spatial queries at scale

## Technology Stack

### Core Database
- **PostgreSQL 18.x**: Primary data store
- **PostGIS 3.6+**: Spatial indexing and operations
- **pgvector**: Vector similarity (supplementary)
- **PL/Python3u**: High-performance reconstruction functions
- **pg_stat_statements**: Query performance monitoring

### Application Layer
- **.NET 10 (C#)**: Primary application framework
- **Entity Framework Core 10**: ORM with PostGIS support
- **Npgsql**: High-performance PostgreSQL driver
- **MediatR**: CQRS pattern implementation
- **Polly**: Resilience and transient fault handling
- **Serilog**: Structured logging
- **OpenTelemetry**: Distributed tracing
- **Prometheus.NET**: Metrics collection

### Infrastructure
- **Docker**: Containerization
- **Kubernetes**: Orchestration (production)
- **Azure Arc**: Hybrid deployment management
- **Redis**: Distributed caching layer
- **RabbitMQ/Azure Service Bus**: Event streaming
- **Grafana**: Observability dashboards
- **Seq/Azure Application Insights**: Log aggregation

## Architecture Layers

```
┌─────────────────────────────────────────────────────┐
│           API Layer (ASP.NET Core)                   │
│  - REST API, gRPC endpoints                          │
│  - Authentication/Authorization (Azure AD)           │
│  - Rate limiting, request validation                 │
│  - OpenAPI/Swagger documentation                     │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│        Application Layer (CQRS + MediatR)            │
│  - Command handlers (writes)                         │
│  - Query handlers (reads)                            │
│  - Domain events                                     │
│  - Business logic validation                         │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│         Domain Layer (Core Business Logic)           │
│  - Atom aggregate root                               │
│  - Hilbert encoding domain services                  │
│  - SDI (Structured Deterministic Identity) generation│
│  - Value objects (immutable)                         │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│      Infrastructure Layer (Data Access)              │
│  - EF Core DbContext with PostGIS                    │
│  - Repository pattern implementation                 │
│  - Unit of Work pattern                              │
│  - Connection pooling (Npgsql)                       │
│  - Retry policies (Polly)                            │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│      PostgreSQL 18 + PostGIS 3.6                     │
│  - atoms table (partitioned by h_xy ranges)          │
│  - GIST spatial indexes                              │
│  - Materialized views for aggregations               │
│  - PL/Python3u reconstruction functions              │
└─────────────────────────────────────────────────────┘
```

## Database Schema (Production)

### Core Atoms Table

```sql
-- Main atoms table (will be partitioned)
CREATE TABLE atoms (
    -- Composite Hilbert Primary Key
    h_xy BIGINT NOT NULL,
    h_yz BIGINT NOT NULL,
    h_zm BIGINT NOT NULL,
    
    -- Geometry payload (the actual data structure)
    geom GEOMETRY(GEOMETRYCOLLECTION, 0) NOT NULL,
    
    -- Raw value for primitive atoms (leaf nodes)
    value BYTEA,
    CONSTRAINT chk_value_size CHECK (value IS NULL OR length(value) <= 64),
    
    -- Modality type (extracted from h_xy high byte)
    modality SMALLINT NOT NULL,
    
    -- Content hash for global deduplication
    content_hash BYTEA NOT NULL,
    CONSTRAINT chk_hash_size CHECK (length(content_hash) = 32),
    
    -- Reference counting for garbage collection
    ref_count BIGINT NOT NULL DEFAULT 1,
    CONSTRAINT chk_ref_count CHECK (ref_count >= 0),
    
    -- Optional metadata (JSONB for flexibility)
    meta JSONB,
    
    -- Audit fields
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by TEXT,
    updated_at TIMESTAMPTZ,
    updated_by TEXT,
    version BIGINT NOT NULL DEFAULT 1,
    
    -- Primary key
    PRIMARY KEY (h_xy, h_yz, h_zm)
) PARTITION BY RANGE (h_xy);

-- Create 256 partitions for horizontal scalability
-- Each partition = 1/256th of Hilbert space
-- Example partitions:
CREATE TABLE atoms_p000 PARTITION OF atoms
    FOR VALUES FROM (0) TO (72057594037927936);  -- 2^56 / 256

CREATE TABLE atoms_p001 PARTITION OF atoms
    FOR VALUES FROM (72057594037927936) TO (144115188075855872);

-- ... (254 more partitions)

-- Indexes per partition
CREATE INDEX idx_atoms_p000_geom ON atoms_p000 USING GIST(geom);
CREATE INDEX idx_atoms_p000_hash ON atoms_p000 (content_hash);
CREATE INDEX idx_atoms_p000_modality ON atoms_p000 (modality);
CREATE INDEX idx_atoms_p000_created ON atoms_p000 (created_at DESC);

-- ... (repeat for all partitions)
```

### Supporting Tables

```sql
-- Ingestion audit trail
CREATE TABLE ingestion_log (
    id BIGSERIAL PRIMARY KEY,
    batch_id UUID NOT NULL,
    source_uri TEXT NOT NULL,
    source_hash BYTEA NOT NULL,
    atoms_created BIGINT NOT NULL DEFAULT 0,
    atoms_updated BIGINT NOT NULL DEFAULT 0,
    atoms_skipped BIGINT NOT NULL DEFAULT 0,
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    error_message TEXT,
    status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    meta JSONB
);

CREATE INDEX idx_ingestion_batch ON ingestion_log (batch_id);
CREATE INDEX idx_ingestion_status ON ingestion_log (status, started_at DESC);

-- Spatial query performance cache (materialized view)
CREATE MATERIALIZED VIEW atom_spatial_summary AS
SELECT 
    modality,
    ST_Extent(geom) AS bbox,
    COUNT(*) AS atom_count,
    SUM(ref_count) AS total_refs,
    AVG(ST_NPoints(geom)) AS avg_complexity
FROM atoms
GROUP BY modality;

CREATE INDEX idx_spatial_summary_modality ON atom_spatial_summary (modality);

-- Refresh strategy: concurrent refresh every 5 minutes via pg_cron
CREATE EXTENSION IF NOT EXISTS pg_cron;
SELECT cron.schedule('refresh-spatial-summary', '*/5 * * * *', 
    'REFRESH MATERIALIZED VIEW CONCURRENTLY atom_spatial_summary');
```

## Configuration Management

### Connection String (Production)
```json
{
  "ConnectionStrings": {
    "AtomStore": "Host=atom-db-primary.postgres.database.azure.com;Port=5432;Database=hartonomous;Username=atom_writer;Password={from-keyvault};SSL Mode=Require;Trust Server Certificate=true;Pooling=true;Minimum Pool Size=10;Maximum Pool Size=100;Connection Idle Lifetime=300;Connection Pruning Interval=10;Timeout=30;Command Timeout=60"
  }
}
```

### Retry Policy Configuration
```csharp
services.AddSingleton<IAsyncPolicy>(Policy
    .Handle<NpgsqlException>()
    .Or<TimeoutException>()
    .WaitAndRetryAsync(
        retryCount: 3,
        sleepDurationProvider: attempt => TimeSpan.FromSeconds(Math.Pow(2, attempt)),
        onRetry: (exception, timeSpan, retryCount, context) =>
        {
            logger.LogWarning(exception, 
                "Retry {RetryCount} after {Delay}s due to {ExceptionType}",
                retryCount, timeSpan.TotalSeconds, exception.GetType().Name);
        }
    ));
```

## Performance Requirements

| Operation | Target Latency | Throughput |
|-----------|---------------|------------|
| Point lookup (by Hilbert ID) | < 1ms | 100k ops/sec |
| Spatial k-NN query | < 10ms | 10k ops/sec |
| Bulk insert (batch 1000) | < 100ms | 10k atoms/sec |
| Reconstruction (1MB file) | < 50ms | 20 MB/sec |
| A* pathfinding (10 hops) | < 100ms | 1k paths/sec |

## Security Model

### Row-Level Security
```sql
ALTER TABLE atoms ENABLE ROW LEVEL SECURITY;

-- Read policy: users can read atoms they created or public atoms
CREATE POLICY atoms_read_policy ON atoms
    FOR SELECT
    USING (created_by = current_user OR meta->>'visibility' = 'public');

-- Write policy: users can only update their own atoms
CREATE POLICY atoms_write_policy ON atoms
    FOR UPDATE
    USING (created_by = current_user)
    WITH CHECK (created_by = current_user);
```

### API Authentication
- Azure AD OAuth 2.0 / OpenID Connect
- JWT bearer tokens with claims-based authorization
- API key authentication for service-to-service
- Rate limiting per client ID

## Observability

### Metrics (Prometheus)
```csharp
public static class Metrics
{
    public static readonly Counter IngestionsTotal = 
        Prometheus.Metrics.CreateCounter(
            "hartonomous_ingestions_total",
            "Total number of ingestion operations",
            new[] { "source_type", "status" });
            
    public static readonly Histogram QueryDuration = 
        Prometheus.Metrics.CreateHistogram(
            "hartonomous_query_duration_seconds",
            "Query execution time",
            new[] { "query_type", "modality" });
            
    public static readonly Gauge AtomCount = 
        Prometheus.Metrics.CreateGauge(
            "hartonomous_atoms_total",
            "Total atoms in store",
            new[] { "modality" });
}
```

### Logging Standards
```csharp
logger.LogInformation(
    "Atom ingested: {@AtomMetadata}",
    new {
        HilbertId = new { h_xy, h_yz, h_zm },
        Modality = modality,
        ValueSize = value?.Length ?? 0,
        GeometryType = geom.GeometryType,
        Duration = stopwatch.ElapsedMilliseconds
    });
```

## Deployment Strategy

### Blue-Green Deployment
1. Deploy new version to "green" environment
2. Run smoke tests and performance validation
3. Gradually shift traffic (10% → 50% → 100%)
4. Monitor error rates and latency
5. Rollback if p99 latency > 2x baseline

### Database Migration Strategy
- Use Flyway or EF Core Migrations
- All migrations must be backwards compatible
- No destructive changes without 2-phase migration
- Test migrations on production-sized dataset clone

## Disaster Recovery

### Backup Strategy
- Continuous WAL archiving to Azure Blob Storage
- Full base backup daily (retention: 30 days)
- Point-in-time recovery capability (7 days)
- Cross-region replication for HA

### RTO/RPO Targets
- **RPO (Recovery Point Objective)**: 5 minutes
- **RTO (Recovery Time Objective)**: 30 minutes
- **Availability SLA**: 99.95% (4.38 hours downtime/year)

