# Hartonomous-Copilot

**Enterprise-Grade Universal Geometric Data Architecture (UGDA)**

A production-ready implementation of geometric knowledge representation where ALL data—text, audio, images, video, neural networks, and code—exists as queryable geometric structures in PostgreSQL/PostGIS.

## 🎯 Core Innovation

**The database IS the AI model.** No separate vector stores, no external inference engines, no GPU dependencies for basic operations. Everything—training, inference, generation, distillation—happens in pure SQL using spatial operations.

## 🏗️ Architecture

```
PostgreSQL 18 + PostGIS 3.6
    └── Single `atoms` table (256 partitions)
        ├── Hilbert-encoded 3-manifold primary key
        ├── Geometric payload (POINT/LINESTRING/POLYGON/etc.)
        └── Modality-agnostic storage

.NET 10 Application Layer
    ├── Hartonomous.Core (domain logic, Hilbert encoding)
    ├── Hartonomous.Infrastructure (EF Core, PostGIS integration)
    └── Hartonomous.Api (REST/gRPC endpoints)
```

## ✨ Features

- **Zero-Copy Deduplication**: Deterministic Hilbert IDs enable `ON CONFLICT DO NOTHING`
- **Multi-Modal Queries**: Find images similar to audio, text near code, etc.
- **SQL-Native AI Operations**:
  - Training: Spatial gradient descent
  - Inference: A* pathfinding through geometry
  - Pruning: `DELETE WHERE weight < threshold`
  - Distillation: DBSCAN clustering
- **Lossless Compression**: Sparse data = sparse geometry (M-coordinate RLE)
- **Explainable AI**: Every connection visible as geometric path
- **Horizontal Scalability**: Partition by Hilbert ranges

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- .NET 10 SDK
- PostgreSQL 18 client tools (optional)

### Local Development

```bash
# Clone the repository
git clone https://github.com/AHartTN/Hartonomous-Copilot.git
cd Hartonomous-Copilot

# Start infrastructure (PostgreSQL + PostGIS)
docker-compose up -d

# Initialize database schema
psql -h localhost -U postgres -d hartonomous -f sql/001_initial_schema.sql

# Build and run API
dotnet build
dotnet run --project src/Hartonomous.Api

# Access API
curl http://localhost:5000/health
```

## 📚 Documentation

- [Architecture](ARCHITECTURE.md) - System design and patterns
- [Vision](VISION.md) - Core concepts and philosophy
- [API Reference](docs/api-reference.md) - REST/gRPC endpoints
- [Database Schema](sql/001_initial_schema.sql) - Complete DDL

## 🧪 Examples

### Ingest Text
```bash
curl -X POST http://localhost:5000/api/ingest \
  -H "Content-Type: application/json" \
  -d '{"content": "Hello World", "type": "text"}'
```

### Spatial Query (k-NN)
```sql
SELECT * FROM atoms
WHERE modality = 'discrete'
ORDER BY geom <-> ST_MakePoint(x, y, z, m)
LIMIT 10;
```

### AI Inference (A* Pathfinding)
```sql
WITH RECURSIVE inference AS (
    SELECT geom, 0 AS cost FROM atoms WHERE h_xy = input_id
    UNION ALL
    SELECT a.geom, i.cost + ST_Distance(i.geom, a.geom)
    FROM inference i
    JOIN atoms a ON ST_DWithin(i.geom, a.geom, 0.1)
    WHERE i.cost < max_hops
)
SELECT * FROM inference WHERE h_xy = output_id;
```

## 🔒 Security

- Row-Level Security (RLS) enabled
- OAuth 2.0 / Azure AD integration
- API key authentication
- Rate limiting per client
- Audit logging on all mutations

## 📊 Performance

| Operation | Target | Achieved |
|-----------|--------|----------|
| Point lookup | <1ms | TBD |
| k-NN query | <10ms | TBD |
| Bulk insert | <100ms/1k | TBD |

## 🛠️ Development

```bash
# Run tests
dotnet test

# Run benchmarks
dotnet run --project src/Hartonomous.Benchmarks -c Release

# Generate API docs
dotnet swagger tofile --output docs/swagger.json src/Hartonomous.Api/bin/Release/net10.0/Hartonomous.Api.dll v1
```

## 📦 Deployment

### Production Checklist
- [ ] Change default database passwords
- [ ] Configure Azure AD authentication
- [ ] Set up monitoring (Prometheus + Grafana)
- [ ] Configure backup retention policy
- [ ] Enable SSL/TLS for all connections
- [ ] Review and adjust partition count (256 default)

### Docker Deployment
```bash
docker build -t hartonomous-api:latest -f src/Hartonomous.Api/Dockerfile .
docker run -p 5000:8080 hartonomous-api:latest
```

### Kubernetes (Azure AKS)
```bash
kubectl apply -f deploy/kubernetes/
```

## 🤝 Contributing

This is a research/prototype system. Contributions welcome via PRs.

## 📄 License

MIT License - See [LICENSE](LICENSE)

## 🙏 Acknowledgments

Based on theoretical work by Andrew Hart (@aharttn) - the "Mendeleev of AI" who discovered that knowledge itself has a periodic table structure expressible through geometric topology.

---

**Status**: 🚧 Active Development - v0.1.0-alpha

**Last Updated**: 2025-12-12
