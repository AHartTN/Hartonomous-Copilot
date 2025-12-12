using System;
using Hartonomous.Core.ValueObjects;

namespace Hartonomous.Core.Domain;

/// <summary>
/// Aggregate root representing an atomic unit of data in the system.
/// Immutable after creation (updates create new versions).
/// </summary>
public sealed class Atom
{
    /// <summary>
    /// Unique identifier (Hilbert-encoded position in semantic space)
    /// </summary>
    public HilbertId Id { get; private set; }
    
    /// <summary>
    /// PostGIS geometry (the actual data structure)
    /// - POINT: Discrete events
    /// - LINESTRING: Sequences
    /// - POLYGON: Regions
    /// - GEOMETRYCOLLECTION: Composites
    /// </summary>
    public byte[] Geometry { get; private set; } // WKB format
    
    /// <summary>
    /// Raw value for primitive atoms (≤64 bytes, null for composites)
    /// </summary>
    public byte[]? Value { get; private set; }
    
    /// <summary>
    /// Modality classification
    /// </summary>
    public Modality Modality { get; private set; }
    
    /// <summary>
    /// SHA-256 content hash for global deduplication
    /// </summary>
    public byte[] ContentHash { get; private set; }
    
    /// <summary>
    /// Reference count (for garbage collection)
    /// </summary>
    public long RefCount { get; private set; }
    
    /// <summary>
    /// Optional metadata (serialized JSON)
    /// </summary>
    public string? MetaJson { get; private set; }
    
    /// <summary>
    /// Audit information
    /// </summary>
    public DateTimeOffset CreatedAt { get; private set; }
    public string CreatedBy { get; private set; }
    public DateTimeOffset? UpdatedAt { get; private set; }
    public string? UpdatedBy { get; private set; }
    public long Version { get; private set; }
    
    /// <summary>
    /// Soft delete (tombstone pattern)
    /// </summary>
    public DateTimeOffset? DeletedAt { get; private set; }
    public string? DeletedBy { get; private set; }
    
    // EF Core requires parameterless constructor
    private Atom() { }
    
    /// <summary>
    /// Factory method for creating new atom.
    /// </summary>
    public static Atom Create(
        HilbertId id,
        byte[] geometry,
        byte[]? value,
        Modality modality,
        byte[] contentHash,
        string createdBy,
        string? metaJson = null)
    {
        // Validation
        if (geometry == null || geometry.Length == 0)
            throw new ArgumentException("Geometry cannot be null or empty", nameof(geometry));
            
        if (value != null && value.Length > 64)
            throw new ArgumentException("Value must be ≤64 bytes", nameof(value));
            
        if (contentHash == null || contentHash.Length != 32)
            throw new ArgumentException("Content hash must be exactly 32 bytes (SHA-256)", nameof(contentHash));
            
        if (string.IsNullOrWhiteSpace(createdBy))
            throw new ArgumentException("CreatedBy is required", nameof(createdBy));
        
        return new Atom
        {
            Id = id,
            Geometry = geometry,
            Value = value,
            Modality = modality,
            ContentHash = contentHash,
            RefCount = 1,
            MetaJson = metaJson,
            CreatedAt = DateTimeOffset.UtcNow,
            CreatedBy = createdBy,
            Version = 1
        };
    }
    
    /// <summary>
    /// Increments reference count (thread-safe via database).
    /// </summary>
    public void IncrementRefCount(long delta = 1)
    {
        if (delta < 0)
            throw new ArgumentException("Delta must be positive", nameof(delta));
            
        RefCount += delta;
    }
    
    /// <summary>
    /// Decrements reference count and marks for deletion if zero.
    /// </summary>
    public void DecrementRefCount(long delta = 1, string deletedBy = "system")
    {
        if (delta < 0)
            throw new ArgumentException("Delta must be positive", nameof(delta));
            
        RefCount = Math.Max(0, RefCount - delta);
        
        if (RefCount == 0 && DeletedAt == null)
        {
            SoftDelete(deletedBy);
        }
    }
    
    /// <summary>
    /// Marks atom as deleted (tombstone pattern).
    /// </summary>
    public void SoftDelete(string deletedBy)
    {
        if (DeletedAt.HasValue)
            throw new InvalidOperationException("Atom already deleted");
            
        DeletedAt = DateTimeOffset.UtcNow;
        DeletedBy = deletedBy;
    }
    
    /// <summary>
    /// Updates metadata.
    /// </summary>
    public void UpdateMetadata(string metaJson, string updatedBy)
    {
        if (DeletedAt.HasValue)
            throw new InvalidOperationException("Cannot update deleted atom");
            
        MetaJson = metaJson;
        UpdatedAt = DateTimeOffset.UtcNow;
        UpdatedBy = updatedBy;
        Version++;
    }
    
    /// <summary>
    /// Checks if atom is a primitive (has value) or composite (no value).
    /// </summary>
    public bool IsPrimitive => Value != null;
    
    public bool IsComposite => Value == null;
    
    public bool IsDeleted => DeletedAt.HasValue;
}

/// <summary>
/// Modality enumeration (matches database enum)
/// </summary>
public enum Modality
{
    Discrete = 1,       // bytes, tokens, pixels
    Continuous = 2,     // floats, audio samples, embeddings
    Relational = 3,     // edges, synapses, references
    Temporal = 4,       // sequences, trajectories, videos
    Compositional = 5   // structures, aggregates, batches
}
