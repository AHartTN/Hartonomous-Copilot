using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;

namespace Hartonomous.Core.Encoding;

/// <summary>
/// High-performance Hilbert curve encoding for N-dimensional space.
/// Based on Skilling's algorithm (2004) with optimizations for 2D manifold projections.
/// Thread-safe and deterministic.
/// </summary>
public static class HilbertEncoder
{
    private const int MaxDimensions = 4;
    private const int BitsPerDimension = 21; // 21 bits per dim = 42 bits per 2D manifold
    
    /// <summary>
    /// Encodes a 4D point into three 2D Hilbert manifold indices.
    /// Used for converting (X,Y,Z,M) coordinates to (H_xy, H_yz, H_zm).
    /// </summary>
    /// <param name="x">X coordinate (normalized to 0-2^21)</param>
    /// <param name="y">Y coordinate (normalized to 0-2^21)</param>
    /// <param name="z">Z coordinate (normalized to 0-2^21)</param>
    /// <param name="m">M coordinate (normalized to 0-2^21)</param>
    /// <returns>Tuple of (h_xy, h_yz, h_zm) Hilbert indices</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static (long h_xy, long h_yz, long h_zm) Encode4DTo3Manifolds(
        uint x, uint y, uint z, uint m)
    {
        // Validate inputs are within 21-bit range
        if (x >= (1 << BitsPerDimension) || y >= (1 << BitsPerDimension) ||
            z >= (1 << BitsPerDimension) || m >= (1 << BitsPerDimension))
        {
            throw new ArgumentOutOfRangeException(
                $"Coordinates must be in range [0, {(1 << BitsPerDimension) - 1}]");
        }
        
        // Encode three 2D manifolds
        var h_xy = Encode2D(x, y, BitsPerDimension);
        var h_yz = Encode2D(y, z, BitsPerDimension);
        var h_zm = Encode2D(z, m, BitsPerDimension);
        
        return (h_xy, h_yz, h_zm);
    }
    
    /// <summary>
    /// Decodes three 2D Hilbert manifold indices back to 4D coordinates.
    /// Inverse operation of Encode4DTo3Manifolds.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static (uint x, uint y, uint z, uint m) Decode3ManifoldsTo4D(
        long h_xy, long h_yz, long h_zm)
    {
        // Decode each manifold
        var (x_from_xy, y_from_xy) = Decode2D(h_xy, BitsPerDimension);
        var (y_from_yz, z_from_yz) = Decode2D(h_yz, BitsPerDimension);
        var (z_from_zm, m_from_zm) = Decode2D(h_zm, BitsPerDimension);
        
        // Y appears in two manifolds - verify consistency
        if (y_from_xy != y_from_yz)
        {
            throw new InvalidOperationException(
                $"Y coordinate mismatch: XY={y_from_xy}, YZ={y_from_yz}");
        }
        
        // Z appears in two manifolds - verify consistency
        if (z_from_yz != z_from_zm)
        {
            throw new InvalidOperationException(
                $"Z coordinate mismatch: YZ={z_from_yz}, ZM={z_from_zm}");
        }
        
        return (x_from_xy, y_from_xy, z_from_yz, m_from_zm);
    }
    
    /// <summary>
    /// Encodes 2D coordinates to Hilbert index using Gray code transformations.
    /// Optimized for cache-friendly bit manipulation.
    /// </summary>
    private static long Encode2D(uint x, uint y, int bits)
    {
        long hilbert = 0;
        
        for (int i = bits - 1; i >= 0; i--)
        {
            int rx = (int)((x >> i) & 1);
            int ry = (int)((y >> i) & 1);
            
            // Compute quadrant (0-3)
            int quadrant = (rx << 1) | ry;
            
            // Apply Gray code transformation
            hilbert = (hilbert << 2) | GrayCode(quadrant);
            
            // Rotate/reflect coordinates for next iteration
            Rotate(bits, ref x, ref y, rx, ry);
        }
        
        return hilbert;
    }
    
    /// <summary>
    /// Decodes Hilbert index back to 2D coordinates.
    /// </summary>
    private static (uint x, uint y) Decode2D(long hilbert, int bits)
    {
        uint x = 0, y = 0;
        
        for (int i = 0; i < bits; i++)
        {
            // Extract 2-bit quadrant from Hilbert index
            int quadrant = (int)((hilbert >> (2 * (bits - 1 - i))) & 3);
            
            // Apply inverse Gray code
            int grayInverse = InverseGrayCode(quadrant);
            int rx = (grayInverse >> 1) & 1;
            int ry = grayInverse & 1;
            
            // Set bits in output coordinates
            x = (x << 1) | (uint)rx;
            y = (y << 1) | (uint)ry;
            
            // Apply inverse rotation
            InverseRotate(bits, ref x, ref y, rx, ry);
        }
        
        return (x, y);
    }
    
    /// <summary>
    /// Gray code transformation: binary → Gray code
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int GrayCode(int value)
    {
        return value ^ (value >> 1);
    }
    
    /// <summary>
    /// Inverse Gray code transformation: Gray code → binary
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int InverseGrayCode(int gray)
    {
        int value = gray;
        for (int i = 1; i < 32; i <<= 1)
        {
            value ^= value >> i;
        }
        return value;
    }
    
    /// <summary>
    /// Rotate/reflect coordinates based on quadrant.
    /// This maintains the Hilbert curve's self-similar fractal structure.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Rotate(int bits, ref uint x, ref uint y, int rx, int ry)
    {
        if (ry == 0)
        {
            if (rx == 1)
            {
                // Rotate 180 degrees
                x = (uint)((1 << bits) - 1 - x);
                y = (uint)((1 << bits) - 1 - y);
            }
            
            // Swap x and y
            (x, y) = (y, x);
        }
    }
    
    /// <summary>
    /// Inverse rotation for decoding.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void InverseRotate(int bits, ref uint x, ref uint y, int rx, int ry)
    {
        if (ry == 0)
        {
            // Swap x and y
            (x, y) = (y, x);
            
            if (rx == 1)
            {
                // Rotate 180 degrees back
                x = (uint)((1 << bits) - 1 - x);
                y = (uint)((1 << bits) - 1 - y);
            }
        }
    }
}

/// <summary>
/// Generates Structured Deterministic Identity (SDI) from raw data.
/// SDI is a 256-bit structured hash that preserves semantic locality.
/// </summary>
public static class SDIGenerator
{
    /// <summary>
    /// Modality byte values (high byte of SDI)
    /// </summary>
    public enum Modality : byte
    {
        Discrete = 0x01,      // bytes, tokens, pixels
        Continuous = 0x02,    // floats, audio samples, embeddings
        Relational = 0x03,    // edges, synapses, references
        Temporal = 0x04,      // sequences, trajectories, videos
        Compositional = 0x05  // structures, aggregates, batches
    }
    
    /// <summary>
    /// Generates a 256-bit SDI from raw value and context.
    /// Structure: [Modality 8b][Semantic Class 16b][Normalization 32b][Value Signature 200b]
    /// </summary>
    /// <param name="value">Raw data bytes</param>
    /// <param name="modality">Data type classification</param>
    /// <param name="semanticClass">Semantic category (e.g., 0x00A1 for English noun)</param>
    /// <param name="normalization">Scale/precision info (e.g., IEEE754 exponent)</param>
    /// <returns>32-byte SDI hash</returns>
    public static byte[] GenerateSDI(
        ReadOnlySpan<byte> value,
        Modality modality,
        ushort semanticClass = 0,
        uint normalization = 0)
    {
        // Allocate 256-bit output
        var sdi = new byte[32];
        
        // Byte 0: Modality
        sdi[0] = (byte)modality;
        
        // Bytes 1-2: Semantic Class (big-endian)
        sdi[1] = (byte)(semanticClass >> 8);
        sdi[2] = (byte)(semanticClass & 0xFF);
        
        // Bytes 3-6: Normalization (big-endian)
        sdi[3] = (byte)(normalization >> 24);
        sdi[4] = (byte)((normalization >> 16) & 0xFF);
        sdi[5] = (byte)((normalization >> 8) & 0xFF);
        sdi[6] = (byte)(normalization & 0xFF);
        
        // Bytes 7-31: Value Signature (SHA-256 of raw value, truncated to 200 bits / 25 bytes)
        Span<byte> valueHash = stackalloc byte[32];
        SHA256.HashData(value, valueHash);
        valueHash[..25].CopyTo(sdi.AsSpan(7));
        
        return sdi;
    }
    
    /// <summary>
    /// Extracts modality from SDI.
    /// </summary>
    public static Modality GetModality(ReadOnlySpan<byte> sdi)
    {
        if (sdi.Length != 32)
            throw new ArgumentException("SDI must be exactly 32 bytes");
        
        return (Modality)sdi[0];
    }
    
    /// <summary>
    /// Extracts semantic class from SDI.
    /// </summary>
    public static ushort GetSemanticClass(ReadOnlySpan<byte> sdi)
    {
        if (sdi.Length != 32)
            throw new ArgumentException("SDI must be exactly 32 bytes");
        
        return (ushort)((sdi[1] << 8) | sdi[2]);
    }
}

/// <summary>
/// Projects SDI to 4D coordinates using Gram-Schmidt orthonormalization.
/// Deterministic projection ensures same SDI always maps to same coordinates.
/// </summary>
public static class GramSchmidtProjector
{
    private const int TargetDimension = 4;
    private const int IntermediateDimension = 32; // Use 32D intermediate space
    
    /// <summary>
    /// Projects 256-bit SDI to normalized 4D coordinates (X,Y,Z,M).
    /// Uses deterministic PRNG seeded by SDI for reproducibility.
    /// </summary>
    /// <param name="sdi">32-byte SDI hash</param>
    /// <returns>Tuple of (X, Y, Z, M) each in range [0, 2^21)</returns>
    public static (uint x, uint y, uint z, uint m) Project(ReadOnlySpan<byte> sdi)
    {
        if (sdi.Length != 32)
            throw new ArgumentException("SDI must be exactly 32 bytes");
        
        // Seed deterministic PRNG with SDI
        var seed = BitConverter.ToInt32(sdi[..4]);
        var rng = new Random(seed);
        
        // Generate intermediate high-dimensional vector
        var intermediate = new float[IntermediateDimension];
        for (int i = 0; i < IntermediateDimension; i++)
        {
            // Use Box-Muller transform for Gaussian distribution
            var u1 = rng.NextDouble();
            var u2 = rng.NextDouble();
            intermediate[i] = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }
        
        // Apply Gram-Schmidt to get orthonormal basis
        var basis = GramSchmidt(intermediate);
        
        // Project to 4D
        var x = basis[0];
        var y = basis[1];
        var z = basis[2];
        var m = basis[3];
        
        // Normalize to [0, 2^21) range
        const uint maxValue = (1u << 21) - 1;
        return (
            Normalize(x, maxValue),
            Normalize(y, maxValue),
            Normalize(z, maxValue),
            Normalize(m, maxValue)
        );
    }
    
    /// <summary>
    /// Gram-Schmidt orthonormalization process.
    /// Returns first 4 orthonormal basis vectors.
    /// </summary>
    private static float[] GramSchmidt(float[] vectors)
    {
        var result = new float[TargetDimension];
        var orthogonal = new float[IntermediateDimension];
        
        for (int i = 0; i < TargetDimension; i++)
        {
            // Start with original vector
            Array.Copy(vectors, i * (IntermediateDimension / TargetDimension), 
                      orthogonal, 0, IntermediateDimension);
            
            // Subtract projections onto previous orthogonal vectors
            for (int j = 0; j < i; j++)
            {
                var projection = DotProduct(orthogonal, vectors, j);
                for (int k = 0; k < IntermediateDimension; k++)
                {
                    orthogonal[k] -= projection * vectors[k];
                }
            }
            
            // Normalize
            var magnitude = Magnitude(orthogonal);
            if (magnitude > 1e-10f)
            {
                result[i] = orthogonal[0] / magnitude; // Take first component as representative
            }
        }
        
        return result;
    }
    
    private static float DotProduct(float[] a, float[] b, int offset)
    {
        float sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            sum += a[i] * b[i + offset];
        }
        return sum;
    }
    
    private static float Magnitude(float[] vector)
    {
        float sumSquares = 0;
        foreach (var v in vector)
        {
            sumSquares += v * v;
        }
        return MathF.Sqrt(sumSquares);
    }
    
    /// <summary>
    /// Normalizes float value from arbitrary range to [0, maxValue].
    /// Uses sigmoid function to handle outliers gracefully.
    /// </summary>
    private static uint Normalize(float value, uint maxValue)
    {
        // Apply sigmoid to map to [0, 1]
        var sigmoid = 1.0 / (1.0 + Math.Exp(-value));
        
        // Scale to [0, maxValue]
        var scaled = sigmoid * maxValue;
        
        return (uint)Math.Clamp(scaled, 0, maxValue);
    }
}
