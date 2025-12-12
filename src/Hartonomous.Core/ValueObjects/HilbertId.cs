using System;

namespace Hartonomous.Core.ValueObjects;

/// <summary>
/// Immutable value object representing a Hilbert-encoded atom identifier.
/// Composite of three 2D manifold projections (h_xy, h_yz, h_zm).
/// </summary>
public readonly record struct HilbertId
{
    /// <summary>
    /// Content × Context manifold (XY projection)
    /// </summary>
    public long HXY { get; init; }
    
    /// <summary>
    /// Entropy × Structure manifold (YZ projection)
    /// </summary>
    public long HYZ { get; init; }
    
    /// <summary>
    /// Causality × Frequency manifold (ZM projection)
    /// </summary>
    public long HZM { get; init; }
    
    public HilbertId(long hxy, long hyz, long hzm)
    {
        if (hxy < 0 || hyz < 0 || hzm < 0)
            throw new ArgumentOutOfRangeException("Hilbert indices must be non-negative");
            
        HXY = hxy;
        HYZ = hyz;
        HZM = hzm;
    }
    
    /// <summary>
    /// Creates HilbertId from 4D coordinates.
    /// </summary>
    public static HilbertId FromCoordinates(uint x, uint y, uint z, uint m)
    {
        var (hxy, hyz, hzm) = Encoding.HilbertEncoder.Encode4DTo3Manifolds(x, y, z, m);
        return new HilbertId(hxy, hyz, hzm);
    }
    
    /// <summary>
    /// Converts back to 4D coordinates.
    /// </summary>
    public (uint x, uint y, uint z, uint m) ToCoordinates()
    {
        return Encoding.HilbertEncoder.Decode3ManifoldsTo4D(HXY, HYZ, HZM);
    }
    
    /// <summary>
    /// String representation for debugging.
    /// </summary>
    public override string ToString() => $"({HXY},{HYZ},{HZM})";
}
