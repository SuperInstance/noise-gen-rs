//! Utility functions for noise generation: gradient vectors, interpolation, and hashing.

/// Fade curve: 6t^5 - 15t^4 + 10t^3 (Quintic Hermite interpolation).
/// Produces a smooth S-curve that has zero first and second derivatives at t=0 and t=1.
#[inline]
pub fn fade(t: f64) -> f64 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// Linear interpolation between `a` and `b` by factor `t`.
#[inline]
pub fn lerp(t: f64, a: f64, b: f64) -> f64 {
    a + t * (b - a)
}

/// 2D gradient vectors used for Perlin noise.
/// These are the 8 unit directions on a circle.
pub const GRADIENTS_2D: [(f64, f64); 8] = [
    (1.0, 0.0),
    (-1.0, 0.0),
    (0.0, 1.0),
    (0.0, -1.0),
    (1.0, 1.0),
    (-1.0, 1.0),
    (1.0, -1.0),
    (-1.0, -1.0),
];

/// 3D gradient vectors used for Perlin noise.
pub const GRADIENTS_3D: [(f64, f64, f64); 12] = [
    (1.0, 1.0, 0.0),
    (-1.0, 1.0, 0.0),
    (1.0, -1.0, 0.0),
    (-1.0, -1.0, 0.0),
    (1.0, 0.0, 1.0),
    (-1.0, 0.0, 1.0),
    (1.0, 0.0, -1.0),
    (-1.0, 0.0, -1.0),
    (0.0, 1.0, 1.0),
    (0.0, -1.0, 1.0),
    (0.0, 1.0, -1.0),
    (0.0, -1.0, -1.0),
];

/// Simple hash function for seeding. Maps an integer to a pseudo-random index.
/// Uses a multiplication-based hash for good distribution.
#[inline]
pub fn hash(seed: u64, x: i64, y: i64) -> usize {
    let h = seed
        .wrapping_mul(374761393)
        .wrapping_add(x as u64)
        .wrapping_mul(668265263)
        .wrapping_add(y as u64)
        .wrapping_mul(1274126177);
    (h >> 16) as usize
}

/// 3D hash function for seeding.
#[inline]
pub fn hash3(seed: u64, x: i64, y: i64, z: i64) -> usize {
    let h = seed
        .wrapping_mul(374761393)
        .wrapping_add(x as u64)
        .wrapping_mul(668265263)
        .wrapping_add(y as u64)
        .wrapping_mul(1274126177)
        .wrapping_add(z as u64)
        .wrapping_mul(1911520717);
    (h >> 16) as usize
}

/// Simple seeded pseudo-random number generator (xoshiro256-like, simplified).
/// Returns a value in [0, 1).
#[inline]
pub fn seeded_random(seed: u64, index: usize) -> f64 {
    let mut h = seed.wrapping_add(index as u64);
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
    h ^= h >> 33;
    (h as f64) / (u64::MAX as f64)
}

/// Skewing factor for 2D Simplex noise: (sqrt(3) - 1) / 2 ≈ 0.3660254037844386.
pub const SIMPLEX_SKEW_2D: f64 = 0.3660254037844386;

/// Unskewing factor for 2D Simplex noise: (3 - sqrt(3)) / 6 ≈ 0.21132486540518713.
pub const SIMPLEX_UNSKEW_2D: f64 = 0.21132486540518713;

/// Skewing factor for 3D Simplex noise: (sqrt(4) - 1) / 3 = 1/3.
pub const SIMPLEX_SKEW_3D: f64 = 1.0 / 3.0;

/// Unskewing factor for 3D Simplex noise: 1/6.
pub const SIMPLEX_UNSKEW_3D: f64 = 1.0 / 6.0;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fade_boundaries() {
        // At t=0 and t=1, fade should be 0 and 1 respectively
        assert!((fade(0.0) - 0.0).abs() < 1e-10);
        assert!((fade(1.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_fade_derivatives_zero_at_boundaries() {
        // Derivative of fade: 30t^4 - 60t^3 + 30t^2
        let d = |t: f64| 30.0 * t * t * t * t - 60.0 * t * t * t + 30.0 * t * t;
        assert!(d(0.0).abs() < 1e-10);
        assert!(d(1.0).abs() < 1e-10);
    }

    #[test]
    fn test_fade_midpoint() {
        // At t=0.5, fade should be 0.5
        assert!((fade(0.5) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_lerp() {
        assert!((lerp(0.0, 5.0, 10.0) - 5.0).abs() < 1e-10);
        assert!((lerp(1.0, 5.0, 10.0) - 10.0).abs() < 1e-10);
        assert!((lerp(0.5, 0.0, 10.0) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_hash_deterministic() {
        let h1 = hash(42, 3, 7);
        let h2 = hash(42, 3, 7);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_hash_different_inputs() {
        let h1 = hash(42, 0, 0);
        let h2 = hash(42, 1, 0);
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_hash3_deterministic() {
        let h1 = hash3(42, 1, 2, 3);
        let h2 = hash3(42, 1, 2, 3);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_seeded_random_range() {
        for i in 0..100 {
            let v = seeded_random(42, i);
            assert!(v >= 0.0 && v < 1.0, "Value {} out of range", v);
        }
    }

    #[test]
    fn test_seeded_random_deterministic() {
        let v1 = seeded_random(123, 456);
        let v2 = seeded_random(123, 456);
        assert!((v1 - v2).abs() < 1e-15);
    }
}
