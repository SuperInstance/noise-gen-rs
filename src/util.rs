//! Shared utilities: permutation tables, gradient look-ups, interpolation, and hashing.

/// A permutation table built from a seed, used to hash grid coordinates.
#[derive(Clone, Debug)]
pub struct PermutationTable {
    perm: [u8; 512],
}

impl PermutationTable {
    /// Build a new permutation table from a 64-bit seed.
    ///
    /// Uses a simple xorshift PRNG to shuffle a base permutation of `0..256`.
    pub fn new(seed: u64) -> Self {
        let mut p: [u8; 256] = core::array::from_fn(|i| i as u8);
        let mut s = seed;
        for i in (1..256).rev() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let j = (s >> 33) as usize % (i + 1);
            p.swap(i, j);
        }
        let mut perm = [0u8; 512];
        perm[..256].copy_from_slice(&p);
        perm[256..].copy_from_slice(&p);
        Self { perm }
    }

    /// Hash a single integer coordinate.
    #[inline]
    pub fn hash1(&self, x: i32) -> u8 {
        self.perm[(x & 0xFF) as usize]
    }

    /// Hash two integer coordinates.
    #[inline]
    pub fn hash2(&self, x: i32, y: i32) -> u8 {
        self.perm[(i32::from(self.hash1(x)) + (y & 0xFF)) as usize & 0x1FF]
    }

    /// Hash three integer coordinates.
    #[inline]
    pub fn hash3(&self, x: i32, y: i32, z: i32) -> u8 {
        self.perm[(i32::from(self.hash2(x, y)) + (z & 0xFF)) as usize & 0x1FF]
    }
}

/// 2D gradient vectors for Perlin/Simplex noise (12 directions).
pub const GRADIENTS_2D: [[f64; 2]; 12] = [
    [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0],
    [1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [-1.0, -1.0],
    [1.0, 0.5], [-1.0, 0.5], [0.5, 1.0], [0.5, -1.0],
];

/// 3D gradient vectors for Perlin noise (12 edge midpoints of a cube).
pub const GRADIENTS_3D: [[f64; 3]; 12] = [
    [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, -1.0, 0.0],
    [1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 0.0, -1.0],
    [0.0, 1.0, 1.0], [0.0, -1.0, 1.0], [0.0, 1.0, -1.0], [0.0, -1.0, -1.0],
];

/// Quintic fade curve: `6t^5 - 15t^4 + 10t^3`. Smooth first and second derivatives.
#[inline]
pub fn fade(t: f64) -> f64 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// Linear interpolation between `a` and `b` with parameter `t`.
#[inline]
pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + t * (b - a)
}

/// Dot product of a 2D gradient and an offset vector.
#[inline]
pub fn dot2(g: &[f64; 2], x: f64, y: f64) -> f64 {
    g[0] * x + g[1] * y
}

/// Dot product of a 3D gradient and an offset vector.
#[inline]
pub fn dot3(g: &[f64; 3], x: f64, y: f64, z: f64) -> f64 {
    g[0] * x + g[1] * y + g[2] * z
}

/// Floor a float to `i32`.
#[inline]
pub fn floori(x: f64) -> i32 {
    x.floor() as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_permutation_deterministic() {
        let t1 = PermutationTable::new(42);
        let t2 = PermutationTable::new(42);
        assert_eq!(t1.perm, t2.perm);
    }

    #[test]
    fn test_permutation_different_seeds() {
        let t1 = PermutationTable::new(42);
        let t2 = PermutationTable::new(99);
        assert_ne!(t1.perm, t2.perm);
    }

    #[test]
    fn test_fade_bounds() {
        assert!((fade(0.0)).abs() < 1e-12);
        assert!((fade(1.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_fade_derivative_zero_at_bounds() {
        // derivative of fade is 30t^4 - 60t^3 + 30t^2
        let d = |t: f64| 30.0 * t * t * t * t - 60.0 * t * t * t + 30.0 * t * t;
        assert!(d(0.0).abs() < 1e-12);
        assert!(d(1.0).abs() < 1e-12);
    }

    #[test]
    fn test_lerp_endpoints() {
        assert!((lerp(2.0, 5.0, 0.0) - 2.0).abs() < 1e-12);
        assert!((lerp(2.0, 5.0, 1.0) - 5.0).abs() < 1e-12);
        assert!((lerp(2.0, 5.0, 0.5) - 3.5).abs() < 1e-12);
    }

    #[test]
    fn test_floori() {
        assert_eq!(floori(1.7), 1);
        assert_eq!(floori(-0.3), -1);
        assert_eq!(floori(0.0), 0);
    }
}
