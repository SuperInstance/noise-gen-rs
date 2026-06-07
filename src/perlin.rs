//! Perlin gradient noise in 2D and 3D.

use crate::util::{
    PermutationTable, GRADIENTS_2D, GRADIENTS_3D, dot2, dot3, fade, floori, lerp,
};

/// Perlin noise generator with a configurable seed.
///
/// Produces smooth, continuous noise values in approximately `[-1, 1]`.
pub struct PerlinNoise {
    perm: PermutationTable,
}

impl PerlinNoise {
    /// Create a new Perlin noise generator with the given seed.
    pub fn new(seed: u64) -> Self {
        Self {
            perm: PermutationTable::new(seed),
        }
    }

    /// Evaluate 2D Perlin noise at `(x, y)`.
    pub fn noise2d(&self, x: f64, y: f64) -> f64 {
        let xi = floori(x);
        let yi = floori(y);
        let xf = x - xi as f64;
        let yf = y - yi as f64;

        let u = fade(xf);
        let v = fade(yf);

        let g00 = &GRADIENTS_2D[self.perm.hash2(xi, yi) as usize % 12];
        let g10 = &GRADIENTS_2D[self.perm.hash2(xi + 1, yi) as usize % 12];
        let g01 = &GRADIENTS_2D[self.perm.hash2(xi, yi + 1) as usize % 12];
        let g11 = &GRADIENTS_2D[self.perm.hash2(xi + 1, yi + 1) as usize % 12];

        let n00 = dot2(g00, xf, yf);
        let n10 = dot2(g10, xf - 1.0, yf);
        let n01 = dot2(g01, xf, yf - 1.0);
        let n11 = dot2(g11, xf - 1.0, yf - 1.0);

        let nx0 = lerp(n00, n10, u);
        let nx1 = lerp(n01, n11, u);
        lerp(nx0, nx1, v)
    }

    /// Evaluate 3D Perlin noise at `(x, y, z)`.
    pub fn noise3d(&self, x: f64, y: f64, z: f64) -> f64 {
        let xi = floori(x);
        let yi = floori(y);
        let zi = floori(z);
        let xf = x - xi as f64;
        let yf = y - yi as f64;
        let zf = z - zi as f64;

        let u = fade(xf);
        let v = fade(yf);
        let w = fade(zf);

        let g000 = &GRADIENTS_3D[self.perm.hash3(xi, yi, zi) as usize % 12];
        let g100 = &GRADIENTS_3D[self.perm.hash3(xi + 1, yi, zi) as usize % 12];
        let g010 = &GRADIENTS_3D[self.perm.hash3(xi, yi + 1, zi) as usize % 12];
        let g110 = &GRADIENTS_3D[self.perm.hash3(xi + 1, yi + 1, zi) as usize % 12];
        let g001 = &GRADIENTS_3D[self.perm.hash3(xi, yi, zi + 1) as usize % 12];
        let g101 = &GRADIENTS_3D[self.perm.hash3(xi + 1, yi, zi + 1) as usize % 12];
        let g011 = &GRADIENTS_3D[self.perm.hash3(xi, yi + 1, zi + 1) as usize % 12];
        let g111 = &GRADIENTS_3D[self.perm.hash3(xi + 1, yi + 1, zi + 1) as usize % 12];

        let n000 = dot3(g000, xf, yf, zf);
        let n100 = dot3(g100, xf - 1.0, yf, zf);
        let n010 = dot3(g010, xf, yf - 1.0, zf);
        let n110 = dot3(g110, xf - 1.0, yf - 1.0, zf);
        let n001 = dot3(g001, xf, yf, zf - 1.0);
        let n101 = dot3(g101, xf - 1.0, yf, zf - 1.0);
        let n011 = dot3(g011, xf, yf - 1.0, zf - 1.0);
        let n111 = dot3(g111, xf - 1.0, yf - 1.0, zf - 1.0);

        let nx00 = lerp(n000, n100, u);
        let nx10 = lerp(n010, n110, u);
        let nx01 = lerp(n001, n101, u);
        let nx11 = lerp(n011, n111, u);

        let nxy0 = lerp(nx00, nx10, v);
        let nxy1 = lerp(nx01, nx11, v);

        lerp(nxy0, nxy1, w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perlin_2d_deterministic() {
        let n = PerlinNoise::new(42);
        let v1 = n.noise2d(1.5, 2.5);
        let v2 = n.noise2d(1.5, 2.5);
        assert!((v1 - v2).abs() < 1e-12);
    }

    #[test]
    fn test_perlin_2d_range() {
        let n = PerlinNoise::new(42);
        for x in 0..20 {
            for y in 0..20 {
                let v = n.noise2d(x as f64 * 0.3, y as f64 * 0.3);
                assert!(v >= -1.1 && v <= 1.1, "value {v} out of range at ({x},{y})");
            }
        }
    }

    #[test]
    fn test_perlin_2d_continuity() {
        let n = PerlinNoise::new(42);
        let v1 = n.noise2d(1.0, 1.0);
        let v2 = n.noise2d(1.001, 1.001);
        assert!((v1 - v2).abs() < 0.05, "discontinuous: {v1} vs {v2}");
    }

    #[test]
    fn test_perlin_2d_different_seeds() {
        let n1 = PerlinNoise::new(42);
        let n2 = PerlinNoise::new(99);
        let v1 = n1.noise2d(1.5, 2.5);
        let v2 = n2.noise2d(1.5, 2.5);
        assert!((v1 - v2).abs() > 0.01, "different seeds produced same value");
    }

    #[test]
    fn test_perlin_2d_zero_at_grid() {
        let n = PerlinNoise::new(42);
        let v = n.noise2d(0.0, 0.0);
        assert!(v.abs() < 1e-10, "expected zero at origin, got {v}");
    }

    #[test]
    fn test_perlin_3d_deterministic() {
        let n = PerlinNoise::new(42);
        let v1 = n.noise3d(1.5, 2.5, 3.5);
        let v2 = n.noise3d(1.5, 2.5, 3.5);
        assert!((v1 - v2).abs() < 1e-12);
    }

    #[test]
    fn test_perlin_3d_range() {
        let n = PerlinNoise::new(42);
        for x in 0..10 {
            for y in 0..10 {
                for z in 0..5 {
                    let v = n.noise3d(x as f64 * 0.4, y as f64 * 0.4, z as f64 * 0.4);
                    assert!(v >= -1.1 && v <= 1.1, "3D value {v} out of range");
                }
            }
        }
    }

    #[test]
    fn test_perlin_3d_continuity() {
        let n = PerlinNoise::new(42);
        let v1 = n.noise3d(1.0, 2.0, 3.0);
        let v2 = n.noise3d(1.001, 2.001, 3.001);
        assert!((v1 - v2).abs() < 0.05, "3D discontinuous: {v1} vs {v2}");
    }

    #[test]
    fn test_perlin_3d_different_seeds() {
        let n1 = PerlinNoise::new(42);
        let n2 = PerlinNoise::new(99);
        let v1 = n1.noise3d(1.5, 2.5, 3.5);
        let v2 = n2.noise3d(1.5, 2.5, 3.5);
        assert!((v1 - v2).abs() > 0.01);
    }
}
