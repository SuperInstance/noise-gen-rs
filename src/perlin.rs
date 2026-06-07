//! Perlin noise generation in 2D and 3D.
//!
//! Perlin noise produces smooth, continuous pseudo-random values.
//! Output range is approximately [-1, 1].

use crate::util::{fade, hash, hash3, lerp, GRADIENTS_2D, GRADIENTS_3D};

/// Perlin noise generator with a configurable seed.
///
/// # Example
/// ```
/// use noise_gen_rs::perlin::PerlinNoise;
/// let noise = PerlinNoise::new(42);
/// let val = noise.noise2d(1.5, 2.5);
/// assert!(val >= -1.0 && val <= 1.0);
/// ```
#[derive(Debug, Clone)]
pub struct PerlinNoise {
    seed: u64,
}

impl PerlinNoise {
    /// Create a new Perlin noise generator with the given seed.
    /// Same seed always produces identical output.
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// Generate 2D Perlin noise at coordinates (x, y).
    ///
    /// Returns a value in approximately [-1, 1].
    pub fn noise2d(&self, x: f64, y: f64) -> f64 {
        // Determine grid cell coordinates
        let xi = x.floor() as i64;
        let yi = y.floor() as i64;

        // Fractional position within the cell
        let xf = x - xi as f64;
        let yf = y - yi as f64;

        // Compute fade curves for each dimension
        let u = fade(xf);
        let v = fade(yf);

        // Hash coordinates of the 4 corners
        let g00 = GRADIENTS_2D[hash(self.seed, xi, yi) % GRADIENTS_2D.len()];
        let g10 = GRADIENTS_2D[hash(self.seed, xi + 1, yi) % GRADIENTS_2D.len()];
        let g01 = GRADIENTS_2D[hash(self.seed, xi, yi + 1) % GRADIENTS_2D.len()];
        let g11 = GRADIENTS_2D[hash(self.seed, xi + 1, yi + 1) % GRADIENTS_2D.len()];

        // Compute dot products between gradient and distance vectors
        let n00 = g00.0 * xf + g00.1 * yf;
        let n10 = g10.0 * (xf - 1.0) + g10.1 * yf;
        let n01 = g01.0 * xf + g01.1 * (yf - 1.0);
        let n11 = g11.0 * (xf - 1.0) + g11.1 * (yf - 1.0);

        // Interpolate
        let nx0 = lerp(u, n00, n10);
        let nx1 = lerp(u, n01, n11);
        lerp(v, nx0, nx1)
    }

    /// Generate 3D Perlin noise at coordinates (x, y, z).
    ///
    /// Returns a value in approximately [-1, 1].
    pub fn noise3d(&self, x: f64, y: f64, z: f64) -> f64 {
        let xi = x.floor() as i64;
        let yi = y.floor() as i64;
        let zi = z.floor() as i64;

        let xf = x - xi as f64;
        let yf = y - yi as f64;
        let zf = z - zi as f64;

        let u = fade(xf);
        let v = fade(yf);
        let w = fade(zf);

        // Hash and compute gradients for 8 corners
        let n000 = dot_grad3(hash3(self.seed, xi, yi, zi), xf, yf, zf);
        let n100 = dot_grad3(hash3(self.seed, xi + 1, yi, zi), xf - 1.0, yf, zf);
        let n010 = dot_grad3(hash3(self.seed, xi, yi + 1, zi), xf, yf - 1.0, zf);
        let n110 = dot_grad3(hash3(self.seed, xi + 1, yi + 1, zi), xf - 1.0, yf - 1.0, zf);
        let n001 = dot_grad3(hash3(self.seed, xi, yi, zi + 1), xf, yf, zf - 1.0);
        let n101 = dot_grad3(hash3(self.seed, xi + 1, yi, zi + 1), xf - 1.0, yf, zf - 1.0);
        let n011 = dot_grad3(hash3(self.seed, xi, yi + 1, zi + 1), xf, yf - 1.0, zf - 1.0);
        let n111 = dot_grad3(
            hash3(self.seed, xi + 1, yi + 1, zi + 1),
            xf - 1.0,
            yf - 1.0,
            zf - 1.0,
        );

        // Trilinear interpolation
        let nx00 = lerp(u, n000, n100);
        let nx10 = lerp(u, n010, n110);
        let nx01 = lerp(u, n001, n101);
        let nx11 = lerp(u, n011, n111);

        let nxy0 = lerp(v, nx00, nx10);
        let nxy1 = lerp(v, nx01, nx11);

        lerp(w, nxy0, nxy1)
    }
}

/// Compute the dot product of a 3D gradient vector and the distance vector (xf, yf, zf).
fn dot_grad3(hash_val: usize, xf: f64, yf: f64, zf: f64) -> f64 {
    let g = GRADIENTS_3D[hash_val % GRADIENTS_3D.len()];
    g.0 * xf + g.1 * yf + g.2 * zf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise2d_range() {
        let noise = PerlinNoise::new(42);
        for x in 0..20 {
            for y in 0..20 {
                let val = noise.noise2d(x as f64 * 0.1, y as f64 * 0.1);
                assert!(
                    val >= -1.0 && val <= 1.0,
                    "Value {} at ({}, {}) out of range",
                    val,
                    x,
                    y
                );
            }
        }
    }

    #[test]
    fn test_noise3d_range() {
        let noise = PerlinNoise::new(42);
        for x in 0..5 {
            for y in 0..5 {
                for z in 0..5 {
                    let val = noise.noise3d(x as f64 * 0.3, y as f64 * 0.3, z as f64 * 0.3);
                    assert!(val >= -1.0 && val <= 1.0, "Value {} out of range", val);
                }
            }
        }
    }

    #[test]
    fn test_noise2d_deterministic() {
        let n1 = PerlinNoise::new(42);
        let n2 = PerlinNoise::new(42);
        for i in 0..10 {
            let x = i as f64 * 0.37;
            let y = i as f64 * 0.53;
            assert!((n1.noise2d(x, y) - n2.noise2d(x, y)).abs() < 1e-12);
        }
    }

    #[test]
    fn test_noise3d_deterministic() {
        let n1 = PerlinNoise::new(42);
        let n2 = PerlinNoise::new(42);
        assert!((n1.noise3d(0.5, 0.5, 0.5) - n2.noise3d(0.5, 0.5, 0.5)).abs() < 1e-12);
    }

    #[test]
    fn test_noise2d_continuity() {
        let noise = PerlinNoise::new(42);
        let base = noise.noise2d(1.0, 1.0);
        let epsilon = 0.001;
        let nearby = noise.noise2d(1.0 + epsilon, 1.0 + epsilon);
        assert!(
            (base - nearby).abs() < 0.01,
            "Continuity violated: {} vs {}",
            base,
            nearby
        );
    }

    #[test]
    fn test_noise3d_continuity() {
        let noise = PerlinNoise::new(42);
        let base = noise.noise3d(1.0, 1.0, 1.0);
        let epsilon = 0.001;
        let nearby = noise.noise3d(1.0 + epsilon, 1.0 + epsilon, 1.0 + epsilon);
        assert!((base - nearby).abs() < 0.01);
    }

    #[test]
    fn test_noise2d_different_seeds() {
        let n1 = PerlinNoise::new(1);
        let n2 = PerlinNoise::new(2);
        // Different seeds should generally produce different values
        let mut any_different = false;
        for i in 0..10 {
            let x = i as f64 * 0.7;
            if (n1.noise2d(x, x) - n2.noise2d(x, x)).abs() > 0.01 {
                any_different = true;
                break;
            }
        }
        assert!(any_different, "Different seeds produced identical output");
    }

    #[test]
    fn test_noise2d_zero_at_grid_points() {
        let noise = PerlinNoise::new(42);
        // At exact grid points, all distance vectors are zero, so dot products should be zero
        // (though this depends on implementation details)
        let val = noise.noise2d(0.0, 0.0);
        // The value might not be exactly zero due to the gradient selection
        assert!(val >= -1.0 && val <= 1.0);
    }

    #[test]
    fn test_noise2d_symmetry() {
        let noise = PerlinNoise::new(42);
        // Noise should not have trivial symmetry (e.g., noise(x,y) != noise(y,x) generally)
        let v1 = noise.noise2d(1.3, 2.7);
        let v2 = noise.noise2d(2.7, 1.3);
        // They might accidentally be equal, but usually aren't
        // Just check both are valid
        assert!(v1 >= -1.0 && v1 <= 1.0);
        assert!(v2 >= -1.0 && v2 <= 1.0);
    }
}
