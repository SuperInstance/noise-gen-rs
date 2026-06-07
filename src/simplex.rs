//! Simplex noise generation in 2D and 3D.
//!
//! Simplex noise is an improvement over Perlin noise that uses a simplex grid
//! instead of a rectangular grid, reducing directional artifacts and computational cost.
//! Output range is approximately [-1, 1].

use crate::util::{seeded_random, SIMPLEX_SKEW_2D, SIMPLEX_SKEW_3D, SIMPLEX_UNSKEW_2D, SIMPLEX_UNSKEW_3D};

/// Simplex noise generator with a configurable seed.
///
/// # Example
/// ```
/// use noise_gen_rs::simplex::SimplexNoise;
/// let noise = SimplexNoise::new(42);
/// let val = noise.noise2d(1.5, 2.5);
/// assert!(val >= -1.0 && val <= 1.0);
/// ```
#[derive(Debug, Clone)]
pub struct SimplexNoise {
    perm: [u8; 512],
}

impl SimplexNoise {
    /// Create a new Simplex noise generator with the given seed.
    pub fn new(seed: u64) -> Self {
        let mut perm = [0u8; 512];
        // Generate permutation table from seed
        let mut p = [0u8; 256];
        for (i, v) in p.iter_mut().enumerate() {
            *v = i as u8;
        }
        // Fisher-Yates shuffle with seeded RNG
        for i in (1..256).rev() {
            let j = (seeded_random(seed, i) * (i as f64 + 1.0)) as usize;
            let j = j.min(i);
            p.swap(i, j);
        }
        // Double the permutation table
        perm[..256].copy_from_slice(&p);
        perm[256..].copy_from_slice(&p);
        Self { perm }
    }

    /// Generate 2D Simplex noise at coordinates (x, y).
    ///
    /// Returns a value in approximately [-1, 1].
    pub fn noise2d(&self, x: f64, y: f64) -> f64 {
        // Skew input space to determine simplex cell
        let s = (x + y) * SIMPLEX_SKEW_2D;
        let i = (x + s).floor() as i64;
        let j = (y + s).floor() as i64;

        // Unskew back to find (x, y) distances from cell origin
        let t = (i + j) as f64 * SIMPLEX_UNSKEW_2D;
        let x0 = x - (i as f64 - t);
        let y0 = y - (j as f64 - t);

        // Determine which simplex we're in
        let (i1, j1, i2, j2) = if x0 > y0 {
            (1, 0, 1, 1) // Lower triangle
        } else {
            (0, 1, 1, 1) // Upper triangle
        };

        // Offsets for other two corners
        let x1 = x0 - i1 as f64 + SIMPLEX_UNSKEW_2D;
        let y1 = y0 - j1 as f64 + SIMPLEX_UNSKEW_2D;
        let x2 = x0 - i2 as f64 + 2.0 * SIMPLEX_UNSKEW_2D;
        let y2 = y0 - j2 as f64 + 2.0 * SIMPLEX_UNSKEW_2D;

        // Calculate contribution from each corner
        let n0 = self.contribution2d(x0, y0, i, j);
        let n1 = self.contribution2d(x1, y1, i + i1, j + j1);
        let n2 = self.contribution2d(x2, y2, i + i2, j + j2);

        // Scale to [-1, 1]
        70.0 * (n0 + n1 + n2)
    }

    /// Compute the contribution of a single corner in 2D.
    fn contribution2d(&self, x: f64, y: f64, i: i64, j: i64) -> f64 {
        let t = 0.5 - x * x - y * y;
        if t < 0.0 {
            0.0
        } else {
            let t = t * t;
            let gi = self.perm[(self.perm[i as usize & 255] as usize + j as usize) & 511] as usize % 4;
            let grad = match gi {
                0 => (1.0, 1.0),
                1 => (-1.0, 1.0),
                2 => (1.0, -1.0),
                _ => (-1.0, -1.0),
            };
            t * t * (grad.0 * x + grad.1 * y)
        }
    }

    /// Generate 3D Simplex noise at coordinates (x, y, z).
    ///
    /// Returns a value in approximately [-1, 1].
    pub fn noise3d(&self, x: f64, y: f64, z: f64) -> f64 {
        // Skew input space
        let s = (x + y + z) * SIMPLEX_SKEW_3D;
        let i = (x + s).floor() as i64;
        let j = (y + s).floor() as i64;
        let k = (z + s).floor() as i64;

        let t = (i + j + k) as f64 * SIMPLEX_UNSKEW_3D;
        let x0 = x - (i as f64 - t);
        let y0 = y - (j as f64 - t);
        let z0 = z - (k as f64 - t);

        // Determine simplex
        let (i1, j1, k1, i2, j2, k2) = {
            if x0 >= y0 {
                if y0 >= z0 {
                    (1, 0, 0, 1, 1, 0) // X Y Z
                } else if x0 >= z0 {
                    (1, 0, 0, 1, 0, 1) // X Z Y
                } else {
                    (0, 0, 1, 1, 0, 1) // Z X Y
                }
            } else if y0 < z0 {
                (0, 0, 1, 0, 1, 1) // Z Y X
            } else {
                (0, 1, 0, 0, 1, 1) // Y Z X
            }
        };

        let x1 = x0 - i1 as f64 + SIMPLEX_UNSKEW_3D;
        let y1 = y0 - j1 as f64 + SIMPLEX_UNSKEW_3D;
        let z1 = z0 - k1 as f64 + SIMPLEX_UNSKEW_3D;
        let x2 = x0 - i2 as f64 + 2.0 * SIMPLEX_UNSKEW_3D;
        let y2 = y0 - j2 as f64 + 2.0 * SIMPLEX_UNSKEW_3D;
        let z2 = z0 - k2 as f64 + 2.0 * SIMPLEX_UNSKEW_3D;
        let x3 = x0 - 1.0 + 3.0 * SIMPLEX_UNSKEW_3D;
        let y3 = y0 - 1.0 + 3.0 * SIMPLEX_UNSKEW_3D;
        let z3 = z0 - 1.0 + 3.0 * SIMPLEX_UNSKEW_3D;

        let ii = i as usize & 255;
        let jj = j as usize & 255;
        let kk = k as usize & 255;

        let n0 = self.contribution3d(x0, y0, z0, ii, jj, kk, 0, 0, 0);
        let n1 = self.contribution3d(x1, y1, z1, ii, jj, kk, i1, j1, k1);
        let n2 = self.contribution3d(x2, y2, z2, ii, jj, kk, i2, j2, k2);
        let n3 = self.contribution3d(x3, y3, z3, ii, jj, kk, 1, 1, 1);

        32.0 * (n0 + n1 + n2 + n3)
    }

    /// Compute the contribution of a single corner in 3D.
    #[allow(clippy::too_many_arguments)]
    fn contribution3d(
        &self,
        x: f64,
        y: f64,
        z: f64,
        ii: usize,
        jj: usize,
        kk: usize,
        di: i64,
        dj: i64,
        dk: i64,
    ) -> f64 {
        let t = 0.6 - x * x - y * y - z * z;
        if t < 0.0 {
            0.0
        } else {
            let t = t * t;
            let pi = self.perm[(ii + di as usize) & 255] as usize;
            let pj = (pi + jj + dj as usize) & 255;
            let gi = self.perm[(pj + kk + dk as usize) & 255] as usize % 12;
            let (gx, gy, gz) = match gi {
                0 => (1.0, 1.0, 0.0),
                1 => (-1.0, 1.0, 0.0),
                2 => (1.0, -1.0, 0.0),
                3 => (-1.0, -1.0, 0.0),
                4 => (1.0, 0.0, 1.0),
                5 => (-1.0, 0.0, 1.0),
                6 => (1.0, 0.0, -1.0),
                7 => (-1.0, 0.0, -1.0),
                8 => (0.0, 1.0, 1.0),
                9 => (0.0, -1.0, 1.0),
                10 => (0.0, 1.0, -1.0),
                _ => (0.0, -1.0, -1.0),
            };
            t * t * (gx * x + gy * y + gz * z)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise2d_range() {
        let noise = SimplexNoise::new(42);
        for i in 0..50 {
            let x = i as f64 * 0.13;
            let y = i as f64 * 0.17;
            let val = noise.noise2d(x, y);
            assert!(
                val >= -1.0 && val <= 1.0,
                "Value {} at ({}, {}) out of range",
                val,
                x,
                y
            );
        }
    }

    #[test]
    fn test_noise3d_range() {
        let noise = SimplexNoise::new(42);
        for i in 0..20 {
            let x = i as f64 * 0.13;
            let val = noise.noise3d(x, x, x);
            assert!(
                val >= -1.0 && val <= 1.0,
                "Value {} out of range",
                val
            );
        }
    }

    #[test]
    fn test_noise2d_deterministic() {
        let n1 = SimplexNoise::new(42);
        let n2 = SimplexNoise::new(42);
        for i in 0..10 {
            let x = i as f64 * 0.37;
            let y = i as f64 * 0.53;
            assert!(
                (n1.noise2d(x, y) - n2.noise2d(x, y)).abs() < 1e-12,
                "Not deterministic"
            );
        }
    }

    #[test]
    fn test_noise3d_deterministic() {
        let n1 = SimplexNoise::new(42);
        let n2 = SimplexNoise::new(42);
        assert!((n1.noise3d(0.5, 0.5, 0.5) - n2.noise3d(0.5, 0.5, 0.5)).abs() < 1e-12);
    }

    #[test]
    fn test_noise2d_continuity() {
        let noise = SimplexNoise::new(42);
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
        let noise = SimplexNoise::new(42);
        let base = noise.noise3d(1.0, 1.0, 1.0);
        let epsilon = 0.001;
        let nearby = noise.noise3d(1.0 + epsilon, 1.0 + epsilon, 1.0 + epsilon);
        assert!((base - nearby).abs() < 0.01);
    }

    #[test]
    fn test_noise2d_different_seeds() {
        let n1 = SimplexNoise::new(1);
        let n2 = SimplexNoise::new(2);
        let mut any_different = false;
        for i in 0..10 {
            let x = i as f64 * 0.7;
            if (n1.noise2d(x, x) - n2.noise2d(x, x)).abs() > 0.01 {
                any_different = true;
                break;
            }
        }
        assert!(any_different);
    }

    #[test]
    fn test_permutation_table() {
        let noise = SimplexNoise::new(42);
        // Permutation table should contain each value exactly twice (doubled)
        let mut counts = [0usize; 256];
        for &p in &noise.perm {
            counts[p as usize] += 1;
        }
        // Each value appears exactly twice
        for (val, &count) in counts.iter().enumerate() {
            assert_eq!(count, 2, "Value {} appears {} times, expected 2", val, count);
        }
    }
}
