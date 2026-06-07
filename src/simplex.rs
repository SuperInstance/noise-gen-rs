//! Simplex noise in 2D and 3D.

use crate::util::{PermutationTable, GRADIENTS_2D, GRADIENTS_3D, dot2, dot3};

/// F2 = 0.5 * (sqrt(3) - 1), the skewing factor for 2D simplex.
const F2: f64 = 0.3660254037844386;
/// G2 = (3 - sqrt(3)) / 6, the unskewing factor for 2D simplex.
const G2: f64 = 0.21132486540518713;

/// F3 = 1/3, the skewing factor for 3D simplex.
const F3: f64 = 1.0 / 3.0;
/// G3 = 1/6, the unskewing factor for 3D simplex.
const G3: f64 = 1.0 / 6.0;

/// Simplex noise generator with a configurable seed.
pub struct SimplexNoise {
    perm: PermutationTable,
}

impl SimplexNoise {
    /// Create a new simplex noise generator seeded with `seed`.
    pub fn new(seed: u64) -> Self {
        Self {
            perm: PermutationTable::new(seed),
        }
    }

    /// Evaluate 2D simplex noise at `(x, y)`.
    pub fn noise2d(&self, x: f64, y: f64) -> f64 {
        // Skew input space
        let s = (x + y) * F2;
        let i = (x + s).floor() as i32;
        let j = (y + s).floor() as i32;

        let t = (i + j) as f64 * G2;
        let x0 = x - (i as f64 - t);
        let y0 = y - (j as f64 - t);

        let (i1, j1, i2, j2) = if x0 > y0 {
            (1, 0, 1, 1)
        } else {
            (0, 1, 1, 1)
        };

        let x1 = x0 - i1 as f64 + G2;
        let y1 = y0 - j1 as f64 + G2;
        let x2 = x0 - i2 as f64 + 2.0 * G2;
        let y2 = y0 - j2 as f64 + 2.0 * G2;

        let mut n = 0.0;

        // Contribution from corner 0
        let t0 = 0.5 - x0 * x0 - y0 * y0;
        if t0 > 0.0 {
            let gi = self.perm.hash2(i, j) as usize % 12;
            n += t0 * t0 * t0 * t0 * dot2(&GRADIENTS_2D[gi], x0, y0);
        }

        // Contribution from corner 1
        let t1 = 0.5 - x1 * x1 - y1 * y1;
        if t1 > 0.0 {
            let gi = self.perm.hash2(i + i1, j + j1) as usize % 12;
            n += t1 * t1 * t1 * t1 * dot2(&GRADIENTS_2D[gi], x1, y1);
        }

        // Contribution from corner 2
        let t2 = 0.5 - x2 * x2 - y2 * y2;
        if t2 > 0.0 {
            let gi = self.perm.hash2(i + i2, j + j2) as usize % 12;
            n += t2 * t2 * t2 * t2 * dot2(&GRADIENTS_2D[gi], x2, y2);
        }

        // Scale to [-1, 1]
        n * 70.0
    }

    /// Evaluate 3D simplex noise at `(x, y, z)`.
    pub fn noise3d(&self, x: f64, y: f64, z: f64) -> f64 {
        let s = (x + y + z) * F3;
        let i = (x + s).floor() as i32;
        let j = (y + s).floor() as i32;
        let k = (z + s).floor() as i32;

        let t = (i + j + k) as f64 * G3;
        let x0 = x - (i as f64 - t);
        let y0 = y - (j as f64 - t);
        let z0 = z - (k as f64 - t);

        // Determine simplex
        let (i1, j1, k1, i2, j2, k2) = if x0 >= y0 {
            if y0 >= z0 {
                (1, 0, 0, 1, 1, 0)
            } else if x0 >= z0 {
                (1, 0, 0, 1, 0, 1)
            } else {
                (0, 0, 1, 1, 0, 1)
            }
        } else if y0 < z0 {
            (0, 0, 1, 0, 1, 1)
        } else {
            (0, 1, 0, 0, 1, 1)
        };

        let x1 = x0 - i1 as f64 + G3;
        let y1 = y0 - j1 as f64 + G3;
        let z1 = z0 - k1 as f64 + G3;
        let x2 = x0 - i2 as f64 + 2.0 * G3;
        let y2 = y0 - j2 as f64 + 2.0 * G3;
        let z2 = z0 - k2 as f64 + 2.0 * G3;
        let x3 = x0 - 1.0 + 3.0 * G3;
        let y3 = y0 - 1.0 + 3.0 * G3;
        let z3 = z0 - 1.0 + 3.0 * G3;

        let mut n = 0.0;

        let t0 = 0.6 - x0 * x0 - y0 * y0 - z0 * z0;
        if t0 > 0.0 {
            let gi = self.perm.hash3(i, j, k) as usize % 12;
            n += t0 * t0 * t0 * t0 * dot3(&GRADIENTS_3D[gi], x0, y0, z0);
        }

        let t1 = 0.6 - x1 * x1 - y1 * y1 - z1 * z1;
        if t1 > 0.0 {
            let gi = self.perm.hash3(i + i1, j + j1, k + k1) as usize % 12;
            n += t1 * t1 * t1 * t1 * dot3(&GRADIENTS_3D[gi], x1, y1, z1);
        }

        let t2 = 0.6 - x2 * x2 - y2 * y2 - z2 * z2;
        if t2 > 0.0 {
            let gi = self.perm.hash3(i + i2, j + j2, k + k2) as usize % 12;
            n += t2 * t2 * t2 * t2 * dot3(&GRADIENTS_3D[gi], x2, y2, z2);
        }

        let t3 = 0.6 - x3 * x3 - y3 * y3 - z3 * z3;
        if t3 > 0.0 {
            let gi = self.perm.hash3(i + 1, j + 1, k + 1) as usize % 12;
            n += t3 * t3 * t3 * t3 * dot3(&GRADIENTS_3D[gi], x3, y3, z3);
        }

        n * 32.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simplex_2d_deterministic() {
        let n = SimplexNoise::new(42);
        assert!((n.noise2d(1.5, 2.5) - n.noise2d(1.5, 2.5)).abs() < 1e-12);
    }

    #[test]
    fn test_simplex_2d_range() {
        let n = SimplexNoise::new(42);
        for x in 0..20 {
            for y in 0..20 {
                let v = n.noise2d(x as f64 * 0.3, y as f64 * 0.3);
                assert!(v >= -1.2 && v <= 1.2, "simplex 2D out of range: {v}");
            }
        }
    }

    #[test]
    fn test_simplex_2d_continuity() {
        let n = SimplexNoise::new(42);
        let v1 = n.noise2d(1.0, 1.0);
        let v2 = n.noise2d(1.001, 1.001);
        assert!((v1 - v2).abs() < 0.05, "simplex 2D discontinuous: {v1} vs {v2}");
    }

    #[test]
    fn test_simplex_2d_different_seeds() {
        let n1 = SimplexNoise::new(42);
        let n2 = SimplexNoise::new(99);
        assert!((n1.noise2d(1.5, 2.5) - n2.noise2d(1.5, 2.5)).abs() > 0.01);
    }

    #[test]
    fn test_simplex_3d_deterministic() {
        let n = SimplexNoise::new(42);
        assert!((n.noise3d(1.5, 2.5, 3.5) - n.noise3d(1.5, 2.5, 3.5)).abs() < 1e-12);
    }

    #[test]
    fn test_simplex_3d_range() {
        let n = SimplexNoise::new(42);
        for x in 0..10 {
            for y in 0..10 {
                for z in 0..5 {
                    let v = n.noise3d(x as f64 * 0.4, y as f64 * 0.4, z as f64 * 0.4);
                    assert!(v >= -1.2 && v <= 1.2, "simplex 3D out of range: {v}");
                }
            }
        }
    }

    #[test]
    fn test_simplex_3d_continuity() {
        let n = SimplexNoise::new(42);
        let v1 = n.noise3d(1.0, 2.0, 3.0);
        let v2 = n.noise3d(1.001, 2.001, 3.001);
        assert!((v1 - v2).abs() < 0.05);
    }

    #[test]
    fn test_simplex_3d_different_seeds() {
        let n1 = SimplexNoise::new(42);
        let n2 = SimplexNoise::new(99);
        // Check at multiple points — different seeds should differ somewhere
        let mut differ = false;
        for i in 0..5 {
            let x = 1.5 + i as f64 * 0.3;
            if (n1.noise3d(x, 2.5, 3.5) - n2.noise3d(x, 2.5, 3.5)).abs() > 0.01 {
                differ = true;
                break;
            }
        }
        assert!(differ, "different seeds produced identical 3D output");
    }

    #[test]
    fn test_simplex_2d_zero_at_origin() {
        let n = SimplexNoise::new(42);
        let v = n.noise2d(0.0, 0.0);
        assert!(v.abs() < 1e-10, "expected zero at origin, got {v}");
    }
}
