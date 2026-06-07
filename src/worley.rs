//! Worley (Voronoi) cellular noise in 2D and 3D.

use crate::util::{PermutationTable, floori};

/// Distance metric for Worley noise.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DistanceMetric {
    /// Euclidean distance (L2 norm).
    Euclidean,
    /// Manhattan distance (L1 norm).
    Manhattan,
    /// Chebyshev distance (L∞ norm).
    Chebyshev,
}

impl DistanceMetric {
    fn dist2(&self, dx: f64, dy: f64) -> f64 {
        match self {
            DistanceMetric::Euclidean => (dx * dx + dy * dy).sqrt(),
            DistanceMetric::Manhattan => dx.abs() + dy.abs(),
            DistanceMetric::Chebyshev => dx.abs().max(dy.abs()),
        }
    }

    fn dist3(&self, dx: f64, dy: f64, dz: f64) -> f64 {
        match self {
            DistanceMetric::Euclidean => (dx * dx + dy * dy + dz * dz).sqrt(),
            DistanceMetric::Manhattan => dx.abs() + dy.abs() + dz.abs(),
            DistanceMetric::Chebyshev => dx.abs().max(dy.abs()).max(dz.abs()),
        }
    }
}

/// Worley (cellular/Voronoi) noise generator.
pub struct WorleyNoise {
    perm: PermutationTable,
    metric: DistanceMetric,
}

impl WorleyNoise {
    /// Create with a seed and default Euclidean distance.
    pub fn new(seed: u64) -> Self {
        Self {
            perm: PermutationTable::new(seed),
            metric: DistanceMetric::Euclidean,
        }
    }

    /// Set the distance metric.
    pub fn with_metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Pseudo-random value in [0, 1) derived from a hash.
    #[inline]
    fn rand2(&self, x: i32, y: i32) -> f64 {
        let h = self.perm.hash2(x, y);
        h as f64 / 256.0
    }

    #[inline]
    fn rand3(&self, x: i32, y: i32, z: i32) -> f64 {
        let h = self.perm.hash3(x, y, z);
        h as f64 / 256.0
    }

    /// Evaluate 2D Worley noise — returns `(F1, F2)` distances to nearest and second-nearest feature points.
    pub fn noise2d(&self, x: f64, y: f64) -> (f64, f64) {
        let xi = floori(x);
        let yi = floori(y);

        let mut f1 = f64::MAX;
        let mut f2 = f64::MAX;

        for dx in -1..=1 {
            for dy in -1..=1 {
                let cx = xi + dx;
                let cy = yi + dy;
                let px = cx as f64 + self.rand2(cx, cy);
                let py = cy as f64 + self.rand2(cx + 317, cy + 101);
                let dist = self.metric.dist2(x - px, y - py);
                if dist < f1 {
                    f2 = f1;
                    f1 = dist;
                } else if dist < f2 {
                    f2 = dist;
                }
            }
        }

        (f1, f2)
    }

    /// Evaluate 3D Worley noise — returns `(F1, F2)`.
    pub fn noise3d(&self, x: f64, y: f64, z: f64) -> (f64, f64) {
        let xi = floori(x);
        let yi = floori(y);
        let zi = floori(z);

        let mut f1 = f64::MAX;
        let mut f2 = f64::MAX;

        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let cx = xi + dx;
                    let cy = yi + dy;
                    let cz = zi + dz;
                    let px = cx as f64 + self.rand3(cx, cy, cz);
                    let py = cy as f64 + self.rand3(cx + 317, cy + 101, cz + 57);
                    let pz = cz as f64 + self.rand3(cx + 79, cy + 211, cz + 173);
                    let dist = self.metric.dist3(x - px, y - py, z - pz);
                    if dist < f1 {
                        f2 = f1;
                        f1 = dist;
                    } else if dist < f2 {
                        f2 = dist;
                    }
                }
            }
        }

        (f1, f2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worley_2d_deterministic() {
        let n = WorleyNoise::new(42);
        let (a, b) = n.noise2d(1.5, 2.5);
        let (c, d) = n.noise2d(1.5, 2.5);
        assert!((a - c).abs() < 1e-12);
        assert!((b - d).abs() < 1e-12);
    }

    #[test]
    fn test_worley_2d_positive_distances() {
        let n = WorleyNoise::new(42);
        for x in 0..10 {
            for y in 0..10 {
                let (f1, f2) = n.noise2d(x as f64 * 0.5, y as f64 * 0.5);
                assert!(f1 >= 0.0, "F1 negative: {f1}");
                assert!(f2 >= f1, "F2 < F1: {f2} < {f1}");
            }
        }
    }

    #[test]
    fn test_worley_2d_f1_small() {
        let n = WorleyNoise::new(42);
        let (f1, _) = n.noise2d(1.5, 2.5);
        assert!(f1 < 2.0, "F1 unexpectedly large: {f1}");
    }

    #[test]
    fn test_worley_3d_deterministic() {
        let n = WorleyNoise::new(42);
        let (a, b) = n.noise3d(1.5, 2.5, 3.5);
        let (c, d) = n.noise3d(1.5, 2.5, 3.5);
        assert!((a - c).abs() < 1e-12);
        assert!((b - d).abs() < 1e-12);
    }

    #[test]
    fn test_worley_3d_positive_distances() {
        let n = WorleyNoise::new(42);
        let (f1, f2) = n.noise3d(1.5, 2.5, 3.5);
        assert!(f1 >= 0.0);
        assert!(f2 >= f1);
    }

    #[test]
    fn test_worley_2d_manhattan() {
        let n = WorleyNoise::new(42).with_metric(DistanceMetric::Manhattan);
        let (f1, f2) = n.noise2d(1.5, 2.5);
        assert!(f1 >= 0.0);
        assert!(f2 >= f1);
    }

    #[test]
    fn test_worley_2d_chebyshev() {
        let n = WorleyNoise::new(42).with_metric(DistanceMetric::Chebyshev);
        let (f1, f2) = n.noise2d(1.5, 2.5);
        assert!(f1 >= 0.0);
        assert!(f2 >= f1);
    }

    #[test]
    fn test_worley_2d_different_seeds() {
        let n1 = WorleyNoise::new(42);
        let n2 = WorleyNoise::new(99);
        let (f1_a, _) = n1.noise2d(1.5, 2.5);
        let (f1_b, _) = n2.noise2d(1.5, 2.5);
        assert!((f1_a - f1_b).abs() > 0.001);
    }
}
