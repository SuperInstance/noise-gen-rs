//! Worley (Voronoi) noise generation in 2D and 3D.
//!
//! Worley noise is based on distance to random feature points, producing
//! cellular/Voronoi-like patterns. Output represents distances, typically [0, ∞).

use crate::util::{hash, hash3, seeded_random};

/// Distance function type for Worley noise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceFunc {
    /// Euclidean distance: sqrt(dx^2 + dy^2)
    Euclidean,
    /// Manhattan distance: |dx| + |dy|
    Manhattan,
    /// Chebyshev distance: max(|dx|, |dy|)
    Chebyshev,
}

/// Worley noise generator with configurable seed and distance function.
///
/// # Example
/// ```
/// use noise_gen_rs::worley::{WorleyNoise, DistanceFunc};
/// let noise = WorleyNoise::new(42);
/// let val = noise.noise2d(0.5, 0.5);
/// assert!(val >= 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct WorleyNoise {
    seed: u64,
    distance_func: DistanceFunc,
    /// Number of feature points per cell (1-4).
    points_per_cell: usize,
}

impl WorleyNoise {
    /// Create a new Worley noise generator with the given seed.
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            distance_func: DistanceFunc::Euclidean,
            points_per_cell: 1,
        }
    }

    /// Set the distance function.
    pub fn with_distance_func(mut self, func: DistanceFunc) -> Self {
        self.distance_func = func;
        self
    }

    /// Set the number of feature points per cell.
    pub fn with_points_per_cell(mut self, n: usize) -> Self {
        self.points_per_cell = n.clamp(1, 4);
        self
    }

    /// Compute distance between two 2D points using the configured distance function.
    fn dist2d(&self, x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
        let dx = x1 - x2;
        let dy = y1 - y2;
        match self.distance_func {
            DistanceFunc::Euclidean => (dx * dx + dy * dy).sqrt(),
            DistanceFunc::Manhattan => dx.abs() + dy.abs(),
            DistanceFunc::Chebyshev => dx.abs().max(dy.abs()),
        }
    }

    /// Compute distance between two 3D points using the configured distance function.
    fn dist3d(&self, x1: f64, y1: f64, z1: f64, x2: f64, y2: f64, z2: f64) -> f64 {
        let dx = x1 - x2;
        let dy = y1 - y2;
        let dz = z1 - z2;
        match self.distance_func {
            DistanceFunc::Euclidean => (dx * dx + dy * dy + dz * dz).sqrt(),
            DistanceFunc::Manhattan => dx.abs() + dy.abs() + dz.abs(),
            DistanceFunc::Chebyshev => dx.abs().max(dy.abs()).max(dz.abs()),
        }
    }

    /// Generate feature points for a given cell.
    fn feature_points2d(&self, cx: i64, cy: i64) -> Vec<(f64, f64)> {
        let base = (cx, cy);
        let count = self.points_per_cell;
        let mut points = Vec::with_capacity(count);
        for k in 0..count {
            let idx = hash(self.seed, cx, cy).wrapping_add(k);
            let px = seeded_random(self.seed, idx * 2) + cx as f64;
            let py = seeded_random(self.seed, idx * 2 + 1) + cy as f64;
            points.push((px, py));
        }
        let _ = base; // used via cx, cy in hash/seeded_random
        points
    }

    /// Generate 2D Worley noise (F1 - distance to nearest feature point).
    ///
    /// Returns a non-negative distance value.
    pub fn noise2d(&self, x: f64, y: f64) -> f64 {
        self.noise2d_f1(x, y)
    }

    /// F1: Distance to the nearest feature point.
    pub fn noise2d_f1(&self, x: f64, y: f64) -> f64 {
        let mut min_dist = f64::MAX;
        let cx = x.floor() as i64;
        let cy = y.floor() as i64;

        // Check 3x3 neighborhood
        for dx in -1..=1 {
            for dy in -1..=1 {
                let cell_x = cx + dx;
                let cell_y = cy + dy;
                for &(px, py) in &self.feature_points2d(cell_x, cell_y) {
                    let d = self.dist2d(x, y, px, py);
                    if d < min_dist {
                        min_dist = d;
                    }
                }
            }
        }
        min_dist
    }

    /// F2: Distance to the second nearest feature point.
    pub fn noise2d_f2(&self, x: f64, y: f64) -> f64 {
        let mut d1 = f64::MAX;
        let mut d2 = f64::MAX;
        let cx = x.floor() as i64;
        let cy = y.floor() as i64;

        for dx in -1..=1 {
            for dy in -1..=1 {
                let cell_x = cx + dx;
                let cell_y = cy + dy;
                for &(px, py) in &self.feature_points2d(cell_x, cell_y) {
                    let d = self.dist2d(x, y, px, py);
                    if d < d1 {
                        d2 = d1;
                        d1 = d;
                    } else if d < d2 {
                        d2 = d;
                    }
                }
            }
        }
        d2
    }

    /// Generate 3D Worley noise (F1).
    pub fn noise3d(&self, x: f64, y: f64, z: f64) -> f64 {
        let mut min_dist = f64::MAX;
        let cx = x.floor() as i64;
        let cy = y.floor() as i64;
        let cz = z.floor() as i64;

        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let cell_x = cx + dx;
                    let cell_y = cy + dy;
                    let cell_z = cz + dz;
                    let count = self.points_per_cell;
                    for k in 0..count {
                        let idx = hash3(self.seed, cell_x, cell_y, cell_z).wrapping_add(k);
                        let px = seeded_random(self.seed, idx * 3) + cell_x as f64;
                        let py = seeded_random(self.seed, idx * 3 + 1) + cell_y as f64;
                        let pz = seeded_random(self.seed, idx * 3 + 2) + cell_z as f64;
                        let d = self.dist3d(x, y, z, px, py, pz);
                        if d < min_dist {
                            min_dist = d;
                        }
                    }
                }
            }
        }
        min_dist
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worley2d_non_negative() {
        let noise = WorleyNoise::new(42);
        for i in 0..20 {
            let x = i as f64 * 0.37;
            let y = i as f64 * 0.53;
            let val = noise.noise2d(x, y);
            assert!(val >= 0.0, "Negative value {} at ({}, {})", val, x, y);
        }
    }

    #[test]
    fn test_worley3d_non_negative() {
        let noise = WorleyNoise::new(42);
        for i in 0..10 {
            let x = i as f64 * 0.37;
            let val = noise.noise3d(x, x, x);
            assert!(val >= 0.0, "Negative value {}", val);
        }
    }

    #[test]
    fn test_worley2d_deterministic() {
        let n1 = WorleyNoise::new(42);
        let n2 = WorleyNoise::new(42);
        for i in 0..10 {
            let x = i as f64 * 0.37;
            assert!((n1.noise2d(x, x) - n2.noise2d(x, x)).abs() < 1e-12);
        }
    }

    #[test]
    fn test_worley_f2_ge_f1() {
        let noise = WorleyNoise::new(42);
        for i in 0..20 {
            let x = i as f64 * 0.31;
            let y = i as f64 * 0.47;
            let f1 = noise.noise2d_f1(x, y);
            let f2 = noise.noise2d_f2(x, y);
            assert!(
                f2 >= f1 - 1e-10,
                "F2 ({}) < F1 ({}) at ({}, {})",
                f2,
                f1,
                x,
                y
            );
        }
    }

    #[test]
    fn test_manhattan_distance() {
        let noise = WorleyNoise::new(42).with_distance_func(DistanceFunc::Manhattan);
        for i in 0..10 {
            let val = noise.noise2d(i as f64 * 0.5, i as f64 * 0.5);
            assert!(val >= 0.0);
        }
    }

    #[test]
    fn test_chebyshev_distance() {
        let noise = WorleyNoise::new(42).with_distance_func(DistanceFunc::Chebyshev);
        for i in 0..10 {
            let val = noise.noise2d(i as f64 * 0.5, i as f64 * 0.5);
            assert!(val >= 0.0);
        }
    }

    #[test]
    fn test_multi_point_per_cell() {
        let noise = WorleyNoise::new(42).with_points_per_cell(3);
        let val = noise.noise2d(0.5, 0.5);
        assert!(val >= 0.0);
    }

    #[test]
    fn test_worley2d_continuity() {
        let noise = WorleyNoise::new(42);
        let base = noise.noise2d(1.0, 1.0);
        let epsilon = 0.001;
        let nearby = noise.noise2d(1.0 + epsilon, 1.0 + epsilon);
        assert!(
            (base - nearby).abs() < 0.1,
            "Continuity violated: {} vs {}",
            base,
            nearby
        );
    }

    #[test]
    fn test_different_seeds_different_output() {
        let n1 = WorleyNoise::new(1);
        let n2 = WorleyNoise::new(2);
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
}
