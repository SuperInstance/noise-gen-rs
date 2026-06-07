//! Fractal Brownian Motion — layering noise octaves for natural detail.

/// Trait for any 2D noise source that can be layered by fBm.
pub trait NoiseSource2D {
    /// Evaluate noise at `(x, y)`. Should ideally return values in `[-1, 1]`.
    fn noise(&self, x: f64, y: f64) -> f64;
}

/// Trait for any 3D noise source.
pub trait NoiseSource3D {
    /// Evaluate noise at `(x, y, z)`.
    fn noise(&self, x: f64, y: f64, z: f64) -> f64;
}

/// Fractal Brownian Motion generator.
///
/// Sums multiple octaves of a base noise at increasing frequencies and
/// decreasing amplitudes, producing natural-looking textures.
pub struct FractalBrownianMotion;

impl FractalBrownianMotion {
    /// Generate 2D fBm by layering `octaves` of a noise source.
    ///
    /// - `noise`: the base noise source
    /// - `x`, `y`: input coordinates
    /// - `octaves`: number of layers (more = finer detail)
    /// - `lacunarity`: frequency multiplier per octave (typically 2.0)
    /// - `persistence`: amplitude multiplier per octave (typically 0.5)
    ///
    /// Returns the summed noise value.
    pub fn fbm2d(
        noise: &dyn NoiseSource2D,
        x: f64,
        y: f64,
        octaves: u32,
        lacunarity: f64,
        persistence: f64,
    ) -> f64 {
        let mut value = 0.0;
        let mut amplitude = 1.0;
        let mut frequency = 1.0;
        let mut max_amp = 0.0;

        for _ in 0..octaves {
            value += amplitude * noise.noise(x * frequency, y * frequency);
            max_amp += amplitude;
            amplitude *= persistence;
            frequency *= lacunarity;
        }

        value / max_amp
    }

    /// Generate 3D fBm.
    pub fn fbm3d(
        noise: &dyn NoiseSource3D,
        x: f64,
        y: f64,
        z: f64,
        octaves: u32,
        lacunarity: f64,
        persistence: f64,
    ) -> f64 {
        let mut value = 0.0;
        let mut amplitude = 1.0;
        let mut frequency = 1.0;
        let mut max_amp = 0.0;

        for _ in 0..octaves {
            value += amplitude * noise.noise(x * frequency, y * frequency, z * frequency);
            max_amp += amplitude;
            amplitude *= persistence;
            frequency *= lacunarity;
        }

        value / max_amp
    }

    /// Turbulence variant — uses the absolute value of each octave.
    pub fn turbulence2d(
        noise: &dyn NoiseSource2D,
        x: f64,
        y: f64,
        octaves: u32,
        lacunarity: f64,
        persistence: f64,
    ) -> f64 {
        let mut value = 0.0;
        let mut amplitude = 1.0;
        let mut frequency = 1.0;
        let mut max_amp = 0.0;

        for _ in 0..octaves {
            value += amplitude * noise.noise(x * frequency, y * frequency).abs();
            max_amp += amplitude;
            amplitude *= persistence;
            frequency *= lacunarity;
        }

        value / max_amp
    }

    /// Ridge noise variant — inverts and squares each octave for ridged textures.
    pub fn ridge2d(
        noise: &dyn NoiseSource2D,
        x: f64,
        y: f64,
        octaves: u32,
        lacunarity: f64,
        persistence: f64,
    ) -> f64 {
        let mut value = 0.0;
        let mut amplitude = 1.0;
        let mut frequency = 1.0;
        let mut max_amp = 0.0;

        for _ in 0..octaves {
            let n = noise.noise(x * frequency, y * frequency);
            value += amplitude * (1.0 - n.abs());
            max_amp += amplitude;
            amplitude *= persistence;
            frequency *= lacunarity;
        }

        value / max_amp
    }
}

// Implement NoiseSource2D for PerlinNoise
impl NoiseSource2D for crate::PerlinNoise {
    fn noise(&self, x: f64, y: f64) -> f64 {
        self.noise2d(x, y)
    }
}

impl NoiseSource3D for crate::PerlinNoise {
    fn noise(&self, x: f64, y: f64, z: f64) -> f64 {
        self.noise3d(x, y, z)
    }
}

impl NoiseSource2D for crate::SimplexNoise {
    fn noise(&self, x: f64, y: f64) -> f64 {
        self.noise2d(x, y)
    }
}

impl NoiseSource3D for crate::SimplexNoise {
    fn noise(&self, x: f64, y: f64, z: f64) -> f64 {
        self.noise3d(x, y, z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PerlinNoise, SimplexNoise};

    #[test]
    fn test_fbm_2d_deterministic() {
        let n = PerlinNoise::new(42);
        let v1 = FractalBrownianMotion::fbm2d(&n, 1.5, 2.5, 4, 2.0, 0.5);
        let v2 = FractalBrownianMotion::fbm2d(&n, 1.5, 2.5, 4, 2.0, 0.5);
        assert!((v1 - v2).abs() < 1e-12);
    }

    #[test]
    fn test_fbm_2d_range() {
        let n = PerlinNoise::new(42);
        for x in 0..10 {
            for y in 0..10 {
                let v = FractalBrownianMotion::fbm2d(&n, x as f64 * 0.3, y as f64 * 0.3, 4, 2.0, 0.5);
                assert!(v >= -1.1 && v <= 1.1, "fBm out of range: {v}");
            }
        }
    }

    #[test]
    fn test_fbm_2d_more_octaves_more_detail() {
        let n = PerlinNoise::new(42);
        let v1 = FractalBrownianMotion::fbm2d(&n, 1.5, 2.5, 1, 2.0, 0.5);
        let v4 = FractalBrownianMotion::fbm2d(&n, 1.5, 2.5, 4, 2.0, 0.5);
        // With different octave counts, values should differ
        assert!((v1 - v4).abs() > 0.001);
    }

    #[test]
    fn test_fbm_3d_deterministic() {
        let n = PerlinNoise::new(42);
        let v1 = FractalBrownianMotion::fbm3d(&n, 1.5, 2.5, 3.5, 4, 2.0, 0.5);
        let v2 = FractalBrownianMotion::fbm3d(&n, 1.5, 2.5, 3.5, 4, 2.0, 0.5);
        assert!((v1 - v2).abs() < 1e-12);
    }

    #[test]
    fn test_fbm_simplex_2d() {
        let n = SimplexNoise::new(42);
        let v = FractalBrownianMotion::fbm2d(&n, 1.5, 2.5, 4, 2.0, 0.5);
        assert!(v >= -1.1 && v <= 1.1);
    }

    #[test]
    fn test_turbulence_2d_positive() {
        let n = PerlinNoise::new(42);
        let v = FractalBrownianMotion::turbulence2d(&n, 1.5, 2.5, 4, 2.0, 0.5);
        assert!(v >= 0.0, "turbulence should be non-negative: {v}");
    }

    #[test]
    fn test_ridge_2d_range() {
        let n = PerlinNoise::new(42);
        let v = FractalBrownianMotion::ridge2d(&n, 1.5, 2.5, 4, 2.0, 0.5);
        assert!(v >= -0.1 && v <= 1.1, "ridge out of range: {v}");
    }
}
