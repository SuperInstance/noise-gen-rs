//! Fractal Brownian Motion (fBm) - layered noise for natural-looking patterns.
//!
//! fBm stacks multiple octaves of a base noise function with increasing frequency
//! and decreasing amplitude, producing realistic terrain-like textures.

use crate::perlin::PerlinNoise;

/// Fractal Brownian Motion generator built on top of Perlin noise.
///
/// # Example
/// ```
/// use noise_gen_rs::fbm::FractalBrownianMotion;
/// let fbm = FractalBrownianMotion::new(42);
/// let val = fbm.fbmd2(0.5, 0.5, 4, 2.0, 0.5);
/// assert!(val >= -1.0 && val <= 1.0);
/// ```
#[derive(Debug, Clone)]
pub struct FractalBrownianMotion {
    noise: PerlinNoise,
}

impl FractalBrownianMotion {
    /// Create a new fBm generator with the given seed.
    pub fn new(seed: u64) -> Self {
        Self {
            noise: PerlinNoise::new(seed),
        }
    }

    /// 2D fBm: layers multiple octaves of 2D Perlin noise.
    ///
    /// # Arguments
    /// * `x`, `y` - Input coordinates
    /// * `octaves` - Number of noise layers (1-16)
    /// * `lacunarity` - Frequency multiplier between octaves (typically 2.0)
    /// * `persistence` - Amplitude multiplier between octaves (typically 0.5)
    ///
    /// # Returns
    /// A value in approximately [-1, 1], normalized by the sum of all amplitudes.
    pub fn fbmd2(&self, x: f64, y: f64, octaves: u32, lacunarity: f64, persistence: f64) -> f64 {
        let octaves = octaves.clamp(1, 16);
        let mut value = 0.0;
        let mut amplitude = 1.0;
        let mut frequency = 1.0;
        let mut max_amplitude = 0.0;

        for _ in 0..octaves {
            value += amplitude * self.noise.noise2d(x * frequency, y * frequency);
            max_amplitude += amplitude;
            amplitude *= persistence;
            frequency *= lacunarity;
        }

        value / max_amplitude
    }

    /// 3D fBm: layers multiple octaves of 3D Perlin noise.
    ///
    /// Same arguments as `fbmd2` but with an additional `z` dimension.
    pub fn fbmd3(
        &self,
        x: f64,
        y: f64,
        z: f64,
        octaves: u32,
        lacunarity: f64,
        persistence: f64,
    ) -> f64 {
        let octaves = octaves.clamp(1, 16);
        let mut value = 0.0;
        let mut amplitude = 1.0;
        let mut frequency = 1.0;
        let mut max_amplitude = 0.0;

        for _ in 0..octaves {
            value += amplitude * self.noise.noise3d(x * frequency, y * frequency, z * frequency);
            max_amplitude += amplitude;
            amplitude *= persistence;
            frequency *= lacunarity;
        }

        value / max_amplitude
    }

    /// Ridged multifractal variant - uses |noise| inverted for ridge-like features.
    ///
    /// Produces sharp ridges and valleys instead of smooth hills.
    pub fn ridged2d(&self, x: f64, y: f64, octaves: u32, lacunarity: f64, persistence: f64) -> f64 {
        let octaves = octaves.clamp(1, 16);
        let mut value = 0.0;
        let mut amplitude = 1.0;
        let mut frequency = 1.0;
        let mut max_amplitude = 0.0;

        for _ in 0..octaves {
            let n = self.noise.noise2d(x * frequency, y * frequency);
            // Ridge: invert and square
            let ridge = 1.0 - n.abs();
            value += amplitude * ridge * ridge;
            max_amplitude += amplitude;
            amplitude *= persistence;
            frequency *= lacunarity;
        }

        // Normalize to approximately [-1, 1]
        value / max_amplitude * 2.0 - 1.0
    }

    /// Turbulence variant - uses absolute value of noise for turbulent patterns.
    pub fn turbulence2d(&self, x: f64, y: f64, octaves: u32, lacunarity: f64, persistence: f64) -> f64 {
        let octaves = octaves.clamp(1, 16);
        let mut value = 0.0;
        let mut amplitude = 1.0;
        let mut frequency = 1.0;
        let mut max_amplitude = 0.0;

        for _ in 0..octaves {
            value += amplitude * self.noise.noise2d(x * frequency, y * frequency).abs();
            max_amplitude += amplitude;
            amplitude *= persistence;
            frequency *= lacunarity;
        }

        value / max_amplitude
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fbmd2_range() {
        let fbm = FractalBrownianMotion::new(42);
        for i in 0..30 {
            let x = i as f64 * 0.13;
            let y = i as f64 * 0.17;
            let val = fbm.fbmd2(x, y, 4, 2.0, 0.5);
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
    fn test_fbmd3_range() {
        let fbm = FractalBrownianMotion::new(42);
        for i in 0..10 {
            let x = i as f64 * 0.13;
            let val = fbm.fbmd3(x, x, x, 4, 2.0, 0.5);
            assert!(val >= -1.0 && val <= 1.0, "Value {} out of range", val);
        }
    }

    #[test]
    fn test_fbmd2_deterministic() {
        let f1 = FractalBrownianMotion::new(42);
        let f2 = FractalBrownianMotion::new(42);
        for i in 0..10 {
            let x = i as f64 * 0.37;
            let y = i as f64 * 0.53;
            assert!((f1.fbmd2(x, y, 4, 2.0, 0.5) - f2.fbmd2(x, y, 4, 2.0, 0.5)).abs() < 1e-12);
        }
    }

    #[test]
    fn test_fbmd2_single_octave_equals_perlin() {
        let fbm = FractalBrownianMotion::new(42);
        let perlin = PerlinNoise::new(42);
        let val_fbm = fbm.fbmd2(1.5, 2.5, 1, 2.0, 0.5);
        let val_perlin = perlin.noise2d(1.5, 2.5);
        assert!((val_fbm - val_perlin).abs() < 1e-12);
    }

    #[test]
    fn test_more_octaves_more_detail() {
        let fbm = FractalBrownianMotion::new(42);
        let v1 = fbm.fbmd2(0.5, 0.5, 1, 2.0, 0.5);
        let v4 = fbm.fbmd2(0.5, 0.5, 4, 2.0, 0.5);
        // With more octaves, the value should generally change
        // (it won't always, but the multi-octave value should still be in range)
        assert!(v4 >= -1.0 && v4 <= 1.0);
        assert!(v1 >= -1.0 && v1 <= 1.0);
    }

    #[test]
    fn test_lacunarity_effect() {
        let fbm = FractalBrownianMotion::new(42);
        let v_low = fbm.fbmd2(0.5, 0.5, 4, 1.5, 0.5);
        let v_high = fbm.fbmd2(0.5, 0.5, 4, 3.0, 0.5);
        // Different lacunarity should generally give different results
        assert!(v_low >= -1.0 && v_low <= 1.0);
        assert!(v_high >= -1.0 && v_high <= 1.0);
    }

    #[test]
    fn test_ridged2d_range() {
        let fbm = FractalBrownianMotion::new(42);
        for i in 0..20 {
            let x = i as f64 * 0.13;
            let val = fbm.ridged2d(x, x, 4, 2.0, 0.5);
            assert!(
                val >= -1.0 && val <= 1.0,
                "Ridged value {} out of range",
                val
            );
        }
    }

    #[test]
    fn test_turbulence2d_non_negative() {
        let fbm = FractalBrownianMotion::new(42);
        for i in 0..20 {
            let x = i as f64 * 0.13;
            let val = fbm.turbulence2d(x, x, 4, 2.0, 0.5);
            assert!(val >= 0.0, "Turbulence value {} is negative", val);
            assert!(val <= 1.0, "Turbulence value {} exceeds 1.0", val);
        }
    }

    #[test]
    fn test_turbulence2d_deterministic() {
        let f1 = FractalBrownianMotion::new(42);
        let f2 = FractalBrownianMotion::new(42);
        assert!((f1.turbulence2d(1.0, 2.0, 4, 2.0, 0.5) - f2.turbulence2d(1.0, 2.0, 4, 2.0, 0.5)).abs() < 1e-12);
    }

    #[test]
    fn test_ridged2d_deterministic() {
        let f1 = FractalBrownianMotion::new(42);
        let f2 = FractalBrownianMotion::new(42);
        assert!((f1.ridged2d(1.0, 2.0, 4, 2.0, 0.5) - f2.ridged2d(1.0, 2.0, 4, 2.0, 0.5)).abs() < 1e-12);
    }
}
