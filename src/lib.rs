//! # noise-gen-rs
//!
//! A pure-Rust noise generation library providing Perlin noise, Simplex noise (2D/3D),
//! Worley/Voronoi noise, and fractal Brownian motion (fBm).
//!
//! # Example
//! ```
//! use noise_gen_rs::perlin::PerlinNoise;
//! use noise_gen_rs::simplex::SimplexNoise;
//! use noise_gen_rs::worley::WorleyNoise;
//! use noise_gen_rs::fbm::FractalBrownianMotion;
//!
//! let perlin = PerlinNoise::new(42);
//! let val = perlin.noise2d(0.5, 0.5);
//! assert!(val >= -1.0 && val <= 1.0);
//!
//! let simplex = SimplexNoise::new(42);
//! let val = simplex.noise2d(0.5, 0.5);
//! assert!(val >= -1.0 && val <= 1.0);
//!
//! let worley = WorleyNoise::new(42);
//! let val = worley.noise2d(0.5, 0.5);
//! assert!(val >= 0.0);
//!
//! let fbm = FractalBrownianMotion::new(42);
//! let val = fbm.fbmd2(0.5, 0.5, 4, 2.0, 0.5);
//! assert!(val >= -1.0 && val <= 1.0);
//! ```

pub mod perlin;
pub mod simplex;
pub mod worley;
pub mod fbm;
pub mod util;
