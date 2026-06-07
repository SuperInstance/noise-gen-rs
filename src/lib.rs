//! # noise-gen-rs
//!
//! A pure-Rust library for procedural noise generation, providing:
//! - **Perlin noise** (2D and 3D)
//! - **Simplex noise** (2D and 3D)
//! - **Worley/Voronoi noise** (2D and 3D)
//! - **Fractal Brownian Motion** (fBm) layering on any noise source
//!
//! All generators are deterministic when seeded, produce outputs in `[-1, 1]`,
//! and require no external dependencies.

/// Perlin gradient noise in 2D and 3D.
pub mod perlin;
/// Simplex noise in 2D and 3D.
pub mod simplex;
/// Worley (Voronoi) cellular noise in 2D and 3D.
pub mod worley;
/// Fractal Brownian Motion — layers any base noise for natural-looking detail.
pub mod fbm;
/// Shared utilities: gradient tables, interpolation, and hashing helpers.
pub mod util;

pub use perlin::PerlinNoise;
pub use simplex::SimplexNoise;
pub use worley::WorleyNoise;
pub use fbm::FractalBrownianMotion;
