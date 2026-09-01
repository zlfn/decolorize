//! Contrast preserving decolorization.
//!
//! Converts color images to grayscale while preserving the contrast between
//! colors that ordinary luminance conversions flatten out. See [`decolorize`]
//! for the algorithm and the entry points.
#![deny(missing_docs)]

pub mod decolorize;
pub mod definitions;

pub use image;
