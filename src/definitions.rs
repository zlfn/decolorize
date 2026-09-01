//! Trait definitions and type aliases.
//!
//! Mirrors the subset of `imageproc::definitions` that [`crate::decolorize`]
//! relies on, so that module needs no edits when upstreamed.

use image::{ImageBuffer, Pixel};

/// An `ImageBuffer` containing Pixels of type P with storage `Vec<P::Subpixel>`.
pub type Image<P> = ImageBuffer<P, Vec<<P as Pixel>::Subpixel>>;
