//! Trait definitions and type aliases.

use image::{ImageBuffer, Pixel};

/// An `ImageBuffer` containing pixels of type `P` with storage
/// `Vec<P::Subpixel>`.
pub type Image<P> = ImageBuffer<P, Vec<<P as Pixel>::Subpixel>>;
