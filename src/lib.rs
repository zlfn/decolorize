//! Contrast preserving decolorization.
//!
//! This crate is a staging area for a module intended to be upstreamed into
//! [`imageproc`]. The whole implementation lives in [`decolorize`], which only
//! depends on `image`, `nalgebra` and (optionally) `rayon` — exactly the
//! dependencies `imageproc` already has — so the file can be moved across
//! unchanged.
//!
//! [`imageproc`]: https://docs.rs/imageproc
#![deny(missing_docs)]

pub mod decolorize;
pub mod definitions;

pub use image;
