//! Contrast preserving decolorization.
//!
//! Ordinary color-to-gray conversions map each color onto its luminance. Colors
//! that differ only in chrominance collapse onto the same gray and the contrast
//! between them is lost. Red text on a green background is the standard
//! example: it comes out as a flat block.
//!
//! The functions here implement the *contrast preserving decolorization* (CPD)
//! of Lu, Xu and Jia, which instead picks the global mapping that best
//! reproduces the color contrast of every neighboring pixel pair. Two variants
//! are provided:
//!
//! * [`decolorize`] fits a degree-2 polynomial mapping by maximum likelihood
//!   under a bimodal Gaussian prior. This is the algorithm of the ICCP 2012 and
//!   IJCV 2014 papers.
//! * [`decolorize_fast`] restricts the model to a convex combination of the
//!   three channels and picks the best of 66 candidates by exhaustive search on
//!   a sparse sample of pixel pairs. This is the SIGGRAPH Asia 2012 technical
//!   brief, and is roughly an order of magnitude cheaper.
//!
//! # The model
//!
//! Write `Δg` for the difference of the output grays of a pixel pair and `δ`
//! for the color contrast of that pair, the Euclidean distance in CIELAB. The
//! sign of `δ` carries no physical meaning, so instead of fixing it in advance
//! CPD models `Δg` as a mixture of two Gaussians centered at `+δ` and `-δ` and
//! lets the optimizer choose:
//!
//! ```text
//! E(ω) = - Σ  ln[ α·G(Δg - δ, σ²) + (1 - α)·G(Δg + δ, σ²) ]
//! ```
//!
//! The *weak color order* `α` collapses the mixture to a single mode for the
//! pairs whose order is unambiguous, namely those where one color dominates the
//! other in all three channels. [`decolorize_fast`] drops that term.
//!
//! # Transparency
//!
//! Transparent pixels hold whatever color was left behind them, so they are
//! weighted out of the objective and out of the final rescaling. Partial alpha
//! counts in proportion.
//!
//! # Relation to `cv::decolor`
//!
//! OpenCV ships the authors' own implementation of the polynomial variant.
//! Across the 24 image benchmark from the project page the two outputs have a
//! mean absolute correlation of 0.992. They stop iterating at different points:
//! `cv::decolor` tests convergence against a smoothed stand-in for the
//! objective and settles after about three iterations, where this crate uses
//! the objective as published.
//!
//! # References
//!
//! * Cewu Lu, Li Xu and Jiaya Jia. *Contrast Preserving Decolorization*. ICCP
//!   2012.
//! * Cewu Lu, Li Xu and Jiaya Jia. *Contrast Preserving Decolorization with
//!   Perception-Based Quality Metrics*. IJCV 2014.
//! * Cewu Lu, Li Xu and Jiaya Jia. *Real-time Contrast Preserving
//!   Decolorization*. SIGGRAPH Asia 2012 Technical Briefs.
//!
//! # Examples
//!
//! ```
//! use decolorize::decolorize;
//! use image::{Rgb, RgbImage};
//!
//! // Red and green with the same luminance under Rec. 601: a naive
//! // conversion maps both onto the same gray.
//! let mut image = RgbImage::new(8, 8);
//! for (x, _y, pixel) in image.enumerate_pixels_mut() {
//!     *pixel = if x < 4 { Rgb([251, 0, 0]) } else { Rgb([0, 128, 0]) };
//! }
//!
//! let gray = decolorize(&image);
//! assert_eq!(gray.dimensions(), (8, 8));
//! // The two regions stay distinguishable.
//! assert!(gray.get_pixel(0, 0)[0].abs_diff(gray.get_pixel(7, 0)[0]) > 32);
//! ```
#![deny(missing_docs)]

use image::{ImageBuffer, Luma, LumaA, Pixel, Primitive, Rgb, Rgba};
use nalgebra::{SMatrix, SVector};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// An `ImageBuffer` with `Vec` storage, the image type these functions take and
/// return.
pub type Image<P> = ImageBuffer<P, Vec<<P as Pixel>::Subpixel>>;

/// Number of monomials spanning the degree-2 polynomial color space.
const NUM_MONOMIALS: usize = 9;

/// Exponent triples `(a, b, c)` of the monomial basis of
/// `Π₂ = span{ rᵃ gᵇ bᶜ : 0 < a + b + c ≤ 2 }`, in the order used by the paper:
/// `r, g, b, rg, rb, gb, r², g², b²`.
const MONOMIALS: [(u32, u32, u32); NUM_MONOMIALS] = [
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 0),
    (1, 0, 1),
    (0, 1, 1),
    (2, 0, 0),
    (0, 2, 0),
    (0, 0, 2),
];

/// Number of pixel pairs handled by one unit of work. Fixing this makes the
/// order of the floating point reductions independent of the `rayon` feature,
/// so results are identical with and without it.
const PAIR_CHUNK: usize = 8192;

type Matrix9 = SMatrix<f64, NUM_MONOMIALS, NUM_MONOMIALS>;
type Vector9 = SVector<f64, NUM_MONOMIALS>;

mod sealed {
    pub trait Sealed {}

    impl Sealed for u8 {}
    impl Sealed for u16 {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}

    impl<T> Sealed for image::Rgb<T> {}
    impl<T> Sealed for image::Rgba<T> {}
    impl<T> Sealed for image::Luma<T> {}
    impl<T> Sealed for image::LumaA<T> {}
}

/// Subpixel types that these functions accept.
///
/// Implemented for `u8`, `u16`, `f32` and `f64`. Integral samples are taken to
/// span the unit interval over the whole of their range, floating point samples
/// to lie in `[0, 1]` already.
///
/// The trait is sealed. Open an issue if you need another sample type.
pub trait DecolorizeSubpixel: Primitive + Send + Sync + sealed::Sealed + 'static {
    /// Maps a sample onto the unit interval.
    fn to_unit(self) -> f64;

    /// Maps a value on the unit interval back to a sample.
    ///
    /// Integral types round and saturate. Floating point types pass the value
    /// through, so an out of range input stays out of range.
    fn from_unit(value: f64) -> Self;
}

macro_rules! impl_integral_subpixel {
    ($t:ty) => {
        impl DecolorizeSubpixel for $t {
            #[inline]
            fn to_unit(self) -> f64 {
                f64::from(self) / f64::from(<$t>::MAX)
            }

            #[inline]
            fn from_unit(value: f64) -> Self {
                let max = f64::from(<$t>::MAX);
                // A non-finite value lands on zero rather than trapping.
                (value * max).round().clamp(0.0, max) as $t
            }
        }
    };
}

macro_rules! impl_float_subpixel {
    ($t:ty) => {
        impl DecolorizeSubpixel for $t {
            #[inline]
            fn to_unit(self) -> f64 {
                self as f64
            }

            #[inline]
            fn from_unit(value: f64) -> Self {
                value as $t
            }
        }
    };
}

impl_integral_subpixel!(u8);
impl_integral_subpixel!(u16);
impl_float_subpixel!(f32);
impl_float_subpixel!(f64);

/// Pixel types these functions accept, and the grayscale type each produces.
///
/// Sealed, and implemented for `Rgb`, `Rgba`, `Luma` and `LumaA`, which is
/// every type `image` implements `Pixel` for.
pub trait DecolorizePixel: Pixel + sealed::Sealed {
    /// Grayscale pixel type produced for this input.
    type Output: Pixel<Subpixel = Self::Subpixel>;

    /// Builds an output pixel from a gray level and the input's alpha.
    fn gray(value: Self::Subpixel, alpha: Self::Subpixel) -> Self::Output;
}

macro_rules! impl_decolorize_pixel {
    ($($t:ty),*) => {
        $(
            impl DecolorizePixel for Rgb<$t> {
                type Output = Luma<$t>;
                #[inline]
                fn gray(value: $t, _alpha: $t) -> Luma<$t> {
                    Luma([value])
                }
            }

            impl DecolorizePixel for Luma<$t> {
                type Output = Luma<$t>;
                #[inline]
                fn gray(value: $t, _alpha: $t) -> Luma<$t> {
                    Luma([value])
                }
            }

            impl DecolorizePixel for Rgba<$t> {
                type Output = LumaA<$t>;
                #[inline]
                fn gray(value: $t, alpha: $t) -> LumaA<$t> {
                    LumaA([value, alpha])
                }
            }

            impl DecolorizePixel for LumaA<$t> {
                type Output = LumaA<$t>;
                #[inline]
                fn gray(value: $t, alpha: $t) -> LumaA<$t> {
                    LumaA([value, alpha])
                }
            }
        )*
    };
}

impl_decolorize_pixel!(u8, u16, f32, f64);

/// Options for [`decolorize_with`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecolorizeOptions {
    /// Standard deviation σ of the two Gaussian modes, in units of gray level.
    ///
    /// Small values demand that the gray contrast match the color contrast
    /// closely; large values let the optimizer trade individual pairs off
    /// against each other. Defaults to `0.02`, the value used by the authors'
    /// reference implementation.
    pub sigma: f64,
    /// Maximum number of fixed point iterations, `kmax` in the paper, which
    /// sets it to `15` empirically. Some images are still drifting at that
    /// point, so this is a budget rather than a convergence criterion.
    pub max_iterations: usize,
    /// Iteration stops once the energy changes by less than this between
    /// successive iterations. Defaults to `1e-4`.
    pub tolerance: f64,
    /// The mapping is fitted on a copy of the image downscaled to roughly this
    /// bound on `width + height`, then applied at full resolution. Defaults to
    /// `800`. Neither side is ever scaled below one pixel, so an extreme aspect
    /// ratio can overshoot the bound by one.
    ///
    /// Fitting cost is quadratic in this value while the result is almost
    /// unaffected, since a global mapping only has nine degrees of freedom.
    pub fitting_size: u32,
}

impl Default for DecolorizeOptions {
    fn default() -> Self {
        Self {
            sigma: 0.02,
            max_iterations: 15,
            tolerance: 1e-4,
            fitting_size: 800,
        }
    }
}

/// Options for [`decolorize_fast_with`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FastDecolorizeOptions {
    /// Width of the two Gaussian modes, entering the score as
    /// `exp(-(Δg ∓ δ)² / σ²)`. Defaults to `0.05`.
    pub sigma: f64,
    /// Pixel pairs are drawn from a grid holding about `samples²` points.
    /// Defaults to `64`.
    pub samples: u32,
    /// Pairs whose color contrast falls below this are discarded, so that the
    /// search is driven by the pairs that actually carry contrast. Defaults to
    /// `0.05`.
    pub contrast_threshold: f64,
    /// Seed for the pairing of the randomly matched samples. Fixed by default,
    /// so the output is reproducible; vary it to check how stable the search is
    /// on a given image.
    pub seed: u64,
}

impl Default for FastDecolorizeOptions {
    fn default() -> Self {
        Self {
            sigma: 0.05,
            samples: 64,
            contrast_threshold: 0.05,
            seed: 0,
        }
    }
}

/// Converts an image to grayscale while preserving color contrast, using the
/// default [`DecolorizeOptions`].
///
/// See [`decolorize_with`] for the details.
#[must_use]
pub fn decolorize<P>(image: &Image<P>) -> Image<P::Output>
where
    P: DecolorizePixel,
    P::Subpixel: DecolorizeSubpixel,
{
    decolorize_with(image, DecolorizeOptions::default())
}

/// Converts an image to grayscale while preserving color contrast.
///
/// Fits the degree-2 polynomial mapping `f(r, g, b) = Σ ωᵢmᵢ` that maximizes
/// the bimodal contrast likelihood of Lu, Xu and Jia, then applies it and
/// rescales the result to the full output range.
///
/// The mapping is global, so equal colors always map to equal grays and no
/// local halos or gradient reversals are introduced.
///
/// Fitting runs on a downscaled copy (see [`DecolorizeOptions::fitting_size`]),
/// so its cost does not grow with the input.
///
/// # Examples
///
/// ```
/// use decolorize::{DecolorizeOptions, decolorize_with};
/// use image::{Rgb, RgbImage};
///
/// let image = RgbImage::from_pixel(4, 4, Rgb([120, 40, 200]));
///
/// // A tighter fit, at the cost of more iterations.
/// let options = DecolorizeOptions {
///     max_iterations: 40,
///     tolerance: 1e-6,
///     ..DecolorizeOptions::default()
/// };
/// let gray = decolorize_with(&image, options);
///
/// assert_eq!(gray.dimensions(), (4, 4));
/// ```
///
/// # Panics
///
/// Panics if `options.sigma` is not positive, or if `options.fitting_size` is
/// zero.
#[must_use]
pub fn decolorize_with<P>(image: &Image<P>, options: DecolorizeOptions) -> Image<P::Output>
where
    P: DecolorizePixel,
    P::Subpixel: DecolorizeSubpixel,
{
    assert!(options.sigma > 0.0, "sigma must be positive");
    assert!(options.fitting_size > 0, "fitting_size must be non-zero");

    if image.width() == 0 || image.height() == 0 {
        return ImageBuffer::new(image.width(), image.height());
    }

    let planes = Planes::downscale(image, options.fitting_size);
    let weights = fit_weights(&planes, options);
    apply_polynomial(image, &weights)
}

/// Converts an image to grayscale while preserving color contrast, using the
/// default [`FastDecolorizeOptions`].
///
/// See [`decolorize_fast_with`] for the details.
#[must_use]
pub fn decolorize_fast<P>(image: &Image<P>) -> Image<P::Output>
where
    P: DecolorizePixel,
    P::Subpixel: DecolorizeSubpixel,
{
    decolorize_fast_with(image, FastDecolorizeOptions::default())
}

/// Converts an image to grayscale while preserving color contrast, using the
/// real-time approximation.
///
/// Restricts the mapping to `f(r, g, b) = w_r·r + w_g·g + w_b·b` with
/// non-negative weights summing to one, quantized to multiples of `0.1`. The
/// resulting 66 candidates are scored by a bimodal likelihood like the one
/// [`decolorize_with`] minimizes, though without its weak color order term,
/// evaluated on a fixed size sample of pixel pairs. The best scoring candidate
/// is applied.
///
/// This is roughly an order of magnitude cheaper than [`decolorize_with`], but
/// the linear model cannot separate colors that lie on a common line through
/// the RGB cube.
///
/// Because the weights are convex the output needs no rescaling, so unlike
/// [`decolorize_with`] this function preserves the absolute brightness of the
/// input.
///
/// Contrast is measured as the Euclidean distance in RGB rather than in CIELAB,
/// following the authors' formulation for this variant.
///
/// # Examples
///
/// ```
/// use decolorize::decolorize_fast;
/// use image::{Rgb, RgbImage};
///
/// let mut image = RgbImage::new(64, 64);
/// for (x, _y, pixel) in image.enumerate_pixels_mut() {
///     *pixel = if x < 32 { Rgb([255, 0, 0]) } else { Rgb([0, 0, 255]) };
/// }
///
/// let gray = decolorize_fast(&image);
/// assert!(gray.get_pixel(0, 0)[0].abs_diff(gray.get_pixel(63, 0)[0]) > 32);
/// ```
///
/// # Panics
///
/// Panics if `options.sigma` is not positive, or if `options.samples` is zero.
#[must_use]
pub fn decolorize_fast_with<P>(image: &Image<P>, options: FastDecolorizeOptions) -> Image<P::Output>
where
    P: DecolorizePixel,
    P::Subpixel: DecolorizeSubpixel,
{
    assert!(options.sigma > 0.0, "sigma must be positive");
    assert!(options.samples > 0, "samples must be non-zero");

    if image.width() == 0 || image.height() == 0 {
        return ImageBuffer::new(image.width(), image.height());
    }

    let channel_weights = search_linear_weights(image, options);
    apply_linear(image, channel_weights)
}

// -------------------------------------------------------------------------
// The polynomial variant (ICCP 2012 / IJCV 2014)
// -------------------------------------------------------------------------

/// A downscaled copy of the input held as planes on the unit interval.
struct Planes {
    width: usize,
    height: usize,
    r: Vec<f64>,
    g: Vec<f64>,
    b: Vec<f64>,
    /// Mean alpha of the source pixels behind each cell. Colors are averaged
    /// weighted by it, so a cell that is partly transparent takes its color
    /// from the part that is actually there.
    coverage: Vec<f64>,
}

impl Planes {
    /// Box filters `image` down until `width + height <= max_extent`.
    ///
    /// Box filtering rather than point sampling matters here. The objective is
    /// a sum over *neighboring* pixel pairs, so aliasing would fabricate
    /// contrast between pixels that are not adjacent in the original.
    fn downscale<P>(image: &Image<P>, max_extent: u32) -> Self
    where
        P: DecolorizePixel,
        P::Subpixel: DecolorizeSubpixel,
    {
        let (source_width, source_height) = (image.width() as usize, image.height() as usize);
        let extent = source_width + source_height;

        let (width, height) = if extent as u32 > max_extent {
            let scale = f64::from(max_extent) / extent as f64;
            (
                ((source_width as f64 * scale).round() as usize).max(1),
                ((source_height as f64 * scale).round() as usize).max(1),
            )
        } else {
            (source_width, source_height)
        };

        let channels = P::CHANNEL_COUNT as usize;
        let raw = image.as_raw();
        let mut planes = Planes {
            width,
            height,
            r: vec![0.0; width * height],
            g: vec![0.0; width * height],
            b: vec![0.0; width * height],
            coverage: vec![0.0; width * height],
        };

        type Row<'a> = (
            usize,
            (
                ((&'a mut [f64], &'a mut [f64]), &'a mut [f64]),
                &'a mut [f64],
            ),
        );
        let fill = |(y, (((row_r, row_g), row_b), row_a)): Row<'_>| {
            // Half open source row range covered by output row `y`.
            let y0 = y * source_height / height;
            let y1 = (((y + 1) * source_height).div_ceil(height)).max(y0 + 1);
            for x in 0..width {
                let x0 = x * source_width / width;
                let x1 = (((x + 1) * source_width).div_ceil(width)).max(x0 + 1);

                let (mut r, mut g, mut b, mut a) = (0.0, 0.0, 0.0, 0.0);
                for sy in y0..y1 {
                    for sx in x0..x1 {
                        let i = (sy * source_width + sx) * channels;
                        let rgba = P::from_slice(&raw[i..i + channels]).to_rgba();
                        let alpha = rgba.0[3].to_unit();
                        r += rgba.0[0].to_unit() * alpha;
                        g += rgba.0[1].to_unit() * alpha;
                        b += rgba.0[2].to_unit() * alpha;
                        a += alpha;
                    }
                }

                let count = ((y1 - y0) * (x1 - x0)) as f64;
                // Undo the premultiplication, leaving the mean color of the
                // part of the cell that is present. A fully transparent cell
                // has no color to speak of and is left at zero, but its
                // coverage keeps it out of the fit entirely.
                let (scaled_r, scaled_g, scaled_b) = if a > 0.0 {
                    (r / a, g / a, b / a)
                } else {
                    (0.0, 0.0, 0.0)
                };
                row_r[x] = scaled_r;
                row_g[x] = scaled_g;
                row_b[x] = scaled_b;
                row_a[x] = a / count;
            }
        };

        // Output rows are independent, and each reads a disjoint band of the
        // source, so splitting by row keeps the result identical either way.
        #[cfg(feature = "rayon")]
        planes
            .r
            .par_chunks_mut(width)
            .zip(planes.g.par_chunks_mut(width))
            .zip(planes.b.par_chunks_mut(width))
            .zip(planes.coverage.par_chunks_mut(width))
            .enumerate()
            .for_each(fill);
        #[cfg(not(feature = "rayon"))]
        planes
            .r
            .chunks_mut(width)
            .zip(planes.g.chunks_mut(width))
            .zip(planes.b.chunks_mut(width))
            .zip(planes.coverage.chunks_mut(width))
            .enumerate()
            .for_each(fill);

        planes
    }

    /// Number of four-neighbor pixel pairs, `|N|` in the paper.
    fn pair_count(&self) -> usize {
        self.width.saturating_sub(1) * self.height + self.width * self.height.saturating_sub(1)
    }
}

/// Combines the two endpoints of every four-neighbor pixel pair with `f`,
/// horizontal pairs first and vertical pairs second.
///
/// The color contrasts, the weak orders, the monomial differences and the
/// coverage weights all run through here, which is what keeps them in step:
/// they are indexed against each other everywhere downstream.
fn map_pairs<F>(plane: &[f64], width: usize, height: usize, out: &mut [f64], f: F)
where
    F: Fn(f64, f64) -> f64,
{
    let horizontal = width.saturating_sub(1);
    for y in 0..height {
        let row = y * width;
        for x in 0..horizontal {
            out[y * horizontal + x] = f(plane[row + x], plane[row + x + 1]);
        }
    }

    let offset = horizontal * height;
    for y in 0..height.saturating_sub(1) {
        let row = y * width;
        for x in 0..width {
            out[offset + row + x] = f(plane[row + x], plane[row + width + x]);
        }
    }
}

/// Writes `p(x) - p(y)` for every four-neighbor pixel pair into `out`.
fn pair_differences(plane: &[f64], width: usize, height: usize, out: &mut [f64]) {
    map_pairs(plane, width, height, out, |a, b| a - b);
}

/// Per-pair weight, the product of the coverage at the two endpoints.
///
/// A pair that touches a fully transparent cell drops out of the fit, and a
/// pair across an antialiased edge counts in proportion to how much of it is
/// really there. For an opaque image every weight is exactly `1.0`, which
/// leaves the fit bit for bit what it would be without any of this.
fn pair_weights(planes: &Planes) -> Vec<f64> {
    let mut weights = vec![0.0; planes.pair_count()];
    map_pairs(
        &planes.coverage,
        planes.width,
        planes.height,
        &mut weights,
        |a, b| a * b,
    );
    weights
}

/// Per-pair color contrast `|δ|`, the CIELAB distance of the pair scaled so
/// that it is commensurate with a gray difference on the unit interval.
fn color_contrast(planes: &Planes) -> Vec<f64> {
    let pixels = planes.width * planes.height;
    let (mut lightness, mut green_red, mut blue_yellow) =
        (vec![0.0; pixels], vec![0.0; pixels], vec![0.0; pixels]);

    for i in 0..pixels {
        let (l, a, b) = srgb_to_lab(planes.r[i], planes.g[i], planes.b[i]);
        lightness[i] = l;
        green_red[i] = a;
        blue_yellow[i] = b;
    }

    let pairs = planes.pair_count();
    let (mut dl, mut da, mut db) = (vec![0.0; pairs], vec![0.0; pairs], vec![0.0; pairs]);
    pair_differences(&lightness, planes.width, planes.height, &mut dl);
    pair_differences(&green_red, planes.width, planes.height, &mut da);
    pair_differences(&blue_yellow, planes.width, planes.height, &mut db);

    // `L` runs over [0, 100], so dividing by 100 brings the distance back onto
    // roughly the same scale as a difference of grays on the unit interval.
    (0..pairs)
        .map(|i| (dl[i] * dl[i] + da[i] * da[i] + db[i] * db[i]).sqrt() / 100.0)
        .collect()
}

/// Threshold below which a channel difference is not taken as evidence of an
/// unambiguous color order.
const WEAK_ORDER_LEVEL: f64 = 0.05;

/// Per-pair weak color order, `+1` where the first color dominates the second
/// in every channel, `-1` where it is dominated in every channel, and `0` where
/// the order is ambiguous.
///
/// This is `2α - 1` for the `α` of the paper. It selects the single-moded prior
/// `G(±δ, σ²)` for the pairs that really are ordered and leaves the mixture
/// evenly weighted, free to choose a sign, everywhere else.
fn weak_order(planes: &Planes) -> Vec<f64> {
    let pairs = planes.pair_count();
    let (mut dr, mut dg, mut db) = (vec![0.0; pairs], vec![0.0; pairs], vec![0.0; pairs]);
    pair_differences(&planes.r, planes.width, planes.height, &mut dr);
    pair_differences(&planes.g, planes.width, planes.height, &mut dg);
    pair_differences(&planes.b, planes.width, planes.height, &mut db);

    (0..pairs)
        .map(|i| {
            let level = WEAK_ORDER_LEVEL;
            if dr[i] > level && dg[i] > level && db[i] > level {
                1.0
            } else if dr[i] < -level && dg[i] < -level && db[i] < -level {
                -1.0
            } else {
                0.0
            }
        })
        .collect()
}

/// Per-pair differences `lᵢ = mᵢ(x) - mᵢ(y)` of each monomial, the matrix `P`
/// whose rows the solver contracts against the weights.
fn monomial_differences(planes: &Planes) -> Vec<Vec<f64>> {
    let pixels = planes.width * planes.height;
    let pairs = planes.pair_count();

    let build = |monomial: usize| {
        let (er, eg, eb) = MONOMIALS[monomial];
        let mut values = vec![0.0; pixels];
        for (i, value) in values.iter_mut().enumerate() {
            *value = planes.r[i].powi(er as i32)
                * planes.g[i].powi(eg as i32)
                * planes.b[i].powi(eb as i32);
        }
        let mut differences = vec![0.0; pairs];
        pair_differences(&values, planes.width, planes.height, &mut differences);
        differences
    };

    #[cfg(feature = "rayon")]
    {
        (0..NUM_MONOMIALS).into_par_iter().map(build).collect()
    }
    #[cfg(not(feature = "rayon"))]
    {
        (0..NUM_MONOMIALS).map(build).collect()
    }
}

/// Running totals of one pass over the pixel pairs.
#[derive(Clone, Copy)]
struct Accumulator {
    /// Right hand side `Σ (2β - 1)·lⱼ·δ` of the linearized normal equations.
    rhs: [f64; NUM_MONOMIALS],
    /// Unnormalized objective `Σ -ln[ α·G(Δg - δ) + (1 - α)·G(Δg + δ) ]`.
    energy: f64,
    /// Total pair weight behind the two sums above.
    weight: f64,
}

impl Accumulator {
    fn zero() -> Self {
        Self {
            rhs: [0.0; NUM_MONOMIALS],
            energy: 0.0,
            weight: 0.0,
        }
    }

    fn merge(mut self, other: Self) -> Self {
        for (total, part) in self.rhs.iter_mut().zip(other.rhs) {
            *total += part;
        }
        self.energy += other.energy;
        self.weight += other.weight;
        self
    }
}

/// Evaluates the objective and the fixed point right hand side over the pairs
/// `[start, start + len)`.
#[allow(clippy::too_many_arguments)]
fn accumulate_pairs(
    basis: &[Vec<f64>],
    contrast: &[f64],
    order: &[f64],
    pair_weight: &[f64],
    weights: &[f64; NUM_MONOMIALS],
    sigma: f64,
    start: usize,
    len: usize,
) -> Accumulator {
    let scale = 1.0 / (2.0 * sigma * sigma);
    let mut acc = Accumulator::zero();

    for i in start..start + len {
        let coverage = pair_weight[i];
        if coverage == 0.0 {
            continue;
        }

        // Δg for this pair under the current mapping.
        let mut difference = 0.0;
        for (weight, row) in weights.iter().zip(basis) {
            difference += weight * row[i];
        }

        let delta = contrast[i];
        // α and 1 - α of Eq. (7), recovered from the weak order.
        let (positive, negative) = ((1.0 + order[i]) / 2.0, (1.0 - order[i]) / 2.0);

        // With σ = 0.02 both Gaussians underflow for any pair whose contrast
        // is badly reproduced, which is where the objective is largest, so the
        // mixture is evaluated in the log domain throughout.
        let low = positive.ln() - (difference - delta).powi(2) * scale;
        let high = negative.ln() - (difference + delta).powi(2) * scale;
        let peak = low.max(high);
        let (low, high) = ((low - peak).exp(), (high - peak).exp());

        // 2β - 1, the signed mixture responsibility. One of the two
        // exponentials is 1 by construction, so the quotient is well
        // conditioned.
        let responsibility = (low - high) / (low + high);

        acc.energy -= coverage * (peak + (low + high).ln());
        acc.weight += coverage;
        for (rhs, row) in acc.rhs.iter_mut().zip(basis) {
            *rhs += coverage * row[i] * delta * responsibility;
        }
    }

    acc
}

/// Solves for the polynomial coefficients by fixed point iteration.
fn fit_weights(planes: &Planes, options: DecolorizeOptions) -> [f64; NUM_MONOMIALS] {
    // ω⁰ = { ⅓, ⅓, ⅓, 0, … }, the plain channel average.
    let mut weights = [0.0; NUM_MONOMIALS];
    weights[0] = 1.0 / 3.0;
    weights[1] = 1.0 / 3.0;
    weights[2] = 1.0 / 3.0;

    let pairs = planes.pair_count();
    if pairs == 0 {
        return weights;
    }

    let contrast = color_contrast(planes);
    let order = weak_order(planes);
    let basis = monomial_differences(planes);
    let pair_weight = pair_weights(planes);

    // A fully transparent image has nothing to fit.
    if pair_weight.iter().all(|w| *w == 0.0) {
        return weights;
    }

    // Only the right hand side of the normal equations depends on ω, so the
    // Gram matrix `P Pᵀ` and its factorization are computed once. Forming it
    // rather than the authors' explicit `(P Pᵀ)⁻¹ P diag(δ)` keeps the working
    // set at nine floats instead of a dense 9 × |N| matrix.
    let mut gram = Matrix9::zeros();
    for i in 0..NUM_MONOMIALS {
        for j in i..NUM_MONOMIALS {
            let dot: f64 = basis[i]
                .iter()
                .zip(&basis[j])
                .zip(&pair_weight)
                .map(|((a, b), w)| w * a * b)
                .sum();
            gram[(i, j)] = dot;
            gram[(j, i)] = dot;
        }
    }
    let factorization = gram.lu();

    let mut previous_energy = f64::INFINITY;
    for _ in 0..options.max_iterations {
        let pass = |(chunk, slice): (usize, &[f64])| {
            accumulate_pairs(
                &basis,
                &contrast,
                &order,
                &pair_weight,
                &weights,
                options.sigma,
                chunk * PAIR_CHUNK,
                slice.len(),
            )
        };

        #[cfg(feature = "rayon")]
        let partials: Vec<Accumulator> = contrast
            .par_chunks(PAIR_CHUNK)
            .enumerate()
            .map(pass)
            .collect();
        #[cfg(not(feature = "rayon"))]
        let partials: Vec<Accumulator> =
            contrast.chunks(PAIR_CHUNK).enumerate().map(pass).collect();

        let total = partials
            .into_iter()
            .fold(Accumulator::zero(), Accumulator::merge);

        let energy = total.energy / total.weight;
        if (energy - previous_energy).abs() <= options.tolerance {
            break;
        }
        previous_energy = energy;

        // A flat image leaves the Gram matrix singular; there is nothing to fit,
        // so keep the weights we have.
        match factorization.solve(&Vector9::from(total.rhs)) {
            Some(next) if next.iter().all(|w| w.is_finite()) => {
                weights.copy_from_slice(next.as_slice());
            }
            _ => break,
        }
    }

    weights
}

/// Evaluates `f(r, g, b) = Σ ωᵢmᵢ`.
#[inline]
fn evaluate(weights: &[f64; NUM_MONOMIALS], r: f64, g: f64, b: f64) -> f64 {
    let mut total = 0.0;
    for (weight, (er, eg, eb)) in weights.iter().zip(MONOMIALS) {
        total += weight * r.powi(er as i32) * g.powi(eg as i32) * b.powi(eb as i32);
    }
    total
}

/// Applies the fitted polynomial at full resolution and rescales the result to
/// the output range.
///
/// The polynomial is unconstrained, so its range depends on the image; the
/// paper's final step maps that range linearly onto the displayable one.
fn apply_polynomial<P>(image: &Image<P>, weights: &[f64; NUM_MONOMIALS]) -> Image<P::Output>
where
    P: DecolorizePixel,
    P::Subpixel: DecolorizeSubpixel,
{
    let channels = P::CHANNEL_COUNT as usize;
    let raw = image.as_raw();
    let pixels = image.width() as usize * image.height() as usize;
    let mut gray = vec![0.0f32; pixels];

    let map = |(out, pixel): (&mut f32, &[P::Subpixel])| {
        let rgba = P::from_slice(pixel).to_rgba();
        *out = evaluate(
            weights,
            rgba.0[0].to_unit(),
            rgba.0[1].to_unit(),
            rgba.0[2].to_unit(),
        ) as f32;
    };

    #[cfg(feature = "rayon")]
    gray.par_iter_mut()
        .zip(raw.par_chunks_exact(channels))
        .for_each(map);
    #[cfg(not(feature = "rayon"))]
    gray.iter_mut()
        .zip(raw.chunks_exact(channels))
        .for_each(map);

    // Transparent pixels carry whatever color happened to be stored behind
    // them, which would otherwise stretch the range and flatten everything
    // visible, so they are left out of it.
    let (mut min, mut max) = (f32::INFINITY, f32::NEG_INFINITY);
    if <P as Pixel>::HAS_ALPHA {
        for (&value, pixel) in gray.iter().zip(raw.chunks_exact(channels)) {
            if P::from_slice(pixel).to_rgba().0[3].to_unit() > 0.0 {
                min = min.min(value);
                max = max.max(value);
            }
        }
    } else {
        for &value in &gray {
            min = min.min(value);
            max = max.max(value);
        }
    }

    let range = f64::from(max - min);
    let normalize = |v: f32| -> f64 {
        if range > 0.0 {
            (f64::from(v - min) / range).clamp(0.0, 1.0)
        } else {
            // Constant output: nothing to stretch, so keep it where it lands.
            f64::from(v).clamp(0.0, 1.0)
        }
    };

    let out_channels = <P::Output as Pixel>::CHANNEL_COUNT as usize;
    let mut data = vec![P::Subpixel::from_unit(0.0); pixels * out_channels];

    type Slot<'a, S> = ((&'a mut [S], &'a f32), &'a [S]);
    let write = |((slot, &value), pixel): Slot<'_, P::Subpixel>| {
        let alpha = if <P as Pixel>::HAS_ALPHA {
            P::from_slice(pixel).to_rgba().0[3]
        } else {
            P::Subpixel::DEFAULT_MAX_VALUE
        };
        *<P::Output as Pixel>::from_slice_mut(slot) =
            P::gray(P::Subpixel::from_unit(normalize(value)), alpha);
    };

    #[cfg(feature = "rayon")]
    data.par_chunks_exact_mut(out_channels)
        .zip(gray.par_iter())
        .zip(raw.par_chunks_exact(channels))
        .for_each(write);
    #[cfg(not(feature = "rayon"))]
    data.chunks_exact_mut(out_channels)
        .zip(gray.iter())
        .zip(raw.chunks_exact(channels))
        .for_each(write);

    ImageBuffer::from_vec(image.width(), image.height(), data)
        .expect("buffer holds exactly one output pixel per input pixel")
}

// -------------------------------------------------------------------------
// The real-time variant (SIGGRAPH Asia 2012)
// -------------------------------------------------------------------------

/// Quantization of the weight simplex: weights are multiples of
/// `1 / WEIGHT_STEPS`.
const WEIGHT_STEPS: u32 = 10;

/// Every weight triple on the quantized simplex, 66 of them for the default
/// ten steps.
fn candidate_weights() -> Vec<[f64; 3]> {
    let steps = f64::from(WEIGHT_STEPS);
    let mut candidates = Vec::with_capacity(66);
    for i in 0..=WEIGHT_STEPS {
        for j in 0..=(WEIGHT_STEPS - i) {
            let k = WEIGHT_STEPS - i - j;
            candidates.push([
                f64::from(i) / steps,
                f64::from(j) / steps,
                f64::from(k) / steps,
            ]);
        }
    }
    candidates
}

/// A pixel pair, as its RGB difference and the color contrast that difference
/// carries.
struct Pair {
    difference: [f64; 3],
    contrast: f64,
    /// Product of the alpha at the two endpoints, so a pair reaching into a
    /// transparent region does not steer the search.
    weight: f64,
}

/// SplitMix64, used only to pair up the sampled pixels.
///
/// A fixed generator rather than one from `rand` keeps the output of
/// [`decolorize_fast_with`] reproducible across platforms and releases.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, bound: usize) -> usize {
        (self.next() % bound as u64) as usize
    }
}

/// Chooses the best of the 66 candidate channel weightings.
fn search_linear_weights<P>(image: &Image<P>, options: FastDecolorizeOptions) -> [f64; 3]
where
    P: DecolorizePixel,
    P::Subpixel: DecolorizeSubpixel,
{
    let equal_weights = [1.0 / 3.0; 3];
    let pairs = sample_pairs(image, options);
    if pairs.is_empty() {
        // Nothing carries enough contrast to tell the candidates apart, so
        // the image is grayscale or near flat. Fall back to the channel
        // average.
        return equal_weights;
    }

    let candidates = candidate_weights();
    let scale = 1.0 / (options.sigma * options.sigma);

    let coverage: f64 = pairs.iter().map(|pair| pair.weight).sum();
    let score = |weights: &[f64; 3]| -> f64 {
        let total: f64 = pairs
            .iter()
            .map(|pair| {
                let difference = weights[0] * pair.difference[0]
                    + weights[1] * pair.difference[1]
                    + weights[2] * pair.difference[2];
                let low = -(difference - pair.contrast).powi(2) * scale;
                let high = -(difference + pair.contrast).powi(2) * scale;
                let peak = low.max(high);
                pair.weight * (peak + ((low - peak).exp() + (high - peak).exp()).ln())
            })
            .sum();
        total / coverage
    };

    #[cfg(feature = "rayon")]
    let scores: Vec<f64> = candidates.par_iter().map(score).collect();
    #[cfg(not(feature = "rayon"))]
    let scores: Vec<f64> = candidates.iter().map(score).collect();

    let best = scores
        .iter()
        .enumerate()
        .filter(|(_, score)| score.is_finite())
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i);

    match best {
        Some(i) => candidates[i],
        None => equal_weights,
    }
}

/// Draws the pixel pairs the candidate search is scored on: one randomly
/// matched pair per grid sample, plus the horizontal and vertical neighbors of
/// a grid half as fine.
///
/// The random pairs supply long range contrasts that a purely local objective
/// misses, between regions that never touch but must still be told apart. The
/// neighbor pairs keep local detail from being flattened.
fn sample_pairs<P>(image: &Image<P>, options: FastDecolorizeOptions) -> Vec<Pair>
where
    P: DecolorizePixel,
    P::Subpixel: DecolorizeSubpixel,
{
    let (width, height) = (image.width() as usize, image.height() as usize);

    // Scale the grid so that it holds about `samples²` points.
    let scale = f64::from(options.samples) / ((width * height) as f64).sqrt();
    let columns = (((width as f64 * scale) + 0.5) as usize).clamp(1, width);
    let rows = (((height as f64 * scale) + 0.5) as usize).clamp(1, height);

    let at = |column: usize, of: usize, extent: usize| -> usize {
        ((column as f64 + 0.5) * extent as f64 / of as f64) as usize
    };

    let mut pairs = Vec::new();
    let push = |a: (usize, usize), b: (usize, usize), pairs: &mut Vec<Pair>| {
        let first = image.get_pixel(a.0 as u32, a.1 as u32).to_rgba();
        let second = image.get_pixel(b.0 as u32, b.1 as u32).to_rgba();
        let weight = first.0[3].to_unit() * second.0[3].to_unit();
        if weight == 0.0 {
            return;
        }
        let difference = [
            first.0[0].to_unit() - second.0[0].to_unit(),
            first.0[1].to_unit() - second.0[1].to_unit(),
            first.0[2].to_unit() - second.0[2].to_unit(),
        ];
        // Scaled by √2, following the authors' reference implementation.
        // That is not the diameter of the RGB cube, which is √3, so a pair can
        // score slightly above 1.
        let contrast = (difference[0] * difference[0]
            + difference[1] * difference[1]
            + difference[2] * difference[2])
            .sqrt()
            / std::f64::consts::SQRT_2;

        if contrast >= options.contrast_threshold {
            pairs.push(Pair {
                difference,
                contrast,
                weight,
            });
        }
    };

    let grid: Vec<(usize, usize)> = (0..columns)
        .flat_map(|i| (0..rows).map(move |j| (i, j)))
        .map(|(i, j)| (at(i, columns, width), at(j, rows, height)))
        .collect();

    // Fisher-Yates over a copy, then pair position `i` with shuffled position `i`.
    let mut shuffled = grid.clone();
    let mut rng = SplitMix64(options.seed);
    for i in (1..shuffled.len()).rev() {
        shuffled.swap(i, rng.below(i + 1));
    }
    for (a, b) in grid.iter().zip(&shuffled) {
        push(*a, *b, &mut pairs);
    }

    let (columns, rows) = (columns / 2, rows / 2);
    for i in 0..columns.saturating_sub(1) {
        for j in 0..rows {
            let y = at(j, rows, height);
            push(
                (at(i, columns, width), y),
                (at(i + 1, columns, width), y),
                &mut pairs,
            );
        }
    }
    for i in 0..columns {
        for j in 0..rows.saturating_sub(1) {
            let x = at(i, columns, width);
            push(
                (x, at(j, rows, height)),
                (x, at(j + 1, rows, height)),
                &mut pairs,
            );
        }
    }

    pairs
}

/// Applies a convex combination of the channels. No rescaling is needed, since
/// the weights sum to one.
fn apply_linear<P>(image: &Image<P>, weights: [f64; 3]) -> Image<P::Output>
where
    P: DecolorizePixel,
    P::Subpixel: DecolorizeSubpixel,
{
    let channels = P::CHANNEL_COUNT as usize;
    let out_channels = <P::Output as Pixel>::CHANNEL_COUNT as usize;
    let raw = image.as_raw();
    let pixels = image.width() as usize * image.height() as usize;
    let mut data = vec![P::Subpixel::from_unit(0.0); pixels * out_channels];

    let map = |(slot, pixel): (&mut [P::Subpixel], &[P::Subpixel])| {
        let rgba = P::from_slice(pixel).to_rgba();
        let value = weights[0] * rgba.0[0].to_unit()
            + weights[1] * rgba.0[1].to_unit()
            + weights[2] * rgba.0[2].to_unit();
        *<P::Output as Pixel>::from_slice_mut(slot) =
            P::gray(P::Subpixel::from_unit(value), rgba.0[3]);
    };

    #[cfg(feature = "rayon")]
    data.par_chunks_exact_mut(out_channels)
        .zip(raw.par_chunks_exact(channels))
        .for_each(map);
    #[cfg(not(feature = "rayon"))]
    data.chunks_exact_mut(out_channels)
        .zip(raw.chunks_exact(channels))
        .for_each(map);

    ImageBuffer::from_vec(image.width(), image.height(), data)
        .expect("buffer holds exactly one output pixel per input pixel")
}

// -------------------------------------------------------------------------
// Color space conversion
// -------------------------------------------------------------------------

/// D65 white point, as used for the sRGB primaries below.
const WHITE_POINT: [f64; 3] = [0.950_456, 1.0, 1.088_754];

/// Breakpoint `ε` below which the CIELAB transfer function is linear.
const LAB_EPSILON: f64 = 216.0 / 24389.0;

/// Scale `κ` of that linear segment, which runs `(κt + 16) / 116`.
const LAB_KAPPA: f64 = 24389.0 / 27.0;

/// Inverts the sRGB transfer function.
#[inline]
fn srgb_to_linear(channel: f64) -> f64 {
    if channel <= 0.040_45 {
        channel / 12.92
    } else {
        ((channel + 0.055) / 1.055).powf(2.4)
    }
}

/// Converts an sRGB triple on the unit interval to CIELAB, with `L` on
/// `[0, 100]`.
fn srgb_to_lab(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let (r, g, b) = (
        srgb_to_linear(r.clamp(0.0, 1.0)),
        srgb_to_linear(g.clamp(0.0, 1.0)),
        srgb_to_linear(b.clamp(0.0, 1.0)),
    );

    let x = 0.412_453 * r + 0.357_580 * g + 0.180_423 * b;
    let y = 0.212_671 * r + 0.715_160 * g + 0.072_169 * b;
    let z = 0.019_334 * r + 0.119_193 * g + 0.950_227 * b;

    let f = |t: f64| -> f64 {
        if t > LAB_EPSILON {
            t.cbrt()
        } else {
            (LAB_KAPPA * t + 16.0) / 116.0
        }
    };

    let (fx, fy, fz) = (
        f(x / WHITE_POINT[0]),
        f(y / WHITE_POINT[1]),
        f(z / WHITE_POINT[2]),
    );

    (116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Luma, LumaA, Rgb, RgbImage, Rgba};

    /// Rec. 601 luminance, the conversion `image` itself performs. Used to
    /// construct inputs whose contrast is invisible to a luminance mapping.
    fn luma601(pixel: Rgb<u8>) -> f64 {
        0.299 * f64::from(pixel.0[0])
            + 0.587 * f64::from(pixel.0[1])
            + 0.114 * f64::from(pixel.0[2])
    }

    /// Two vertical bands of the given colors.
    fn two_bands(left: Rgb<u8>, right: Rgb<u8>, width: u32, height: u32) -> RgbImage {
        ImageBuffer::from_fn(
            width,
            height,
            |x, _| {
                if x < width / 2 { left } else { right }
            },
        )
    }

    #[test]
    fn lab_matches_reference_values() {
        // Reference values for the sRGB primaries under D65.
        let cases = [
            ([1.0, 1.0, 1.0], [100.0, 0.0, 0.0]),
            ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            ([1.0, 0.0, 0.0], [53.24, 80.09, 67.20]),
            ([0.0, 1.0, 0.0], [87.73, -86.18, 83.18]),
            ([0.0, 0.0, 1.0], [32.30, 79.19, -107.86]),
        ];

        for (rgb, expected) in cases {
            let (l, a, b) = srgb_to_lab(rgb[0], rgb[1], rgb[2]);
            for (actual, expected) in [l, a, b].iter().zip(expected) {
                assert!(
                    (actual - expected).abs() < 0.05,
                    "sRGB {rgb:?} gave ({l}, {a}, {b}), expected {expected:?}"
                );
            }
        }
    }

    #[test]
    fn pair_differences_are_forward_differences() {
        // 3 x 2 plane:  0 1 2
        //               3 4 5
        let plane = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let mut out = [0.0; 7]; // 2 * 2 horizontal + 3 * 1 vertical
        pair_differences(&plane, 3, 2, &mut out);

        assert_eq!(&out[..4], &[-1.0, -1.0, -1.0, -1.0], "horizontal pairs");
        assert_eq!(&out[4..], &[-3.0, -3.0, -3.0], "vertical pairs");
    }

    #[test]
    fn pair_weights_line_up_with_the_differences() {
        // 3 x 2, with the middle of the top row transparent. Every pair that
        // touches it must be zeroed, and no others, which only holds while the
        // weights and the differences share one traversal.
        let planes = Planes {
            width: 3,
            height: 2,
            r: vec![0.0; 6],
            g: vec![0.0; 6],
            b: vec![0.0; 6],
            coverage: vec![1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        };

        assert_eq!(
            pair_weights(&planes),
            // horizontal (0,1) (1,2) (3,4) (4,5), then vertical (0,3) (1,4) (2,5)
            vec![0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0]
        );
    }

    #[test]
    fn pair_count_covers_every_four_neighbour_pair() {
        for (width, height, expected) in [(1, 1, 0), (2, 1, 1), (1, 2, 1), (3, 2, 7), (4, 4, 24)] {
            let planes = Planes {
                width,
                height,
                r: vec![0.0; width * height],
                g: vec![0.0; width * height],
                b: vec![0.0; width * height],
                coverage: vec![1.0; width * height],
            };
            assert_eq!(planes.pair_count(), expected, "{width} x {height}");
        }
    }

    #[test]
    fn separates_isoluminant_colours() {
        // Red and green chosen so that Rec. 601 maps both onto the same gray:
        // decolorization by luminance destroys this edge entirely.
        let (left, right) = (Rgb([251u8, 0, 0]), Rgb([0u8, 128, 0]));
        assert!((luma601(left) - luma601(right)).abs() < 2.0);

        let gray = decolorize(&two_bands(left, right, 32, 32));
        let contrast = gray.get_pixel(0, 0)[0].abs_diff(gray.get_pixel(31, 0)[0]);
        assert!(contrast > 200, "contrast was only {contrast}");
    }

    #[test]
    fn fast_variant_separates_isoluminant_colours() {
        let (left, right) = (Rgb([251u8, 0, 0]), Rgb([0u8, 128, 0]));
        let gray = decolorize_fast(&two_bands(left, right, 64, 64));
        let contrast = gray.get_pixel(0, 0)[0].abs_diff(gray.get_pixel(63, 0)[0]);
        assert!(contrast > 64, "contrast was only {contrast}");
    }

    #[test]
    fn preserves_the_order_of_a_grey_ramp() {
        // Steps of 16 clear the weak order threshold, so the sign of every pair
        // is pinned and the ramp must come out increasing rather than inverted.
        let ramp: RgbImage = ImageBuffer::from_fn(16, 4, |x, _| {
            let v = (x * 16) as u8;
            Rgb([v, v, v])
        });

        let gray = decolorize(&ramp);
        for x in 1..16 {
            assert!(
                gray.get_pixel(x, 0)[0] > gray.get_pixel(x - 1, 0)[0],
                "not increasing at x = {x}: {:?}",
                (0..16).map(|x| gray.get_pixel(x, 0)[0]).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn stretches_to_the_full_output_range() {
        let image: RgbImage =
            ImageBuffer::from_fn(16, 16, |x, y| Rgb([(x * 16) as u8, (y * 16) as u8, 128]));

        let gray = decolorize(&image);
        let values: Vec<u8> = gray.pixels().map(|p| p.0[0]).collect();
        assert_eq!(values.iter().copied().min(), Some(0));
        assert_eq!(values.iter().copied().max(), Some(255));
    }

    #[test]
    fn fast_variant_preserves_brightness() {
        // The candidate weights are convex, so a flat color maps onto a
        // weighted average of its own channels and stays in range.
        let image = RgbImage::from_pixel(64, 64, Rgb([90, 90, 90]));
        let gray = decolorize_fast(&image);
        assert!(gray.pixels().all(|p| p.0[0] == 90));
    }

    #[test]
    fn is_deterministic() {
        let image: RgbImage = ImageBuffer::from_fn(48, 48, |x, y| {
            Rgb([(x * 5) as u8, (y * 5) as u8, ((x ^ y) * 5) as u8])
        });

        assert_eq!(decolorize(&image), decolorize(&image));
        assert_eq!(decolorize_fast(&image), decolorize_fast(&image));
    }

    #[test]
    fn handles_degenerate_sizes() {
        for (width, height) in [(0, 0), (0, 4), (4, 0), (1, 1), (1, 7), (7, 1), (2, 2)] {
            let image = RgbImage::from_pixel(width, height, Rgb([10, 200, 30]));
            assert_eq!(decolorize(&image).dimensions(), (width, height));
            assert_eq!(decolorize_fast(&image).dimensions(), (width, height));
        }
    }

    #[test]
    fn handles_a_constant_image() {
        let image = RgbImage::from_pixel(16, 16, Rgb([70, 140, 210]));
        let gray = decolorize(&image);
        let first = gray.get_pixel(0, 0)[0];
        assert!(gray.pixels().all(|p| p.0[0] == first));
    }

    #[test]
    fn supports_u16_and_f32_subpixels() {
        let wide: Image<Rgb<u16>> = ImageBuffer::from_fn(32, 32, |x, _| {
            if x < 16 {
                Rgb([52_800, 0, 0])
            } else {
                Rgb([0, 32_768, 0])
            }
        });
        let gray = decolorize(&wide);
        assert!(gray.get_pixel(0, 0)[0].abs_diff(gray.get_pixel(31, 0)[0]) > 50_000);

        let float: Image<Rgb<f32>> = ImageBuffer::from_fn(32, 32, |x, _| {
            if x < 16 {
                Rgb([0.81, 0.0, 0.0])
            } else {
                Rgb([0.0, 0.5, 0.0])
            }
        });
        let gray = decolorize(&float);
        assert!((gray.get_pixel(0, 0)[0] - gray.get_pixel(31, 0)[0]).abs() > 0.8);
    }

    #[test]
    fn accepts_pixel_types_other_than_rgb() {
        // An opaque RGBA image decolorizes to the same grays as its RGB
        // counterpart, with an alpha channel carried alongside.
        let rgba: Image<Rgba<u8>> = ImageBuffer::from_fn(32, 32, |x, _| {
            if x < 16 {
                Rgba([251, 0, 0, 255])
            } else {
                Rgba([0, 128, 0, 255])
            }
        });
        let rgb = two_bands(Rgb([251, 0, 0]), Rgb([0, 128, 0]), 32, 32);

        let with_alpha = decolorize(&rgba);
        let without = decolorize(&rgb);
        for (a, b) in with_alpha.pixels().zip(without.pixels()) {
            assert_eq!(a.0[0], b.0[0]);
            assert_eq!(a.0[1], 255);
        }

        // A luminance image is already gray and should survive intact.
        let luma: Image<Luma<u8>> = ImageBuffer::from_fn(16, 4, |x, _| Luma([(x * 16) as u8]));
        let gray = decolorize(&luma);
        assert_eq!(gray.get_pixel(0, 0)[0], 0);
        assert_eq!(gray.get_pixel(15, 0)[0], 255);
    }

    /// Two contrasting visible bands, plus a fully transparent one filled with
    /// `hidden`.
    fn padded_with_hidden(hidden: Rgb<u8>) -> Image<Rgba<u8>> {
        ImageBuffer::from_fn(48, 32, |x, _| match x {
            0..16 => Rgba([251, 0, 0, 255]),
            16..32 => Rgba([0, 128, 0, 255]),
            _ => Rgba([hidden.0[0], hidden.0[1], hidden.0[2], 0]),
        })
    }

    #[test]
    fn transparent_padding_is_as_good_as_not_being_there() {
        // Every pair touching the transparent strip is weighted out, which
        // leaves exactly the pairs of the visible part, and the rescaling sees
        // exactly its pixels. So the visible half has to come out the same as
        // decolorizing it on its own, to the bit.
        let cropped: RgbImage = ImageBuffer::from_fn(32, 32, |x, _| {
            if x < 16 {
                Rgb([251, 0, 0])
            } else {
                Rgb([0, 128, 0])
            }
        });
        let padded = padded_with_hidden(Rgb([255, 255, 255]));

        let expected = decolorize(&cropped);
        let actual = decolorize(&padded);
        for (x, y, pixel) in expected.enumerate_pixels() {
            assert_eq!(pixel.0[0], actual.get_pixel(x, y).0[0], "at {x}, {y}");
        }
    }

    #[test]
    fn what_hides_under_transparent_pixels_does_not_show() {
        // Same visible content, wildly different colors stored behind the
        // transparent strip. Neither the fit nor the rescaling may notice.
        let white = padded_with_hidden(Rgb([255, 255, 255]));
        let black = padded_with_hidden(Rgb([0, 0, 0]));

        for (a, b) in decolorize(&white).pixels().zip(decolorize(&black).pixels()) {
            if a.0[1] > 0 {
                assert_eq!(a.0[0], b.0[0]);
            }
        }

        for (a, b) in decolorize_fast(&white)
            .pixels()
            .zip(decolorize_fast(&black).pixels())
        {
            if a.0[1] > 0 {
                assert_eq!(a.0[0], b.0[0]);
            }
        }
    }

    #[test]
    fn transparent_pixels_do_not_eat_the_output_range() {
        // The hidden white would otherwise pin the top of the range and leave
        // the visible half compressed into the bottom of it.
        let image: Image<Rgba<u8>> = ImageBuffer::from_fn(32, 32, |x, _| match x {
            0..16 => Rgba([251, 0, 0, 255]),
            16..24 => Rgba([0, 128, 0, 255]),
            _ => Rgba([255, 255, 255, 0]),
        });

        let gray = decolorize(&image);
        let visible: Vec<u8> = gray
            .pixels()
            .filter(|p| p.0[1] > 0)
            .map(|p| p.0[0])
            .collect();
        assert_eq!(visible.iter().copied().min(), Some(0));
        assert_eq!(visible.iter().copied().max(), Some(255));
        assert!(visible.len() > 1);
    }

    #[test]
    fn alpha_survives_both_variants() {
        let image: Image<Rgba<u8>> =
            ImageBuffer::from_fn(32, 32, |x, y| Rgba([200, 40, 90, (x * 8 + y) as u8]));

        for gray in [decolorize(&image), decolorize_fast(&image)] {
            for (x, y, pixel) in gray.enumerate_pixels() {
                assert_eq!(pixel.0[1], (x * 8 + y) as u8, "alpha at {x}, {y}");
            }
        }
    }

    #[test]
    fn handles_a_fully_transparent_image() {
        let image = Image::<Rgba<u8>>::from_pixel(16, 16, Rgba([90, 10, 200, 0]));

        for gray in [decolorize(&image), decolorize_fast(&image)] {
            assert_eq!(gray.dimensions(), (16, 16));
            assert!(gray.pixels().all(|p| p.0[1] == 0));
        }
    }

    #[test]
    fn luma_alpha_input_keeps_its_alpha() {
        let image: Image<LumaA<u8>> =
            ImageBuffer::from_fn(16, 4, |x, _| LumaA([(x * 16) as u8, 128]));
        let gray = decolorize(&image);
        assert!(gray.pixels().all(|p| p.0[1] == 128));
    }

    #[test]
    fn there_are_sixty_six_candidate_weights() {
        let candidates = candidate_weights();
        assert_eq!(candidates.len(), 66);
        for weights in &candidates {
            let sum: f64 = weights.iter().sum();
            assert!((sum - 1.0).abs() < 1e-12, "{weights:?} does not sum to one");
            assert!(weights.iter().all(|w| *w >= 0.0));
        }
    }

    #[test]
    fn downscaling_bounds_the_fitting_size() {
        let image = RgbImage::from_pixel(1000, 600, Rgb([10, 20, 30]));
        let planes = Planes::downscale(&image, 800);
        assert!(planes.width + planes.height <= 800);
        assert!(planes.width > 0 && planes.height > 0);

        // Small images are left alone.
        let small = RgbImage::from_pixel(30, 20, Rgb([10, 20, 30]));
        let planes = Planes::downscale(&small, 800);
        assert_eq!((planes.width, planes.height), (30, 20));

        // Neither side is scaled below one pixel, so an extreme aspect ratio
        // overshoots the bound by one.
        let sliver = RgbImage::from_pixel(100_000, 10, Rgb([10, 20, 30]));
        let planes = Planes::downscale(&sliver, 800);
        assert_eq!((planes.width, planes.height), (800, 1));
    }

    #[test]
    fn downscaling_averages_rather_than_samples() {
        // A one pixel checkerboard averages to mid gray; point sampling would
        // return one of the two extremes instead.
        let checker: RgbImage = ImageBuffer::from_fn(64, 64, |x, y| {
            if (x + y) % 2 == 0 {
                Rgb([0, 0, 0])
            } else {
                Rgb([255, 255, 255])
            }
        });
        let planes = Planes::downscale(&checker, 16);
        assert!(planes.r.iter().all(|v| (v - 0.5).abs() < 0.02));
    }

    #[test]
    fn the_search_is_stable_across_seeds() {
        let image: RgbImage =
            ImageBuffer::from_fn(96, 96, |x, y| Rgb([(x * 2) as u8, (y * 2) as u8, 200]));

        let first = decolorize_fast_with(&image, FastDecolorizeOptions::default());
        let second = decolorize_fast_with(
            &image,
            FastDecolorizeOptions {
                seed: 12345,
                ..FastDecolorizeOptions::default()
            },
        );

        // Different draws of the random pairs, but the search is stable enough
        // that they agree on the weighting.
        assert_eq!(first, second);
    }

    #[test]
    #[should_panic(expected = "fitting_size must be non-zero")]
    fn rejects_a_zero_fitting_size() {
        let image = RgbImage::from_pixel(4, 4, Rgb([1, 2, 3]));
        let _ = decolorize_with(
            &image,
            DecolorizeOptions {
                fitting_size: 0,
                ..DecolorizeOptions::default()
            },
        );
    }

    #[test]
    #[should_panic(expected = "sigma must be positive")]
    fn rejects_a_non_positive_sigma() {
        let image = RgbImage::from_pixel(4, 4, Rgb([1, 2, 3]));
        let _ = decolorize_with(
            &image,
            DecolorizeOptions {
                sigma: 0.0,
                ..DecolorizeOptions::default()
            },
        );
    }

    #[test]
    #[should_panic(expected = "sigma must be positive")]
    fn rejects_a_non_positive_sigma_in_the_fast_variant() {
        // A zero sigma used to make every candidate score NaN, and the search
        // then quietly settled on a pure red channel conversion.
        let image = RgbImage::from_pixel(4, 4, Rgb([1, 2, 3]));
        let _ = decolorize_fast_with(
            &image,
            FastDecolorizeOptions {
                sigma: 0.0,
                ..FastDecolorizeOptions::default()
            },
        );
    }

    #[test]
    #[should_panic(expected = "samples must be non-zero")]
    fn rejects_zero_samples() {
        let image = RgbImage::from_pixel(4, 4, Rgb([1, 2, 3]));
        let _ = decolorize_fast_with(
            &image,
            FastDecolorizeOptions {
                samples: 0,
                ..FastDecolorizeOptions::default()
            },
        );
    }
}
