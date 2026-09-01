# decolorize

[![Crates.io Version](https://img.shields.io/crates/v/decolorize?style=for-the-badge&logo=rust&color=dea584)](https://crates.io/crates/decolorize)
[![docs.rs](https://img.shields.io/docsrs/decolorize?style=for-the-badge&logo=docsdotrs&color=%23000000)](https://docs.rs/decolorize)
[![Crates.io License](https://img.shields.io/crates/l/decolorize?style=for-the-badge&logo=opensourceinitiative&logoColor=white&color=3DA639)](https://github.com/zlfn/decolorize/blob/main/LICENSE)

Converts color images to grayscale while preserving the contrast between colors
that ordinary luminance conversions flatten out.

![Comparison on Monet's Impression, Sunrise](https://raw.githubusercontent.com/zlfn/decolorize/main/comparison.png)

## Usage

```rust
use decolorize::{decolorize, decolorize_fast};

let gray = decolorize(&image);       // fits a degree-2 polynomial mapping
let gray = decolorize_fast(&image);  // picks the best of 66 channel weightings
```

`decolorize` reproduces the color contrast most closely; `decolorize_fast` is
about an order of magnitude cheaper, because it searches a fixed number of pixel
pairs instead of solving for a mapping. Both take any `image` pixel type
(`Rgb`, `Rgba`, `Luma`, `LumaA`) with `u8`, `u16`, `f32` or `f64` samples, and
return `Luma`, or `LumaA` where the input carries alpha.

`decolorize_with` and `decolorize_fast_with` take an options struct if you need
to tune σ, the iteration budget or the sampling.

Enabling the default `rayon` feature parallelizes the work without changing the
output. Results are byte-identical either way.

## References

Both variants are ports of the work of Cewu Lu, Li Xu and Jiaya Jia, whose
[project page][project] collects the papers, the benchmark data and their
reference implementation.

* [*Contrast Preserving Decolorization*][iccp12] — ICCP 2012. The algorithm
  behind `decolorize`: a degree-2 polynomial color mapping whose nine
  coefficients are fitted by maximum likelihood under a bimodal Gaussian prior
  on the contrast of each neighboring pixel pair, solved by fixed point
  iteration.
* [*Contrast Preserving Decolorization with Perception-Based Quality
  Metrics*][ijcv14] — IJCV 2014. The journal version, which adds the CCPR and
  E-score metrics used to evaluate this port.
* [*Real-time Contrast Preserving Decolorization*][siga12] — SIGGRAPH Asia 2012
  Technical Briefs. The algorithm behind `decolorize_fast`: the same idea with
  the weak color order dropped and the mapping restricted to convex
  combinations of the three channels, quantized to 66 candidates and scored on
  a sparse sample of pixel pairs.

The authors' own implementation ships in OpenCV as [`cv::decolor`][opencv].
`decolorize` matches it to a mean absolute correlation of 0.992 across the
24 image benchmark, differing only in that it evaluates the published objective
for its convergence test rather than the smoothed variant OpenCV uses.

[project]: https://www.cse.cuhk.edu.hk/~leojia/projects/color2gray/
[iccp12]: http://www.cse.cuhk.edu.hk/~leojia/papers/decolorization_iccp12.pdf
[ijcv14]: http://www.cse.cuhk.edu.hk/~leojia/papers/decolorization_ijcv14.pdf
[siga12]: https://dl.acm.org/doi/10.1145/2407746.2407780
[opencv]: https://docs.opencv.org/4.x/d4/d32/group__photo__decolor.html
