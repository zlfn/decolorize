//! Decolorizes images with both variants and reports the time each took.
//!
//! Also writes `image`'s own `to_luma8` conversion as a baseline to compare
//! against.
//!
//! ```text
//! cargo run --release --example decolorize_image -- OUT_DIR IMAGE [IMAGE ...]
//! ```
//!
//! `DECOLORIZE_ITERATIONS` and `DECOLORIZE_TOLERANCE` override the solver
//! settings, which is useful when comparing against other implementations.

use std::path::{Path, PathBuf};
use std::time::Instant;

use decolorize::decolorize::{DecolorizeOptions, decolorize_fast, decolorize_with};
use image::{DynamicImage, ImageReader};

fn main() {
    let mut args = std::env::args().skip(1);
    let out_dir = PathBuf::from(args.next().expect("usage: OUT_DIR IMAGE [IMAGE ...]"));
    std::fs::create_dir_all(&out_dir).expect("could not create output directory");

    let mut options = DecolorizeOptions::default();
    if let Ok(v) = std::env::var("DECOLORIZE_ITERATIONS") {
        options.max_iterations = v.parse().expect("DECOLORIZE_ITERATIONS");
    }
    if let Ok(v) = std::env::var("DECOLORIZE_TOLERANCE") {
        options.tolerance = v.parse().expect("DECOLORIZE_TOLERANCE");
    }

    for path in args {
        let path = Path::new(&path);
        let image = ImageReader::open(path)
            .and_then(|r| r.with_guessed_format())
            .expect("could not open image")
            .decode()
            .expect("could not decode image")
            .to_rgb8();

        let stem = path.file_stem().unwrap().to_string_lossy().to_string();
        let (width, height) = (image.width(), image.height());

        let start = Instant::now();
        let cpd = decolorize_with(&image, options);
        let cpd_time = start.elapsed();

        let start = Instant::now();
        let fast = decolorize_fast(&image);
        let fast_time = start.elapsed();

        cpd.save(out_dir.join(format!("{stem}_cpd.png"))).unwrap();
        fast.save(out_dir.join(format!("{stem}_fast.png"))).unwrap();
        DynamicImage::ImageRgb8(image)
            .to_luma8()
            .save(out_dir.join(format!("{stem}_luma.png")))
            .unwrap();

        println!("{stem}: {width}x{height}  cpd {cpd_time:>8.1?}  fast {fast_time:>8.1?}");
    }
}
