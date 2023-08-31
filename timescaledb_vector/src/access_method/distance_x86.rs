//! Calculate the distance by vector arithmetic optimized for x86

use simdeez::scalar::*;
use simdeez::sse2::*;
use simdeez::sse41::*;
use simdeez::*;
//use simdeez::avx::*;
use simdeez::avx2::*;

simdeez::simd_runtime_generate!(
    pub fn distance_opt(x: &[f32], y: &[f32]) -> f32 {
        let mut res = S::setzero_ps();

        assert!(x.len() == y.len());
        let mut x = &x[..];
        let mut y = &y[..];

        // Operations have to be done in terms of the vector width
        // so that it will work with any size vector.
        // the width of a vector type is provided as a constant
        // so the compiler is free to optimize it more.
        // S::VF32_WIDTH is a constant, 4 when using SSE, 8 when using AVX2, etc
        while x.len() >= S::VF32_WIDTH {
            //load data from your vec into an SIMD value
            let xv = S::loadu_ps(&x[0]);
            let yv = S::loadu_ps(&y[0]);

            let mut diff = S::sub_ps(xv, yv);
            diff *= diff;

            res = res + diff;

            // Move each slice to the next position
            x = &x[S::VF32_WIDTH..];
            y = &y[S::VF32_WIDTH..];
        }

        let mut dist = S::horizontal_add_ps(res);

        // compute for the remaining elements
        for i in 0..x.len() {
            let diff = x[i] - y[i];
            dist += diff * diff;
        }

        assert!(dist >= 0.);
        dist.sqrt()
    }
);
