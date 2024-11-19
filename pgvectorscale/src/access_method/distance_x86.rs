//! Calculate the distance by vector arithmetic optimized for x86

use simdeez::scalar::*;
use simdeez::sse2::*;
use simdeez::sse41::*;
//use simdeez::avx::*;
use simdeez::avx2::*;

#[cfg(not(target_feature = "avx2"))]
#[cfg(not(doc))]
compile_error!(
    "On x86, the AVX2 feature must be enabled. Set RUSTFLAGS=\"-C target-feature=+avx2,+fma\""
);

//note: without fmadd, the performance degrades pretty badly. Benchmark before disbaling
#[cfg(not(target_feature = "fma"))]
#[cfg(not(doc))]
compile_error!(
    "On x86, the fma feature must be enabled. Set RUSTFLAGS=\"-C target-feature=+avx2,+fma\""
);

simdeez::simd_runtime_generate!(
    pub fn distance_l2_x86(x: &[f32], y: &[f32]) -> f32 {
        let mut accum0 = S::setzero_ps();
        let mut accum1 = S::setzero_ps();
        let mut accum2 = S::setzero_ps();
        let mut accum3 = S::setzero_ps();

        //assert!(x.len() == y.len());
        let mut x = &x[..];
        let mut y = &y[..];

        // Operations have to be done in terms of the vector width
        // so that it will work with any size vector.
        // the width of a vector type is provided as a constant
        // so the compiler is free to optimize it more.
        // S::VF32_WIDTH is a constant, 4 when using SSE, 8 when using AVX2, etc
        while x.len() >= S::VF32_WIDTH * 4 {
            //load data from your vec into an SIMD value
            accum0 = accum0
                + ((S::loadu_ps(&x[S::VF32_WIDTH * 0]) - S::loadu_ps(&y[S::VF32_WIDTH * 0]))
                    * (S::loadu_ps(&x[S::VF32_WIDTH * 0]) - S::loadu_ps(&y[S::VF32_WIDTH * 0])));
            accum1 = accum1
                + ((S::loadu_ps(&x[S::VF32_WIDTH * 1]) - S::loadu_ps(&y[S::VF32_WIDTH * 1]))
                    * (S::loadu_ps(&x[S::VF32_WIDTH * 1]) - S::loadu_ps(&y[S::VF32_WIDTH * 1])));
            accum2 = accum2
                + ((S::loadu_ps(&x[S::VF32_WIDTH * 2]) - S::loadu_ps(&y[S::VF32_WIDTH * 2]))
                    * (S::loadu_ps(&x[S::VF32_WIDTH * 2]) - S::loadu_ps(&y[S::VF32_WIDTH * 2])));
            accum3 = accum3
                + ((S::loadu_ps(&x[S::VF32_WIDTH * 3]) - S::loadu_ps(&y[S::VF32_WIDTH * 3]))
                    * (S::loadu_ps(&x[S::VF32_WIDTH * 3]) - S::loadu_ps(&y[S::VF32_WIDTH * 3])));

            // Move each slice to the next position
            x = &x[S::VF32_WIDTH * 4..];
            y = &y[S::VF32_WIDTH * 4..];
        }

        let mut dist = S::horizontal_add_ps(accum0)
            + S::horizontal_add_ps(accum1)
            + S::horizontal_add_ps(accum2)
            + S::horizontal_add_ps(accum3);

        // compute for the remaining elements
        for i in 0..x.len() {
            let diff = x[i] - y[i];
            dist += diff * diff;
        }

        assert!(dist >= 0.);
        //dist.sqrt()
        dist
    }
);

simdeez::simd_runtime_generate!(
    pub fn inner_product_x86(x: &[f32], y: &[f32]) -> f32 {
        let mut accum0 = S::setzero_ps();
        let mut accum1 = S::setzero_ps();
        let mut accum2 = S::setzero_ps();
        let mut accum3 = S::setzero_ps();

        let mut x = &x[..];
        let mut y = &y[..];

        //assert!(x.len() == y.len());

        // Operations have to be done in terms of the vector width
        // so that it will work with any size vector.
        // the width of a vector type is provided as a constant
        // so the compiler is free to optimize it more.
        // S::VF32_WIDTH is a constant, 4 when using SSE, 8 when using AVX2, etc
        while x.len() >= S::VF32_WIDTH * 4 {
            accum0 = S::fmadd_ps(
                S::loadu_ps(&x[S::VF32_WIDTH * 0]),
                S::loadu_ps(&y[S::VF32_WIDTH * 0]),
                accum0,
            );
            accum1 = S::fmadd_ps(
                S::loadu_ps(&x[S::VF32_WIDTH * 1]),
                S::loadu_ps(&y[S::VF32_WIDTH * 1]),
                accum1,
            );
            accum2 = S::fmadd_ps(
                S::loadu_ps(&x[S::VF32_WIDTH * 2]),
                S::loadu_ps(&y[S::VF32_WIDTH * 2]),
                accum2,
            );
            accum3 = S::fmadd_ps(
                S::loadu_ps(&x[S::VF32_WIDTH * 3]),
                S::loadu_ps(&y[S::VF32_WIDTH * 3]),
                accum3,
            );

            // Move each slice to the next position
            x = &x[S::VF32_WIDTH * 4..];
            y = &y[S::VF32_WIDTH * 4..];
        }

        let mut dist = S::horizontal_add_ps(accum0)
            + S::horizontal_add_ps(accum1)
            + S::horizontal_add_ps(accum2)
            + S::horizontal_add_ps(accum3);

        // compute for the remaining elements
        for i in 0..x.len() {
            dist += x[i] * y[i];
        }

        dist
    }

    /// Calculate the cosine distance between two normal vectors
    pub fn cosine_distance_x86(x: &[f32], y: &[f32]) -> f32 {
        (1.0 - inner_product_x86(x, y)).max(0.0)
    }
);

#[cfg(test)]
mod tests {
    #[test]
    fn distances_equal() {
        let r: Vec<f32> = (0..2000).map(|v| v as f32 + 1.0).collect();
        let l: Vec<f32> = (0..2000).map(|v| v as f32 + 2.0).collect();

        let r_size = r.iter().map(|v| v * v).sum::<f32>().sqrt();
        let l_size = l.iter().map(|v| v * v).sum::<f32>().sqrt();

        let r: Vec<f32> = r.iter().map(|v| v / r_size).collect();
        let l: Vec<f32> = l.iter().map(|v| v / l_size).collect();

        assert!(
            (unsafe { super::distance_cosine_x86_avx2(&r, &l) }
                - super::super::distance::distance_cosine_unoptimized(&r, &l))
            .abs()
                < 0.000001
        );
        assert!(
            (unsafe { super::distance_l2_x86_avx2(&r, &l) }
                - super::super::distance::distance_l2_unoptimized(&r, &l))
            .abs()
                < 0.000001
        );
    }
}
