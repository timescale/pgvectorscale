//! Calculate the distance by vector arithmetic optimized for x86

use simdeez::avx2::*;
use simdeez::scalar::*;
use simdeez::sse2::*;
use simdeez::sse41::*;

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
        super::distance::distance_l2_simd_body!(x, y)
    }
);

simdeez::simd_runtime_generate!(
    pub fn inner_product_x86(x: &[f32], y: &[f32]) -> f32 {
        super::distance::inner_product_simd_body!(x, y)
    }
);

/// Calculate the cosine distance between two normal vectors
pub unsafe fn distance_cosine_x86_avx2(x: &[f32], y: &[f32]) -> f32 {
    (1.0 - inner_product_x86_avx2(x, y)).max(0.0)
}

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
