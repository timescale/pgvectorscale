//! Calculate the distance by vector arithmetic optimized for aarch64 neon intrinsics

use core::arch::aarch64::{self, *};
use std::ops;

#[cfg(not(target_feature = "neon"))]
#[cfg(not(doc))]
compile_error!(
    "On arm, the neon feature must be enabled. Set RUSTFLAGS=\"-C target-feature=+neon\""
);

// Naming and impl done to match simdeez and the options used in the distance_l2_simd_body and
// distance_cosine_simd_body macros
struct S(float32x4_t);

impl S {
    const VF32_WIDTH: usize = 4; // 128bit register for ARM NEON

    unsafe fn setzero_ps() -> S {
        let zero: f32 = 0.0;
        S(aarch64::vld1q_dup_f32(&zero))
    }

    unsafe fn loadu_ps(a: &f32) -> S {
        S(aarch64::vld1q_f32(a))
    }

    unsafe fn horizontal_add_ps(a: S) -> f32 {
        aarch64::vaddvq_f32(a.0)
    }

    unsafe fn fmadd_ps(a: S, b: S, c: S) -> S {
        S(aarch64::vfmaq_f32(c.0, a.0, b.0))
    }
}

impl ops::Add<S> for S {
    type Output = S;

    fn add(self, rhs: S) -> Self::Output {
        unsafe { S(aarch64::vaddq_f32(self.0, rhs.0)) }
    }
}

impl ops::Sub<S> for S {
    type Output = S;

    fn sub(self, rhs: S) -> Self::Output {
        unsafe { S(aarch64::vsubq_f32(self.0, rhs.0)) }
    }
}

impl ops::Mul<S> for S {
    type Output = S;

    fn mul(self, rhs: S) -> Self::Output {
        unsafe { S(aarch64::vmulq_f32(self.0, rhs.0)) }
    }
}

pub unsafe fn distance_l2_aarch64_neon(x: &[f32], y: &[f32]) -> f32 {
    super::distance::distance_l2_simd_body!(x, y)
}

pub unsafe fn distance_cosine_aarch64_neon(x: &[f32], y: &[f32]) -> f32 {
    (1.0 - super::distance::inner_product_simd_body!(x, y)).max(0.0)
}

pub unsafe fn inner_product_aarch64_neon(x: &[f32], y: &[f32]) -> f32 {
    super::distance::inner_product_simd_body!(x, y)
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
            (unsafe { super::distance_cosine_aarch64_neon(&r, &l) }
                - super::super::distance::distance_cosine_unoptimized(&r, &l))
            .abs()
                < 0.000001
        );
        assert!(
            (unsafe { super::distance_l2_aarch64_neon(&r, &l) }
                - super::super::distance::distance_l2_unoptimized(&r, &l))
            .abs()
                < 0.000001
        );
    }
}
