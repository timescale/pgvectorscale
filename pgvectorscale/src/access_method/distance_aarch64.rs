//! Calculate the distance by vector arithmetic optimized for aarch64 neon intrinsics

use core::arch::aarch64::{self, *};

#[cfg(not(target_feature = "neon"))]
#[cfg(not(doc))]
compile_error!(
    "On arm, the neon feature must be enabled. Set RUSTFLAGS=\"-C target-feature=+neon\""
);

// Eventually would like to upstream changed into simdeez, so to simplify the transition,
// creating this struct for an easy refactor
struct S;

impl S {
    const VF32_WIDTH: usize = 4; // 128bit register for ARM NEON

    fn setzero_ps() -> float32x4_t {
        let x: float32x4_t;
        unsafe {
            let zero: f32 = 0.0;
            x = aarch64::vld1q_dup_f32(&zero);
        }
        x
    }

    fn loadu_ps(a: &f32) -> float32x4_t {
        unsafe { aarch64::vld1q_f32(a) }
    }

    fn horizontal_add_ps(a: float32x4_t) -> f32 {
        unsafe { aarch64::vaddvq_f32(a) }
    }

    fn fmadd_ps(a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t {
        unsafe { aarch64::vfmaq_f32(c, a, b) }
    }

    fn mult_ps(a: float32x4_t, b: float32x4_t) -> float32x4_t {
        unsafe { aarch64::vmulq_f32(a, b) }
    }

    fn add_ps(a: float32x4_t, b: float32x4_t) -> float32x4_t {
        unsafe { aarch64::vaddq_f32(a, b) }
    }

    fn sub_ps(a: float32x4_t, b: float32x4_t) -> float32x4_t {
        unsafe { aarch64::vsubq_f32(a, b) }
    }
}

pub fn distance_l2_aarch64_neon(x: &[f32], y: &[f32]) -> f32 {
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
    while x.len() >= S::VF32_WIDTH * 4 {
        //load data from your vec into an SIMD value
        accum0 = S::add_ps(
            accum0,
            S::mult_ps(
                S::sub_ps(
                    S::loadu_ps(&x[S::VF32_WIDTH * 0]),
                    S::loadu_ps(&y[S::VF32_WIDTH * 0]),
                ),
                S::sub_ps(
                    S::loadu_ps(&x[S::VF32_WIDTH * 0]),
                    S::loadu_ps(&y[S::VF32_WIDTH * 0]),
                ),
            ),
        );
        accum1 = S::add_ps(
            accum1,
            S::mult_ps(
                S::sub_ps(
                    S::loadu_ps(&x[S::VF32_WIDTH * 1]),
                    S::loadu_ps(&y[S::VF32_WIDTH * 1]),
                ),
                S::sub_ps(
                    S::loadu_ps(&x[S::VF32_WIDTH * 1]),
                    S::loadu_ps(&y[S::VF32_WIDTH * 1]),
                ),
            ),
        );
        accum2 = S::add_ps(
            accum2,
            S::mult_ps(
                S::sub_ps(
                    S::loadu_ps(&x[S::VF32_WIDTH * 2]),
                    S::loadu_ps(&y[S::VF32_WIDTH * 2]),
                ),
                S::sub_ps(
                    S::loadu_ps(&x[S::VF32_WIDTH * 2]),
                    S::loadu_ps(&y[S::VF32_WIDTH * 2]),
                ),
            ),
        );
        accum3 = S::add_ps(
            accum3,
            S::mult_ps(
                S::sub_ps(
                    S::loadu_ps(&x[S::VF32_WIDTH * 3]),
                    S::loadu_ps(&y[S::VF32_WIDTH * 3]),
                ),
                S::sub_ps(
                    S::loadu_ps(&x[S::VF32_WIDTH * 3]),
                    S::loadu_ps(&y[S::VF32_WIDTH * 3]),
                ),
            ),
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
        let diff = x[i] - y[i];
        dist += diff * diff;
    }

    assert!(dist >= 0.);
    //dist.sqrt()
    dist
}

pub fn distance_cosine_aarch64_neon(x: &[f32], y: &[f32]) -> f32 {
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

    (1.0 - dist).max(0.0)
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
            (super::distance_cosine_aarch64_neon(&r, &l)
                - super::super::distance::distance_cosine_unoptimized(&r, &l))
            .abs()
                < 0.000001
        );
        assert!(
            (super::distance_l2_aarch64_neon(&r, &l)
                - super::super::distance::distance_l2_unoptimized(&r, &l))
            .abs()
                < 0.000001
        );
    }
}
