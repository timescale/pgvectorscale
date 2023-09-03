
use std::ops::{Div, Mul, Sub};

#[inline]
pub fn dot(from: &[f32], to: &[f32]) -> f32 {
    from.iter().zip(to.iter()).map(|(x, y)| x.mul(*y)).sum()
}
/// Calculate the L2 distance between two vectors, using scalar operations.
///
/// Rely on compiler auto-vectorization.
#[inline]
 fn l2_scalar(from: &[f32], to: &[f32]) -> f32 {
    from.iter()
        .zip(to.iter())
        .map(|(a, b)| (a.sub(*b).powi(2)))
        .sum::<f32>()
}

/// Fallback non-SIMD implementation
#[allow(dead_code)] // Does not fallback on aarch64.
#[inline]
fn cosine_scalar(x: &[f32], y: &[f32], x_norm: f32) -> f32 {
    let y_sq = dot(y, y);
    let xy = dot(x, y);
    // 1 - xy / (sqrt(x_sq) * sqrt(y_sq))
    1f32.sub(xy.div(x_norm.mul(y_sq.sqrt())))
}

#[inline]
pub fn l2_dist(from: &[f32], to: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        // TODO: Only known platform that does not support FMA is Github Action Mac(Intel) Runner.
        // However, it introduces one more branch, which may affect performance.
        if is_x86_feature_detected!("avx2") {
            // AVX2 / FMA is the lowest x86_64 CPU requirement (released from 2011) for Lance.
            use x86_64::avx::l2_f32;
            return l2_f32(from, to);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // Neon is the lowest aarch64 CPU requirement (available in all Apple Silicon / Arm V7+).
        use aarch64::neon::l2_f32;
        l2_f32(from, to)
    }

    // Fallback on x86_64 without AVX2 / FMA, or other platforms.
    #[cfg(not(target_arch = "aarch64"))]
    l2_scalar(from, to)
}

#[inline]
pub fn cosine(from: &[f32], to: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")] {
        use crate::access_method::distance::aarch64::neon::norm_l2;
        let x_norm = norm_l2(from);
        cosine_fast(x_norm, from, to)
    }


    #[cfg(target_arch = "x86_64")] {
        use crate::access_method::distance::x86_64::avx::norm_l2_f32;
        let x_norm = norm_l2_f32(from);
        cosine_fast(x_norm, from, to)
    }

}

#[inline]
fn cosine_fast(x_norm: f32, from: &[f32], other: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        use crate::access_method::distance::aarch64::neon::norm_l2;
        aarch64::neon::cosine_f32(from, other, x_norm)
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("fma") {
            return x86_64::avx::cosine_f32(from, other, x_norm);
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    cosine_scalar(from, other, x_norm)
}

#[cfg(target_arch = "x86_64")]
mod x86_64 {
    use std::arch::x86_64::*;
    pub mod avx {
        use super::*;
        use crate::access_method::distance::l2_scalar;
        #[inline]
        pub unsafe fn add_f32_register(x: std::arch::x86_64::__m256) -> f32 {
            use std::arch::x86_64::*;

            let mut sums = x;
            let mut shift = _mm256_permute2f128_ps(sums, sums, 1);
            // [x0+x4, x1+x5, ..]
            sums = _mm256_add_ps(sums, shift);
            shift = _mm256_permute_ps(sums, 14);
            sums = _mm256_add_ps(sums, shift);
            sums = _mm256_hadd_ps(sums, sums);
            let mut results: [f32; 8] = [0f32; 8];
            _mm256_storeu_ps(results.as_mut_ptr(), sums);
            results[0]
        }

        pub fn norm_l2_f32(vector: &[f32]) -> f32 {
            let len = vector.len() / 8 * 8;
            let mut sum = unsafe {
                let mut sums = _mm256_setzero_ps();
                vector.chunks_exact(8).for_each(|chunk| {
                    let x = _mm256_loadu_ps(chunk.as_ptr());
                    sums = _mm256_fmadd_ps(x, x, sums);
                });
                add_f32_register(sums)
            };
            sum += vector[len..].iter().map(|v| v * v).sum::<f32>();
            sum.sqrt()
        }
        #[inline]
        pub fn cosine_f32(x_vector: &[f32], y_vector: &[f32], x_norm: f32) -> f32 {
            unsafe {
                let len = x_vector.len() / 8 * 8;
                let mut xy = _mm256_setzero_ps();
                let mut y_sq = _mm256_setzero_ps();
                for i in (0..len).step_by(8) {
                    let x = _mm256_loadu_ps(x_vector.as_ptr().add(i));
                    let y = _mm256_loadu_ps(y_vector.as_ptr().add(i));
                    xy = _mm256_fmadd_ps(x, y, xy);
                    y_sq = _mm256_fmadd_ps(y, y, y_sq);
                }
                // handle remaining elements
                let mut dotprod = add_f32_register(xy);
                dotprod += dot_f32(&x_vector[len..], &y_vector[len..]);
                let mut y_sq_sum = add_f32_register(y_sq);
                y_sq_sum += norm_l2_f32(&y_vector[len..]).powi(2);
                1.0 - dotprod / (x_norm * y_sq_sum.sqrt())
            }
        }
        #[inline]
        pub fn l2_f32(from: &[f32], to: &[f32]) -> f32 {
            unsafe {
                use std::arch::x86_64::*;
                debug_assert_eq!(from.len(), to.len());

                // Get the potion of the vector that is aligned to 32 bytes.
                let len = from.len() / 8 * 8;
                let mut sums = _mm256_setzero_ps();
                for i in (0..len).step_by(8) {
                    let left = _mm256_loadu_ps(from.as_ptr().add(i));
                    let right = _mm256_loadu_ps(to.as_ptr().add(i));
                    let sub = _mm256_sub_ps(left, right);
                    // sum = sub * sub + sum
                    sums = _mm256_fmadd_ps(sub, sub, sums);
                }
                // Shift and add vector, until only 1 value left.
                // sums = [x0-x7], shift = [x4-x7]
                let mut shift = _mm256_permute2f128_ps(sums, sums, 1);
                // [x0+x4, x1+x5, ..]
                sums = _mm256_add_ps(sums, shift);
                shift = _mm256_permute_ps(sums, 14);
                sums = _mm256_add_ps(sums, shift);
                sums = _mm256_hadd_ps(sums, sums);
                let mut results: [f32; 8] = [0f32; 8];
                _mm256_storeu_ps(results.as_mut_ptr(), sums);

                // Remaining
                results[0] += l2_scalar(&from[len..], &to[len..]);
                results[0]
            }
        }
        pub fn dot_f32(x: &[f32], y: &[f32]) -> f32 {
            let len = x.len() / 8 * 8;
            let mut sum = unsafe {
                let mut sums = _mm256_setzero_ps();
                x.chunks_exact(8).zip(y.chunks_exact(8)).for_each(|(a, b)| {
                    let x = _mm256_loadu_ps(a.as_ptr());
                    let y = _mm256_loadu_ps(b.as_ptr());
                    sums = _mm256_fmadd_ps(x, y, sums);
                });
                add_f32_register(sums)
            };
            sum += x[len..]
                .iter()
                .zip(y[len..].iter())
                .map(|(a, b)| a * b)
                .sum::<f32>();
            sum
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use std::arch::aarch64::*;

    pub mod neon {
        use super::super::l2_scalar;
        use super::*;
        use crate::access_method::distance::dot;
        use std::arch::aarch64::*;

        // TODO: learn rust macro and refactor to macro instead of manually unroll
        const UNROLL_FACTOR: usize = 4;
        const INSTRUCTION_WIDTH: usize = 4;

        const STEP_SIZE: usize = UNROLL_FACTOR * INSTRUCTION_WIDTH;
        #[inline]
        pub fn l2_f32(from: &[f32], to: &[f32]) -> f32 {
            unsafe {
                let len_aligned_to_unroll = from.len() / STEP_SIZE * STEP_SIZE;
                let buf = [0.0_f32; 4];
                let mut sum1 = vld1q_f32(buf.as_ptr());
                let mut sum2 = vld1q_f32(buf.as_ptr());
                let mut sum3 = vld1q_f32(buf.as_ptr());
                let mut sum4 = vld1q_f32(buf.as_ptr());
                for i in (0..len_aligned_to_unroll).step_by(STEP_SIZE) {
                    let left = vld1q_f32(from.as_ptr().add(i));
                    let right = vld1q_f32(to.as_ptr().add(i));
                    let sub1 = vsubq_f32(left, right);
                    sum1 = vfmaq_f32(sum1, sub1, sub1);

                    let left = vld1q_f32(from.as_ptr().add(i + 4));
                    let right = vld1q_f32(to.as_ptr().add(i + 4));
                    let sub2 = vsubq_f32(left, right);
                    sum2 = vfmaq_f32(sum2, sub2, sub2);

                    let left = vld1q_f32(from.as_ptr().add(i + 8));
                    let right = vld1q_f32(to.as_ptr().add(i + 8));
                    let sub3 = vsubq_f32(left, right);
                    sum3 = vfmaq_f32(sum3, sub3, sub3);

                    let left = vld1q_f32(from.as_ptr().add(i + 12));
                    let right = vld1q_f32(to.as_ptr().add(i + 12));
                    let sub4 = vsubq_f32(left, right);
                    sum4 = vfmaq_f32(sum4, sub4, sub4);
                }

                let mut sum = vaddq_f32(vaddq_f32(sum1, sum2), vaddq_f32(sum3, sum4));

                let len_aligned_to_instruction = from.len() / INSTRUCTION_WIDTH * INSTRUCTION_WIDTH;

                // non-unrolled tail
                for i in
                    (len_aligned_to_unroll..len_aligned_to_instruction).step_by(INSTRUCTION_WIDTH)
                {
                    let left = vld1q_f32(from.as_ptr().add(i));
                    let right = vld1q_f32(to.as_ptr().add(i));
                    let sub = vsubq_f32(left, right);
                    sum = vfmaq_f32(sum, sub, sub);
                }

                // non vectorized tail
                let mut sum = vaddvq_f32(sum);
                sum += l2_scalar(
                    &from[len_aligned_to_instruction..],
                    &to[len_aligned_to_instruction..],
                );
                sum
            }
        }

        #[inline]
        pub fn cosine_f32(x: &[f32], y: &[f32], x_norm: f32) -> f32 {
            unsafe {
                let len = x.len() / 16 * 16;
                let buf = [0.0_f32; 4];
                let mut xy = vld1q_f32(buf.as_ptr());
                let mut y_sq = xy;

                let mut xy1 = vld1q_f32(buf.as_ptr());
                let mut y_sq1 = xy1;

                let mut xy2 = vld1q_f32(buf.as_ptr());
                let mut y_sq2 = xy2;

                let mut xy3 = vld1q_f32(buf.as_ptr());
                let mut y_sq3 = xy3;
                for i in (0..len).step_by(16) {
                    let left = vld1q_f32(x.as_ptr().add(i));
                    let right = vld1q_f32(y.as_ptr().add(i));
                    xy = vfmaq_f32(xy, left, right);
                    y_sq = vfmaq_f32(y_sq, right, right);

                    let left1 = vld1q_f32(x.as_ptr().add(i + 4));
                    let right1 = vld1q_f32(y.as_ptr().add(i + 4));
                    xy1 = vfmaq_f32(xy1, left1, right1);
                    y_sq1 = vfmaq_f32(y_sq1, right1, right1);

                    let left2 = vld1q_f32(x.as_ptr().add(i + 8));
                    let right2 = vld1q_f32(y.as_ptr().add(i + 8));
                    xy2 = vfmaq_f32(xy2, left2, right2);
                    y_sq2 = vfmaq_f32(y_sq2, right2, right2);

                    let left3 = vld1q_f32(x.as_ptr().add(i + 12));
                    let right3 = vld1q_f32(y.as_ptr().add(i + 12));
                    xy3 = vfmaq_f32(xy3, left3, right3);
                    y_sq3 = vfmaq_f32(y_sq3, right3, right3);
                }
                xy = vaddq_f32(vaddq_f32(xy, xy3), vaddq_f32(xy1, xy2));
                y_sq = vaddq_f32(vaddq_f32(y_sq, y_sq3), vaddq_f32(y_sq1, y_sq2));
                // handle remaining elements
                let mut dotprod = vaddvq_f32(xy);
                dotprod += dot(&x[len..], &y[len..]);
                let mut y_sq_sum = vaddvq_f32(y_sq);
                y_sq_sum += norm_l2(&y[len..]).powi(2);
                1.0 - dotprod / (x_norm * y_sq_sum.sqrt())
            }
        }
        pub fn norm_l2(vector: &[f32]) -> f32 {
            let len = vector.len() / 4 * 4;
            let mut sum = unsafe {
                let buf = [0.0_f32; 4];
                let mut sum = vld1q_f32(buf.as_ptr());
                for i in (0..len).step_by(4) {
                    let x = vld1q_f32(vector.as_ptr().add(i));
                    sum = vfmaq_f32(sum, x, x);
                }
                vaddvq_f32(sum)
            };
            sum += vector[len..].iter().map(|v| v.powi(2)).sum::<f32>();
            sum.sqrt()
        }
    }
}
