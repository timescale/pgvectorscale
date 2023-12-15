use criterion::{black_box, criterion_group, criterion_main, Criterion};
use timescale_vector::access_method::distance::{distance_cosine, distance_l2};

//copy and use qdrants simd code, purely for benchmarking purposes
//not used in the actual extension
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
#[target_feature(enable = "fma")]
unsafe fn hsum256_ps_avx(x: __m256) -> f32 {
    let x128: __m128 = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    let x64: __m128 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    let x32: __m128 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    _mm_cvtss_f32(x32)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
#[target_feature(enable = "fma")]
pub unsafe fn dot_similarity_avx_qdrant(v1: &[f32], v2: &[f32]) -> f32 {
    let n = v1.len();
    let m = n - (n % 32);
    let mut ptr1: *const f32 = v1.as_ptr();
    let mut ptr2: *const f32 = v2.as_ptr();
    let mut sum256_1: __m256 = _mm256_setzero_ps();
    let mut sum256_2: __m256 = _mm256_setzero_ps();
    let mut sum256_3: __m256 = _mm256_setzero_ps();
    let mut sum256_4: __m256 = _mm256_setzero_ps();
    let mut i: usize = 0;
    while i < m {
        sum256_1 = _mm256_fmadd_ps(_mm256_loadu_ps(ptr1), _mm256_loadu_ps(ptr2), sum256_1);
        sum256_2 = _mm256_fmadd_ps(
            _mm256_loadu_ps(ptr1.add(8)),
            _mm256_loadu_ps(ptr2.add(8)),
            sum256_2,
        );
        sum256_3 = _mm256_fmadd_ps(
            _mm256_loadu_ps(ptr1.add(16)),
            _mm256_loadu_ps(ptr2.add(16)),
            sum256_3,
        );
        sum256_4 = _mm256_fmadd_ps(
            _mm256_loadu_ps(ptr1.add(24)),
            _mm256_loadu_ps(ptr2.add(24)),
            sum256_4,
        );

        ptr1 = ptr1.add(32);
        ptr2 = ptr2.add(32);
        i += 32;
    }

    let mut result = hsum256_ps_avx(sum256_1)
        + hsum256_ps_avx(sum256_2)
        + hsum256_ps_avx(sum256_3)
        + hsum256_ps_avx(sum256_4);

    for i in 0..n - m {
        result += (*ptr1.add(i)) * (*ptr2.add(i));
    }
    result
}

/// Copy of Diskann's distance function. again just for benchmarking
/// not used in the actual extension
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(never)]
pub unsafe fn distance_l2_vector_f32(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();

    // make sure the addresses are bytes aligned
    debug_assert_eq!(a.as_ptr().align_offset(32), 0);
    debug_assert_eq!(b.as_ptr().align_offset(32), 0);

    unsafe {
        let mut sum = _mm256_setzero_ps();

        // Iterate over the elements in steps of 8
        for i in (0..n).step_by(8) {
            let a_vec = _mm256_load_ps(&a[i]);
            let b_vec = _mm256_load_ps(&b[i]);
            let diff = _mm256_sub_ps(a_vec, b_vec);
            sum = _mm256_fmadd_ps(diff, diff, sum);
        }

        let x128: __m128 = _mm_add_ps(_mm256_extractf128_ps(sum, 1), _mm256_castps256_ps128(sum));
        /* ( -, -, x1+x3+x5+x7, x0+x2+x4+x6 ) */
        let x64: __m128 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
        /* ( -, -, -, x0+x1+x2+x3+x4+x5+x6+x7 ) */
        let x32: __m128 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
        /* Conversion to float is a no-op on x86-64 */
        _mm_cvtss_f32(x32)
    }
}

//only used for alignment
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[repr(C, align(32))]
struct Vector32ByteAligned {
    v: [f32; 2000],
}

//the diskann version requires alignment so run benchmarks with aligned vectors
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn benchmark_distance_x86_aligned_vectors(c: &mut Criterion) {
    let a = Box::new(Vector32ByteAligned {
        v: [(); 2000].map(|_| 100.1),
    });

    let b = Box::new(Vector32ByteAligned {
        v: [(); 2000].map(|_| 22.1),
    });

    let l = a.v;
    let r = b.v;

    assert_eq!(r.as_ptr().align_offset(32), 0);
    assert_eq!(l.as_ptr().align_offset(32), 0);

    c.bench_function("distance comparison qdrant (aligned)", |b| {
        b.iter(|| unsafe { dot_similarity_avx_qdrant(black_box(&r), black_box(&l)) })
    });
    c.bench_function("distance comparison diskann (aligned)", |b| {
        b.iter(|| unsafe { distance_l2_vector_f32(black_box(&r), black_box(&l)) })
    });
}

//compare qdrant on unaligned vectors (we don't have alignment so this is apples to apples with us)
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn benchmark_distance_x86_unaligned_vectors(c: &mut Criterion) {
    let r: Vec<f32> = (0..2000).map(|v| v as f32 + 1000.1).collect();
    let l: Vec<f32> = (0..2000).map(|v| v as f32 + 2000.2).collect();

    c.bench_function("distance comparison qdrant (unaligned)", |b| {
        b.iter(|| unsafe { dot_similarity_avx_qdrant(black_box(&r), black_box(&l)) })
    });
}

fn benchmark_distance(c: &mut Criterion) {
    let r: Vec<f32> = (0..2000).map(|v| v as f32 + 1000.1).collect();
    let l: Vec<f32> = (0..2000).map(|v| v as f32 + 2000.2).collect();

    let mut group = c.benchmark_group("Distance");
    group.bench_function("distance l2", |b| {
        b.iter(|| distance_l2(black_box(&r), black_box(&l)))
    });
    group.bench_function("distance cosine", |b| {
        b.iter(|| distance_cosine(black_box(&r), black_box(&l)))
    });
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
criterion_group!(
    benches,
    benchmark_distance,
    benchmark_distance_x86_unaligned_vectors,
    benchmark_distance_x86_aligned_vectors
);
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
criterion_group!(benches, benchmark_distance);

criterion_main!(benches);
