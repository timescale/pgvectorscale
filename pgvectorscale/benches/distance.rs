use criterion::{black_box, criterion_group, criterion_main, Criterion};
use vectorscale::access_method::distance::{
    distance_cosine, distance_cosine_unoptimized, distance_l2,
    distance_l2_optimized_for_few_dimensions, distance_l2_unoptimized, distance_xor_optimized,
};

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
    group.bench_function("distance l2 unoptimized", |b| {
        b.iter(|| distance_l2_unoptimized(black_box(&r), black_box(&l)))
    });
    group.bench_function("distance cosine", |b| {
        b.iter(|| distance_cosine(black_box(&r), black_box(&l)))
    });
    group.bench_function("distance cosine unoptimized", |b| {
        b.iter(|| distance_cosine_unoptimized(black_box(&r), black_box(&l)))
    });
}

#[inline(always)]
pub fn distance_l2_fixed_size_opt(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), 6);
    let norm: f32 = a[..6]
        .iter()
        .zip(b[..6].iter())
        .map(|t| ({ *t.0 } - { *t.1 }) * ({ *t.0 } - { *t.1 }))
        .sum();
    assert!(norm >= 0.);
    //don't sqrt for performance. These are only used for ordering so sqrt not needed
    norm
}

//PQ uses l2 distance on small vectors (6 dims or so). Benchmark that.
fn benchmark_distance_few_dimensions(c: &mut Criterion) {
    let r: Vec<f32> = (0..6).map(|v| v as f32 + 1000.1).collect();
    let l: Vec<f32> = (0..6).map(|v| v as f32 + 2000.2).collect();

    let mut group = c.benchmark_group("Distance");
    group.bench_function("pq distance l2 optimized for many dimensions", |b| {
        b.iter(|| distance_l2(black_box(&r), black_box(&l)))
    });
    group.bench_function("pq distance l2 unoptimized", |b| {
        b.iter(|| distance_l2_unoptimized(black_box(&r), black_box(&l)))
    });
    group.bench_function(
        "pq distance l2 auto-vectorized for 6 dimensionl arrays",
        |b| b.iter(|| distance_l2_fixed_size_opt(black_box(&r), black_box(&l))),
    );
    group.bench_function(
        "pq distance l2 optimized for few dimensions (what's used in the code now)",
        |b| b.iter(|| distance_l2_optimized_for_few_dimensions(black_box(&r), black_box(&l))),
    );
}

fn pack_bools_to_u8(bools: Vec<bool>) -> Vec<u8> {
    let mut bytes = vec![0u8; (bools.len() + 7) / 8];

    for (i, &b) in bools.iter().enumerate() {
        let byte_index = i / 8;
        let bit_index = i % 8;

        if b {
            bytes[byte_index] |= 1 << bit_index;
        }
    }

    bytes
}

fn pack_bools_to_u64(bools: Vec<bool>) -> Vec<u64> {
    let mut u64s = vec![0u64; (bools.len() + 63) / 64];

    for (i, &b) in bools.iter().enumerate() {
        let u64_index = i / 64;
        let bit_index = i % 64;

        if b {
            u64s[u64_index] |= 1 << bit_index;
        }
    }

    u64s
}

fn pack_bools_to_u128(bools: Vec<bool>) -> Vec<u128> {
    let mut u128s = vec![0u128; (bools.len() + 127) / 128];

    for (i, &b) in bools.iter().enumerate() {
        let u128_index = i / 128;
        let bit_index = i % 128;

        if b {
            u128s[u128_index] |= 1 << bit_index;
        }
    }

    u128s
}

fn xor_unoptimized_u8(v1: &[u8], v2: &[u8]) -> usize {
    let mut result = 0;
    for (b1, b2) in v1.iter().zip(v2.iter()) {
        result += (b1 ^ b2).count_ones() as usize;
    }
    result
}

fn xor_unoptimized_u8_fixed_size(v1: &[u8], v2: &[u8]) -> usize {
    let mut result = 0;
    for (b1, b2) in v1[..192].iter().zip(v2[..192].iter()) {
        result += (b1 ^ b2).count_ones() as usize;
    }
    result
}

fn xor_unoptimized_u64(v1: &[u64], v2: &[u64]) -> usize {
    let mut result = 0;
    for (b1, b2) in v1.iter().zip(v2.iter()) {
        result += (b1 ^ b2).count_ones() as usize;
    }
    result
}

fn xor_unoptimized_u64_fixed_size(v1: &[u64], v2: &[u64]) -> usize {
    let mut result = 0;
    for (b1, b2) in v1[..24].iter().zip(v2[..24].iter()) {
        result += (b1 ^ b2).count_ones() as usize;
    }
    result
}

fn xor_unoptimized_u64_fixed_size_map(v1: &[u64], v2: &[u64]) -> usize {
    v1[..24]
        .iter()
        .zip(v2[..24].iter())
        .map(|(&l, &r)| (l ^ r).count_ones() as usize)
        .sum()
}

fn xor_unoptimized_u128(v1: &[u128], v2: &[u128]) -> usize {
    let mut result = 0;
    for (b1, b2) in v1.iter().zip(v2.iter()) {
        result += (b1 ^ b2).count_ones() as usize;
    }
    result
}

fn xor_unoptimized_u128_fixed_size(v1: &[u128], v2: &[u128]) -> usize {
    let mut result = 0;
    for (b1, b2) in v1[..12].iter().zip(v2[..12].iter()) {
        result += (b1 ^ b2).count_ones() as usize;
    }
    result
}

fn benchmark_distance_xor(c: &mut Criterion) {
    let r: Vec<bool> = (0..1536).map(|v| v as u64 % 2 == 0).collect();
    let l: Vec<bool> = (0..1536).map(|v| v as u64 % 3 == 0).collect();
    let r_u8 = pack_bools_to_u8(r.clone());
    let l_u8 = pack_bools_to_u8(l.clone());
    let r_u64 = pack_bools_to_u64(r.clone());
    let l_u64 = pack_bools_to_u64(l.clone());
    let r_u128 = pack_bools_to_u128(r.clone());
    let l_u128 = pack_bools_to_u128(l.clone());

    let mut group = c.benchmark_group("Distance xor");
    group.bench_function("xor unoptimized u8", |b| {
        b.iter(|| xor_unoptimized_u8(black_box(&r_u8), black_box(&l_u8)))
    });
    group.bench_function("xor unoptimized u64", |b| {
        b.iter(|| xor_unoptimized_u64(black_box(&r_u64), black_box(&l_u64)))
    });
    group.bench_function("xor unoptimized u128", |b| {
        b.iter(|| xor_unoptimized_u128(black_box(&r_u128), black_box(&l_u128)))
    });

    assert!(r_u8.len() == 192);
    group.bench_function("xor unoptimized u8 fixed size", |b| {
        b.iter(|| xor_unoptimized_u8_fixed_size(black_box(&r_u8), black_box(&l_u8)))
    });
    assert!(r_u64.len() == 24);
    group.bench_function("xor unoptimized u64 fixed size", |b| {
        b.iter(|| xor_unoptimized_u64_fixed_size(black_box(&r_u64), black_box(&l_u64)))
    });
    group.bench_function("xor unoptimized u64 fixed size_map", |b| {
        b.iter(|| xor_unoptimized_u64_fixed_size_map(black_box(&r_u64), black_box(&l_u64)))
    });
    group.bench_function("xor optimized version we use in code", |b| {
        b.iter(|| distance_xor_optimized(black_box(&r_u64), black_box(&l_u64)))
    });
    assert!(r_u128.len() == 12);
    group.bench_function("xor unoptimized u128 fixed size", |b| {
        b.iter(|| xor_unoptimized_u128_fixed_size(black_box(&r_u128), black_box(&l_u128)))
    });
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
criterion_group!(
    benches,
    benchmark_distance,
    benchmark_distance_few_dimensions,
    benchmark_distance_x86_unaligned_vectors,
    benchmark_distance_x86_aligned_vectors,
    benchmark_distance_xor,
);
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
criterion_group!(
    benches,
    benchmark_distance,
    benchmark_distance_few_dimensions,
    benchmark_distance_xor,
);

criterion_main!(benches);
