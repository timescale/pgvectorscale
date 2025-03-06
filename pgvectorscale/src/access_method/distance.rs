use pgrx::pg_extern;

pub type DistanceFn = fn(&[f32], &[f32]) -> f32;

#[derive(Debug, PartialEq)]
pub enum DistanceType {
    Cosine = 0,
    L2 = 1,
    InnerProduct = 2,
}

impl DistanceType {
    pub fn from_u16(value: u16) -> Self {
        match value {
            0 => DistanceType::Cosine,
            1 => DistanceType::L2,
            2 => DistanceType::InnerProduct,
            _ => panic!("Unknown DistanceType number {}", value),
        }
    }

    pub fn get_operator(&self) -> &str {
        match self {
            DistanceType::Cosine => "<=>",
            DistanceType::L2 => "<->",
            DistanceType::InnerProduct => "<#>",
        }
    }

    pub fn get_operator_class(&self) -> &str {
        match self {
            DistanceType::Cosine => "vector_cosine_ops",
            DistanceType::L2 => "vector_l2_ops",
            DistanceType::InnerProduct => "vector_ip_ops",
        }
    }

    pub fn get_distance_function(&self) -> DistanceFn {
        match self {
            DistanceType::Cosine => distance_cosine,
            DistanceType::L2 => distance_l2,
            DistanceType::InnerProduct => distance_inner_product,
        }
    }
}

#[pg_extern(immutable, parallel_safe, create_or_replace)]
pub fn distance_type_cosine() -> i16 {
    DistanceType::Cosine as i16
}

#[pg_extern(immutable, parallel_safe, create_or_replace)]
pub fn distance_type_l2() -> i16 {
    DistanceType::L2 as i16
}

#[pg_extern(immutable, parallel_safe, create_or_replace)]
pub fn distance_type_inner_product() -> i16 {
    DistanceType::InnerProduct as i16
}

/* we use the avx2 version of x86 functions. This verifies that's kosher */
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(not(target_feature = "avx2"))]
#[cfg(not(doc))]
compile_error!(
    "On x86, the AVX2 feature must be enabled. Set RUSTFLAGS=\"-C target-feature=+avx2,+fma\""
);

pub fn init() {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        panic!("On x86, pgvectorscale requires the CPU to support AVX2 and FMA. See https://github.com/timescale/pgvectorscale/issues/115");
    }

    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    if !std::arch::is_aarch64_feature_detected!("neon") {
        panic!("On aarch64, pgvectorscale requires the CPU to support Neon. See https://github.com/timescale/pgvectorscale/issues/115");
    }
}

#[inline]
pub fn distance_l2(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    //note safety is guraranteed by compile_error above
    unsafe {
        return super::distance_x86::distance_l2_x86_avx2(a, b);
    }

    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    unsafe {
        return super::distance_aarch64::distance_l2_aarch64_neon(a, b);
    }

    #[allow(unreachable_code)]
    {
        distance_l2_unoptimized(a, b)
    }
}

#[inline(always)]
pub fn distance_l2_unoptimized(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let norm: f32 = a
        .iter()
        .zip(b.iter())
        .map(|t| ({ *t.0 } - { *t.1 }) * ({ *t.0 } - { *t.1 }))
        .sum();
    assert!(norm >= 0.);
    //don't sqrt for performance. These are only used for ordering so sqrt not needed
    norm
}

/* PQ computes distances on subsegments that have few dimensions (e.g. 6). This function optimizes that.
* We optimize by telling the compiler exactly how long the slices are. This allows the compiler to figure
* out SIMD optimizations. Look at the benchmark results. */
#[inline]
pub fn distance_l2_optimized_for_few_dimensions(a: &[f32], b: &[f32]) -> f32 {
    let norm: f32 = match a.len() {
        0 => 0.,
        1 => a[..1]
            .iter()
            .zip(b[..1].iter())
            .map(|t| ({ *t.0 } - { *t.1 }) * ({ *t.0 } - { *t.1 }))
            .sum(),
        2 => a[..2]
            .iter()
            .zip(b[..2].iter())
            .map(|t| ({ *t.0 } - { *t.1 }) * ({ *t.0 } - { *t.1 }))
            .sum(),
        3 => a[..3]
            .iter()
            .zip(b[..3].iter())
            .map(|t| ({ *t.0 } - { *t.1 }) * ({ *t.0 } - { *t.1 }))
            .sum(),
        4 => a[..4]
            .iter()
            .zip(b[..4].iter())
            .map(|t| ({ *t.0 } - { *t.1 }) * ({ *t.0 } - { *t.1 }))
            .sum(),
        5 => a[..5]
            .iter()
            .zip(b[..5].iter())
            .map(|t| ({ *t.0 } - { *t.1 }) * ({ *t.0 } - { *t.1 }))
            .sum(),
        6 => a[..6]
            .iter()
            .zip(b[..6].iter())
            .map(|t| ({ *t.0 } - { *t.1 }) * ({ *t.0 } - { *t.1 }))
            .sum(),
        7 => a[..7]
            .iter()
            .zip(b[..7].iter())
            .map(|t| ({ *t.0 } - { *t.1 }) * ({ *t.0 } - { *t.1 }))
            .sum(),
        8 => a[..8]
            .iter()
            .zip(b[..8].iter())
            .map(|t| ({ *t.0 } - { *t.1 }) * ({ *t.0 } - { *t.1 }))
            .sum(),
        _ => distance_l2(a, b),
    };
    assert!(norm >= 0.);
    //don't sqrt for performance. These are only used for ordering so sqrt not needed
    norm
}

#[inline]
/// Negative inner product for use as distance function
pub fn distance_inner_product(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    unsafe {
        return -super::distance_x86::inner_product_x86_avx2(a, b);
    }

    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    unsafe {
        return -super::distance_aarch64::inner_product_aarch64_neon(a, b);
    }

    #[allow(unreachable_code)]
    {
        -inner_product_unoptimized(a, b)
    }
}

#[inline]
pub fn distance_cosine(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    //note safety is guraranteed by compile_error above
    unsafe {
        return super::distance_x86::distance_cosine_x86_avx2(a, b);
    }

    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    unsafe {
        return super::distance_aarch64::distance_cosine_aarch64_neon(a, b);
    }

    #[allow(unreachable_code)]
    {
        distance_cosine_unoptimized(a, b)
    }
}

#[inline(always)]
pub fn inner_product_unoptimized(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a, b)| *a * *b).sum()
}

#[inline(always)]
pub fn distance_cosine_unoptimized(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    debug_assert!(preprocess_cosine_get_norm(a).is_none());
    debug_assert!(preprocess_cosine_get_norm(b).is_none());
    let res: f32 = inner_product_unoptimized(a, b);
    (1.0 - res).max(0.0)
}

pub fn preprocess_cosine_get_norm(a: &[f32]) -> Option<f32> {
    let norm = a.iter().map(|v| v * v).sum::<f32>();
    //adjust the epsilon to the length of the vector
    let adj_epsilon = f32::EPSILON * a.len() as f32;

    /* this mainly handles the zero-vector case */
    if norm < f32::EPSILON {
        return None;
    }
    /* no need to renormalize if norm around 1.0 */
    if norm >= 1.0 - adj_epsilon && norm <= 1.0 + adj_epsilon {
        return None;
    }
    Some(norm.sqrt())
}

pub fn preprocess_cosine(a: &mut [f32]) {
    let norm = preprocess_cosine_get_norm(a);
    match norm {
        None => (),
        Some(norm) => {
            a.iter_mut().for_each(|v| *v /= norm);
            debug_assert!(
                preprocess_cosine_get_norm(a).is_none(),
                "preprocess_cosine isn't idempotent",
            );
        }
    }
}

macro_rules! xor_arm {
    ($a: expr, $b: expr, $sz: expr) => {
        $a[..$sz]
            .iter()
            .zip($b[..$sz].iter())
            .map(|(&l, &r)| (l ^ r).count_ones() as usize)
            .sum()
    };
}

#[inline(always)]
pub fn distance_xor_optimized(a: &[u64], b: &[u64]) -> usize {
    match a.len() {
        1 => xor_arm!(a, b, 1),
        2 => xor_arm!(a, b, 2),
        3 => xor_arm!(a, b, 3),
        4 => xor_arm!(a, b, 4),
        5 => xor_arm!(a, b, 5),
        6 => xor_arm!(a, b, 6),
        7 => xor_arm!(a, b, 7),
        8 => xor_arm!(a, b, 8),
        9 => xor_arm!(a, b, 9),
        10 => xor_arm!(a, b, 10),
        11 => xor_arm!(a, b, 11),
        12 => xor_arm!(a, b, 12),
        13 => xor_arm!(a, b, 13),
        14 => xor_arm!(a, b, 14),
        15 => xor_arm!(a, b, 15),
        16 => xor_arm!(a, b, 16),
        17 => xor_arm!(a, b, 17),
        18 => xor_arm!(a, b, 18),
        19 => xor_arm!(a, b, 19),
        20 => xor_arm!(a, b, 20),
        21 => xor_arm!(a, b, 21),
        22 => xor_arm!(a, b, 22),
        23 => xor_arm!(a, b, 23),
        24 => xor_arm!(a, b, 24),
        25 => xor_arm!(a, b, 25),
        26 => xor_arm!(a, b, 26),
        27 => xor_arm!(a, b, 27),
        28 => xor_arm!(a, b, 28),
        29 => xor_arm!(a, b, 29),
        30 => xor_arm!(a, b, 30),
        31 => xor_arm!(a, b, 31),
        32 => xor_arm!(a, b, 32),
        33 => xor_arm!(a, b, 33),
        34 => xor_arm!(a, b, 34),
        35 => xor_arm!(a, b, 35),
        36 => xor_arm!(a, b, 36),
        37 => xor_arm!(a, b, 37),
        38 => xor_arm!(a, b, 38),
        39 => xor_arm!(a, b, 39),
        40 => xor_arm!(a, b, 40),
        41 => xor_arm!(a, b, 41),
        42 => xor_arm!(a, b, 42),
        43 => xor_arm!(a, b, 43),
        44 => xor_arm!(a, b, 44),
        45 => xor_arm!(a, b, 45),
        46 => xor_arm!(a, b, 46),
        47 => xor_arm!(a, b, 47),
        48 => xor_arm!(a, b, 48),
        49 => xor_arm!(a, b, 49),
        _ => a
            .iter()
            .zip(b.iter())
            .map(|(&l, &r)| (l ^ r).count_ones() as usize)
            .sum(),
    }
}

macro_rules! distance_l2_simd_body {
    ($x:ident, $y:ident) => {{
        let mut accum0 = S::setzero_ps();
        let mut accum1 = S::setzero_ps();
        let mut accum2 = S::setzero_ps();
        let mut accum3 = S::setzero_ps();

        //assert!(x.len() == y.len());
        let mut x = &$x[..];
        let mut y = &$y[..];

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
    }};
}
pub(crate) use distance_l2_simd_body;

macro_rules! inner_product_simd_body {
    ($x:ident, $y:ident) => {{
        let mut accum0 = S::setzero_ps();
        let mut accum1 = S::setzero_ps();
        let mut accum2 = S::setzero_ps();
        let mut accum3 = S::setzero_ps();
        let mut x = &$x[..];
        let mut y = &$y[..];

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

        dist
    }};
}
pub(crate) use inner_product_simd_body;
