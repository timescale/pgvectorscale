use pgrx::prelude::*;

pgrx::pg_module_magic!();

pub mod access_method;
mod util;

#[allow(non_snake_case)]
#[pg_guard]
pub unsafe extern "C" fn _PG_init() {
    access_method::options::init();
    access_method::guc::init();

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let sz = pg_sys::NBuffers as libc::size_t * pg_sys::BLCKSZ as libc::size_t;
        let mres = libc::madvise(pg_sys::BufferBlocks, sz, libc::MADV_RANDOM);
        if mres != 0 {
            let err = std::io::Error::last_os_error();
            error!("Error in madvise: {}", err);
        }
    }
}

#[allow(non_snake_case)]
#[pg_guard]
pub extern "C" fn _PG_fini() {
    // noop
}

/// This module is required by `cargo pgrx test` invocations.
/// It must be visible at the root of your extension crate.
#[cfg(test)]
pub mod pg_test {
    pub fn setup(_options: Vec<&str>) {
        //let (mut client, _) = pgrx_tests::client().unwrap();

        // perform one-off initialization when the pg_test framework starts
    }

    pub fn postgresql_conf_options() -> Vec<&'static str> {
        // return any postgresql.conf settings that are required for your tests
        vec![]
    }
}
