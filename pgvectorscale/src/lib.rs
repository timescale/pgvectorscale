#![allow(unexpected_cfgs)]
use pgrx::prelude::*;

pgrx::pg_module_magic!();

pub mod access_method;
mod util;

#[allow(non_snake_case)]
#[pg_guard]
pub unsafe extern "C" fn _PG_init() {
    access_method::distance::init();
    access_method::options::init();
    access_method::guc::init();
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

    #[cfg(feature = "build_parallel")]
    pub fn postgresql_conf_options() -> Vec<&'static str> {
        vec!["maintenance_work_mem = '640MB'"]
    }
    #[cfg(not(feature = "build_parallel"))]
    pub fn postgresql_conf_options() -> Vec<&'static str> {
        vec![]
    }
}
