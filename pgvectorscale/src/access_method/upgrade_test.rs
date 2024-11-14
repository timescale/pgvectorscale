#[cfg(test)]
#[pgrx::pg_schema]
pub mod tests {
    use pgrx::*;
    use std::{fs, path::Path, process::Stdio};

    fn copy_dir_all(src: impl AsRef<Path>, dst: impl AsRef<Path>) -> std::io::Result<()> {
        fs::create_dir_all(&dst)?;
        for entry in fs::read_dir(src)? {
            let entry = entry?;
            let ty = entry.file_type()?;
            if ty.is_dir() {
                if entry.file_name() == "target" {
                    continue;
                }
                copy_dir_all(entry.path(), dst.as_ref().join(entry.file_name()))?;
            } else {
                fs::copy(entry.path(), dst.as_ref().join(entry.file_name()))?;
            }
        }
        Ok(())
    }

    #[test]
    ///This function is only a mock to bring up the test framewokr in test_delete_vacuum
    fn test_upgrade_from_0_0_2() {
        if cfg!(feature = "pg17") {
            // PG17 is only supported for one version
            return;
        }
        pgrx_tests::run_test(
            "test_delete_mock_fn",
            None,
            crate::pg_test::postgresql_conf_options(),
        )
        .unwrap();

        let (mut client, _) = pgrx_tests::client().unwrap();

        client
            .execute(
                &"DROP EXTENSION IF EXISTS vectorscale CASCADE;".to_string(),
                &[],
            )
            .unwrap();

        let current_file = file!();

        // Convert the file path to an absolute path
        let current_dir = std::env::current_dir().unwrap();
        let mut absolute_path = std::path::Path::new(&current_dir).join(current_file);
        absolute_path = absolute_path.ancestors().nth(4).unwrap().to_path_buf();

        let temp_dir = tempfile::tempdir().unwrap();
        let temp_path = temp_dir.path();

        copy_dir_all(absolute_path.clone(), temp_dir.path()).unwrap();

        let pgrx = pgrx_pg_config::Pgrx::from_config().unwrap();
        let pg_version = pg_sys::get_pg_major_version_num();
        let pg_config = pgrx.get(&format!("pg{}", pg_version)).unwrap();

        let version = "0.0.2";
        let res = std::process::Command::new("git")
            .current_dir(temp_path)
            .arg("checkout")
            .arg("-f")
            .arg(version)
            .output()
            .unwrap();
        assert!(
            res.status.success(),
            "failed: {:?} {:?} {:?}",
            res,
            absolute_path,
            temp_dir.path()
        );

        let res = std::process::Command::new("cargo")
            .current_dir(temp_path.join("timescale_vector"))
            .args([
                "install",
                "cargo-pgrx",
                "--version",
                "=0.11.1",
                "--force",
                "--root",
                temp_path.join("pgrx-0.11.1").to_str().unwrap(),
                "cargo-pgrx",
            ])
            .stdout(Stdio::inherit())
            .stderr(Stdio::piped())
            .output()
            .unwrap();
        assert!(res.status.success(), "failed: {:?}", res);

        let res = std::process::Command::new(
            temp_path
                .join("pgrx-0.11.1/bin/cargo-pgrx")
                .to_str()
                .unwrap(),
        )
        .current_dir(temp_path.join("timescale_vector"))
        .arg("pgrx")
        .arg("install")
        .arg("--test")
        .arg("--pg-config")
        .arg(pg_config.path().unwrap())
        .stdout(Stdio::inherit())
        .stderr(Stdio::piped())
        .output()
        .unwrap();
        assert!(res.status.success(), "failed: {:?}", res);

        client
            .execute(
                &format!(
                    "CREATE EXTENSION timescale_vector VERSION '{}' CASCADE;",
                    version
                ),
                &[],
            )
            .unwrap();

        let suffix = (1..=253)
            .map(|i| format!("{}", i))
            .collect::<Vec<String>>()
            .join(", ");

        client
            .batch_execute(&format!(
                "CREATE TABLE test(embedding vector(256));

        select setseed(0.5);
        -- generate 300 vectors
        INSERT INTO test(embedding)
        SELECT
         *
        FROM (
            SELECT
        ('[ 0 , ' || array_to_string(array_agg(random()), ',', '0') || ']')::vector AS embedding
        FROM
         generate_series(1, 255 * 300) i
        GROUP BY
        i % 300) g;

        INSERT INTO test(embedding) VALUES ('[1,2,3,{suffix}]'), ('[4,5,6,{suffix}]'), ('[7,8,10,{suffix}]');

        CREATE INDEX idxtest
              ON test
           USING tsv(embedding);
            "
            ))
            .unwrap();

        client.execute("set enable_seqscan = 0;", &[]).unwrap();
        let cnt: i64 = client.query_one(&format!("WITH cte as (select * from test order by embedding <=> '[1,1,1,{suffix}]') SELECT count(*) from cte;"), &[]).unwrap().get(0);
        assert_eq!(cnt, 303, "count before upgrade");

        //reinstall myself
        let res = std::process::Command::new("cargo")
            .arg("pgrx")
            .arg("install")
            .arg("--test")
            .arg("--pg-config")
            .arg(pg_config.path().unwrap())
            .stdout(Stdio::inherit())
            .stderr(Stdio::piped())
            .output()
            .unwrap();
        assert!(res.status.success(), "failed: {:?}", res);

        client
            .execute(
                &"UPDATE pg_extension SET extname='vectorscale' WHERE extname = 'timescale_vector';".to_string(),
                &[],
            )
            .unwrap();

        //need to recreate the client to avoid double load of GUC. Look into this later.
        let (mut client, _) = pgrx_tests::client().unwrap();
        client
            .execute(
                &format!(
                    "ALTER EXTENSION vectorscale UPDATE TO '{}'",
                    env!("CARGO_PKG_VERSION")
                ),
                &[],
            )
            .unwrap();

        client.execute("set enable_seqscan = 0;", &[]).unwrap();
        let cnt: i64 = client.query_one(&format!("WITH cte as (select * from test order by embedding <=> '[1,1,1,{suffix}]') SELECT count(*) from cte;"), &[]).unwrap().get(0);
        assert_eq!(cnt, 303, "count after upgrade");
    }

    #[test]
    fn test_upgrade_from_0_2_0() {
        test_upgrade_base("0.2.0", "0.11.4");
    }

    #[test]
    fn test_upgrade_from_0_3_0() {
        test_upgrade_base("0.3.0", "0.11.4");
    }

    #[test]
    fn test_upgrade_from_0_4_0() {
        test_upgrade_base("0.4.0", "0.12.5");
    }

    /// Common upgrade test logic for versions 0.2.0, 0.3.0, and 0.4.0
    fn test_upgrade_base(version: &str, pgrx_version: &str) {
        if cfg!(feature = "pg17") {
            // PG17 is only supported for one version
            return;
        }
        pgrx_tests::run_test(
            "test_delete_mock_fn",
            None,
            crate::pg_test::postgresql_conf_options(),
        )
        .unwrap();

        let (mut client, _) = pgrx_tests::client().unwrap();

        client
            .execute(
                &"DROP EXTENSION IF EXISTS vectorscale CASCADE;".to_string(),
                &[],
            )
            .unwrap();

        let current_file = file!();

        // Convert the file path to an absolute path
        let current_dir = std::env::current_dir().unwrap();
        let mut absolute_path = std::path::Path::new(&current_dir).join(current_file);
        absolute_path = absolute_path.ancestors().nth(4).unwrap().to_path_buf();

        let temp_dir = tempfile::tempdir().unwrap();
        let temp_path = temp_dir.path();

        // TODO: just checkout to the target directory, skip this massive copy
        copy_dir_all(absolute_path.clone(), temp_dir.path()).unwrap();

        let pgrx = pgrx_pg_config::Pgrx::from_config().unwrap();
        let pg_version = pg_sys::get_pg_major_version_num();
        let pg_config = pgrx.get(&format!("pg{}", pg_version)).unwrap();

        let res = std::process::Command::new("git")
            .current_dir(temp_path)
            .arg("checkout")
            .arg("-f")
            .arg(version)
            .output()
            .unwrap();
        assert!(
            res.status.success(),
            "failed: {:?} {:?} {:?}",
            res,
            absolute_path,
            temp_dir.path()
        );

        let pgrx_path = format!("pgrx-{pgrx_version}");
        let res = std::process::Command::new("cargo")
            .current_dir(temp_path.join("pgvectorscale"))
            .args([
                "install",
                "cargo-pgrx",
                "--version",
                format!("={pgrx_version}").as_str(),
                "--force",
                "--root",
                temp_path.join(pgrx_path.clone()).to_str().unwrap(),
                "cargo-pgrx",
            ])
            .stdout(Stdio::inherit())
            .stderr(Stdio::piped())
            .output()
            .unwrap();
        assert!(res.status.success(), "failed: {:?}", res);

        let res = std::process::Command::new(
            temp_path
                .join(pgrx_path)
                .join("bin/cargo-pgrx")
                .to_str()
                .unwrap(),
        )
        .current_dir(temp_path.join("pgvectorscale"))
        .arg("pgrx")
        .arg("install")
        .arg("--test")
        .arg("--pg-config")
        .arg(pg_config.path().unwrap())
        .stdout(Stdio::inherit())
        .stderr(Stdio::piped())
        .output()
        .unwrap();
        assert!(res.status.success(), "failed: {:?}", res);

        client
            .execute(
                &format!(
                    "CREATE EXTENSION vectorscale VERSION '{}' CASCADE;",
                    version
                ),
                &[],
            )
            .unwrap();

        let suffix = (1..=253)
            .map(|i| format!("{}", i))
            .collect::<Vec<String>>()
            .join(", ");

        client
            .batch_execute(&format!(
                "CREATE TABLE test(embedding vector(256));

        select setseed(0.5);
        -- generate 300 vectors
        INSERT INTO test(embedding)
        SELECT
         *
        FROM (
            SELECT
        ('[ 0 , ' || array_to_string(array_agg(random()), ',', '0') || ']')::vector AS embedding
        FROM
         generate_series(1, 255 * 300) i
        GROUP BY
        i % 300) g;

        INSERT INTO test(embedding) VALUES ('[1,2,3,{suffix}]'), ('[4,5,6,{suffix}]'), ('[7,8,10,{suffix}]');

        CREATE INDEX idxtest
              ON test
           USING diskann(embedding);
            "
            ))
            .unwrap();

        client.execute("set enable_seqscan = 0;", &[]).unwrap();
        let cnt: i64 = client.query_one(&format!("WITH cte as (select * from test order by embedding <=> '[1,1,1,{suffix}]') SELECT count(*) from cte;"), &[]).unwrap().get(0);
        assert_eq!(cnt, 303, "count before upgrade");

        //reinstall myself
        let res = std::process::Command::new("cargo")
            .arg("pgrx")
            .arg("install")
            .arg("--test")
            .arg("--pg-config")
            .arg(pg_config.path().unwrap())
            .stdout(Stdio::inherit())
            .stderr(Stdio::piped())
            .output()
            .unwrap();
        assert!(res.status.success(), "failed: {:?}", res);

        client
            .execute(
                &"UPDATE pg_extension SET extname='vectorscale' WHERE extname = 'timescale_vector';".to_string(),
                &[],
            )
            .unwrap();

        //need to recreate the client to avoid double load of GUC. Look into this later.
        let (mut client, _) = pgrx_tests::client().unwrap();
        client
            .execute(
                &format!(
                    "ALTER EXTENSION vectorscale UPDATE TO '{}'",
                    env!("CARGO_PKG_VERSION")
                ),
                &[],
            )
            .unwrap();

        client.execute("set enable_seqscan = 0;", &[]).unwrap();
        let cnt: i64 = client.query_one(&format!("WITH cte as (select * from test order by embedding <=> '[1,1,1,{suffix}]') SELECT count(*) from cte;"), &[]).unwrap().get(0);
        assert_eq!(cnt, 303, "count after upgrade");
    }
}
