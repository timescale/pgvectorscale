#[cfg(test)]
#[pgrx::pg_schema]
pub mod tests {
    use pgrx::*;
    use serial_test::serial;
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

    fn test_upgrade_base(
        version: &str,
        pgrx_version: &str,
        subdirname: &str,
        extname: &str,
        amname: &str,
    ) {
        if cfg!(feature = "pg17") && version != "0.4.0" {
            // PG17 was not supported before 0.4.0
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

        let pgrx_str = format!("={pgrx_version}");
        let pgrx_dir = format!("pgrx-{pgrx_version}");

        let res = std::process::Command::new("cargo")
            .current_dir(temp_path.join(subdirname))
            .args([
                "install",
                "cargo-pgrx",
                "--version",
                pgrx_str.as_str(),
                "--force",
                "--root",
                temp_path.join(pgrx_dir.as_str()).to_str().unwrap(),
                "cargo-pgrx",
            ])
            .stdout(Stdio::inherit())
            .stderr(Stdio::piped())
            .output()
            .unwrap();

        assert!(res.status.success(), "failed: {:?}", res);

        let res = std::process::Command::new(
            temp_path
                .join(pgrx_dir.as_str())
                .join("bin/cargo-pgrx")
                .to_str()
                .unwrap(),
        )
        .current_dir(temp_path.join(subdirname))
        .env(
            "CARGO_TARGET_DIR",
            temp_path.join(subdirname).join("target"),
        )
        .env("CARGO_PKG_VERSION", version)
        .arg("pgrx")
        .arg("install")
        .arg("--test")
        .arg("--pg-config")
        .arg(pg_config.path().unwrap())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .output()
        .unwrap();
        assert!(res.status.success(), "failed: {:?}", res);

        client
            .execute(
                &format!("CREATE EXTENSION {extname} VERSION '{}' CASCADE;", version),
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
           USING {amname}(embedding);
            "
            ))
            .unwrap();

        client.execute("set enable_seqscan = 0;", &[]).unwrap();
        let cnt: i64 = client.query_one(&format!("WITH cte as (select * from test order by embedding <=> '[1,1,1,{suffix}]') SELECT count(*) from cte;"), &[]).unwrap().get(0);
        assert_eq!(cnt, 303, "count before upgrade");

        if extname == "timescale_vector" {
            client
                .execute(
                    &"UPDATE pg_extension SET extname='vectorscale' WHERE extname = 'timescale_vector';".to_string(),
                    &[],
                )
                .unwrap();
        }

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

        // Recreate client to pick up system catalog changes
        let (mut client, _) = pgrx_tests::client().unwrap();

        client.execute("set enable_seqscan = 0;", &[]).unwrap();
        let cnt: i64 = client.query_one(&format!("WITH cte as (select * from test order by embedding <=> '[1,1,1,{suffix}]') SELECT count(*) from cte;"), &[]).unwrap().get(0);
        assert_eq!(cnt, 303, "count after upgrade");

        client.execute("DROP INDEX idxtest;", &[]).unwrap();
        client
            .execute(
                "CREATE INDEX idxtest_cosine ON test USING diskann(embedding vector_cosine_ops);",
                &[],
            )
            .unwrap();
        client
            .execute(
                "CREATE INDEX idxtest_l2 ON test USING diskann(embedding vector_l2_ops);",
                &[],
            )
            .unwrap();
    }

    #[ignore]
    #[serial]
    #[test]
    fn test_upgrade_from_0_0_2() {
        test_upgrade_base(
            "0.0.2",
            "0.11.1",
            "timescale_vector",
            "timescale_vector",
            "tsv",
        );
    }

    #[ignore]
    #[serial]
    #[test]
    fn test_upgrade_from_0_2_0() {
        test_upgrade_base("0.2.0", "0.11.4", "pgvectorscale", "vectorscale", "diskann");
    }

    #[ignore]
    #[serial]
    #[test]
    fn test_upgrade_from_0_3_0() {
        test_upgrade_base("0.3.0", "0.11.4", "pgvectorscale", "vectorscale", "diskann");
    }

    #[ignore]
    #[serial]
    #[test]
    fn test_upgrade_from_0_4_0() {
        test_upgrade_base("0.4.0", "0.12.5", "pgvectorscale", "vectorscale", "diskann");
    }
}
