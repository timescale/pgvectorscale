#[cfg(test)]
mod tests {
    use rkyv::{Archive, Deserialize, Serialize};

    #[derive(Archive, Serialize, Deserialize, Clone, Debug)]
    #[archive_attr(derive(Debug))]
    #[archive(check_bytes)]
    struct TestValue {
        data: Vec<u8>,
    }

    #[test]
    fn test_direct_serialization() {
        let value = TestValue {
            data: vec![1, 2, 3],
        };

        // Serialize
        let bytes = rkyv::to_bytes::<_, 256>(&value).unwrap();
        println!("Serialized {} bytes", bytes.len());
        println!("First 32 bytes: {:?}", &bytes[..32.min(bytes.len())]);

        // Check the archived value directly
        let archived = unsafe { rkyv::archived_root::<TestValue>(&bytes) };
        println!("Archived data: {:?}", archived.data.as_slice());

        // Deserialize
        let deserialized: TestValue = rkyv::from_bytes(&bytes).unwrap();
        println!("Deserialized data: {:?}", deserialized.data);

        assert_eq!(deserialized.data, vec![1, 2, 3]);
    }

    #[test]
    fn test_manual_memory_copy() {
        let value = TestValue {
            data: vec![1, 2, 3],
        };

        // Serialize
        let bytes = rkyv::to_bytes::<_, 256>(&value).unwrap();

        // Allocate memory and copy
        let mut buffer = vec![0u8; bytes.len()];
        buffer.copy_from_slice(&bytes);

        // Read back
        let archived = unsafe { rkyv::archived_root::<TestValue>(&buffer) };
        println!("After copy - archived data: {:?}", archived.data.as_slice());

        assert_eq!(archived.data.as_slice(), &[1, 2, 3]);
    }
}
