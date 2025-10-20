#[cfg(test)]
mod debug_tests {
    use rkyv::{Archive, Deserialize, Serialize};

    #[derive(Archive, Serialize, Deserialize, Hash, Eq, PartialEq, Clone, Debug)]
    #[archive(compare(PartialEq))]
    #[archive_attr(derive(Debug))]
    #[archive(check_bytes)]
    struct TestKey {
        id: u64,
    }

    #[test]
    fn test_serialization_basic() {
        let key = TestKey { id: 42 };

        // Try to serialize using to_bytes
        let bytes = rkyv::to_bytes::<_, 256>(&key).expect("Failed to serialize");
        println!("Serialized to {} bytes using to_bytes", bytes.len());

        // Now try to deserialize
        let deserialized: TestKey = rkyv::from_bytes(&bytes).expect("Failed to deserialize");
        println!("Deserialized: {:?}", deserialized);
        assert_eq!(key, deserialized);
    }
}
