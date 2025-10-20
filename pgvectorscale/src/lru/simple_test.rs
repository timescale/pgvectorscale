#[cfg(test)]
mod simple_tests {
    use crate::lru::allocator::MockAllocator;
    use crate::lru::cache::SharedMemoryLru;
    use rkyv::{Archive, Deserialize, Serialize};

    #[derive(Archive, Serialize, Deserialize, Hash, Eq, PartialEq, Clone, Debug)]
    #[archive(compare(PartialEq))]
    #[archive_attr(derive(Debug))]
    #[archive(check_bytes)]
    struct TestKey {
        id: u64,
    }

    #[derive(Archive, Serialize, Deserialize, Clone, Debug)]
    #[archive_attr(derive(Debug))]
    #[archive(check_bytes)]
    struct TestValue {
        data: Vec<u8>,
    }

    #[test]
    fn test_create_cache() {
        println!("Creating allocator...");
        let allocator = MockAllocator::new();
        println!("Creating cache...");
        let cache = SharedMemoryLru::<TestKey, TestValue, MockAllocator>::new(
            allocator,
            1024,
            "test".to_string(),
            None,
        );
        println!("Cache created successfully!");
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_insert_only() {
        println!("Creating cache...");
        let allocator = MockAllocator::new();
        let cache = SharedMemoryLru::<TestKey, TestValue, MockAllocator>::new(
            allocator,
            1024,
            "test".to_string(),
            None,
        );

        println!("Creating key and value...");
        let key = TestKey { id: 42 };
        let value = TestValue {
            data: vec![1, 2, 3],
        };

        println!("Inserting into cache...");
        let result = cache.insert(key.clone(), value.clone());
        println!("Insert result: {:?}", result);

        assert!(result.is_ok());
        println!("Checking cache length...");
        assert_eq!(cache.len(), 1);
        println!("Test completed!");
    }

    #[test]
    fn test_get_after_insert() {
        println!("Creating cache...");
        let allocator = MockAllocator::new();
        let cache = SharedMemoryLru::<TestKey, TestValue, MockAllocator>::new(
            allocator,
            1024,
            "test".to_string(),
            None,
        );

        println!("Inserting...");
        let key = TestKey { id: 42 };
        let value = TestValue {
            data: vec![1, 2, 3],
        };
        cache.insert(key.clone(), value).unwrap();

        println!("Getting from cache...");
        let pinned = cache.get(&key);
        println!("Get result: {:?}", pinned.is_some());

        assert!(pinned.is_some());

        if let Some(p) = pinned {
            println!("Accessing value bytes...");
            let value_bytes = p.value_bytes();
            println!("Value bytes len: {}", value_bytes.len());
            println!(
                "Value bytes (first 32): {:?}",
                &value_bytes[..32.min(value_bytes.len())]
            );

            println!("Accessing archived value...");
            let archived = p.get();
            println!("Got archived value!");
            println!("Archived data: {:?}", archived.data.as_slice());
            println!("Archived data length: {}", archived.data.len());
            assert_eq!(archived.data.as_slice(), &[1, 2, 3]);
        }
    }
}
