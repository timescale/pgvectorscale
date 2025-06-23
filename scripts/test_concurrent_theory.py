#!/usr/bin/env python3
"""
Test script to validate the theory about concurrent page modifications causing corruption.

This script tests different scenarios:
1. Low parallelism with small batches (should work)
2. High parallelism with large batches (should reproduce issue)
3. Single inserts with high parallelism (test if issue is batch-specific)
"""

import asyncio
import sys
from issue_193_repro import _main, parser

async def test_scenario(description, **kwargs):
    """Test a specific scenario and report results"""
    print(f"\n=== Testing: {description} ===")
    print(f"Parameters: {kwargs}")
    
    # Create args object with the test parameters
    args = parser.parse_args([])
    for key, value in kwargs.items():
        setattr(args, key.replace('-', '_'), value)
    
    try:
        await _main(args)
        print("✅ SUCCESS: No errors encountered")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

async def main():
    """Run a series of tests to validate the concurrent access theory"""
    
    scenarios = [
        # Safe scenarios (should work)
        ("Single threaded, small batches", {
            "batch_size": 50, "batches": 10, "parallelism": 1
        }),
        ("Low parallelism, small batches", {
            "batch_size": 25, "batches": 8, "parallelism": 2
        }),
        
        # Risky scenarios (likely to fail)
        ("High parallelism, medium batches", {
            "batch_size": 50, "batches": 20, "parallelism": 4
        }),
        ("Very high parallelism, small batches", {
            "batch_size": 10, "batches": 50, "parallelism": 8
        }),
        ("Medium parallelism, large batches", {
            "batch_size": 100, "batches": 30, "parallelism": 3
        }),
        
        # Edge cases
        ("Many small individual inserts", {
            "batch_size": 1, "batches": 100, "parallelism": 6
        }),
    ]
    
    results = []
    for description, params in scenarios:
        success = await test_scenario(description, **params)
        results.append((description, success))
        
        # Short pause between tests to avoid overwhelming the database
        await asyncio.sleep(1)
    
    print("\n" + "="*60)
    print("SUMMARY OF RESULTS:")
    print("="*60)
    
    for description, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{status:4s} | {description}")
    
    failed_count = sum(1 for _, success in results if not success)
    total_count = len(results)
    
    print(f"\nTotal: {total_count} tests, {failed_count} failed, {total_count - failed_count} passed")
    
    if failed_count > 0:
        print(f"\n🔍 Analysis: {failed_count} tests failed, confirming concurrent access issues")
        print("The pattern suggests the issue occurs with parallelism > 1")
    else:
        print("\n✅ All tests passed - issue might be intermittent or environment-specific")

if __name__ == "__main__":
    asyncio.run(main())