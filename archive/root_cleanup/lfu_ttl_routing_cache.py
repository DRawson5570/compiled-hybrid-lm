"""lfu_ttl_routing_cache.py

Implements Phase 12.2: Least Frequently Used (LFU) Routing Cache with Virtual TTL age.
Ensures we do not exceed the memory thresholds of the system under large sandbox runs.
"""
from __future__ import annotations

import time

class LFUTTLRoutingCache:
    def __init__(self, capacity: int = 4):
        self.capacity = capacity
        # maps key -> value
        self.cache: dict[str, any] = {}
        # maps key -> frequency
        self.freqs: dict[str, int] = {}
        # maps key -> virtual_timestamp
        self.timestamps: dict[str, int] = {}
        # virtual clock tick
        self.clock = 0

    def get(self, key: str) -> any:
        self.clock += 1
        if key in self.cache:
            self.freqs[key] += 1
            self.timestamps[key] = self.clock
            return self.cache[key]
        return None

    def set(self, key: str, value: any) -> None:
        self.clock += 1
        if self.capacity <= 0:
            return

        if key in self.cache:
            self.cache[key] = value
            self.freqs[key] += 1
            self.timestamps[key] = self.clock
            return

        # Eviction condition
        if len(self.cache) >= self.capacity:
            # Step A: Find min frequency
            min_freq = min(self.freqs.values())
            # Step B: Filter keys matching min frequency
            candidates = [k for k, f in self.freqs.items() if f == min_freq]
            
            # Step C: Break tie with virtual timestamp (evict candidate with oldest/smallest clock tick)
            evict_key = min(candidates, key=lambda k: self.timestamps[k])
            
            # Evict key
            del self.cache[evict_key]
            del self.freqs[evict_key]
            del self.timestamps[evict_key]
            print(f"  [Cache Eviction] Evicted cache key: '{evict_key}' (min_freq={min_freq})")

        # Insert new key
        self.cache[key] = value
        self.freqs[key] = 1
        self.timestamps[key] = self.clock

def run_cache_tests():
    print("=" * 80)
    print("         CMI MEMORY-CAPPED LFU-TTL ROUTING CACHE SUITE")
    print("=" * 80)
    
    # 1. Initialize cache with a tiny capacity of 3
    cache = LFUTTLRoutingCache(capacity=3)
    print("Initializing cache with capacity = 3...")
    
    # 2. Fill the cache
    print("\nAdding first 3 entries...")
    cache.set("math_expr_12", [0.0, 0.0, 0.0, 1.0])
    cache.set("translate_dog", [0.95, 0.05, 0.0, 0.0])
    cache.set("code_def_sum", [0.0, 0.0, 0.9, 0.1])
    
    # Check contents
    print("Current keys in cache:", list(cache.cache.keys()))
    
    # 3. Access some elements to increase frequency (making them popular)
    print("\nAccessing 'translate_dog' twice and 'math_expr_12' once...")
    cache.get("translate_dog")
    cache.get("translate_dog")
    cache.get("math_expr_12")
    
    # Print frequency stats
    for k in cache.cache:
        print(f"  Key: '{k}' | Freq: {cache.freqs[k]} | Clock: {cache.timestamps[k]}")
        
    # 4. Insert 4th element (Requires eviction of LFU or timestamp tie-breaker)
    # Frequency:
    #   code_def_sum: 1 (clock tick = 3)
    #   math_expr_12: 2
    #   translate_dog: 3
    # Expected eviction: 'code_def_sum' (freq = 1)
    print("\nAdding 'reasoning_template' (requires eviction)...")
    cache.set("reasoning_template", [0.0, 1.0, 0.0, 0.0])
    print("Current keys in cache after eviction:", list(cache.cache.keys()))
    
    # 5. Access reasoning_template to increment its freq to 2
    cache.get("reasoning_template")
    
    # Let's insert another one.
    # Current state:
    #   math_expr_12: freq = 2, clock = 6
    #   translate_dog: freq = 3, clock = 5
    #   reasoning_template: freq = 2, clock = 8
    # Freq min is 2: nominees are math_expr_12 and reasoning_template
    # Timestamp min is math_expr_12 (6 < 8)
    # Expected eviction: 'math_expr_12'
    print("\nAdding 'code_oop' (should evict 'math_expr_12' via timestamp tie-breaker)...")
    cache.set("code_oop", [0.0, 0.0, 1.0, 0.0])
    print("Current keys in cache after tie-breaker eviction:", list(cache.cache.keys()))
    
    assert "math_expr_12" not in cache.cache, "Eviction logic assertion failed!"
    print("\nStep 2 Completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    run_cache_tests()
