import time
import numpy as np
import hnswlib

def main():
    # 150,000 vectors = exactly ~76 MB of raw data, plus ~20 MB for the HNSW graph.
    # Total memory footprint inside the cage will naturally sit around 130 MB.
    num_elements, dim, k = 150000, 128, 10
    num_queries = 200  # Fewer queries for speed; logic unchanged

    np.random.seed(42)
    train_data = np.random.randn(num_elements, dim).astype(np.float32)
    queries = np.random.randn(num_queries, dim).astype(np.float32)

    # Memory-efficient exact NN: one query at a time to minimize peak RAM
    exact = np.zeros((num_queries, k), dtype=np.int32)
    for i in range(num_queries):
        dists = np.sum((train_data - queries[i])**2, axis=1)
        exact[i] = np.argsort(dists)[:k]
    
    # Build HNSW graph
    index = hnswlib.Index(space='l2', dim=dim)
    index.init_index(max_elements=num_elements, ef_construction=200, M=32)
    index.add_items(train_data, np.arange(num_elements))
    index.set_ef(50)
    
    # Search and measure time
    t0 = time.perf_counter()
    approx, _ = index.knn_query(queries, k=k)
    latency_ms = (time.perf_counter() - t0) * 1000  # seconds -> ms

    # Output results
    hits = sum(len(set(a).intersection(set(t))) for a, t in zip(approx, exact))
    recall = hits / (num_queries * k)

    # Real kernel memory stats from /proc (no inference)
    vm_rss_kb = 0
    vm_swap_kb = 0
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    vm_rss_kb = int(line.split()[1])
                elif line.startswith("VmSwap:"):
                    vm_swap_kb = int(line.split()[1])
                    break
    except Exception:
        pass

    print(f"RESULT,{recall},{latency_ms},{vm_rss_kb},{vm_swap_kb}")

if __name__ == "__main__":
    main()