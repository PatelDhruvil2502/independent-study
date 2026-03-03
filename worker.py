import time
import numpy as np
import hnswlib

def main():
    # Dataset: 300k vectors (~150 MB dataset + ~60 MB Graph)
    num_elements, dim, k = 300000, 128, 10
    
    # 1,000,000 queries. (This single array is 512 MB of physical RAM).
    num_queries_total = 1000000  
    
    # We only brute-force the exact answer key for the first 1,000 so the script doesn't take days
    num_queries_eval = 1000 

    rng = np.random.default_rng(42)
    print("Generating train_data...")
    train_data = rng.standard_normal((num_elements, dim), dtype=np.float32)
    print("Generating queries...")
    queries = rng.standard_normal((num_queries_total, dim), dtype=np.float32)

    # 1. Exact Answer Key (Only first 1,000)
    print("Computing exact answer key...")
    exact = np.zeros((num_queries_eval, k), dtype=np.int32)
    diffs = np.empty_like(train_data)
    for i in range(num_queries_eval):
        np.subtract(train_data, queries[i], out=diffs)
        np.square(diffs, out=diffs)
        dists = np.sum(diffs, axis=1)
        exact[i] = np.argsort(dists)[:k]
    
    del diffs
    import gc; gc.collect()
    
    # 2. Build HNSW graph
    print("Building HNSW graph...")
    index = hnswlib.Index(space='l2', dim=dim)
    index.init_index(max_elements=num_elements, ef_construction=200, M=32)
    index.add_items(train_data, np.arange(num_elements))
    index.set_ef(50)
    
    # 3. MASSIVE SEARCH: 1 Million queries!
    print("Starting massive search...")
    t0 = time.perf_counter()
    approx, distances = index.knn_query(queries, k=k, num_threads=1)
    
    latency_ms = ((time.perf_counter() - t0) / num_queries_total) * 1000

    # 4. Accuracy Grade (Only grade the first 1,000)
    print("Grading accuracy...")
    approx_eval = approx[:num_queries_eval]
    hits = sum(len(set(a).intersection(set(t))) for a, t in zip(approx_eval, exact))
    recall = hits / (num_queries_eval * k)

    # 5. Kernel RAM Stats (Real, undeniable OS hardware data)
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

    # Extract REAL hexadecimal memory addresses as they exist inside the Linux Docker container!
    q_start = hex(queries.__array_interface__['data'][0])
    q_size_mb = queries.nbytes / (1024 * 1024)
    t_start = hex(train_data.__array_interface__['data'][0])
    t_size_mb = train_data.nbytes / (1024 * 1024)
    
    print(f"MEMORY_LAYOUT,{q_start},{q_size_mb:.1f},{t_start},{t_size_mb:.1f}")
    print(f"RESULT,{recall},{latency_ms},{vm_rss_kb},{vm_swap_kb}")

if __name__ == "__main__":
    main()