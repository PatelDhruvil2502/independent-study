import time
import numpy as np
import hnswlib

def main():
    print("Loading real_world_dataset.npy...")
    train_data = np.load("/app/real_world_dataset.npy")
    if train_data.ndim == 1:
        train_data = train_data.reshape(1, -1)
    num_elements, dim = train_data.shape[0], train_data.shape[1]
    k = 20
    num_queries_total = 1000000

    rng = np.random.default_rng(42)
    print("Generating queries...")
    # Sample real points so queries follow the dataset distribution
    indices = rng.choice(num_elements, size=num_queries_total, replace=True)
    queries = np.empty((num_queries_total, dim), dtype=np.float32)
    chunk_size = 50000
    for i in range(0, num_queries_total, chunk_size):
        end = min(i + chunk_size, num_queries_total)
        idx = indices[i:end]
        queries[i:end] = train_data[idx] + rng.normal(scale=0.01, size=(len(idx), dim)).astype(np.float32)

    num_passes = 1
    t0 = time.perf_counter()
    print("Building HNSW graph...")
    index = hnswlib.Index(space='l2', dim=dim)
    index.init_index(max_elements=num_elements, ef_construction=64, M=16)
    index.add_items(train_data, np.arange(num_elements))
    index.set_ef(100) # Deeper search = more nodes touched = more page faults
    print("Starting search...")
    
    # Process queries in randomly ordered batches to intentionally scatter memory access
    # This ensures that when memory is swapped out, each batch hits a different page,
    # defeating sequential OS prefetching and inducing severe swap thrashing latency.
    eval_batch_size = 5000 
    random_order = rng.permutation(num_queries_total)
    
    for _ in range(num_passes):
        for i in range(0, num_queries_total, eval_batch_size):
            batch_idx = random_order[i : i + eval_batch_size]
            batch_queries = queries[batch_idx]
            index.knn_query(batch_queries, k=k, num_threads=1)
            
    elapsed_s = time.perf_counter() - t0
    total_queries_done = num_queries_total * num_passes
    latency_ms = (elapsed_s / total_queries_done) * 1000

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
    print(f"RESULT,{latency_ms},{vm_rss_kb},{vm_swap_kb}")

if __name__ == "__main__":
    main()