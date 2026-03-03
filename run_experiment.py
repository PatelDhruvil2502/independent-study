import subprocess
import matplotlib.pyplot as plt

def main():
    print("Step 1: Rebuilding Docker image...")
    dockerfile_content = """FROM python:3.10-slim
RUN apt-get update && apt-get install -y build-essential python3-dev
RUN pip install hnswlib numpy
COPY worker.py /app/worker.py
WORKDIR /app
CMD ["python", "worker.py"]
"""
    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)

    subprocess.run(["docker", "build", "-t", "hnsw-stress-test", "."], check=True)
    
    # The payload is ~850 MB minimum. 
    # Giving 3000 MB Total guarantees it survives the middle phases so you see the latency spike.
    configs = [
        (1500, 3000), # Phase 1: 1.5 GB RAM -> Plenty of room (FAST)
        (950,  3000), # Phase 1: 950 MB RAM -> Snug, but fits entirely in RAM (FAST)
        (650,  3000), # Phase 2: 650 MB RAM -> 200 MB too small! OS must use SSD Swap (HIGH LATENCY)
        (400,  3000), # Phase 2: 400 MB RAM -> Extreme SSD thrashing (MASSIVE LATENCY)
        (400,  400),  # Phase 3: 400 MB Total -> Out of RAM & Out of Swap -> CRASH
    ]
    
    plot_limits, plot_latencies, plot_recalls = [], [], []
    crash_point = None

    print("\n--- Starting 1-MILLION QUERY Stress Test (Single-Threaded to Expose SSD) ---")
    print(f"{'RAM Limit':<10} | {'Total Limit':<12} | {'Memory Status':<18} | {'Avg Latency (ms)':<16} | {'Queries @ Addr'}")
    print("-" * 80)

    for ram, total_mem in configs:
        cmd = [
            "docker", "run", "--rm",
            f"--memory={ram}m",
            f"--memory-swap={total_mem}m",
            "hnsw-stress-test"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 137:
                print(f"{ram} MB     | {total_mem} MB      | Crashed (OOM)       | -")
                plot_limits.append(ram)
                plot_latencies.append(0)
                plot_recalls.append(0)
                crash_point = ram
                break

            output = result.stdout.strip().splitlines()
            mem_line = [line for line in output if line.startswith("MEMORY_LAYOUT")]
            res_line = [line for line in output if line.startswith("RESULT")]

            if res_line:
                parts = res_line[0].split(",")
                recall = float(parts[1])
                latency = float(parts[2])
                vm_swap_kb = int(parts[4]) if len(parts) >= 5 else 0
                
                q_start = mem_line[0].split(",")[1] if mem_line else "-"
                
                # Real data reported directly by the kernel
                memory_status = "RAM only" if vm_swap_kb == 0 else f"Swapping ({vm_swap_kb/1024:.1f} MB)"
                print(f"{ram} MB     | {total_mem} MB      | {memory_status:<18} | {latency:<16.4f} | {q_start}")
                
                plot_limits.append(ram)
                plot_recalls.append(float(recall))
                plot_latencies.append(latency)
            else:
                print(f"{ram} MB     | {total_mem} MB      | ❌ FAILED UNEXPECTEDLY")
                print("\n--- EXACT ERROR LOG ---")
                print(result.stderr if result.stderr else result.stdout)
                break

        except Exception as e:
            print(f"Error: {e}")
            break

    # --- Draw the Graph ---
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.invert_xaxis()
    ax1.set_xlabel("Physical RAM Allowed (MB) — Decreasing →", fontweight='bold')
    ax1.set_ylabel("Recall (Accuracy)", color="tab:blue", fontweight='bold')
    ax1.plot(plot_limits, plot_recalls, "o-", color="tab:blue", linewidth=2.5, label="Recall@10")
    ax1.set_ylim(-0.05, 1.05)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel("Avg Query Latency (ms)", color="tab:orange", fontweight='bold')
    ax2.plot(plot_limits, plot_latencies, "s-", color="tab:orange", linewidth=2.5, label="Latency (ms)")
    
    if crash_point is not None:
        ax1.axvline(x=crash_point, color="red", linestyle="--", linewidth=3)
        ax1.text(
            crash_point, 0.5,
            " OOM KILL",
            color="white", 
            bbox=dict(facecolor='red', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.5'),
            fontweight="bold", va="center"
        )

    fig.suptitle("1 Million Queries: True Hardware Swapping & Latency", fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig("true_memory_crash_plot.png", dpi=300)
    print(f"\n✅ Graph successfully saved as: true_memory_crash_plot.png")
    plt.close(fig)

if __name__ == "__main__":
    main()