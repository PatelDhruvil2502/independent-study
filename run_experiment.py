import subprocess
import matplotlib.pyplot as plt

def main():
    print("Step 1: Rebuilding Docker image...")
    subprocess.run(["docker", "build", "-t", "hnsw-stress-test", "."], check=True)
    
    configs = [
        (250, 250),
        (180, 300),
        (120, 300),
        (80,  300),
        (80,  80),
    ]
    
    plot_limits, plot_latencies, plot_recalls = [], [], []
    crash_point = None

    print("\n--- Starting True Memory Stress Test ---")
    print(f"{'RAM Limit':<10} | {'Total Limit':<12} | {'Memory':<18} | {'Latency (ms)'}")
    print("-" * 60)

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
                # Real OOM: kernel killed the process for exceeding memory
                print(f"{ram} MB     | {total_mem} MB      | Crashed (OOM)       | -")
                plot_limits.append(ram)
                plot_latencies.append(0)
                plot_recalls.append(0)
                crash_point = ram
                break

            output = result.stdout.strip().splitlines()
            res_line = [line for line in output if line.startswith("RESULT")]

            if res_line:
                parts = res_line[0].split(",")
                recall = float(parts[1])
                latency = float(parts[2])
                vm_swap_kb = int(parts[4]) if len(parts) >= 5 else 0
                # Memory status from kernel VmSwap (real data)
                memory_status = "RAM only" if vm_swap_kb == 0 else "Swapping"
                print(f"{ram} MB     | {total_mem} MB      | {memory_status:<18} | {latency:.2f} ms")
                
                plot_limits.append(ram)
                plot_recalls.append(float(recall))
                plot_latencies.append(latency)
            else:
                print(f"{ram} MB     | {total_mem} MB      | ❌ FAILED UNEXPECTEDLY")
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
    ax2.set_ylabel("Query Latency (ms)", color="tab:orange", fontweight='bold')
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

    fig.suptitle("HNSW Performance vs Memory Limit (Real Docker Constraints)", fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig("true_memory_crash_plot.png", dpi=300)
    print(f"\n✅ Graph successfully saved as: true_memory_crash_plot.png")
    plt.close(fig)

if __name__ == "__main__":
    main()