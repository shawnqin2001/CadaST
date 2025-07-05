import os
from anndata import read_h5ad
from cadaST import CadaST
import time
import scanpy as sc
import pandas as pd
import psutil
import threading
import signal

beta, alpha, theta, init_alpha = 10, 0.6, 0.2, 6
max_iter = 2
kneighbors = 24
n_jobs = 16
scale = False
dataPath = "/home/qinxianhan/project/spatial/dataset/MouseEmbryo"
n_top = 2000
memory_usage = {}


def get_memory_usage():
    """Get current memory usage of this process in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024 * 1024)  # Convert bytes to GB


def track_max_memory(max_memory, stop_event):
    """Track maximum memory usage in a separate thread"""
    while not stop_event.is_set():
        current_memory = get_memory_usage()
        max_memory[0] = max(max_memory[0], current_memory)
        time.sleep(0.1)  # Check every 100ms


def timeout_handler(signum, frame):
    """Handle timeout signal"""
    raise TimeoutError("Model processing exceeded the time limit")


def process(sample):
    datadir = f"{dataPath}/{sample}.h5ad"
    # path = f"../../output/embryo/{today}/{sample}/"
    # os.makedirs(path, exist_ok=True)

    # Setup memory tracking
    max_memory = [0]  # Using list to make it mutable in the thread
    stop_event = threading.Event()
    memory_thread = threading.Thread(target=track_max_memory, args=(max_memory, stop_event))
    memory_thread.daemon = True
    memory_thread.start()

    # Set timeout (3 minutes)
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(180)

    try:
        adata = read_h5ad(datadir)
        print(adata.shape)
        print("Preprocessing adata")
        if scale:
            sc.pp.scale(adata)

        model = CadaST(
            adata=adata,
            kneighbors=kneighbors,
            alpha=alpha,
            theta=theta,
            max_iter=max_iter,
            n_top=n_top,
            n_jobs=n_jobs,
        )

        print("Start running:")
        model.construct_graph()
        model.filter_genes(n_top=n_top)
        adata = model.fit()
        print(f"Finished processing {sample}")
        # adata.write(filename=f"{path}adata.h5ad")  # type: ignore
    except TimeoutError as e:
        print(f"Timeout for {sample}: {e}")
    except Exception as e:
        print(f"Error processing {sample}: {e}")
    finally:
        # Disable the alarm
        signal.alarm(0)

        # Stop memory tracking
        stop_event.set()
        memory_thread.join()

        # Record the maximum memory used
        memory_usage[sample] = max_memory[0]

        print(f"Maximum memory usage for {sample}: {max_memory[0]:.2f} GB")

        # Clean up memory
        if "adata" in locals():
            del adata


def main():
    samples = [
        "E9.5.h5ad",
        "E10.5.h5ad",
        "E11.5.h5ad",
        "E12.5.h5ad",
        "E13.5.h5ad",
        "E14.5.h5ad",
        "E15.5.h5ad",
        "E16.5.h5ad",
    ]
    samples = [os.path.splitext(sample)[0] for sample in samples]
    for sample in samples:
        print(f"Processing {sample}")
        process(sample)

    memory_df = pd.DataFrame(memory_usage.items(), columns=["Sample", "Memory_GB"])

    memory_df.to_csv("./memory.csv", index=False)


if __name__ == "__main__":
    main()
