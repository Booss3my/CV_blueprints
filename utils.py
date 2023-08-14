import gc 
import torch
import os
import subprocess

def clear_memory_and_display_gpu_info():
    """Clears Python and CUDA memory caches and prints GPU information."""
    try:
        # Clearing Python's garbage collector
        gc.collect(generation=2)
        gc.collect()

        # Clearing CUDA cache
        torch.cuda.empty_cache()

        # Getting GPU info using nvidia-smi command
        gpu_info = subprocess.getoutput("nvidia-smi")
        
        # Printing GPU info and memory reserved by caching allocator
        print(gpu_info)
        print("Memory reserved by caching allocator:", torch.cuda.memory_reserved() / (1024 * 1024 * 1024), "GB")
    except Exception as e:
        print("An error occurred:", str(e))