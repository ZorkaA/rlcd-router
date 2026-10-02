import threading
import mlx.core as mx
import numpy as np

# Create in main thread
a = mx.array([1, 2, 3])

def run():
    print(np.array(a))
    
threading.Thread(target=run).start()
