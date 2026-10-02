import threading
import mlx.core as mx
import numpy as np

def run():
    a = mx.array([1, 2, 3])
    print(np.array(a))
    
threading.Thread(target=run).start()
