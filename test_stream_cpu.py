import threading
import mlx.core as mx

def run():
    try:
        mx.set_default_device(mx.cpu)
        a = mx.array([1, 2, 3], mx.cpu)
        print(a)
    except Exception as e:
        print(e)
        
threading.Thread(target=run).start()
