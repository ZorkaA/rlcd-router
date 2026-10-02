import threading
import mlx.core as mx

def run():
    mx.eval(mx.array([1, 2, 3]))
    print('success')

threading.Thread(target=run).start()
