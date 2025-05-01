import multiprocessing as mp
import tensorflow as tf

@tf.function
def div(x):
    return x / x

with tf.device("/GPU:0"):    # rm this line can wrok
    div(tf.ones((1, 1)))      # rm this line can wrok

def run():
    with tf.device("/GPU:0"):
        print(div(tf.ones((1, 1)))) # No print as the program exited with -6.


process = mp.Process(target=run)
process.start()
process.join()
print(f"Sub-process exited with {process.exitcode}")
