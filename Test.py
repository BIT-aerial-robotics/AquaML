# import tensorflow as tf


class Test:
    def __init__(self, tf_handle):
        self.tf = tf_handle

        a = self.tf.zeros(shape=(12, 1), dtype=self.tf.float32)

        print(a)


if __name__ == "__main__":
    import multiprocessing as mp

    def running(id):
        import os
        import tensorflow as tf
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        test = Test(tf)

        print(id)


    processes = [mp.Process(target=running, args=(1,)) for i in range(10)]

    for proces in processes:
        proces.start()

    proces.join()

