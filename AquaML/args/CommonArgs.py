class DatesetArgs:
    def __init__(self, batch_size, test_size, tf_set=False):
        """

        :param batch_size:
        :param test_size: ration 0~1.
        :param tf_set: Using tf_tf_set flag.
        """
        self.batch_size = batch_size
        self.test_size = test_size
        self.tf_set = tf_set
