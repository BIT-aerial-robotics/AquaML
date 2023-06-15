import tensorflow as tf

def initialize_model_weights(model, io_info,expand_dims_idx=None):
    """
    初始化网络权值，如果是循环神经网络，需要指定expand_dims_idx，以便于扩展维度。

    Args:
        model (tf.keras.Model): tensorflow keras model.
        expand_dims_idx (tuple, optional): 需要扩展维度的index. Defaults to None.
    """
    
    input_data_name = model.input_names
    
    # create tensor according to input data name
    input_data = []

    for name in input_data_name:
        try:
            shape, _ = io_info(name)
        except:
            shape = (1, 1)
        data = tf.zeros(shape=shape, dtype=tf.float32)
        input_data.append(data)
    if expand_dims_idx is not None:
        for idx in expand_dims_idx:
            input_data[idx] = tf.expand_dims(input_data[idx], axis=1)

        model(*input_data)