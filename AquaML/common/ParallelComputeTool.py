
def ComputeDataBlock(total_length, worker_num, worker_id):
    one_worker_block_size = int(total_length / worker_num)

    start_point = one_worker_block_size * (worker_id - 1)

    end_point = one_worker_block_size * worker_id - 1

    return start_point, end_point
