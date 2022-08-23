from multiprocessing import shared_memory

shm = shared_memory.SharedMemory(name='test5_probs', size=8000)
# shm = shared_memory.SharedMemory(create=True, name='test4_obs', size=4)
shm.unlink()
shm.close()
print(shm)
