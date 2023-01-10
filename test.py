from AquaML.data.DataUnit import DataUnit


b = DataUnit('test')

b.read_shared_memory((4,1))

print(b.buffer)

b.close()