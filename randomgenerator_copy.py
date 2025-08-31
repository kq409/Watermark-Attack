import numpy as np

def generate():
    # randi_arr = np.random.randint(0, 256, size=(32, 32))
    randi_arr = np.random.random(size=(32, 32))*256
    return randi_arr



for i in range(10000):
    noise = generate()
    path = './noise/' + str(i) + '.npy'
    # print(path)
    np.save(path , noise)
    # print(np.load(path))
print(np.load('./noise/0.npy'))
print(np.load('./noise/0.npy').shape)
print(np.load('./noise/0.npy').min)
print(np.load('./noise/0.npy').max)