import numpy as np

def generate():
    randi_arr = np.random.randint(0, 256, size=(32, 32))
    return randi_arr



for i in range(10000):
    noise = generate()
    path = './noise/' + str(i) + '.npy'
    # print(path)
    np.save(path , noise)
    # print(np.load(path))