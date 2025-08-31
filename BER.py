import numpy as np
import cv2
import os

# Error rate of symbols
def get_Pe(demodu: np.ndarray, ans: np.ndarray) -> np.float64:
    error_symbol_num = sum(demodu.reshape(-1) == ans.reshape(-1))
    return 1 - error_symbol_num / len(demodu.reshape(-1))

# Error rate of bits,and inputs should be np.array which is formed by 0~3
def get_BER(demodu: np.ndarray, ans: np.ndarray) -> np.float64:
    demod_list = [[0, 0], [0, 1], [1, 0], [1, 1]]
    demodu_bits = np.array([demod_list[idx] for idx in demodu.reshape(-1)]).reshape(-1)
    ans_bits = np.array([demod_list[idx] for idx in ans.reshape(-1)]).reshape(-1)
    return get_Pe(demodu_bits, ans_bits)


def qamdemod(array: np.ndarray, maplist=None, init_phase=0) -> np.ndarray:

    if maplist is None:
        maplist = [1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j]
    maplist = np.array(maplist) * np.exp(+2 * np.pi * 1j * init_phase)  # To correct the phase

    vfunc = np.vectorize(lambda x:
                         np.argmin(np.abs(x - maplist))  # x will be broadcast automatically
                         )

    demod = vfunc(array)
    demod = demod.astype(int)

    return demod

# list_BER = []
# for orig_name in os.listdir('/Users/huangyuanyuan/Desktop/extraction/original/'):
#     for re_name in os.listdir('/Users/huangyuanyuan/Desktop/extraction/127/'):
#         orig_domain = os.path.abspath('/Users/huangyuanyuan/Desktop/extraction/original/')
#         orig_full_path = os.path.join(orig_domain,orig_name)
#         orig_image = cv2.imread(orig_full_path)
#
#         re_domain = os.path.abspath('/Users/huangyuanyuan/Desktop/extraction/127/')
#         re_full_path = os.path.join(re_domain, re_name)
#         re_image = cv2.imread(re_full_path)
#
#         re_demodu = qamdemod(re_image)
#         orig_demodu = qamdemod(orig_image)
#         BER_num = get_BER(re_demodu,orig_demodu)
#         list_BER.append(BER_num)


# print("平均BER:", np.mean(list_BER))


