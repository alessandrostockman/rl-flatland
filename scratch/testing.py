import pickle

with open("./test-envs2/Test_1/Level_0.pkl", 'rb') as handle:
    file_r = pickle.load(handle)
    print(file_r)