# def generate_pairs(norm, MOS):
#     All = []
#     MeanPerPerson = []

#     for i in range(MOS.shape[0]):
#         for j in range(MOS.shape[1]):
#             All.append([norm[i], MOS[i][j]])

#     print(All)


# import numpy as np

# np.random.seed(42)

# MOS = np.random.randint(1, 5, (5, 5))
# norm = np.array([-1, -2, -3, -4, -5])

# print(MOS)
# generate_pairs(norm, MOS)


import numpy as np

numbers = np.random.choice(range(1, 16), size=15, replace=False)
print(numbers)


