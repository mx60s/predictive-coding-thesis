import numpy as np

def remove_consecutive_repeats(np_file_path):
    arrays = np.load(np_file_path, allow_pickle=True)
    unique_arrays = []
    print(len(arrays))
    for i in range(1, len(arrays) - 1):
        if not np.array_equal(arrays[i], arrays[i + 1]):
            unique_arrays.append(arrays[i])

    if len(arrays) > 0:
        unique_arrays.append(arrays[-1])

    print(len(unique_arrays))
    return unique_arrays
    #np.save(np_file_path, unique_arrays)