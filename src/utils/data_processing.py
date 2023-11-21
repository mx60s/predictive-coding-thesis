import numpy as np
import os 

def remove_consecutive_repeats(np_file_path):
    """
    Returns an array of indices with the consecutive repeating frames removed. To be
    used on coordinate files and then applied to the frame files.
    """
    unique_idx = []
    data = np.load(np_file_path, allow_pickle=True)
    for i in range(len(data) - 1):
        if not np.array_equal(data[i], data[i + 1]):
            unique_idx.append(i)

    if len(data) > 0:
        unique_idx.append(len(data) - 1)
    del data
    return unique_idx
    
    
def map_files_to_chunks(source_directory, target_directory, file_start, seq_len):
    print('Indexing files to', target_directory)
    os.makedirs(target_directory)

    file_index = 0
    for filename in os.listdir(source_directory):
        if filename.startswith(file_start):
            filepath = os.path.join(source_directory, filename)
            data = np.load(filepath, mmap_mode='r')
            data = np.load(filepath, mmap_mode='r')
    
            for i in range(len(data) - (seq_len + 1)):
                chunk = data[i:i + seq_len + 1]
                chunk_fp = os.path.join(target_directory, f'{file_index}.npy')
                np.save(chunk_fp, chunk)
                file_index += 1
    
            del data
            os.remove(filepath)

    return file_index
