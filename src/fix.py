from utils.data_processing import remove_consecutive_repeats
import os
import numpy as np

data_dir = 'data/test'
prefix = 'coords' 
for filename in os.listdir(data_dir):
    if filename.startswith(prefix):
        print(filename)
        file_suffix = filename[len(prefix):]
        coords_path = os.path.join(data_dir, filename)
        frames_path = os.path.join(data_dir, 'frames' + file_suffix)

        coords = np.load(coords_path)
        coords = np.delete(coords, 0)

        for i in range(len(coords) - 1):
            if not np.array_equal(coords[i], coords[i + 1]):
                unique_idx.append(i)

        if len(coords) > 0:
            unique_idx.append(len(coords) - 1)

        frames = np.load(frames_path)
        frames = np.delete(frames, 0)
        if len(coords) > len(frames):
            

        np.save(coords_path, coords[unique_idx])
        np.save(frames_path, frames[unique_idx])
        print(len(coords))
        print(len(frames))
        del coords
        del frames
