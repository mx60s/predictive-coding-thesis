import numpy as np
import glob

def yaw_difference(yaw1, yaw2):
    return min(abs(yaw1 - yaw2), 360 - abs(yaw1 - yaw2))

def find_valid_sequences(coordinates):
    valid_sequences = []
    sequence_length = 7
    n = coordinates.shape[0]

    for i in range(n - sequence_length + 1):
        # Extract the current sequence of yaws
        yaws = coordinates[i:i+sequence_length, 2]
        
        # Check the yaw differences between consecutive coordinates
        valid_sequence = True
        for j in range(1, len(yaws)):
            if yaw_difference(yaws[j-1], yaws[j]) > 50:
                valid_sequence = False
                break

        if valid_sequence:
            valid_sequences.append(i)

    return valid_sequences

#coordinates = np.load('data/coords_continuous_detached_2024-03-22-12-45-54.npy')
#valid_indexes = find_valid_sequences(coordinates)
#print(f'can retrieve {len(valid_indexes)} valid sequences from coords')

def process_files(directory):
    """
    Processes all NumPy files in the given directory that start with "coords_continuous".
    """
    pattern = f"{directory}/frames_continuous*.npy"
    #total = 0
    for file_path in glob.glob(pattern):
        coordinates = np.load(file_path)
        print(file_path, coordinates.shape)
        #valid_indexes = len(find_valid_sequences(coordinates))
        #total += valid_indexes
        #print(f"Valid sequences in {file_path}: {valid_indexes}")
    #print(f"total: {total}")

# Example usage
# Replace 'your_directory_path_here' with the actual directory path.
process_files('data')
