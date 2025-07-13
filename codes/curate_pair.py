
import os

def get_matching_pairs(dir1, dir2,output_file, file_ext='.npy'):
    """
    Recursively walks through `dir1`, and for each file found, constructs
    the corresponding path in `dir2`. Returns a list of (file1, file2) pairs.

    Args:
        dir1 (str): Path to first directory.
        dir2 (str): Path to second directory.
        file_ext (str): File extension to match (default: '.npy').

    Returns:
        List of (file_path_in_dir1, file_path_in_dir2) pairs.
    """
    pairs = []

    for root, _, files in os.walk(dir1):
        for file in files:
            if file.lower().endswith(file_ext):  # Only process specific file types
                abs_path1 = os.path.join(root, file)  # Absolute path in dir1
                rel_path = os.path.relpath(abs_path1, dir1)  # Relative path
                abs_path2 = os.path.join(dir2, rel_path)  # Corresponding path in dir2

                #print(abs_path1,abs_path2)
                if os.path.isfile(abs_path2):  # Ensure corresponding file exists
                    pairs.append((abs_path1, abs_path2))

    # Save pairs to a text file
    with open(output_file, 'w') as f:
        for p1, p2 in pairs:
            f.write(f"{p1} {p2}\n")

    print(f"Saved {len(pairs)} pairs to {output_file}")


get_matching_pairs(dir1="/afs/crc.nd.edu/user/a/abhatta/face_matchers_result/cancellable_exps_v2/features/C_M/model2",
                   dir2="/afs/crc.nd.edu/user/a/abhatta/face_matchers_result/cancellable_exps_v2/features/C_M/model5",
                   output_file="/afs/crc.nd.edu/user/a/abhatta/Revocable_Journal_Extension/txt_pairs/model2_5.txt")