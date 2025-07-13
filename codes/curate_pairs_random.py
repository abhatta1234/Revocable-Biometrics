import os
import numpy as np
from itertools import combinations

def get_all_directory_pairs(base_dirs, output_dir, file_ext='.npy'):
    """
    Find matching files across all pairs of directories and save the entire output in one text file.

    Args:
        base_dirs (list): List of directory paths to consider.
        output_dir (str): Directory to save the output file.
        file_ext (str): File extension to match.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all possible directory pairs
    all_pairs = list(combinations(base_dirs, 2))
    print(f"Processing {len(all_pairs)} directory pairs...")
    print("Directory pairs:", all_pairs)

    # Dictionary to track file counts for each pair
    pair_counts = {}
    all_matched_files = []

    # Process each directory pair
    for i, (dir1, dir2) in enumerate(all_pairs):
        pair_name = f"{os.path.basename(dir1)}_{os.path.basename(dir2)}"
        print(f"Processing pair {i + 1}/{len(all_pairs)}: {pair_name}")

        # Find matching files between these directories
        matched_files = []
        for root, _, files in os.walk(dir1):
            for file in files:
                if file.lower().endswith(file_ext):
                    abs_path1 = os.path.join(root, file)
                    rel_path = os.path.relpath(abs_path1, dir1)
                    abs_path2 = os.path.join(dir2, rel_path)
                    if os.path.isfile(abs_path2):
                        matched_files.append((abs_path1, abs_path2))

        # Store count and add to overall collection
        pair_counts[pair_name] = len(matched_files)
        all_matched_files.extend([(f1, f2, pair_name) for f1, f2 in matched_files])
        print(f"  Found {len(matched_files)} matching files")

    # Save all pairs in one file
    output_file = os.path.join(output_dir, "all_pairs.txt")
    with open(output_file, 'w') as f:
        for f1, f2, pair_label in all_matched_files:
            f.write(f"{f1} {f2} {pair_label}\n")

    # Print summary statistics
    total_files = len(all_matched_files)
    print(f"\nTotal matched files: {total_files}")
    print("\nPair distribution:")
    for pair_name, count in pair_counts.items():
        print(f"  {pair_name}: {count} files")

# Example usage:
# base_dirs = ['dirA', 'dirB', 'dirC']
# output_dir = 'output'
# get_all_directory_pairs(base_dirs, output_dir, file_ext='.npy')




# Example usage

# base_directories = [
#     "/afs/crc.nd.edu/user/a/abhatta/face_matchers_result/cancellable_exps_v2/features/C_M/model1",
#     "/afs/crc.nd.edu/user/a/abhatta/face_matchers_result/cancellable_exps_v2/features/C_M/model2",
#     "/afs/crc.nd.edu/user/a/abhatta/face_matchers_result/cancellable_exps_v2/features/C_M/model3",
# ]


base_directories = [
    "/afs/crc.nd.edu/user/a/abhatta/face_matchers_result/cancellable_exps_v2/features/C_M/model1",
    "/afs/crc.nd.edu/user/a/abhatta/face_matchers_result/cancellable_exps_v2/features/C_M/model2",
    "/afs/crc.nd.edu/user/a/abhatta/face_matchers_result/cancellable_exps_v2/features/C_M/model3",
    "/afs/crc.nd.edu/user/a/abhatta/face_matchers_result/cancellable_exps_v2/features/C_M/model4",
    "/afs/crc.nd.edu/user/a/abhatta/face_matchers_result/cancellable_exps_v2/features/C_M/model5",
    "/afs/crc.nd.edu/user/a/abhatta/face_matchers_result/cancellable_exps_v2/features/C_M/model6",
    "/afs/crc.nd.edu/user/a/abhatta/face_matchers_result/cancellable_exps_v2/features/C_M/model7",
    "/afs/crc.nd.edu/user/a/abhatta/face_matchers_result/cancellable_exps_v2/features/C_M/model8",
    "/afs/crc.nd.edu/user/a/abhatta/face_matchers_result/cancellable_exps_v2/features/C_M/model9",
    "/afs/crc.nd.edu/user/a/abhatta/face_matchers_result/cancellable_exps_v2/features/C_M/model10"
]

output_directory = "/afs/crc.nd.edu/user/a/abhatta/Revocable_Journal_Extension/stratified_splits_1"
get_all_directory_pairs(base_directories, output_directory)