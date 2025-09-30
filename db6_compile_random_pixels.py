import os
import random
import argparse

def choose_lines(folder_path, num_lines, output_file):
    all_lines = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"): 
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()
                all_lines.extend(lines)
    
    selected_lines = random.sample(all_lines, min(num_lines, len(all_lines)))
    
    with open(output_file, 'w') as file:
        for line in selected_lines:
            file.write(line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compiles random lines from text files in a folder")
    parser.add_argument("--folder_path", type=str, required=True, help="Folder containing .txt files")
    parser.add_argument("--output_file", type=str, required=True, help="Output file")
    parser.add_argument("--num_lines", type=int, required=True, help="Number of lines to extract")
    args = parser.parse_args()

    choose_lines(args.folder_path, args.num_lines, args.output_file)
    print(f"Extracted {args.num_lines} lines to {args.output_file}.")
