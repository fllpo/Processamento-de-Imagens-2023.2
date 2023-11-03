import os

INPUT_FOLDER = "input"
OUTPUT_FOLDER = "output"

def input_file(filename):
    return os.path.join(INPUT_FOLDER, filename)

def output_file(filename):
    return os.path.join(OUTPUT_FOLDER, filename)