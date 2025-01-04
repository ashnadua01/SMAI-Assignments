#!/bin/bash

# Check if the test data is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path_to_test_data>"
    exit 1
fi

test_data_path="$1"

# Check the knn script
if [ ! -f "knn.py" ]; then
    echo "Python script 'knn.py' not found in the current directory."
    exit 1
fi

# Throw error if the test data does not exist
if [ ! -f "$test_data_path" ]; then
    echo "Test data file not found: $test_data_path"
    exit 1
fi

# Run the Python script and capture output
metrics=$(python3 knn.py "$test_data_path")

# Check if metrics was populated properly
if echo "$metrics" | grep -i -E 'error|exception'; then
    echo "Python script encountered errors. Please check the script and test data."
    exit 1
fi

echo "$metrics"
