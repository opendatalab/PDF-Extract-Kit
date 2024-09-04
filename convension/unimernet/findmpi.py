import os

def find_libmpi():
    # Get the current conda environment path
    conda_env_path = os.environ.get('CONDA_PREFIX')
    if not conda_env_path:
        return 'No conda environment found.'
    
    # Search for libmpi.so files
    libmpi_files = []
    for root, dirs, files in os.walk(conda_env_path):
        for file in files:
            if 'libmpi.so' in file:
                libmpi_files.append(os.path.join(root, file))
    
    return libmpi_files

# Execute the function and print the results
libmpi_locations = find_libmpi()
print(libmpi_locations)