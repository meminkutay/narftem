import os
import ast
import shutil

def find_imports(py_file):
    """Find all imports in a Python file."""
    imports = set()
    try:
        with open(py_file, "r") as f:
            tree = ast.parse(f.read(), filename=py_file)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split('.')[0])  # Get the top-level module
                elif isinstance(node, ast.ImportFrom):
                    if node.module:  # Handle "from module import something"
                        imports.add(node.module.split('.')[0])
    except Exception as e:
        print(f"Error processing {py_file}: {e}")
    return imports

def resolve_files(imports, base_dir):
    """Resolve imported modules to actual file paths in the base directory."""
    resolved = set()
    for imp in imports:
        # Check for .py files in the base directory
        py_file = os.path.join(base_dir, f"{imp}.py")
        if os.path.isfile(py_file):
            resolved.add(py_file)
    return resolved

def gather_files(entry_file, base_dir, clean_dir):
    """Gather all necessary files and copy them to the 'clean' folder."""
    os.makedirs(clean_dir, exist_ok=True)
    processed = set()
    to_process = {entry_file}
    all_files = set()

    while to_process:
        current_file = to_process.pop()
        if current_file in processed:
            continue
        processed.add(current_file)
        all_files.add(current_file)

        # Find imports in the current file
        imports = find_imports(current_file)
        resolved_files = resolve_files(imports, base_dir)
        to_process.update(resolved_files)

    # Copy files to the clean directory
    for file in all_files:
        relative_path = os.path.relpath(file, base_dir)
        dest_path = os.path.join(clean_dir, relative_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy(file, dest_path)

    print(f"Clean folder created at {clean_dir} with {len(all_files)} files.")

# Example usage:
entry_file = "lambda_entry_updapsac.py"  # Replace with the entry point of your Python application
base_dir = os.getcwd()  # Replace with the base directory of your project
clean_dir = os.path.join(base_dir, "clean")  # Directory for cleaned files

gather_files(entry_file, base_dir, clean_dir)
