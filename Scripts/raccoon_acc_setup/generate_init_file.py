import os
import ast

def generate_init_file(directory: str):
    """
    Automatically generate the __init__.py file for a package.
    Includes all functions from Python files in the directory.

    :param directory: Path to the package directory
    """
    init_file_path = os.path.join(directory, "__init__.py")
    imports = []

    # Iterate through all Python files in the directory
    for file_name in os.listdir(directory):
        if file_name.endswith(".py") and file_name != "__init__.py":
            module_name = file_name[:-3]  # Strip the .py extension
            file_path = os.path.join(directory, file_name)

            # Parse the Python file to extract function names
            with open(file_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=file_path)
                functions = [
                    node.name for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef)
                ]

            # Add import statements for all functions
            for func in functions:
                imports.append(f"from .{module_name} import {func}")

    # Write the imports to the __init__.py file
    with open(init_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(imports))
    print(f"__init__.py has been generated in {directory}.")

# Example usage
generate_init_file("./")