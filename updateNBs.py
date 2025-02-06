import json
import re
import os


def copy_and_modify_notebook(
    original_notebook, new_notebook, original_instance, new_instance
):
    # Read the original notebook
    with open(original_notebook, "r", encoding="utf-8") as f:
        notebook_data = json.load(f)

    # Regex pattern to match "instance*p1" and replace "p1" with "pX"
    pattern = re.compile(rf"instance\*{original_instance}")

    # Modify the notebook content
    for cell in notebook_data.get("cells", []):
        if cell.get("cell_type") in {"code", "markdown"}:
            cell["source"] = [
                pattern.sub(f"instance*{new_instance}", line) for line in cell["source"]
            ]

    # Write the modified notebook to a new file
    with open(new_notebook, "w", encoding="utf-8") as f:
        json.dump(notebook_data, f, indent=2)


# Directories
notebooks_dir = "./MTGANotebooks"
problems_dir = "./problems"

# Original notebook
original_notebook = os.path.join(notebooks_dir, "MTGANotebook-p1.ipynb")

# Scan the problems directory for JSON files
for filename in os.listdir(problems_dir):
    if filename.endswith(".json"):
        instance_name = filename.split(".json")[0]  # Extract "pX" from "pX.json"
        new_notebook = os.path.join(
            notebooks_dir, f"MTGANotebook-{instance_name}.ipynb"
        )
        copy_and_modify_notebook(original_notebook, new_notebook, "p1", instance_name)

print("Notebooks copied and modified successfully.")
