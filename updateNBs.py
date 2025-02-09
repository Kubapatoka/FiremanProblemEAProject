import json
import re
import os


def copy_and_modify_notebook(original_notebook, new_notebook, original_instance, new_instance):
    print(f"comparing {original_notebook} with {new_notebook}, by instances {original_instance} -> {new_instance}")
    # Read the original notebook
    with open(original_notebook, "r", encoding="utf-8") as f:
        notebook_data = json.load(f)

    # Flexible regex pattern: match "instance*p1", where "*" could be any character(s)
    pattern = re.compile(rf"problems/{original_instance}")

    def sub(line):
        subed = pattern.sub(f"problems/{new_instance}", line)
        if subed != line:
            print("<", line.rstrip("\n"))
            print(">", subed.rstrip("\n"))
        return subed

    # Modify the notebook content
    for cell in notebook_data.get("cells", []):
        if cell.get("cell_type") in {"code", "markdown"}:
            cell["source"] = [sub(line) for line in cell["source"]]

    # Write the modified notebook to a new file
    with open(new_notebook, "w", encoding="utf-8") as f:
        json.dump(notebook_data, f, indent=2)


# Directories
notebooks_dir = "./MTGA-Notebooks"
problems_dir = "./problems"

# Original notebook
original_notebook = os.path.join(notebooks_dir, "MTGANotebook-p1.ipynb")

# Scan the problems directory for JSON files
for filename in os.listdir(problems_dir):
    if filename.endswith(".json"):
        instance_name = filename.split(".json")[0]
        if instance_name == "p1":
            continue

        new_notebook = os.path.join(notebooks_dir, f"MTGANotebook-{instance_name}.ipynb")
        copy_and_modify_notebook(original_notebook, new_notebook, "p1", instance_name)

print("Notebooks copied and modified successfully.")
