from pathlib import Path
import toml

project_root = Path(__file__).parent


# Extract version
def get_version():
    init_path = project_root / "hyperion" / "__init__.py"
    with open(init_path) as f:
        for line in f:
            if line.startswith("__version__"):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


# Extract dependencies
requirements_path = project_root / "requirements.txt"
with open(requirements_path) as f:
    requirements = f.read().splitlines()

# Generate console_scripts dynamically
binaries = (project_root / "hyperion" / "bin").glob("*.py")
console_scripts = {}
for binary in binaries:
    stem = binary.stem
    script_name = stem.replace("hyperion_", "").replace("_", "-")
    if script_name.startswith("-"):
        continue
    module = f"hyperion.bin.{stem}:main"
    console_scripts[f"hyperion-{script_name}"] = module

# Load existing pyproject.toml
pyproject_proto_path = project_root / "proto_pyproject.toml"
pyproject_path = project_root / "pyproject.toml"
pyproject_data = toml.load(pyproject_proto_path)

# Update fields dynamically
pyproject_data["project"]["version"] = get_version()
pyproject_data["project"]["dependencies"] = requirements
pyproject_data["project"]["scripts"] = console_scripts

# Save updated pyproject.toml
with open(pyproject_path, "w") as f:
    toml.dump(pyproject_data, f)

print("pyproject.toml updated successfully!")
