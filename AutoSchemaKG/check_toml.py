# check_toml.py
import tomli

try:
    with open("pyproject.toml", "rb") as f:
        tomli.load(f)
    print("pyproject.toml is syntactically valid")
except tomli.TOMLDecodeError as e:
    print(f"Invalid TOML: {e}")
except FileNotFoundError:
    print("pyproject.toml not found")