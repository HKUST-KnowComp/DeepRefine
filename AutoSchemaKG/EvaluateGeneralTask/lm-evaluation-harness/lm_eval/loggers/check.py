import os
import utils
print("Current working directory:", os.getcwd())



import subprocess
try:
    subprocess.check_output(["git", "describe", "--always"])
except subprocess.CalledProcessError as e:
    git_hash = utils.get_commit_from_path(os.getcwd())
    print("Git command failed:", e)