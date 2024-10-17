import sys
import os

# To run: 
# source ../../dl_env/bin/activate  (myenv is older one)
# TODO

# Add the parent directory (deep-learning) to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Import the main function from run_wiki.py
from nano_gpt.run_wiki import main as run_wiki_main  # type: ignore

if __name__ == "__main__":
    run_wiki_main()