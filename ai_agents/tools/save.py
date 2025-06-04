import os
from dotenv import load_dotenv
from langchain_core.tools import tool
import sys

# Load environment variables (though save_tool doesn't strictly need them from .env)
load_dotenv()

@tool
def save_tool(filename: str, text: str) -> str:
    """Saves the given text to a file with the specified filename.
    Useful for persisting research findings, summaries, or generated content.
    The file will be saved in the current working directory unless an absolute path is given.
    """
    try:
        # Ensure the directory exists if filename includes a path
        # For simplicity, this example assumes filename is just a name or relative path
        # that doesn't require new directory creation.
        # For more robust saving, consider using os.makedirs(os.path.dirname(filename), exist_ok=True)
        # if filename can be a deeper path.
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)
        return f"Text saved to {filename} successfully."
    except Exception as e:
        return f"Error saving file '{filename}': {str(e)}"

if __name__ == '__main__':
    # Basic test
    print("Testing save.py...", file=sys.stderr)
    test_filename = "test_save_output.txt"
    test_content = "This is content saved by save.py test."
    print(f"Saving to: {test_filename}", file=sys.stderr)
    result = save_tool.invoke({"filename": test_filename, "text": test_content})
    print(result, file=sys.stderr)
    if os.path.exists(test_filename):
        print(f"File '{test_filename}' created. Verifying content...", file=sys.stderr)
        with open(test_filename, "r") as f:
            content = f.read()
        assert content == test_content
        print("Content verified.", file=sys.stderr)
        os.remove(test_filename)
        print(f"File '{test_filename}' removed.", file=sys.stderr)
    else:
        print(f"File '{test_filename}' NOT created.", file=sys.stderr) 