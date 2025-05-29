// existing code...

## Running the Agents

This project provides a Command Line Interface (CLI) to interact with the implemented AI agents.

### Prerequisites

1.  Ensure Python 3.9+ is installed.
2.  Clone the repository.
3.  Navigate to the project root directory (`AI-AGENTS`).
4.  Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
5.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
6.  Set up your environment variables:
    - Copy the `.env.example` file to `.env` in the project root: `cp .env.example .env`
    - Edit the `.env` file and fill in your API keys and other configurations.

### CLI Commands

All commands are run from the project root directory.

**1. Research Agent**

To perform research on a topic:
