# AI-Email-Tasks-Extractor by @omairionz

## What this program does

1. Running `email_database.py` creates a vector ChromaDB that stores email chunks. Sample emails are in `email.txt`, but feel free to replace with your own.
2. Running `query_email.py` runs the program and creates a table of extracted tasks found in `email.txt` in the CLI.
3. Table sorts by `Priority`, but also contains information on `Category`, `Status`, `Deadline`, and `Item #`.
4. Items can be marked as done and removed, and the table can be saved inside `tasks.md` through arrow key inputs all through CLI.

## Install dependencies before running

1. Do the following before installing the dependencies found in `requirements.txt` file because of current challenges installing `onnxruntime` through `pip install onnxruntime`. 

     - For Windows users, follow the guide [here](https://github.com/bycloudai/InstallVSBuildToolsWindows?tab=readme-ov-file) to install the Microsoft C++ Build Tools. Be sure to follow through to the last step to set the environment variable path.

2. Now run this command to install dependenies in the `requirements.txt` file. 

```python
pip install -r requirements.txt
```

3. Install markdown depenendies with: 

```python
pip install "unstructured[md]"
```

## How to run this project

1. Create Chroma database

```python
python email_database.py
```

2. Query the database

```python
python query_email.py
```

## Credits and Additional Information

You'll also need to set up an OpenAI account (and set the OpenAI key in your environment variable) for this to work. You can do that [here](https://platform.openai.com/api-keys).

> Here is a step-by-step tutorial video I used to create a portion of this project: [RAG+Langchain Python Project: Easy AI/Chat For Your Docs](https://www.youtube.com/watch?v=tcqEUSNCn8I&ab_channel=pixegami).

