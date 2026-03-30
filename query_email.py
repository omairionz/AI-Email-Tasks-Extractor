from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma 
from langchain_openai import OpenAIEmbeddings 
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
from datetime import datetime
from enum import Enum
from rich.console import Console
from rich.table import Table
from rich import box
from InquirerPy import inquirer
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

load_dotenv()

console = Console()

CHROMA_PATH = "chroma-database"

PROMPT_TEMPLATE = """
You are an Executive Assistant. Extract tasks, deadlines, priorities, and categories.

Categorization Rules:
- Work: Anything related to reports, meetings, or marketing.
- Personal: Anything related to home, family, or health.

Priority Rules - MUST follow exactly:
- 🟥: Use ONLY for "Immediately," "EOD," "Emergency," "Today," or "Tomorrow Morning."
- 🟨: Use for anything due "Tomorrow," "Next week," or "By Friday." If it's important for work, default to High.
- 🟩: Use for "End of month," "Sometime," or non-essential personal tasks.

Status Rules - MUST FOLLOW EXACTLY:
- Status may either be ⏳ or ✅ ONLY. 
- Defaults to ⏳

Extract EVERY distinct task found in the text. Do not summarize multiple emails into one task. 
If a deadline is a specific day of the week and today is that day of the week, assume the deadline is Today (the current date) unless the text implies otherwise.
ALWAYS convert relative terms (today, tomorrow, next week, EOD) into actual dates based on Today's Date.
Output dates in 'YYYY-MM-DD' format.
Ignore email signitures.

CRITICAL: Under no circumstances are you allowed to return 'Not specified' for a deadline. If the timeline is vague (e.g., 'sometime next week' or 'the week after'), you MUST perform the calculation.

Today's Date: {current_date}
Context: {emails}
Query: {query}
"""

class PriorityEnum(str, Enum):
     urgent = "🟥"
     high = "🟨"
     low = "🟩"

class CategoryEnum(str, Enum):
     work = "Work"
     personal = "Personal"

class Task(BaseModel):
     task_description: str = Field(description="The action item found in the email")
     deadline: str = Field(description="The deadline in ISO YYYY-MM-DD format. Calculate this based on Today's Date. If the email says 'next week' with no day or date, provide the date for the following Monday. If the email gives a 'end of month' day, provide the date for the last day of the month. If it says 'sometime next week', default to 7 days from today unless otherwise specified. If it says 'sometime next month', default to the first day of next month unless otherwise specified. If it says 'sometime next year', default to the first day of next year unless otherwise specified. NEVER return 'not specified'.")
     priority: str = Field(description="How much immediate attention does this task require? The priority level. MUST be exactly one of these three emojis: 🟥, 🟨, or 🟩.")
     category: str = Field(description="The type of work based on content")
     status: str = "⏳"

class TaskList(BaseModel):
     tasks: List[Task]

def interactive_menu(task_list):
     while True:
          PRIORITY_WEIGHTS = {
               "🟥": 1, 
               "🟨": 2, 
               "🟩": 3}

          task_list.tasks.sort(key=lambda x: (PRIORITY_WEIGHTS.get(x.priority, 99), x.deadline))

          display_tasks(task_list)

          action = inquirer.select(
               message="Select an action:",
               choices=["Mark Task as Done", "Delete Task", "Save & Exit"],
          ).execute()

          if action == "Save & Exit":
               save_to_markdown(task_list)
               console.print("[bold green]Progress saved to tasks.md. Goodbye![/bold green]")
               break

          if action == "Mark Task as Done":
               pending_choices = [t.task_description for t in task_list.tasks if "⏳" in t.status]
               choices_to_show = pending_choices if pending_choices else ["-- No Pending Tasks (Go Back) --"]

               task_choice = inquirer.select(
                    message="Which task did you finish?",
                    choices=choices_to_show
               ).execute()

               if task_choice in pending_choices:
                    for t in task_list.tasks:
                         if t.task_description == task_choice:
                              t.status = "✅"
                              print(f"✅ Updated: {task_choice}")

          if action == "Delete Task":
               if not task_list.tasks:
                    console.print("[bold red]⚠️ No tasks left to delete![/bold red]")
                    continue

               task_to_delete = inquirer.select(
                    message="Which task would you like to remove?",
                    choices=[t.task_description for t in task_list.tasks] + ["Cancel"]
               ).execute()

               if task_to_delete != "Cancel":
                    original_count = len(task_list.tasks)
                    task_list.tasks = [t for t in task_list.tasks if t.task_description != task_to_delete]
               
                    if len(task_list.tasks) < original_count:
                         console.print(f"[bold red]🗑️ Deleted:[/bold red] {task_to_delete}")

def display_tasks(task_list: TaskList):

     PRIORITY_WEIGHTS = {
         "🟥": 1, 
         "🟨": 2, 
         "🟩": 3}

     sorted_tasks = sorted(
          task_list.tasks, 
          key=lambda x: (PRIORITY_WEIGHTS.get(x.priority, 99), x.deadline)
     )

     print("\n")
     table = Table(title="[bold cyan]AI Task Extraction Results[/bold cyan]", show_lines=True, box=box.HEAVY, header_style="bold white", border_style="dim")

     table.add_column("Item #", justify="center", width=6, style="bold yellow")
     table.add_column("Status", justify="center", width=8)
     table.add_column("Priority", justify="center", width=10)
     table.add_column("Category", justify="center", style="bold blue", width=10)
     table.add_column("Task Description", justify="left", style="white")
     table.add_column("Deadline", justify="center", style="underline green")

     for i, task in enumerate(sorted_tasks, 1):
          
          style = "dim" if "✅" in task.status else ""
          status_icon = "✅" if "✅" in task.status else "⏳"

          table.add_row(
               str(i),
               status_icon,
               task.priority,
               task.category,
               task.task_description,
               task.deadline,
               style=style
          )

     console.print(table)
     print("\n")

def save_to_markdown(task_list, filename="tasks.md"):
     with open(filename, "a", encoding="utf-8") as f:
          f.write(f"\n## Tasks Extracted on {datetime.now().strftime('%Y-%m-%d')}\n")
          for task in task_list.tasks:
               f.write(f"- {task.category}({task.priority}) | **{task.deadline}** | {task.task_description}\n")

def main():
     
     query_text = "List all tasks found in the emails."
     
     db = Chroma(persist_directory=CHROMA_PATH, embedding_function=OpenAIEmbeddings())

     results = db.get()
     all_emails = results['documents']

     if not all_emails:
          print("Database is empty. :(")

     email_text = "\n\n---\n\n".join(all_emails)
     prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

     prompt = prompt_template.format(
          current_date=datetime.now().strftime("%A, %B %d, %Y"),
          emails=email_text,
          query=query_text
     )

     model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

     structured_llm = model.with_structured_output(TaskList)
     response = structured_llm.invoke(prompt)

     interactive_menu(response)

if __name__ == "__main__":
    main()

