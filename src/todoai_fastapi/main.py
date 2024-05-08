import json
import os
from openai import OpenAI
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from fastapi import FastAPI
from pydantic import BaseModel
from pydantic.json import pydantic_encoder


API_KEY = os.environ["API_KEY"]
MODEL = "gpt-4-turbo"
TOOLS = [
  {
    "type": "function",
    "function": {
      "name": "add",
      "description": "Add a new todo to the list",
      "parameters": {
        "type": "object",
        "properties": {
          "title": {
            "type": "string",
            "description": "The brief and short description of the todo"
          },
          "description": {
            "type": "string",
            "description": "The concise and full description of the todo"
          }
        }
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "mark_as_done",
      "description": "Marks a todo from the list as done or completed. It marks only 1 entry. If need to mark multiple todos as done, then we need to call this function several times. If user uses word like first, then assume it's 0.",
      "parameters": {
        "type": "object",
        "properties": {
          "index": {
            "type": "integer",
            "description": "If index is 0 then mark the latest todo. If index is 1 then mark the second most recent todo."
          }
        }
      }
    }
  }
]
MESSAGES = [
    {
        "role": "system",
        "content": "I need you to convert user requests to function calls for my 'todo' list app, users can either add new stuffs 'to do' or mark existing todos as completed. Please be smart, if user is using past tense, it probably means he wants to mark something as complete. Otherwise if it's in the future, it probably means he wants to add something to his to do list. If the user tells you to add new todo, remember to split contextually different items in different todos. Here are the user todos in JSON for more context: "
    },
    {
        "role": "user",
        "content": "I need to buy some groceries then study for my exams. I've also just ran in the park."
    }
]


class Todo(BaseModel):
    index: int
    title: str
    description: str | None
    completed: bool


class Question(BaseModel):
    message: str
    todos: list[Todo]


client = OpenAI(api_key=API_KEY)


def ask_openai(message: str, todos: list[Todo]):
    messages = [*MESSAGES]
    messages[0]["content"] += json.dumps(todos, default=pydantic_encoder)
    messages[1]["content"] = message
    return client.chat.completions.create(
        model=MODEL,
        tools=TOOLS,
        messages=messages,
        tool_choice="required",
        temperature=0.8,
    )


app = FastAPI()


@app.post("/ask")
async def ask(question: Question) -> list[ChatCompletionMessageToolCall]:
    answer = ask_openai(question.message, question.todos)
    return answer.choices[0].message.tool_calls
