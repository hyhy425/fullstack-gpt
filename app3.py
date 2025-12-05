from langchain.utilities.wikipedia import WikipediaAPIWrapper
from openai import OpenAI
import streamlit as st
import json
import time
import requests
from bs4 import BeautifulSoup


def get_wiki(inputs):
    wiki = WikipediaAPIWrapper()
    query = inputs["query"]
    return wiki.run(query)


def get_web_text(inputs):
    url = inputs["url"]
    response = requests.get(url, headers={"User-Agent": "Mozilla"})
    html = response.text

    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(" ")
    text = " ".join(text.split())

    return text


functions_map = {
    "get_wiki": get_wiki,
    "get_web_text": get_web_text,
}


functions = [
    {
        "type": "function",
        "function": {
            "name": "get_wiki",
            "description": "Search Wikipedia and return article text.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_web_text",
            "description": "Fetch readable text from a webpage.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
]


def get_thread_id(client):
    if "thread_id" not in st.session_state:
        thread = client.beta.threads.create()
        st.session_state["thread_id"] = thread.id
    return st.session_state["thread_id"]


def get_assistant_id(client):
    if "assistant_id" not in st.session_state:
        assistant = client.beta.assistants.create(
            name="Research Assistant",
            instructions="Use tools to gather info from Wikipedia and web pages.",
            model="gpt-4o-mini",
            tools=functions,
        )
        st.session_state["assistant_id"] = assistant.id
    return st.session_state["assistant_id"]


def get_run(run_id, thread_id):
    return client.beta.threads.runs.retrieve(
        run_id=run_id,
        thread_id=thread_id,
    )


def send_message(thread_id, content):
    return client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=content,
    )


def get_messages(thread_id):
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    messages = list(messages)
    messages.reverse()
    return messages


def get_tool_outputs(run_id, thread_id):
    run = get_run(run_id, thread_id)
    outputs = []

    for action in run.required_action.submit_tool_outputs.tool_calls:
        action_id = action.id
        function = action.function
        inputs = json.loads(function.arguments)

        result = functions_map[function.name](inputs)

        outputs.append(
            {
                "output": result,
                "tool_call_id": action_id,
            }
        )

    return outputs


def submit_tool_outputs(run_id, thread_id):
    outputs = get_tool_outputs(run_id, thread_id)
    return client.beta.threads.runs.submit_tool_outputs(
        run_id=run_id,
        thread_id=thread_id,
        tool_outputs=outputs,
    )


def run_assistant(thread_id, assistant_id, content):

    send_message(thread_id, content)

    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
    )

    while True:
        run = get_run(run.id, thread_id)

        if run.status == "requires_action":
            submit_tool_outputs(run.id, thread_id)

        elif run.status == "completed":
            break

        elif run.status in ["failed", "expired", "cancelled"]:
            st.error(run.status)
            break

        time.sleep(0.5)

    return run


@st.cache_data(show_spinner="Validating API key...")
def validate_api_key(api_key):
    client = OpenAI(api_key=api_key)
    client.models.list()
    return True


st.set_page_config(
    page_title="AssistantsGPT",
    page_icon="👀",
)


st.markdown(
    """
    # AssistantsGPT

    An assistant that helps you research using Wikipedia and web pages.
"""
)


with st.sidebar:
    st.markdown("📦 [GitHub Repository](https://github.com/hyhy425/fullstack-gpt)")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
    )

if not api_key:
    st.warning("Please enter your API key in the sidebar.")
    st.stop()

try:
    validate_api_key(api_key)
except Exception:
    st.error("Invalid API Key.")
    st.stop()


client = OpenAI(api_key=api_key)
thread_id = get_thread_id(client)
assistant_id = get_assistant_id(client)


query = st.chat_input("Ask a research question.")

messages = get_messages(thread_id)
for message in messages:
    text = message.content[0].text.value
    with st.chat_message(message.role):
        st.markdown(text)

if query:
    with st.chat_message("user"):
        st.markdown(query)

    with st.spinner("Researching..."):
        run_assistant(thread_id, assistant_id, query)

    last_message = get_messages(thread_id)[-1]
    answer = last_message.content[0].text.value

    with st.chat_message("assistant"):
        st.markdown(answer)
