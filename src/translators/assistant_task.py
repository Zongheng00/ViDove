import time


def Assistant_task(input, task, client, thread_id, assistant_id, temp = 0.15):
    """
    TODO: setup changable domain since the domain is fix in the prompt

    Translates input sentence with desired LLM.

    :param input: Sentence for translation.
    :param task: Prompt.
    :param client: OpenAI client.
    :param thread_id: Thread id.
    :param temp: Model temperature.
    """

    thread_message = client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content= task + "/n" + input
    )

    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
    )

    print("sending message to assistant")

    run = wait_on_run(client, run, thread_id)

    # retrieve all messages added after our last user message

    messages = client.beta.threads.messages.list(
        thread_id=thread_id, order="asc", after=thread_message.id
    ).data

    # print(messages)
    # messages = messages.data

    print(messages[0].content[0].text.value.strip())

    return messages[0].content[0].text.value.strip()

def wait_on_run(client, run, thread_id):
    while run.status == "queued" or run.status == "in_progress":
        run = client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run.id,
        )
        time.sleep(0.5)
    return run
    