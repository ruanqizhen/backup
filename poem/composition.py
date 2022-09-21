import openai


def gen_composition(title):
    openai.api_key = "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"

    response = openai.Completion.create(
        engine="davinci-instruct-beta",
        prompt=f"""Generate a 7th grader's 300 words English essay with title "{title}":""",
        temperature=0.7,
        max_tokens=800,
        top_p=1.0,
        frequency_penalty=0.7,
        presence_penalty=0.1,
    )

    return response["choices"][0]["text"]



def gen_pycode(requirements):
    openai.api_key = "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"

    response = openai.Completion.create(
        engine="davinci-codex",
        prompt=f"""
        '''
        {requirements}
        The python function:
        '''
        """,
        temperature=0,
        max_tokens=750,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )

    return response["choices"][0]["text"]
