"""Make requests to OpenAI models to create dataset."""

from openai import OpenAI

client = OpenAI()


def get_chat_response():
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
            {"role": "user",
             "content": "Compose a poem that explains the concept of recursion in programming."}
        ]
    )
    return completion.choices[0].message


if __name__ == '__main__':
    # prompt = ('What is the capital of Zimbabwe?')
    response = get_chat_response()
    print(response)
