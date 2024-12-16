import time
from typing import Literal

import keyboard
import tiktoken
from openai import OpenAI
from typing import Annotated
import json


def count_tokens(text: str, model: str = 'gpt-4'):
    # Get the appropriate tokenizer for the model
    encoding = tiktoken.encoding_for_model(model)
    # Encode the text to get tokens
    tokens = encoding.encode(text)
    return len(tokens)


class DatasetGenerator:
    def __init__(self):
        self.openai = OpenAI()

    def list_models(self):
        return self.openai.models.list()

    def call_model(self,
                   messages: list[dict],
                   frequency_penalty: Annotated[float, (-2, 2)] | None = None,
                   logprobs: bool = False,
                   top_logprobs: Annotated[int, (0, 20)] | None = None,
                   max_tokens: int | None = None,
                   n: int = 1,
                   presence_penalty: Annotated[float, (-2.0, 2.0)] = 0,
                   model: Literal['gpt-3.5-turbo', 'gpt-4o'] = 'gpt-3.5-turbo',
                   stop: str | list[str] | None = None,
                   temperature: Annotated[float, (0, 2)] | None = None,
                   top_p: Annotated[float, (0, 1)] | None = None):
        completion = self.openai.chat.completions.create(
            model=model,
            messages=messages,
            frequency_penalty=frequency_penalty,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            max_tokens=max_tokens,
            n=n,
            presence_penalty=presence_penalty,
            stop=stop,
            temperature=temperature,
            top_p=top_p
        )

        return completion

    def generate(self, n: int = 1):

        prompt = """
        Generate a data that can be used to train an NER model.
        
        Create a sentence that can be used as the input to the model model. The sentence should look like a
        set of fields in an unstructured text format. The objective is to train a model that can extract the desired
        entities from this unstructured sentence. It should only contain the data fields unless otherwise specified.

        An example of a sentence would be: 'Leonardo da Vinci, "Vitruvian Man" - oil on canvas; 14x16in - 1648'. The
        separators such as ',', ';', '-', should always be different. Don't follow this example word for word. The
        order of the fields should also vary in each response.

        The sentence should contain the following fields in any order:

        The name of an artist. The artist should be a real person that currently exists or has existed. It should not
        be a made up person. The artist must have made fine art. Examples of fine art in this context include painters,
        photographers, print makers, or any other relevant fine art format. The artist can also be from any time
        period. The artist's name can be misspelled. The artist name can be followed or preceded by other text, but
        the indices that represent the entity should only acknowledge the name itself. For example, 'signed by Pablo
        Picasso' can appear in the sentence but only Pablo Picasso should be used in the named entity section.

        A date. The date can be just a year. The date can otherwise be a month and year or a day, month, and year. The
        date cannot be only a day or only a month. Choose one of these options at random.

        The title of a work of art. The work of art must have been done by the artists that you chose. The work of art
        can be quoted or unquoted.

        A medium representing the medium that was used to create the work of art.

        A dimension specification. This should depend on the artwork. It can contain length, height, width, and depth
        where applicable. It may or may not include a unit of measurement. Unit of measurement, if included, can be
        provided for each dimension or for the entire dimension range.
        
        Introduce irregularities such as spelling errors or misplaced separators into any of the fields randomly. This
        should simulate a dirty string that needs to be cleaned.
        
        The output should be in CoNLL format. Do not include any unnecessary formatting or extraneous information. 
        Include only the data. Do not include code or text fences in the output. Example:
        
        oil B-MEDIUM
        on I-MEDIUM
        canvas I-MEDIUM
        - O
        'Girl B-ART_TITLE
        with I-ART_TITLE
        a I-ART_TITLE
        Pearl I-ART_TITLE
        Earring'; I-ART_TITLE
        1665, B-DATE
        Johannes B-ARTIST
        Vermeer, I-ARTIST
        44.5cm B-DIMENSION
        x I-DIMENSION
        39cm I-DIMENSION

        Names should be given the entity tag 'ARTIST'.
        Dates should be given the entity tag 'DATE'.
        Artwork titles should be given the entity tag 'ART_TITLE'.
        Mediums should be given the entity tag 'MEDIUM'.
        Dimensions should be given the entity tag 'DIMENSION'.
        
        When I submit a follow up response saying 'Do it again.', repeat these instructions with a different work of art.
        """

        messages = [
            {'role': 'system', 'content': 'You are a dataset generator.'},
            {'role': 'user', 'content': prompt}
        ]

        responses = []

        original_message = [
            {'role': 'system', 'content': 'You are a dataset generator.'},
            {'role': 'user', 'content': prompt}
        ]
        original_context_length = count_tokens(". ".join([x["content"] for x in original_message]))

        messages = original_message.copy()

        # max_context_length = 8192
        # Set limit lower than max for faster responses.
        max_context_length = 5000
        current_context_length = original_context_length

        try:
            print("Press 'q' to trigger a keyboard interrupt.")
            for i in range(n):

                print(f'Current context length: {current_context_length}')
                print(f'Response: {i}')
                print('Sending request.')
                try:
                    response = self.call_model(messages, model='gpt-4o', temperature=0.4, presence_penalty=-1)
                except Exception as e:
                    print(f'Error sending request. Terminating early. {e}')
                    return responses
                print('Exacting response.')
                message = response.choices[0].message.content
                print('Storing response.')
                responses.append(message)
                current_context_length += count_tokens(message)
                if current_context_length < max_context_length:
                    print('Updating message context.')
                    messages.append({'role': 'assistant', 'content': message})
                    if n > 0:
                        print('Requesting another response.', end='\n\n')
                        messages.append({'role': 'user', 'content': 'Do it again.'})
                        current_context_length += count_tokens('Do it again.')
                else:
                    messages = original_message.copy()
                    current_context_length = original_context_length
        except KeyboardInterrupt:
            print("\nPerforming graceful exit...")
            return responses

        return responses


if __name__ == '__main__':
    generator = DatasetGenerator()
    result = generator.generate(n=100)

    print('Writing responses to file', end='\n\n')
    with open('datasets/CoNLL/003/001.txt', 'w') as f:
        # f.write(json.dumps(responses, indent=4))
        f.write("\n\n".join(result))

    for item in result:
        print(item, end='\n\n')