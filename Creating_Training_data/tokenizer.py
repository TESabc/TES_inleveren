import random

from random import randint
from itertools import islice
from transformers import AutoTokenizer # this one needs to be installed
from Creating_Training_data.query_components import SpecialTokens


def reload(cls):
    cls._reload()

    return cls


@reload
class Tokenizer:

    def __init__(self, *args, **kwargs):
        raise Exception('Do not initialize this class.')

    def __new__(cls):
        return cls

    @classmethod
    def _reload(cls):
        special_tokens = {special_token.value for special_token in SpecialTokens}
        cls.tokens = [
            key for key in AutoTokenizer.from_pretrained('kadasterdst/querygenerator').get_vocab().keys()
            if key not in special_tokens and 'extra_id' not in key
        ]

    @staticmethod
    def random_chunk(li, min_chunk=1, max_chunk=3):
        it = iter(li)
        while True:
            nxt = list(islice(it, randint(min_chunk, max_chunk)))
            if nxt:
                yield nxt
            else:
                break

    @classmethod
    def get_random_text(cls, min_token_count=1, max_token_count=5):
        token_count = random.randint(min_token_count, max_token_count)
        tokens = random.sample(cls.tokens, token_count)
        text = ' '.join([''.join(word) for word in cls.random_chunk(tokens)])
        text = text.replace('▁', ' ')

        return text
