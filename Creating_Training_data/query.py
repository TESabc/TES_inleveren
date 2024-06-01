import os
import yaml
import string
import random

from collections import defaultdict
from abc import ABC, abstractmethod
from Creating_Training_data.tokenizer import Tokenizer
from Creating_Training_data.query_components import MORE_INFORMATION, TEMPLATE_OPTIONS


class Query(ABC):
    _related = {}
    _weight = 1
    _related_key = None

    _dependencies = {}

    questions = None

    '''
    For brt_filter_query.py we do have 'template' in kwargs.keys()
    So in that case we set self.questions to the contents of the YAML file belonging to the query template.
    ( I think it will be a list with all the templates? Perhaps not though. )

    We also take the adresses if they are in the kwargs... and assign them to self.adresses
    '''

    def __init__(self, *args, **kwargs):
        if 'addresses' in kwargs.keys():
            self.addresses = kwargs['addresses']
        if 'template' in kwargs.keys():
            self.questions = self.read_questions(kwargs['template'])

        self.typos = kwargs.get('typos', True)

    def get_dependencies(self):
        return self._dependencies

    @abstractmethod
    def get_answer(self, *args, **kwargs) -> str:
        pass

    def get_question(self, *args, **kwargs) -> str:
        pass

    '''
    Goes to the YML templates in the correct folder and selects the file specified by template_name.

    The yaml.full_load() function is used to load the entire YAML content from the file into a Python data structure. 
    This could be a dictionary, list, or other appropriate data types depending on the content of the YAML file.
    In our case it will be a dictionary. That is why we return data['questions'].    
    '''

    @staticmethod
    def read_questions(template_name):
        directory_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')

        with open(os.path.join(directory_name, f'{template_name}.yml'), encoding='utf-8') as fh:
            data = yaml.full_load(fh)

        return data['questions']


    '''
    It selects one random questions from the templates.

    It takes the query (natural language question template) and splits it into a list of words.

    Then if it is a placeholder like {brtproperty} it keeps the word as it is. The other words it can
    introduce a typo inside.

    The function returns the list with words in the template (with typos.. )
    It also returns a dictionary that maps each word (with typos) to the index in the list.
    A word can be mapped to multiple indices.

    It also returns the original question dictionary.


    '''

    def get_question_tokens(self, typo_count=0):
        question = random.choice(self.questions)

        # We split the question in the template into a list of words.
        template_words = list(question['query'].split())

        template_words = [
            word if word in TEMPLATE_OPTIONS else self.introduce_typo(word, max_count=typo_count)
            for word in template_words
        ]

        token_to_indices = defaultdict(list)
        for i, word in enumerate(template_words):
            token_to_indices[word].append(i)

        return template_words, token_to_indices, question

    '''
    Looks for the indices in which the property is (we input placeholder).
    Then we change the words in the list of words of the template to the synonym.
    '''

    @staticmethod
    def update_tokens(template_words, token_indices, prop, synonym, index=None):
        if index is None:
            for i in token_indices[prop]:
                template_words[i] = synonym if synonym else ''
        else:
            template_words[index] = synonym if synonym else ''

    @staticmethod
    def tokens_to_question(template_words):
        return ' '.join([token for token in template_words if token])

    @abstractmethod
    def get_question_answer(self, *args, **kwargs) -> str:
        pass

    @staticmethod
    def format_query(query):
        tokens = (token for line in query.strip().splitlines() for token in line.split())

        return ' '.join((token for token in tokens if token))

    def introduce_typo(self, word, max_count=1, min_count=1, always=False):
        if self.typos is False:
            return word

        if word and max_count >= min_count and (random.randint(0, 3) == 0 or always):
            for _ in range(random.randint(min_count, max_count)):
                char_list = list(word)
                char_list[random.randint(0, len(word) - 1)] = random.choice(string.ascii_lowercase)
                word = ''.join(char_list)

        return word

    def introduce_typo_template(self, word, max_count=1, min_count=1, always=False):
        if self.typos is False:
            return word

        if word and (random.randint(0, 3) == 0 or always):
            for _ in range(random.randint(min_count, max_count)):
                char_list = list(word)
                char_index = random.randint(0, len(word) - 1)
                if char_list[char_index] not in '{}':
                    char_list[char_index] = random.choice(string.ascii_lowercase)
                    word = ''.join(char_list)

        return word

    @staticmethod
    def get_last_answer_kwargs(answer_kwargs_stack):
        if answer_kwargs_stack:
            return answer_kwargs_stack[-1]
        else:
            return {}

    '''

    '''

    @staticmethod
    def get_text_by_classes(connectors, probability_mask=None, return_components=False):
        stack = []

        for i, connector in enumerate(connectors):
            probability = 50 if probability_mask is None or probability_mask[i] is None else probability_mask[i]
            if random.randint(0, 100) < probability:
                part = random.choice(tuple(connector))
                stack.append(part)

        text = ' '.join([item for item in stack if item])
        result = (text, stack) if return_components else text

        return result

    @staticmethod
    def get_key_value(attribute):
        key = random.choice(list(attribute.keys()))

        return key, random.choice(list(attribute[key]))

    @staticmethod
    def get_enum_key_value(attribute):
        key = random.choice(list(attribute))

        return key.name, random.choice(list(key.value))

    @staticmethod
    def get_random_attribute(*args, **kwargs):
        return Tokenizer.get_random_text(*args, **kwargs)

    @staticmethod
    def format_more_information(*args):
        args_first = ', '.join(args[:-2])
        args_last = ' of '.join(args[-2:])

        return MORE_INFORMATION.format(args_first + args_last)

    def get_weight(self):
        return self._weight

    @abstractmethod
    def get_related(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_related_options(self, *args, **kwargs):
        pass

    def get_address(self, street=True, number=True, generate_postal_code=True, typo_count=0):
        # Picks a random adress
        item = random.choice(self.addresses)
        address_details, address_details_str = {}, []

        # There is a 50-50 probability to start generating, or if you set generate_postal_code to False it will gnerate it
        if random.randint(0, 1) or not generate_postal_code:
            # There is a higher chance here to do something
            if street and random.randint(0, 4):
                # Note that item was defined at the beginning. With a certain probability it generates a typo
                street_name = self.introduce_typo(item['straatNaam'].lower(), max_count=typo_count, always=True)

                address_details['street'] = street_name
                address_details_str.append(street_name)
                # Again with a certain probability a housenumber is included.
                if number and random.randint(0, 4):
                    house_number = item['huisnummer'].lower()
                    address_details['number'] = house_number
                    address_details_str.append(house_number)

                    # If there is an huisletter this happens
                    addition = item['huisletter']
                    # if it is not empty:
                    if addition:
                        addition = addition.lower()
                        address_details['addition'] = addition
                        # with a certain probability it will be a seperate item in the list or added to the number
                        if random.randint(0, 1):
                            address_details_str.append(addition)
                        else:
                            address_details_str[-1] += addition

            # if the adress_details dictionary is not empty
            # we might add a connector to our list that we will use to create a string (certain prob)
            if address_details:
                connector = random.choice(['in', 'te'])
                if random.randint(0, 2):
                    address_details_str.append(connector)

            # We add a city name
            city_name = self.introduce_typo(item['woonplaats'].lower(), max_count=typo_count, always=True)
            address_details['city'] = city_name

            address_details_str.append(city_name)

            # HERE WE GENERATE A POSTAL CODE INSTEAD
        else:
            postal_code = ''.join(random.choice(string.digits) for _ in range(4))
            address_details['postal_code'] = postal_code
            address_details_str.append(postal_code)

            letters = ''.join(random.choice(string.ascii_lowercase) for _ in range(2))
            address_details['postal_code_letters'] = letters
            address_details_str.append(letters)

            if number and random.randint(0, 3):
                address_details['number'] = item['huisnummer'].lower()
                address_details_str.append(item['huisnummer'])

                addition = item['huisletter']
                if addition:
                    address_details['addition'] = addition.lower()
                    if random.randint(0, 1):
                        address_details_str.append(addition)
                    else:
                        address_details_str[-1] += addition

        # we return dictionary and also the joined string
        return address_details, ' '.join(address_details_str).lower()
