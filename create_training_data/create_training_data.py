import os
import json
import random
from Creating_Training_data.functions import Functions

os.environ['CURL_CA_BUNDLE'] = ''
# Either gets environment variable OUTPUT_DIR or otherwise sets this equal to the directory of the current file
OUTPUT_DIR = os.getenv('OUTPUT_DIR', os.path.dirname(os.path.realpath(__file__)))

"""
This file invokes methods from `brt_filter_query.py` to generate 
and save the training data.
"""


class QueryRegistry:
    _encoding = 'utf-8'

    def __init__(self, addresses_name='addresses.json'):
        """
        Initialize the class with the specified addresses file.

        Args:
            addresses_name (str): The name of the file containing addresses. Defaults to 'addresses.json'.

        Attributes:
            weights (list): A list to store weights (initially empty).
            variants (list): A list to store variants (initially empty).
            addresses (list): The list of addresses loaded from the specified file.
        """
        self.weights = []
        self.variants = []
        self.addresses = self.read_addresses(addresses_name)

    @classmethod
    def read_addresses(cls, addresses_name):
        """
        Read and load addresses from a specified file.

        Args:
            addresses_name (str): The name of the file containing addresses.

        Returns:
            list[dict]: A list of dictionaries, where each dictionary represents an address.

        Notes:
            - The file is expected to be in JSON format and located in the `OUTPUT_DIR` directory.
            - The file is read using the class-defined encoding (`cls._encoding`).

        """
        file_name = os.path.join(OUTPUT_DIR, addresses_name)
        with open(file_name, 'r', encoding=cls._encoding) as fh:
            return json.load(fh)

    def register(self, *query_objects_and_template_as_tuple, typos=False):
        """
        Register query variants for training data generation.

        This method is used to register query variants that are used in the training data creation process.
        While our research currently only uses the query variant `brt_filter_query.py`, the codebase
        is designed to be modular, allowing for the creation of additional query-generation files (query variants) with
        different training data. All files that will be used for training data generation need to be registered
        using this method.

        Each registered query variant should be passed as a tuple where:
            - The first element is the query object (used to generate the training data).
            - The second element is the relative path to the `.yml` file containing natural language templates
              corresponding to the query.

        Additionally, each query variant has an associated `weights` property, which specifies the weight
        with which the query variant should be sampled during the generation of training data.

        Args:
            *query_objects_and_template_as_tuple (tuple):
                One or more tuples, each containing:
                - A query object that can generate training data.
                - A relative path to a `.yml` file containing the natural language templates for the query.
            typos (bool, optional):
                A flag indicating whether to include typos in the generated training data. Defaults to `False`.

        Example:
            query_registry.register(
                (BrtFilterQuery, 'path/to/templates/first_template.yml'),
                (AnotherQuery, 'path/to/templates/second_template.yml'),
                typos=True
            )

        """
        self.variants = [query_object[0](template=query_object[1], addresses=self.addresses, typos=typos) for
                         query_object in query_objects_and_template_as_tuple]
        self.weights = [variant.get_weight() for variant in self.variants]

    def get_random_variant(self):
        """
        Select a random query variant based on predefined weights.

        This method returns a randomly chosen query variant from the list of registered
        query variants. The selection process takes into account the associated weights,
        allowing for weighted sampling. The variant with a higher weight has a higher chance
        of being selected.

        Returns:
            object: A random query variant selected from the registered variants.

        Notes:
            - The sampling is performed using the `random.choices` function, which supports
              weighted random selection.
            - If no variants are registered, this function will raise an error.
        """
        return next(iter(random.choices(self.variants, weights=self.weights, k=1)))

    def get_query(self):
        """
        Generates a single training example consisting of a natural language question and its corresponding SPARQL query.

        This method selects a random query variant from the available variants (using the associated weights),
        and then uses the query variant's logic to generate a training example, including a natural language question,
        the corresponding SPARQL query, and additional metadata.


        Returns:
            tuple: A tuple containing:
                - question (str): The generated natural language question.
                - answer (str or None): The corresponding SPARQL query.
                - answer_kwargs (dict or None): A dictionary containing settings or parameters used within the broader
                  query variant from which this question was sampled.
                - golden_classes (list): A list of golden classes used in the SPARQL query.
                - golden_relations (list): A list of golden relations used in the SPARQL query.

        Raises:
            Exception: If no query variants have been registered in `self.variants`.
        """
        questions, answer, answer_kwargs, golden_classes, golden_relations = [], None, None, [], []
        # The `self.variants` list must be initialized, as it is required for generating training data.
        if self.variants:
            dependency_to_value = {}

            query_variant_stack, answer_kwargs_stack = [], []

            query_variant = self.get_random_variant()

            question, answer, answer_kwargs, golden_classes, golden_relations = query_variant.get_question_answer(
                existing_dependencies=dependency_to_value, query_variant_stack=query_variant_stack,
                answer_kwargs_stack=answer_kwargs_stack, related=None,
            )
            questions.append(question)
            query_variant_stack.append(query_variant)
            answer_kwargs_stack.append(answer_kwargs)

        else:
            # Throw an exception if no query variants have been registered.
            raise Exception('No query variant has been registered.')

        question = questions[0]

        return question, answer, answer_kwargs, golden_classes, golden_relations


def get_location_entity_instance_data_description(granularity, location_code_or_preflabel, label_corresponding_to_code):
    """
    Generates a (partial) prompt that describes the location entity used in a SPARQL query,
    based on the corresponding granularity.

    Args:
        granularity (str): The level of granularity of the location. It can be one of the following:
                           - 'gemeente': Refers to a municipality.
                           - 'provincie': Refers to a province.
                           - 'woonplaats': Refers to a city.
                           - 'land': Refers to a country.
        location_code_or_preflabel (str): The location code or prefabel that identifies the location in the query
        label_corresponding_to_code (str): The human-readable label that corresponds to the location code.

    Returns:
        str: A string describing the location entity instance in the context of the SPARQL query.
        The returned string varies based on the granularity specified.
        If the granularity is invalid, the function prints an error message and returns `None`.

    Raises:
        ValueError: If an invalid granularity is provided.
    """
    if granularity == 'gemeente':
        instance_prompt_string = """
        You are provided with a triple of the form: (class relation instancedata)
        You should interpret this as: there are instances of wbk:Gemeente that have this identifier, and this
        identifier corresponds to the gemeente %s . \n
        """ % (label_corresponding_to_code)
        instance_prompt_string = instance_prompt_string + """wbk:Gemeente sdo0:identifier "%s" """ % (
            location_code_or_preflabel)
        return instance_prompt_string

    elif granularity == 'provincie':
        instance_prompt_string = """
        You are provided with a triple of the form: (class relation instancedata)
        You should interpret this as: there are instances of sor:Gemeente that is located in
        an instance of sor:Provincie. The instance of sor:Provincie corresponds to province %s . \n
        """ % (location_code_or_preflabel)
        instance_prompt_string = instance_prompt_string + """sor:Gemeente geo:sfWithin provincie:%s""" % (
            location_code_or_preflabel)
        return instance_prompt_string

    elif granularity == 'woonplaats':
        instance_prompt_string = """
        You are provided with a triple of the form: (class relation instancedata)
        You should interpret this as: there are instances of sor:Woonplaats that are that have the prefLabel
        mentioned in the triple. \n
        """
        instance_prompt_string = instance_prompt_string + """sor:Woonplaats skos:prefLabel "%s"@nl""" % (
            location_code_or_preflabel)
        return instance_prompt_string
    elif granularity == 'land':
        instance_prompt_string = ""
        return instance_prompt_string
    else:
        print('FATAL ERROR')
        print(granularity)
        return None


class Counter:
    """
    A utility class to maintain and ensure balanced counts of generated training data across different granularities.

    This class is designed to track the number of training data samples generated for each granularity level
    ('woonplaats', 'gemeente', 'provincie', 'land').

    Granularities refer to different levels of geographic or administrative divisions used in the training data.
    """

    def __init__(self, target_for_counters):
        """
        Initializes the Counter_new class with a target count for training data samples for each granularity.

        Args:
            target_for_counters (int): The target number of training data samples to be generated for each granularity
            (e.g., 'woonplaats', 'gemeente', 'provincie', 'land') before the generation process terminates.
        """
        # Initialize granularity counters
        self.woonplaats_counter = 0
        self.gemeente_counter = 0
        self.provincie_counter = 0
        self.land_counter = 0

        # target that was passed as input to the constructor is also set as attribute
        self.target_for_counters = target_for_counters

        # booleans which indicate whether we should keep generating a specific query.
        # set to false once target is reached

        self.woonplaats_generate_boolean = True

        self.gemeente_generate_boolean = True
        self.provincie_generate_boolean = True

        self.land_generate_boolean = True

        self.all_counters_reached_target = False

    def get_booleans(self, granularity):
        granularity_boolean = True

        if granularity == 'woonplaats':
            granularity_boolean = self.woonplaats_generate_boolean
        if granularity == 'gemeente':
            granularity_boolean = self.gemeente_generate_boolean
        if granularity == 'provincie':
            granularity_boolean = self.provincie_generate_boolean
        if granularity == 'land':
            granularity_boolean = self.land_generate_boolean

        return granularity_boolean

    def increment(self, granularity):
        if granularity == 'woonplaats':
            self.woonplaats_counter += 1
        if granularity == 'gemeente':
            self.gemeente_counter += 1
        if granularity == 'provincie':
            self.provincie_counter += 1
        if granularity == 'land':
            self.land_counter += 1

        if self.woonplaats_counter >= self.target_for_counters:
            self.woonplaats_generate_boolean = False
        if self.gemeente_counter >= self.target_for_counters:
            self.gemeente_generate_boolean = False
        if self.provincie_counter >= self.target_for_counters:
            self.provincie_generate_boolean = False
        if self.land_counter >= self.target_for_counters:
            self.land_generate_boolean = False

        if all(not getattr(self, attr) for attr in [
            'woonplaats_generate_boolean',
            'gemeente_generate_boolean',
            'provincie_generate_boolean',
            'land_generate_boolean'
        ]):
            self.all_counters_reached_target = True


def create_main_training_data_for_production(target_amount_of_questions_for_each_granularity=6000):
    """
    Creates the main training dataset used in our research, saving it in JSON format.

    Args:
        target_amount_of_questions_for_each_granularity (int, optional): The number of training samples
            to generate for each granularity (e.g., 'woonplaats', 'gemeente', 'provincie', 'land').
            Default is 6000.

    Returns:
        None

    Output:
        The training data is saved at:
        "Training_data/main_training_data/training_data_brt_filter.json"
    """
    # We register the query variant that we use in our research.
    from Creating_Training_data.brt_filter_query import BrtFilterQuery
    query_registry = QueryRegistry()
    query_registry.register((BrtFilterQuery, r'Production_templates\brt_filter_query'))

    # Generate the specified number of training data examples
    qid_counter = 0
    list_with_questions = []
    counter = Counter(target_for_counters=target_amount_of_questions_for_each_granularity)
    while not counter.all_counters_reached_target:
        x, y, answer_kwargs, golden_classes, golden_relations = query_registry.get_query()

        granularity_key = answer_kwargs['granularity_key']

        # Assume the meta-data represents "gebouwen" (buildings) if the property key is not present
        if answer_kwargs['property_key'] is None:
            property_key = 'gebouwen'
        else:
            property_key = answer_kwargs['property_key']

        granularity_boolean = counter.get_booleans(granularity=granularity_key)
        if granularity_key not in ['straat', 'wijk', 'buurt']:
            # Include the IRI of the generated location in the SPARQL query.
            y_adjusted, location_code_or_preflabel, label_corresponding_to_code = Functions.apply_functions(y)
            print([counter.provincie_counter,
                   counter.gemeente_counter,
                   counter.land_counter,
                   counter.woonplaats_counter])
            # If the generated query is valid we include it.
            if y_adjusted != "Wat is het adres waar u in geïnteresseerd bent?":
                if granularity_boolean:
                    instance_prompt_string = get_location_entity_instance_data_description(granularity_key,
                                                                                           location_code_or_preflabel,
                                                                                           label_corresponding_to_code)
                    qid_counter += 1
                    question_dict = {
                        'qid': qid_counter,
                        'question': x,
                        'sparql_query': y_adjusted,
                        'golden_classes': golden_classes,
                        'golden_relations': golden_relations,
                        'property_key': property_key,
                        'granularity_key': granularity_key,
                        'info_dictionary': answer_kwargs,
                        'instance_prompt_string': instance_prompt_string
                    }
                    list_with_questions.append(question_dict)
                    counter.increment(granularity=granularity_key)

    print(qid_counter)
    # Save the training data to a JSON file.
    parent_directory = os.path.dirname(__file__)
    parent_of_parent_directory = os.path.dirname(parent_directory)

    json_filename = os.path.join(parent_of_parent_directory,
                                 r'Training_data\Main_Training_data\training_data_brt_filter.json')

    with open(json_filename, 'w') as json_file:
        json.dump(list_with_questions, json_file, indent=2)


if __name__ == '__main__':
    create_main_training_data_for_production()
