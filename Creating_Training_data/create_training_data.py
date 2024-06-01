'''
This is the file from which we are able to create trainingdata.
'''
import os
import json
import random

from Creating_Training_data.query_components import SpecialTokens
from Creating_Training_data.functions import Functions

import requests

'''This following code will set the CURL_CA_BUNDLE environment variable to an empty string in the Python os module'''

import os

os.environ['CURL_CA_BUNDLE'] = ''

# Either gets environment variable OUTPUT_DIR or otherwise sets this equal to the directory of the current file
OUTPUT_DIR = os.getenv('OUTPUT_DIR', os.path.dirname(os.path.realpath(__file__)))


class QueryRegistry:
    _encoding = 'utf-8'

    def __init__(self, addresses_name='addresses.json'):
        '''
        constructor

        :param addresses_name: this should be the name of a file containing adresses
        '''
        self.weights = []
        self.variants = []
        self.addresses = self.read_addresses(addresses_name)

    @classmethod
    def read_addresses(cls, addresses_name):
        '''
        :param addresses_name: name of the file containing adresses
        :return: returns a list with dictionaries. The dictionaries are adresses.
        '''
        file_name = os.path.join(OUTPUT_DIR, addresses_name)
        with open(file_name, 'r', encoding=cls._encoding) as fh:
            return json.load(fh)

    def register(self, *query_objects_and_template_as_tuple, typos=False):
        '''
        Takes the list of query objects past as parameter and uses their constructor to create instances and saves
        them to a list which is hold in an attribute of this class.

        We also store the (sampling) weights for the specific type of query

        :param query_objects: these are the query objects that represents different sets of queries on the
        kadaster knowledge graph.
        Note we only work with the largest and most important one (BrtFilterQuery)
        :param typos:
        :return:
        '''
        self.variants = [query_object[0](template=query_object[1], addresses=self.addresses, typos=typos) for
                         query_object in query_objects_and_template_as_tuple]
        self.weights = [variant.get_weight() for variant in self.variants]

    def get_random_variant(self):
        '''

        :return: returns a random query variant from the registered query variants.
        Note the sampling is done with the provided weights.
        '''
        return next(iter(random.choices(self.variants, weights=self.weights, k=1)))

    def get_query(self, max_queries=1, suggestions=False, return_variants=False):
        '''
        This is the main function that is used to generate a piece of trainingdata.

        :param max_queries: DISABLED FOR THE PURPOSE OF THIS DEMONSTRATION.
        This code provides the possibility to train a model based on entire conversations histories.
        This can be helpful if we want to allow for the possibility that a user of the chatbot perhaps asks a question
        which doesn't directly include a location. For example: "How many hospitals are there in my province". Training
        the model on conversations would allow a machine learning model to extract the last mentioned location from
        the conversation history. I set this parameter to a default value of 1,
        and therefore disabled the creation of conversation histories for now.

        :param suggestions: DISABLED FOR THE PURPOSE OF THIS DEMONSTRATION.
        Within the deployment of the chatbot there is the possibility to give buttons with
        suggestions after the chatbot has succesfully answered a question. So for example suppose a user asks
        "How many churches are there in Rotterdam?" and the chatbot succesfully asnwers this question. The chatbot
        could provide suggestion button such as "hospitals", "universities", etc. These buttons allow a user to ask the
        previous question with a slight modification. The parameter suggestions allows for such interactions in the
        conversation history. I disabled this feature for now.

        :param return_variants: NOT IMPORTANT FOR THE PURPOSE OF THIS DEMONSTRATION
        This would make the function also return a list with all the query variants (in
        chronological order) that were asked in the conversation history.

        :return:
        question: for the purpose of this demonstration this is simply the question in natural language as a string.
        if we would work with conversation histories it would return the entire history of natural language questions
        concatenated as a string with seperator tokens in between each natural language question.

        answer: a string containing the SPARQL query that belongs to the question in natural language.

        answer_kwargs: a dictionary containing the settings used for this specific query within the broader
        query variant it was sampled from.

        golden_classes: the classes from the ontology that are used in this specific SPARQL query returned in a list.
        Note that this will later be helpful for creating dense retriever neural networks to identify parts of the
        ontology needed for a natural language question.

        golden_relations: the relations from the ontology that are used in this specific SPARQL query returned in a
        list. Note that this will later be helpful for creating dense retriever neural networks to
        identify parts of the ontology needed for a natural language question.

        '''
        questions, answer, answer_kwargs, golden_classes, golden_relations = [], None, None, [], []
        # self.variants must be intialized
        if self.variants:
            dependency_to_value = {}

            query_variant_stack, answer_kwargs_stack = [], []

            # For a number of iterations between 1 & max_queries we do the following
            # Note that in main we seem to do this with max_queries = 1..
            for _ in range(random.randint(1, max_queries)):
                # We select some random query variant
                query_variant = self.get_random_variant()

                '''
                Note that in all the iterations , the variable 'related' is actually None.
                In this case, if we for example look at brt_filter_query.py we see that when 'related' is None
                with some probability:
                -we generate some BRT_PROPERTY and some SYNONYM for it.
                -we generate some location
                -we generate some filter option & filter synonym
                -we generate some recent_key and recent synonym

                Each iterations our lists and stacks do get updated though. But every time related is None
                so we always do the above generating new things.

                we use all this to generate some question and answer. We return the question, answer, and answer_kwargs

                We update our lists and stacks.
                '''
                question, answer, answer_kwargs, golden_classes, golden_relations = query_variant.get_question_answer(
                    existing_dependencies=dependency_to_value, query_variant_stack=query_variant_stack,
                    answer_kwargs_stack=answer_kwargs_stack, related=None,
                )
                questions.append(question)
                query_variant_stack.append(query_variant)
                answer_kwargs_stack.append(answer_kwargs)

                if suggestions and random.randint(0, 100) > 75:
                    '''
                    we explain the get_related() function here:

                    related is basically one of the buttons that can be pressed in Loki.

                    If adress_data is in answer_kwargs this function will return something (otherwise None)

                    If adress_data is in answer_kwargs, we next check if own_adress is in answer_kwargs:
                    1) if own_adress is in answer_kwargs
                    we randomly select a granularity_key (straat, buurt, wijk, woonplaats, gemeente, provincie, land)
                    And returns it in a tuple
                    1st element string 'granularity_key'
                    2nd element the randomly selected granularity_key
                    Note that the selected granularity_key is NOT equal to the granularity_key in the answer_kwargs

                    2)if own_adress is NOT in answer_kwargs
                    25% chance we return
                    ('amount_key', 'aantal')
                    75% chance we return
                    ('property_key', some randomly selected property not equal to what was in answer_kwargs)
                    '''
                    related = query_variant.get_related(answer_kwargs)

                    if related:
                        question, answer, answer_kwargs, golden_classes, golden_relations = query_variant.get_question_answer(
                            existing_dependencies=dependency_to_value, query_variant_stack=query_variant_stack,
                            answer_kwargs_stack=answer_kwargs_stack, related=related,
                        )

                        _, question = related
                        questions.append(question)
                        query_variant_stack.append(query_variant)
                        answer_kwargs_stack.append(answer_kwargs)
        else:
            raise Exception('No query variant has been registered.')

        if questions[0] is not None:
            question = f' {SpecialTokens.INTERACTION.value} '.join(questions[::-1])
        else:
            question = questions[0]

        if return_variants:
            return question, answer, query_variant_stack
        else:
            return question, answer, answer_kwargs, golden_classes, golden_relations

    def get_related(self, max_queries=15, return_variants=False):
        '''
        This function can be ignored for now. It is only needed when we enable the feature of the suggestion buttons
        in the chatbot.

        :param max_queries:
        :param return_variants:
        :return:
        '''
        answer = None
        questions = []
        if self.variants:
            dependency_to_value = {}

            query_variant_stack, answer_kwargs_stack = [], []
            for _ in range(random.randint(1, max_queries)):
                query_variant = self.get_random_variant()

                question, _, answer_kwargs = query_variant.get_question_answer(
                    existing_dependencies=dependency_to_value, query_variant_stack=query_variant_stack,
                    answer_kwargs_stack=answer_kwargs_stack, related=None,
                )
                answer = query_variant.get_related_options(
                    answer_kwargs=answer_kwargs, query_variant_stack=query_variant_stack
                )

                questions.append(question)
                query_variant_stack.append(query_variant)
                answer_kwargs_stack.append(answer_kwargs)

                if random.randint(0, 100) > 75:
                    related = query_variant.get_related(answer_kwargs)

                    if related:
                        _, _, answer_kwargs = query_variant.get_question_answer(
                            existing_dependencies=dependency_to_value, query_variant_stack=query_variant_stack,
                            answer_kwargs_stack=answer_kwargs_stack, related=related,
                        )
                        _, question = related
                        answer = query_variant.get_related_options(
                            answer_kwargs=answer_kwargs, query_variant_stack=query_variant_stack
                        )

                        questions.append(question)
                        query_variant_stack.append(query_variant)
                        answer_kwargs_stack.append(answer_kwargs)
        else:
            raise Exception('No query variant has been registered.')

        question = f'{SpecialTokens.RELATED.value} ' + f' {SpecialTokens.INTERACTION.value} '.join(questions[::-1])

        if answer is None:
            answer = ''

        if return_variants:
            return question, answer, query_variant_stack
        else:
            return question, answer


def get_location_entity_instance_data_description(granularity, location_code_or_preflabel, label_corresponding_to_code):
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


class Counter_new:
    '''
    I create a simple class which keeps track of what kind of queries we have generated.
    This is used during generation to obtain enough (and relatively balanced) trainingsdata
     of each kind (based on granularities and types of physical geographical properties).
    '''

    def __init__(self, target_for_counters, development_train_set=False):
        '''
        Constructor for the class.
        :param target_for_counters: target for each of the counters that need be generated before terminating
        :param development_train_set: a boolean which indicates whether we are using the counter for the training set
        that is used in the train-dev split for hyperparameter tuning and epoch determination.
        This is important since in this trainingset I changed the templates such that no queries will be generated
        for monumenten, gebouwen, wijk, buurt, gemeente, and provincie.
        '''
        # Initialize granularity counters
        self.woonplaats_counter = 0
        self.gemeente_counter = 0
        self.provincie_counter = 0
        self.land_counter = 0

        # we set the targets for each of them here

        # target that was passed as input to the constructor is also set as attribute
        self.target_for_counters = target_for_counters

        # booleans which indicate whether we should keep generating a specific query.
        # set to false once target is reached

        self.woonplaats_generate_boolean = True
        if development_train_set:
            self.gemeente_generate_boolean = False
            self.provincie_generate_boolean = False

        else:
            self.gemeente_generate_boolean = True
            self.provincie_generate_boolean = True

        self.land_generate_boolean = True

        self.all_counters_reached_target = False

    def get_booleans(self, granularity, property):
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

    def increment(self, granularity, property):
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


class Counter:
    '''
    I create a simple class which keeps track of what kind of queries we have generated.
    This is used during generation to obtain enough (and relatively balanced) trainingsdata
     of each kind (based on granularities and types of physical geographical properties).
    '''

    def __init__(self, target_for_counters, development_train_set=False):
        '''
        Constructor for the class.
        :param target_for_counters: target for each of the counters that need be generated before terminating
        :param development_train_set: a boolean which indicates whether we are using the counter for the training set
        that is used in the train-dev split for hyperparameter tuning and epoch determination.
        This is important since in this trainingset I changed the templates such that no queries will be generated
        for monumenten, gebouwen, wijk, buurt, gemeente, and provincie.
        '''
        # Initialize granularity counters
        self.woonplaats_counter = 0
        self.gemeente_counter = 0
        self.provincie_counter = 0
        self.straat_counter = 0
        self.land_counter = 0

        # initialize property counters
        self.percelen_counter = 0
        self.monumenten_counter = 0
        self.huizen_counter = 0
        self.verblijfsobjecten_counter = 0
        self.gebouwen_counter = 0

        # we set the targets for each of them here

        # target that was passed as input to the constructor is also set as attribute
        self.target_for_counters = target_for_counters

        # booleans which indicate whether we should keep generating a specific query.
        # set to false once target is reached

        self.woonplaats_generate_boolean = True
        if development_train_set:
            self.gemeente_generate_boolean = False
            self.provincie_generate_boolean = False
            self.monumenten_generate_boolean = False
            self.gebouwen_generate_boolean = False
        else:
            self.gemeente_generate_boolean = True
            self.provincie_generate_boolean = True
            self.monumenten_generate_boolean = True
            self.gebouwen_generate_boolean = True
        self.straat_generate_boolean = True
        self.land_generate_boolean = True
        self.percelen_generate_boolean = True
        self.huizen_generate_boolean = True
        self.verblijfsobjecten_generate_boolean = True

        self.all_counters_reached_target = False

    def get_booleans(self, granularity, property):
        granularity_boolean = True
        property_boolean = True

        if granularity == 'woonplaats':
            granularity_boolean = self.woonplaats_generate_boolean
        if granularity == 'gemeente':
            granularity_boolean = self.gemeente_generate_boolean
        if granularity == 'provincie':
            granularity_boolean = self.provincie_generate_boolean
        if granularity == 'straat':
            granularity_boolean = self.straat_generate_boolean
        if granularity == 'land':
            granularity_boolean = self.land_generate_boolean

        if property == 'percelen':
            property_boolean = self.percelen_generate_boolean
        if property == 'monumenten':
            property_boolean = self.monumenten_generate_boolean
        if property == 'huizen':
            property_boolean = self.huizen_generate_boolean
        if property == 'verblijfsobjecten':
            property_boolean = self.verblijfsobjecten_generate_boolean
        if property == 'gebouwen':
            property_boolean = self.gebouwen_generate_boolean

        return granularity_boolean, property_boolean

    def increment(self, granularity, property):
        if granularity == 'woonplaats':
            self.woonplaats_counter += 1
        if granularity == 'gemeente':
            self.gemeente_counter += 1
        if granularity == 'provincie':
            self.provincie_counter += 1
        if granularity == 'straat':
            self.straat_counter += 1
        if granularity == 'land':
            self.land_counter += 1

        if property == 'percelen':
            self.percelen_counter += 1
        if property == 'monumenten':
            self.monumenten_counter += 1
        if property == 'huizen':
            self.huizen_counter += 1
        if property == 'verblijfsobjecten':
            self.verblijfsobjecten_counter += 1
        if property == 'gebouwen':
            self.gebouwen_counter += 1

        if self.woonplaats_counter >= self.target_for_counters:
            self.woonplaats_generate_boolean = False
        if self.gemeente_counter >= self.target_for_counters:
            self.gemeente_generate_boolean = False
        if self.provincie_counter >= self.target_for_counters:
            self.provincie_generate_boolean = False
        if self.straat_counter >= self.target_for_counters:
            self.straat_generate_boolean = False
        if self.land_counter >= self.target_for_counters:
            self.land_generate_boolean = False
        if self.percelen_counter >= self.target_for_counters:
            self.percelen_generate_boolean = False
        if self.monumenten_counter >= self.target_for_counters:
            self.monumenten_generate_boolean = False
        if self.huizen_counter >= self.target_for_counters:
            self.huizen_generate_boolean = False
        if self.verblijfsobjecten_counter >= self.target_for_counters:
            self.verblijfsobjecten_generate_boolean = False
        if self.gebouwen_counter >= self.target_for_counters:
            self.gebouwen_generate_boolean = False

        if all(not getattr(self, attr) for attr in [
            'woonplaats_generate_boolean',
            'gemeente_generate_boolean',
            'provincie_generate_boolean',
            'straat_generate_boolean',
            'land_generate_boolean',
            'percelen_generate_boolean',
            'monumenten_generate_boolean',
            'huizen_generate_boolean',
            'verblijfsobjecten_generate_boolean',
            'gebouwen_generate_boolean'
        ]):
            self.all_counters_reached_target = True

    def print_counters(self):
        print("Woonplaats Counter:", self.woonplaats_counter)
        print("Gemeente Counter:", self.gemeente_counter)
        print("Provincie Counter:", self.provincie_counter)
        print("Straat Counter:", self.straat_counter)
        print("Land Counter:", self.land_counter)
        print("Perceel Counter:", self.percelen_counter)
        print("Monument Counter:", self.monumenten_counter)
        print("Huis Counter:", self.huizen_counter)
        print("Verblijfsobject Counter:", self.verblijfsobjecten_counter)
        print("Gebouw Counter:", self.gebouwen_counter)


def create_main_training_data_for_production():
    '''
    Note this function creates the MAIN trainingdata which we will use to train the neural networks in production.
    This includes all templates within brt_filter.

    :return:
    '''
    # Note these are simply the different type of queries (BRT_filter etc... )
    # it seems the names refer to the python files themselves.
    # from data.queries import queries
    # We create a QueryRegistry object
    # query_registry = QueryRegistry()
    # we construct instances of all different queries (and input adresses with typos) and put
    # the list in self.variants and we also put the weights in self.weights
    # query_registry.register(*queries)

    from Creating_Training_data.brt_filter_query import BrtFilterQuery

    query_registry = QueryRegistry()
    query_registry.register((BrtFilterQuery, r'Production_templates\brt_filter_query'))

    # we create a question id counter
    qid_counter = 0
    list_with_questions = []
    counter = Counter_new(target_for_counters=6000)
    question_id_counter = 0
    while not counter.all_counters_reached_target:
        # Note that max_queries=1 in the default code i took from github!
        x, y, answer_kwargs, golden_classes, golden_relations = query_registry.get_query(max_queries=1,

                                                                                         suggestions=False)

        # Here we take the query and format the relevant location (code) into the SPARQL query

        granularity_key = answer_kwargs['granularity_key']

        if answer_kwargs['property_key'] is None:
            property_key = 'gebouwen'
        else:
            property_key = answer_kwargs['property_key']

        granularity_boolean = counter.get_booleans(granularity=granularity_key,
                                                   property=property_key)
        if granularity_key not in ['straat', 'wijk', 'buurt']:
            y_adjusted, location_code_or_preflabel, label_corresponding_to_code = Functions.apply_functions(y)
            print([counter.provincie_counter,
            counter.gemeente_counter,
            counter.land_counter,
            counter.woonplaats_counter])
            if y_adjusted != "Wat is het adres waar u in geïnteresseerd bent?":
                if granularity_boolean:
                    # We apply a function which formats the part of the prompt describing the location entity instance data for the query
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
                    counter.increment(granularity=granularity_key, property=property_key)

        # counter.print_counters()
    print(qid_counter)
    # Now we write our list with questions (where the questions are in dictionaries)
    # to a json file in the main folder
    parent_directory = os.path.dirname(__file__)
    parent_of_parent_directory = os.path.dirname(parent_directory)

    # Specify the filename for the JSON file in the parent directory
    json_filename = os.path.join(parent_of_parent_directory,
                                 r'Training_data\Main_Training_data\training_data_brt_filter.json')

    # Write the list of dictionaries to the JSON file
    with open(json_filename, 'w') as json_file:
        json.dump(list_with_questions, json_file, indent=2)


def create_training_and_development_data_for_dense_retriever():
    '''
    Note this function uses 3 yml files that I manually adjusted from the original brt_filter templates set.
    One will be used for creating training data for the dense retriever
    The other two will be used for validating the dense retriever (development set)

    I created a new training yml file by removing:
    -removing certain templates from the original main trainingdata
    -excludeorder restrictions,
    -excludegranularity restrictions,
    -and excludeproperty restrictions,
    By doing this I made sure NONE of the trainingdata contains questions containing
    sor:Gebouw class and sor:oorspronkelijkBouwjaar property.
    This way i can include such questions in the development set, making sure my development set is
    somewhat representative of the production data
    where we will use it to retrieve schema items not in the trainingdata.
    By doing this we decrease overfitting for better results.

    For the development set, i created two yml file.
    The first yml contains templates without the restrictions applied on the training set described above.
    The second yml template only contains templates that definitely use
    sor:Gebouw class and sor:oorspronkelijkBouwjaar property.
    Then when creating the development set i randomly sample a template from one of the files.
    This way i make sure AT LEAST 50% on the trainingdata definitely contains the schema items that are not in
    the training set.


    :return:
    '''

    '''

    FIRST WE CREATE THE TRAINING DATA FOR THE DENSE RETRIEVER

    '''

    from Creating_Training_data.brt_filter_query import BrtFilterQuery

    query_registry = QueryRegistry()
    query_registry.register(
        (BrtFilterQuery, r'train_dev_split_for_hyperparameter_tuning_templates\brt_filter_query_train'))

    # we create a question id counter
    qid_counter = 0
    list_with_questions = []
    counter = Counter(target_for_counters=50, development_train_set=True)
    question_id_counter = 0
    while not counter.all_counters_reached_target:
        # Note that max_queries=1 in the default code i took from github!
        x, y, answer_kwargs, golden_classes, golden_relations = query_registry.get_query(max_queries=1,
                                                                                         suggestions=False)
        # Here we take the query and format the relevant location (code) into the SPARQL query
        y_adjusted, location_code_or_preflabel, label_corresponding_to_code = Functions.apply_functions(y)
        granularity_key = answer_kwargs['granularity_key']
        if answer_kwargs['property_key'] is None:
            property_key = 'gebouwen'
        else:
            property_key = answer_kwargs['property_key']

        granularity_boolean = counter.get_booleans(granularity=granularity_key,
                                                   property=property_key)
        if granularity_key != 'straat':
            if granularity_boolean:
                if y_adjusted != "Wat is het adres waar u in geïnteresseerd bent?":
                    qid_counter += 1
                    question_dict = {
                        'qid': qid_counter,
                        'question': x,
                        'sparql_query': y,
                        'golden_classes': golden_classes,
                        'golden_relations': golden_relations,
                        'property_key': property_key,
                        'granularity_key': granularity_key,
                        'info_dictionary': answer_kwargs
                    }
                    list_with_questions.append(question_dict)
                    counter.increment(granularity=granularity_key, property=property_key)

        # counter.print_counters()
    print('train for train/dev split')
    print(qid_counter)
    # Now we write our list with questions (where the questions are in dictionaries)
    # to a json file in the main folder
    parent_directory = os.path.dirname(__file__)
    parent_of_parent_directory = os.path.dirname(parent_directory)

    # Specify the filename for the JSON file in the parent directory
    json_filename = os.path.join(parent_of_parent_directory,
                                 r'Training_data\Train_dev_split_for_hyperparameter_tuning\train_data_brt_filter.json')

    # Write the list of dictionaries to the JSON file
    with open(json_filename, 'w') as json_file:
        json.dump(list_with_questions, json_file, indent=2)

    '''
    
    NOW WE CREATE THE DEVELOPMENT DATA FOR THE DENSE RETRIEVER
    
    '''

    from Creating_Training_data.brt_filter_query import BrtFilterQuery

    query_registry = QueryRegistry()
    query_registry.register(
        (BrtFilterQuery, r'train_dev_split_for_hyperparameter_tuning_templates\brt_filter_query_dev')
        , (BrtFilterQuery, r'train_dev_split_for_hyperparameter_tuning_templates\brt_filter_query_dev_BUILDINGS'))

    # we create a question id counter
    qid_counter = 0
    list_with_questions = []
    counter = Counter(target_for_counters=10)
    question_id_counter = 0
    while not counter.all_counters_reached_target:
        # Note that max_queries=1 in the default code i took from github!
        x, y, answer_kwargs, golden_classes, golden_relations = query_registry.get_query(max_queries=1,
                                                                                         suggestions=False)

        granularity_key = answer_kwargs['granularity_key']
        if answer_kwargs['property_key'] is None:
            property_key = 'gebouwen'
        else:
            property_key = answer_kwargs['property_key']

        granularity_boolean, property_boolean = counter.get_booleans(granularity=granularity_key,
                                                                     property=property_key)

        if granularity_boolean or property_boolean:
            qid_counter += 1
            question_dict = {
                'qid': qid_counter,
                'question': x,
                'sparql_query': y,
                'golden_classes': golden_classes,
                'golden_relations': golden_relations,
                'property_key': property_key,
                'granularity_key': granularity_key,
                'info_dictionary': answer_kwargs
            }
            list_with_questions.append(question_dict)
            counter.increment(granularity=granularity_key, property=property_key)

        # counter.print_counters()

    print('dev for train/dev split')
    print(qid_counter)
    # Now we write our list with questions (where the questions are in dictionaries)
    # to a json file in the main folder
    parent_directory = os.path.dirname(__file__)
    parent_of_parent_directory = os.path.dirname(parent_directory)

    # Specify the filename for the JSON file in the parent directory
    json_filename = os.path.join(parent_of_parent_directory,
                                 r'Training_data\Train_dev_split_for_hyperparameter_tuning\dev_data_brt_filter.json')

    # Write the list of dictionaries to the JSON file
    with open(json_filename, 'w') as json_file:
        json.dump(list_with_questions, json_file, indent=2)


if __name__ == '__main__':
    create_main_training_data_for_production()
    # create_training_and_development_data_for_dense_retriever()
