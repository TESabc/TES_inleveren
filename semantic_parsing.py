'''
SELECT THE GPT DEPLOYMENT YOU WANT TO USE

options: OPENAI_GPT_3.5_TURBO     OPENAI_GPT_4_32K    OPENAI_GPT_4_TURBO
'''
from langchain_core._api import LangChainDeprecationWarning

gpt_option = 'OPENAI_GPT_4_32K'

import warnings

# To ignore specific warning categories
warnings.filterwarnings("ignore")

from langchain_openai import AzureOpenAIEmbeddings
import ast
import asyncio
from langchain_community.vectorstores import FAISS
# from langchain.vectorstores import FAISS
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
import configparser
import requests
from typing import List, Dict, Any
import os
import json
from langchain_openai import AzureChatOpenAI
# from langchain_community.chat_models import AzureChatOpenAI
# from langchain.chat_models import AzureChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langchain.prompts.example_selector import (
    MaxMarginalRelevanceExampleSelector,
    SemanticSimilarityExampleSelector,
)
import pandas as pd
import openai
from concurrent.futures import ThreadPoolExecutor
from shortest_path_algorithms.shortest_path import network_and_ontology_store

'''
Here i set up access to the SPARQL endpoint of the KKG.
I also set up the keys needed for access to Azure hosted OpenAI models.
I also define functions that help us get results.
'''
config = configparser.ConfigParser()
# Put the absolute path to the .ini file here:
current_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(current_dir, 'secrets.ini')
config.read(path)

locatie_url = config['KADASTER']['LOC_SEARCH_API']
lookup_url = config['KADASTER']['LOC_LOOKUP_API']
sparql_url = config['KADASTER']['KKG_API']
os.environ['AZURE_OPENAI_API_KEY'] = config[gpt_option]['AZURE_API_KEY']
# os.environ['AZURE_API_TYPE'] = config[gpt_option]['OPENAI_API_TYPE']
os.environ['OPENAI_API_VERSION'] = config[gpt_option]['AZURE_API_VERSION']
os.environ['AZURE_OPENAI_ENDPOINT'] = config[gpt_option]['AZURE_API_BASE']


def get_nested_value(o: dict, path: list) -> any:
    """
    This function allows us to traverse a nested dictionary according to some specified path to retrieve data
    """
    current = o
    for key in path:
        try:
            current = current[key]
        except:
            return None
    return current


def run_sparql(query: str, url=sparql_url,
               user_agent_header: str = None, internal=False) -> List[Dict[str, Any]]:
    """
    This function does API calls.
    """
    headers = {
        'Accept': 'application/sparql-results+json',
        'Content-Type': 'application/sparql-query'
    }
    if user_agent_header is not None:
        headers['User-Agent'] = user_agent_header

    # print(query)
    response = requests.get(url, headers=headers, params={'query': query.replace("```", ""), 'format': 'json'})
    # print(response.status_code)

    if response.status_code != 200 and internal == False:
        return "That query failed. Perhaps you could try a different one?"
    else:
        if response.status_code != 200 and internal == True:
            return None

    # print(response.json())
    results = get_nested_value(response.json(), ['results', 'bindings'])

    return results


def get_prompt_for_selecting_relevant_schema_items_based_on_natural_language_question(natural_language_question,
                                                                                      ):
    '''
    This function returns the prompts needed for schema retrieval.
    :param natural_language_question: the natural language questions for which the schema items must be retrieved
    :return:
    '''
    prompt_string_template_without_description_classes = """
    You will be given all the %s from an ontology belonging to a knowledge graph along with some entities. You will also be provided with a
    question in natural language. 


    Your job is to select the top %s and entities that you expect to be needed to write a SPARQL
    query corresponding to the natural language question.

    ONLY SELECT %s AND ENTITIES THAT ARE IN THE LIST I PROVIDE TO YOU.
    BE VERY CAREFUL TO NOT MAKE UP PREFIXES! I GIVE YOU THE ONLY ALLOWED PREFIXES

    ALWAYS RETRIEVE THE 'sor:' CLASSES AT LEAST!
    ALSO RETRIEVE SCHEMA ITEMS WITH OTHER PREFIXES, BUT DO NOT LET ME SEE YOU ONLY RETRIEVE bag: SCHEMA ITEM
    WHILE NOT SELECTING THE SAME sor: SCHEMA ITEM!
    THEREFORE, FOR EXAMPLE, IF YOU SELECT bag:Woonplaats ALWAYS ALSO SELECT sor:Woonplaats.
    BUT ONLY IF THE sor: VERSION IS IN THE LIST I SUPPLIED TO YOU!

    Give your answer in a python list format as following:
    ['schema_item1','schema_item2', 'schema_item3']
    Make sure to enclose the items within the list with quotation marks as i demonstrated.
    Do not include anything else in your answer.



    You are given some examples.


    %s AND ENTITIES:
    %s
    --------------------------
    EXAMPLES:
    %s
    ---------------------------
    QUESTION IN NATURAL LANGUAGE:
    %s
    """


    prompt_string_template_without_description_properties = """
    You will be given all the %s from an ontology belonging to a knowledge graph. You will also be provided with a
    question in natural language. 


    Your job is to select the top %s that you expect to be needed to write a SPARQL
    query corresponding to the natural language question.

    ONLY SELECT %s THAT ARE IN THE LIST I PROVIDE TO YOU.
    BE VERY CAREFUL TO NOT MAKE UP PREFIXES! I GIVE YOU THE ONLY ALLOWED PREFIXES

    ALWAYS RETRIEVE THE 'sor:' RELATIONS AT LEAST!
    ALSO RETRIEVE SCHEMA ITEMS WITH OTHER PREFIXES, BUT DO NOT LET ME SEE YOU ONLY RETRIEVE bag: SCHEMA ITEM
    WHILE NOT SELECTING THE SAME sor: SCHEMA ITEM!
    BUT ONLY IF THE sor: VERSION IS IN THE LIST I SUPPLIED TO YOU!

    Give your answer in a python list format as following:
    ['schema_item1','schema_item2', 'schema_item3']
    Make sure to enclose the items within the list with quotation marks as i demonstrated.
    Do not include anything else in your answer.



    You are given some examples.


    %s:
    %s
    --------------------------
    EXAMPLES:
    %s
    ---------------------------
    QUESTION IN NATURAL LANGUAGE:
    %s
    """
    examples_class = """
    QUESTION IN NATURAL LANGUAGE: Wat is het gemiddelde bouwjaar van gebouwen in provincie Friesland?
    ANSWER: ['sor:Gebouw', 'sor:Provincie', 'bag:Provincie']

    QUESTION IN NATURAL LANGUAGE: Hoeveel verblijfsobjecten zijn er in Nederland?
    ANSWER: ['sor:Verblijfsobject', 'bag:Verblijfsobject']
    
    QUESTION IN NATURAL LANGUAGE: Hoeveel kerken zijn er in Groningen?
    ANSWER: ['sor:Woonplaats', 'bag:Woonplaats', 'kad-con:kerk', 'sor:Gebouw']
    
    QUESTION IN NATURAL LANGUAGE: Hoeveel brandweerkazernes zijn er in Groningen?
    ['sor:Woonplaats', 'bag:Woonplaats', 'kad-con:brandweerkazerne', 'sor:Gebouw']
    """

    examples_relation = """
    QUESTION IN NATURAL LANGUAGE: Wat is het gemiddelde bouwjaar van gebouwen in provincie Friesland?
    ANSWER: ['sor:oorspronkelijkBouwjaar']
    """
    network_ontology_store = network_and_ontology_store(5)

    class_query = prompt_string_template_without_description_classes % (
        'classes',  'classes', 'classes', 'CLASSES',
        network_ontology_store.classes_without_comments_string, examples_class,
        natural_language_question)
    relation_query = prompt_string_template_without_description_properties % (
        'relations',  'relations', 'relations', 'RELATIONS',
        network_ontology_store.relations_without_comments_string, examples_relation,
        natural_language_question)

    '''
    The following code will allow us to use concurrent programming to send two API calls to
    GPT models at the same time (This allows us to get the results roughly twice as fast).
    '''

    class_messages = [
        HumanMessage(
            content=class_query
        )
    ]

    relation_messages = [
        HumanMessage(
            content=relation_query
        )
    ]

    # def ai_response(message):
    #     response = llm(message)
    #     return response.content
    # result_list = []
    # with ThreadPoolExecutor() as executor:
    #     results = executor.map(ai_response, args_list)
    #     for result in results:
    #         result_list.append(result)
    #
    # result_list_string_converted_to_list = [ast.literal_eval(list_as_string) for list_as_string in result_list ]
    # print('Relevant classes:')
    # print(result_list_string_converted_to_list[0])
    # print('relevant relations:')
    # print(result_list_string_converted_to_list[1])
    # print()
    return class_messages, relation_messages


def concurrent_executor(messages_list, llm):
    '''
    This function allows us to use concurrent progamming to send multiple API calls to OpenAI at the same time.
    We wil us it to send 3 starting API calls at the beginning of our method.
    :return: a tuple containing all 3 LLM responses
    '''

    def ai_response(message):
        response = llm(message)
        return response.content

    result_list = []
    with ThreadPoolExecutor() as executor:
        results = executor.map(ai_response, messages_list)
        for result in results:
            result_list.append(result)
    return tuple(result_list)


class location_and_granularity_object:
    '''
    An object which stores some functions which allows us to work with location instance data in the knowledge graph.

    One function gives us a prompt to:
    1) Extract a specific location
    2) Determine the granularity of said location (Woonplaats, Gemeente, Nederland, etc.)
    3) Retrieve a modified version of the natural language question where the location has been removed.

    Another function:
    Helps us to use the determined granularity, to select the correct suggest_url for the location server
    (which helps us do a targeted search on the granularity)

    Another function:
    Helps us to retrieve the location identifiers from a location server based on the suggest_url


    '''

    def __init__(self):
        pass

    @staticmethod
    def get_prompt_for_location_aggregation_and_questionWithoutLocation(question):
        '''
        This function takes in a question asked by the user.
        It extracts the mentioned location in the question,
        it classifies the scope (gemeente, woonplaats, provincie, or nederland) of the extracted location,
        and it reformulates the original question by removing the location.

        :param temperature_for_splitting_and_classifying_question: temperature parameter for the LLM
        :return: prompt for getting the information mentioned above
        '''

        prompt_string_template = """
        You are given a question.
        Your task is to first determine whether the question refers to a municipality, province, woonplaats, or the entirety of the Netherlands.
        Then you will need to extract the location and also reformulate the question without location.
        Your tasks are as following:

        1) extract a location from the question and write it behind "LOCATION:" in your answer. Remove any spelling mistakes in the location.
        If no location is provided you should say that the location is "Nederland".
        2) You should classify this location as being either "Woonplaats", "Gemeente", "Provincie", or "Nederland". Only strictly use these classifications.
        Note that if something is a Gemeente or Provincie the question will typically specify that it belongs to this category by directly mentioning this in the question.
        If nothing is mentioned you should classify it as a Woonplaats. Note that if a question mentions a "gemeentehuis", this does not mean the scope is gemeente! This has nothing to do with scope.
        If the previously extracted location is a direct mention to the entire Netherlands you should classify it as "Nederland". 
        Put the answer to this part behind "SCOPE:" in your answer.
        3) You should reformulate the question where you remove the location from the question. Write this is Dutch!
        Put the answer to this part behind "QUESTION-NO-ADRESS".
        Remove any spelling mistakes that are in the original question.


        You are provided with some examples of correct execution of you task and also the answer format that is required:


        ---------------------------------------------------------------------------------------------
        Here some examples are given:
        Question: Hoeveel brandweertorens zijn er in Ugchelen gebouwd na 1980?
        LOCATION: Ugchelen
        SCOPE: Woonplaats
        QUESTION-NO_ADDRESS: Hoeveel brandweertorens zijn gebouwd na 1980?

        Question: Hoeveel kerken kan ik vinden in groningen?
        LOCATION: Groningen
        SCOPE: Woonplaats
        QUESTION-NO_ADDRESS: Hoeveel kerken kan ik vinden

        Question: Hoeveel ziekenhuizen staan er in provincie groningen?
        LOCATION: Groningen
        SCOPE: Provincie
        QUESTION-NO-ADDRESS: Hoeveel ziekenhuizen staan er?

        Question: geef mij alle gemeentehuizen in hellevoetsluis
        LOCATION: Hellevoetsluis
        SCOPE: Woonplaats
        QUESTION-NO-ADDRESS: geef mij alle gemeentehuizen

        Question: Hoeveel ziekenhuizen staan er in groningen?
        LOCATION: Groningen
        SCOPE: Woonplaats
        QUESTION-NO-ADDRESS: Hoeveel ziekenhuizen staan er?

        Question: Hoeveel ziekenhuizen zijn er in nederland
        LOCATION: Nederland
        SCOPE: Nederland
        QUESTION-NO-ADDRESS: Hoeveel ziekenhuizen zijn er?

        Question: Hoeveel gevangenissen zijn er?
        LOCATION: Nederland
        SCOPE: Nederland
        QUESTION-NO-ADDRESS: Hoeveel gevangenissen zijn er?

        Question: Hoeveel begraafplaatsen zijn er in gemeente Utrecht?
        LOCATION: Utrecht
        SCOPE: Gemeente
        QUESTION-NO-ADDRESS: Hoeveel begraafplaatsen zijn er?

        Question: geef mij het grootste gemeentehuis in groningen
        LOCATION: groningen
        SCOPE: woonplaats
        QUESTION-NO-ADDRESS: Geef mij het grootste gemeentehuis


        ---------------------------------------------------------------------------------------------
        Question: {vraag}
        LOCATION:
        SCOPE: 
        QUESTION-NO-ADDRESS: 
        """
        messages = [
            HumanMessage(
                content=prompt_string_template.format(vraag=question)
            )

        ]

        # response = self.large_language_model(messages)
        # generated_response = response.content
        #
        # info_dict = {}
        #
        # # Split the input string into lines
        # lines = generated_response.strip().split("\n")
        #
        # # Iterate through the lines and extract information
        # for line in lines:
        #     key, value = line.split(": ", 1)  # Split each line at the first ": " occurrence
        #     key = key.strip()
        #     info_dict[key] = value
        # info_dict['ORIGINAL-QUESTION'] = question
        return messages

    @staticmethod
    def get_suggest_url(granularity):
        '''
        chooses relevant suggest url based on granularity for IRI retrieval for entity.
        :param granularity:
        :return:
        '''
        vector_store = None
        location_server_url = None
        scope = granularity

        # Based on the scope, we select the appropriate FAISS index, and also set the location server URL to the proper
        # filter.
        if granularity == "Woonplaats":
            # vector_store = storage_vector_stores_dict["faiss_index_woonplaats_ALL_EXAMPLES_NO_LOCATION"]
            location_server_url = 'http://api.pdok.nl/bzk/locatieserver/search/v3_1/free' \
                                  '?fq=type:(woonplaats)&q={}'




        elif granularity == "Gemeente":
            # vector_store = storage_vector_stores_dict["faiss_index_gemeente_ALL_EXAMPLES_NO_LOCATION"]
            location_server_url = 'http://api.pdok.nl/bzk/locatieserver/search/v3_1/free' \
                                  '?fq=type:(gemeente)&q={}'


        elif granularity == "Provincie":
            # vector_store = storage_vector_stores_dict["faiss_index_provincie_ALL_EXAMPLES_NO_LOCATION"]
            location_server_url = 'http://api.pdok.nl/bzk/locatieserver/search/v3_1/free' \
                                  '?fq=type:(provincie)&q={}'
        elif granularity == "Nederland":
            # vector_store = storage_vector_stores_dict["faiss_index_nederland_ALL_EXAMPLES_NO_LOCATION"]
            location_server_url = None
        elif granularity == "Straatnaam":
            # THESE NEED TO BE CHANGED
            # vector_store = storage_vector_stores_dict["faiss_index_straat_ALL_EXAMPLES_NO_LOCATION"]
            location_server_url = 'http://api.pdok.nl/bzk/locatieserver/search/v3_1/free' \
                                  '?fq=type:(gemeente%20OR%20woonplaats%20OR%20adres%20OR%20provincie)&q={}'
            test_string = 'https://api.pdok.nl/bzk/locatieserver/search/v3_1/free?fq=type:(gemeente%20OR%20woonplaats%20OR%20adres%20OR%20provincie)%20AND%20bron:BAG&q={}'

            # we need something like this: https://api.pdok.nl/bzk/locatieserver/search/v3_1/free?fq=type:(gemeente%20OR%20woonplaats%20OR%20adres%20OR%20provincie)%20AND%20bron:BAG&q=Het%20Nieuwe%20Diep%20Den%20Helder
        else:
            raise ValueError("Invalid granularity provided: {}".format(granularity))
        return location_server_url

    @staticmethod
    def retrieve_location_data(location, SUGGEST_URL):
        '''
        Takes suggest URL and the search query to perform an API call to the location server to retrieves location entity IRI
        :param location:
        :param SUGGEST_URL:
        :return:
        '''
        """
        This function does a call to the location server and retrieves information about a specific location such as numerical codes/identifiers.

        :param location: The location you want to retrieve codes about.
        :param SUGGEST_URL: The suggest URL that defines in which location types you want to search.
        :return: dictionary with information about the location.
        """
        url = SUGGEST_URL.format(location)
        address = requests.get(url).json()['response']['docs'][0]

        return address



def semantic_parsing_workflow(question, k_shortest_routes):
    '''
    THIS IS FOR THE ABLATION WITHOUT IN-CONTEXT EXAMPLES.



    :param question:
    :param k_shortest_routes:
    :return:
    '''
    # llm = AzureChatOpenAI(deployment_name="gpt-35-turbo", model_name="gpt-35-turbo", verbose=False, max_retries=2,
    #                       temperature=0, openai_api_version="2023-05-15")

    llm = AzureChatOpenAI(azure_deployment="gpt-4-32k", model_name="gpt-4-32k", verbose=False, max_retries=2,
                          temperature=0, openai_api_version="2023-05-15")

    print('Processing concurrent API calls to select relevant schema items and extract location..')
    print()
    # We create an object which stores some useful functions we will use later
    location_and_granularity_obj = location_and_granularity_object()

    # we retrieve the 3 prompt we want to execute concurrently
    retrieve_class_prompt, retrieve_relation_prompt = get_prompt_for_selecting_relevant_schema_items_based_on_natural_language_question(
        question)
    extract_location_prompt = location_and_granularity_obj.get_prompt_for_location_aggregation_and_questionWithoutLocation(
        question)

    # we process the 3 prompts concurrently
    class_response, relation_response, location_extractor_response = concurrent_executor(
        [retrieve_class_prompt, retrieve_relation_prompt, extract_location_prompt], llm=llm)

    # now we define some functions which process the responses of the llm
    def process_schema_response(list_as_string):
        '''
        We give it an LLM response which is a list formatted as a string.
        We convert it to an actual list object in python for further processing.
        :param list_as_string: the list as a string
        :return: an actual list retrieved from the string representation
        '''
        response_as_list = ast.literal_eval(list_as_string)
        return response_as_list

    def process_location_response_and_return_dictionary(response, question):
        '''
        We take the LLM response and turn it into a dictionary for further processing.



        :param response: the LLM response we want to process
        :param question: the natural language question (we also put it in the dictionary.)
        :return: A dictionary with the following keys: LOCATION, SCOPE, QUESTION-NO-ADRESS, ORIGINAL-QUESTION
        '''
        info_dict = {}

        # Split the input string into lines
        lines = response.strip().split("\n")

        # Iterate through the lines and extract information
        for line in lines:
            key, value = line.split(": ", 1)  # Split each line at the first ": " occurrence
            key = key.strip()
            info_dict[key] = value
        info_dict['ORIGINAL-QUESTION'] = question

        return info_dict

    # We use our functions to actually transform the LLM response
    # List with selected classes
    list_with_selected_classes = process_schema_response(class_response)
    # List with selected relations
    list_with_selected_relations = process_schema_response(relation_response)
    # Dictionary with decomposed question
    dictionary_with_decomposed_question = process_location_response_and_return_dictionary(location_extractor_response,
                                                                                          question)
    print('SELECTED CLASSES:')
    print(list_with_selected_classes)
    print('SELECTED RELATIONS')
    print(list_with_selected_relations)
    print(dictionary_with_decomposed_question)
    # We take the classes of the retrieved location and add it to our list of selected classes
    if dictionary_with_decomposed_question['SCOPE'] == 'Woonplaats':
        list_with_selected_classes.append('sor:Woonplaats')
    elif dictionary_with_decomposed_question['SCOPE'] == 'Gemeente':
        list_with_selected_classes.append('wbk:Gemeente')
    elif dictionary_with_decomposed_question['SCOPE'] == 'Provincie':
        list_with_selected_classes.append('sor:Provincie')
    elif dictionary_with_decomposed_question['SCOPE'] == 'Nederland':
        pass
    elif dictionary_with_decomposed_question['SCOPE'] == 'Straatnaam':
        list_with_selected_classes.append('sor:Woonplaats')
        list_with_selected_classes.append('sor:OpenbareRuimte')

    print('Condensing ontology based on relevant classes and relations..')
    print()
    # we create this object which stores a lot of information about our ontology and also a graph network representation
    network_ontology_store = network_and_ontology_store(k_shortest_routes)

    ontology_string_object_part, ontology_string_datatype_part = network_ontology_store.condense_ontology_based_on_retrieved_schema_items(
        list_with_selected_classes, list_with_selected_relations)

    print('Processing API call to location server to retrieve location identifier..')
    print()
    targeted_search_suggest_url = location_and_granularity_object.get_suggest_url(
        dictionary_with_decomposed_question['SCOPE'])
    response_location_server = None
    # depending on the scope we need to format the search URL in a different way
    if dictionary_with_decomposed_question['SCOPE'] == 'Straatnaam':
        # if the scope/granularity is about streets, we format our location such that we have %20 instead of space
        # this way we can change the search url to make it search for street & Woonplaats at the same time.
        # otherwise we cannot identify a street and woonplaats at the same time
        response_location_server = location_and_granularity_object.retrieve_location_data(
            dictionary_with_decomposed_question['LOCATION'].replace(" ", "%20"),
            targeted_search_suggest_url)
    elif dictionary_with_decomposed_question['SCOPE'] == 'Nederland':
        response_location_server = None
    else:
        # if we don't consider streets we can directly format the single location into the URL that performs the search
        response_location_server = location_and_granularity_object.retrieve_location_data(
            dictionary_with_decomposed_question['LOCATION'], targeted_search_suggest_url)

    # We create instance triples which which specify the desired location.
    # Note that by adding such triples we explain to the model where in the ontology structure this data resides
    # This instance data represents the location that is mentioned in the question.
    # NOTE: I also manually create some ontology triples here.
    # Later we check if these ontology triples are in our condensed ontology for completeness. (this is not crucial)
    # This is basically the ontology triples (often datatype edge) that precisely represents
    # the same part of the instance triple.
    # Due to the mechanism of the graph traversal algoritm we might get the entire route except the
    # endpoint which points to the datatype. So i make sure it is added for completeness.
    instance_prompt_string = ""
    corresponding_datatype_ontology_triples_list = []
    corresponding_object_ontology_triples_list = []
    location_to_return = None
    if dictionary_with_decomposed_question['SCOPE'] == 'Woonplaats':
        location_to_return = response_location_server["woonplaatsnaam"]
        instance_triple_1 = """sor:Woonplaats skos:prefLabel "%s"@nl""" % (response_location_server["woonplaatsnaam"])

        instance_prompt_string = """
        You are provided with a triple of the form: (class relation instancedata)
        You should interpret this as: there are instances of sor:Woonplaats that are that have the prefLabel
        mentioned in the triple. \n
        """
        instance_prompt_string = instance_prompt_string + instance_triple_1
        corresponding_datatype_ontology_triples_list.append('sor:Woonplaats skos:prefLabel rdf:langString')

    elif dictionary_with_decomposed_question['SCOPE'] == 'Gemeente':
        location_to_return = response_location_server["gemeentecode"]
        instance_triple_1 = """wbk:Gemeente sdo0:identifier "%s" """ % ('GM' + response_location_server["gemeentecode"])
        instance_prompt_string = """
        You are provided with a triple of the form: (class relation instancedata)
        You should interpret this as: there are instances of wbk:Gemeente that have this identifier, and this
        identifier corresponds to the gemeente %s . \n
        """ % (dictionary_with_decomposed_question['LOCATION'])
        instance_prompt_string = instance_prompt_string + instance_triple_1
        corresponding_datatype_ontology_triples_list.append('wbk:Gemeente sdo0:identifier xsd:string')

    elif dictionary_with_decomposed_question['SCOPE'] == 'Provincie':
        location_to_return = response_location_server["provinciecode"].replace("PV", "00")
        instance_triple_1 = """sor:Gemeente geo:sfWithin provincie:%s""" % (
            response_location_server["provinciecode"].replace("PV", "00"))
        instance_prompt_string = """
        You are provided with a triple of the form: (class relation instancedata)
        You should interpret this as: there are instances of sor:Gemeente that is located in
        an instance of sor:Provincie. The instance of sor:Provincie corresponds to province %s . \n
        """ % (dictionary_with_decomposed_question['LOCATION'])
        instance_prompt_string = instance_prompt_string + instance_triple_1
        corresponding_object_ontology_triples_list.append('sor:Gemeente geo:sfWithin sor:Provincie')

    elif dictionary_with_decomposed_question['SCOPE'] == 'Nederland':
        location_to_return = 'Nederland'
        pass
    elif dictionary_with_decomposed_question['SCOPE'] == 'Straatnaam':
        instance_triple_1 = """sor:Woonplaats skos:prefLabel "%s"@nl\n""" % (response_location_server["woonplaatsnaam"])
        instance_triple_2 = """sor:OpenbareRuimte skos:prefLabel "%s"@nl""" % (response_location_server["straatnaam"])
        instance_prompt_string = """
        You are provided with some triples of the form: (class relation instancedata)
        The first triple you should interpret as: there are instances of sor:Woonplaats that are that have the prefLabel
        mentioned in the triple.
        The second triple you should interpret as: there are instance of sor:OpenbareRuimte that have the prefLab
        as mentioned in the triple. These are typically streets. \n
        """
        instance_prompt_string = instance_prompt_string + instance_triple_1 + instance_triple_2
        corresponding_datatype_ontology_triples_list.append('sor:Woonplaats skos:prefLabel rdf:langString')
        corresponding_datatype_ontology_triples_list.append('sor:OpenbareRuimte skos:prefLabel rdf:langString')





    # we check if the corresponding datatype ontology triples are in our condensed ontology, if not we add it
    for trip in corresponding_datatype_ontology_triples_list:
        if trip not in ontology_string_datatype_part:
            ontology_string_datatype_part = ontology_string_datatype_part + '\n' + trip
    # we check if the corresponding object ontology triples are in our condensed ontology, if not we add it
    for trip in corresponding_object_ontology_triples_list:
        if trip not in ontology_string_object_part:
            ontology_string_datatype_part = ontology_string_datatype_part + '\n' + trip

    print('Processing API call to generate SPARQL..')
    print()
    system_string = """
    You should act as a knowledge graph professional with excellent expertise in writing SPARQL queries.
    You will be provided with a schema of a knowledge graph in the geographical domain where the schema items
    are in Dutch. 

    You will be given a natural language question. Using the provided schema, 
    you should write SPARQL that best matches the natural language question.

    1)
    The section called "INSTANCE DATA REGARDING LOCATIONS" has triples which show position of certain instance
    data in the ontology. This instance data is always about geographical locations. this way you will be able to generate
    SPARQL regarding certain geographical locations.
    If the question is about the entire netherlands this place will be empty since you don't have
    to link it to certain locations.

    2)
    The section called "OBJECT/CLASS NAVIGATION PATH ONTOLOGY" contains lines where we see how classes are related to eachother
    through properties in the following format:
    class1 property1 class2
    
    -We define relationships between classes by specifying how they are linked to each other through properties.
    -In our ontology, the first class mentioned in a relationship represents the starting point, 
    while the second class represents the destination. This distinction is crucial for understanding the directionality of relationships.
    -By default, relationships are interpreted from left to right, with the first class acting as the subject and the second class as the object. 
    However, when using the SPARQL operator (^), relationships can be traversed from right to left, reversing the directionality.
    We provide examples demonstrating how to use the directionality in SPARQL.
    
    -When you apply the RELATION 'property1' to an instance of the CLASS 'class1', you take path towards
    and instance of CLASS 'class2'. SO IT TELLS YOU WHERE A RELATION PATH LEADS TO!
    
    -THESE PATHS ONLY GO IN ONE DIRECTION!
    IF YOU WANT TO NAVIGATE THE OTHER WAY AROUND USE THE "^" OPERATOR IN SPARQL.
    
    -The FIRST and THIRD words in such lines are CLASSES. USE THEM AS SUCH! DO NOT USE THEM AS RELATIONS!
    The SECOND word in such lines are RELATIONS. USE ONLY THOSE AS RELATIONS!

    -ONLY USE THE NAVIGATION PATHS THAT ARE DESCRIBED! THIS IS VERY IMPORTANT!
    YOU CANNOT SKIP INTERMEDIATE PATHS!
    
    -ALWAYS PRIORITIZE CLASSES, RELATIONS, AND NAVIGATION PATHS WITH THE "sor:" PREFIX IF THESE ARE AVAILABLE! 
    THESE SHOULD BE PREFERRED OVER 'bag:' and "kad:" IF POSSIBLE!
    
    -ALWAYS PRIORITIZE CLASSES, RELATIONS, AND NAVIGATION PATHS WITH THE "kad:" PREFIX over "bag:" 
    EXAMPLE: USE kad:Ligplaats  INSTEAD OF bag:Ligplaats !!
    
    WE GIVE SOME EXAMPLE, OBSERVE HOW WE ONLY USE EXISTING ROUTES:
    
    ------------------
    Example 1:
    
    ONTOLOGY:
    sor:Gebouw geo:sfWithin wbk:Buurt
    wbk:Buurt geo:sfWithin wbk:Wijk
    wbk:Wijk geo:sfWithin wbk:Gemeente
    sor:Gemeente owl:sameAs wbk:Gemeente
    sor:Gemeente geo:sfWithin sor:Provincie
    
    TASK:
    You want to link instances of  'sor:Gebouw' to 'sor:Provincie'
    
    PATH IN SPARQL:
    ?gebouw a sor:Gebouw ;
        geo:sfWithin/geo:sfWithin/geo:sfWithin/^owl:sameAs/geo:sfWithin ?provincie .
    ?gemeente a sor:Provincie.
    
    WRONG PATH IN SPARQL, NEVER DO THIS ( HERE YOU USE CLASSES AS RELATION. THIS IS INVALID ):
    ?gebouw a sor:Gebouw ;
        geo:sfWithin/wbk:Buurt/geo:sfWithin/wbk:Wijk/geo:sfWithin/wbk:Gemeente/^owl:sameAs/sor:Gemeente/geo:sfWithin/sor:Provincie ?provincie .
    -------------------------------------
    -------------------------------------
    Example 2:
    
    ONTOLOGY:
    sor:Verblijfsobject sor:maaktDeelUitVan sor:Gebouw
    sor:Verblijfsobject sor:hoofdadres sor:Nummeraanduiding
    sor:Nummeraanduiding sor:ligtAan sor:OpenbareRuimte
    sor:OpenbareRuimte sor:ligtIn sor:Woonplaats
    
    TASK:
    you want to link isntances of 'sor:Gebouw' to 'sor:Woonplaats':
    
    PATH IN SPARQL:
    ?gebouw a sor:Gebouw ;
        ^sor:maaktDeelUitVan/sor:hoofdadres/sor:ligtAan/sor:ligtIn ?woonplaats .
    ?woonplaats a sor:Woonplaats.
    
    -------------------------------------
    Example 3:
    
    ONTOLOGY:
    sor:Perceel geo:sfWithin sor:Gemeente
    sor:Gemeente geo:sfWithin sor:Provincie
    bag:Perceel geo:sfWithin bag:Gemeente
    bag:Gemeente geo:sfWithin bag:Provincie
    
    TASK
    you want to link percelen to provinces.
    Remember we put priority on sor: paths.
    
    PATH IN SPARQL:
    ?perceel a sor:Perceel ;
        geo:sfWithin/geo:sfWithin ?provincie
    ?provincie a sor:Provincie.
    ---------------------------------------


    WHEN USING CLASSES SPECIFICALLY DEFINE THE CLASSES THAT YOU USE, FOR EXAMPLE, LIKE THIS:
    ?variable a sor:Gebouw


    3)
    The section called "DATATYPE ONTOLOGY" contains lines where we see how classes are related to certain datatypes
    of instance data through properties using the following format:
    class1 property2 xsd:PositiveInteger
    
    Use this information: For example, take special care of variables of type xsd:gYear. 
    Ensure that any comparisons or operations involving ?bouwjaar are appropriate for this data type. 
    Additionally, consider any necessary adjustments to the query to accommodate the specific characteristics of xsd:gYear.
    A good option is to use the YEAR() function to extract the actual year.
    

    !!!!!
    DO NOT USE SCHEMA ITEMS/NAVIGATION PATHS THAT ARE: NOT IN THE ONTOLOGY I PROVIDED YOU WITH OR NOT IN THE EXAMPLES
    !!!!!
    
    USE COMPLETE RELATION PATH, DO NOT SKIP RELATIONS IN THE PATH!!
    
    When performing COUNT() operations always include DISTINCT INSIDE OF IT: COUNT( DISTINCT )
    
    
    If you're filtering based on the count of values in a HAVING clause, 
    make sure to use the aggregate function directly in the HAVING clause rather than referencing an aliased variable.

    
    Only respond with a SPARQL query, do not put any other comments in your response.
    """

    partial_prompt = """
    INSTANCE DATA REGARDING LOCATIONS:
    %s

    OBJECT ONTOLOGY:
    %s

    DATATYPE ONTOLOGY:
    %s

    QUESTION IN NATURAL LANGUAGE:
    %s
    """ % (instance_prompt_string, ontology_string_object_part, ontology_string_datatype_part, question)
    print(partial_prompt)

    # Here we combine the partial prompt with a prefix we put in a SystemMessage.
    # To use ChatModels in langchain we need the prompt in this format.
    messages = [
        SystemMessage(
            content=system_string
        ),
        # THIS NEEDS TO BE CHANGED STILL!
        HumanMessage(
            content=str(partial_prompt)
        )
    ]
    llm = AzureChatOpenAI(azure_deployment="gpt-4-32k", model_name="gpt-4-32k", verbose=False, max_retries=2,
                          temperature=0, openai_api_version="2023-05-15")
    response = llm(messages)
    generated_query = response.content
    print('Generated SPARQL')
    print(generated_query)

    return {'query': generated_query, 'granularity':dictionary_with_decomposed_question['SCOPE'], 'location':location_to_return }


def semantic_parsing_workflow_few_shot(question,
                                       k_shortest_routes,
                                       number_of_few_shot_examples=2,
                                       meta_data_filtering_for_few_shot_examples=True,
                                       retrieve_examples_with_question_without_location=False,
                                       example_retrieval_style='max_marginal_similarity',
                                       naive_schema_linking_without_graph_traversal = False,
                                       verbose = False,
                                       condense_ontology = True):
    '''

    THIS ONE INCLUDES IN-CONTEXT EXAMPLES.

    :param question:
    :param k_shortest_routes:
    :param number_of_few_shot_examples:
    :param meta_data_filtering_for_few_shot_examples:
    :param retrieve_examples_with_question_without_location:
    :param example_retrieval_style:
    :param naive_schema_linking_without_graph_traversal:
    :param verbose:
    :param condense_ontology: Whether you want to condense the ontology or include full ontology
    :return:
    '''
    # llm = AzureChatOpenAI(deployment_name="gpt-35-turbo", model_name="gpt-35-turbo", verbose=False, max_retries=2,
    #                       temperature=0, openai_api_version="2023-05-15")

    llm = AzureChatOpenAI(azure_deployment="gpt-4-32k", model_name="gpt-4-32k", verbose=False, max_retries=2,
                          temperature=0, openai_api_version="2023-05-15")
    if verbose == True:
        print('Processing concurrent API calls to select relevant schema items and extract location..')
    # We create an object which stores some useful functions we will use later
    location_and_granularity_obj = location_and_granularity_object()

    # we retrieve the 3 prompt we want to execute concurrently
    retrieve_class_prompt, retrieve_relation_prompt = get_prompt_for_selecting_relevant_schema_items_based_on_natural_language_question(
        question)
    extract_location_prompt = location_and_granularity_obj.get_prompt_for_location_aggregation_and_questionWithoutLocation(
        question)

    # we process the 3 prompts concurrently
    class_response, relation_response, location_extractor_response = concurrent_executor(
        [retrieve_class_prompt, retrieve_relation_prompt, extract_location_prompt], llm=llm)

    # now we define some functions which process the responses of the llm
    def process_schema_response(list_as_string):
        '''
        We give it an LLM response which is a list formatted as a string.
        We convert it to an actual list object in python for further processing.
        :param list_as_string: the list as a string
        :return: an actual list retrieved from the string representation
        '''
        response_as_list = ast.literal_eval(list_as_string)
        return response_as_list

    def process_location_response_and_return_dictionary(response, question):
        '''
        We take the LLM response and turn it into a dictionary for further processing.



        :param response: the LLM response we want to process
        :param question: the natural language question (we also put it in the dictionary.)
        :return: A dictionary with the following keys: LOCATION, SCOPE, QUESTION-NO-ADRESS, ORIGINAL-QUESTION
        '''
        info_dict = {}

        # Split the input string into lines
        lines = response.strip().split("\n")

        # Iterate through the lines and extract information
        for line in lines:
            key, value = line.split(": ", 1)  # Split each line at the first ": " occurrence
            key = key.strip()
            info_dict[key] = value
        info_dict['ORIGINAL-QUESTION'] = question

        return info_dict

    # We use our functions to actually transform the LLM response
    # List with selected classes
    list_with_selected_classes = process_schema_response(class_response)
    # List with selected relations
    list_with_selected_relations = process_schema_response(relation_response)
    # Dictionary with decomposed question
    dictionary_with_decomposed_question = process_location_response_and_return_dictionary(location_extractor_response,
                                                                                         question)
    if verbose == True:
        print(dictionary_with_decomposed_question)

        print('SELECTED CLASSES:')
        print(list_with_selected_classes)
        print('SELECTED RELATIONS:')
        print(list_with_selected_relations)
        print()

        print('Retrieving few-shot examples based on granularity..')
    # We select the appropriate vector_store with training data based on the granularity
    # We also take the classes of the retrieved location and add it to our list of selected classes

    embeddings = AzureOpenAIEmbeddings(
        azure_deployment='text-embedding-ada-002',
        openai_api_version=os.environ['OPENAI_API_VERSION'],
        # chunk_size=1
    ),
    vector_store = None

    if dictionary_with_decomposed_question['SCOPE'] == 'Woonplaats':
        list_with_selected_classes.append('sor:Woonplaats')
        os.path.join(current_dir, '/vector_stores/woonplaats_vector_store')
        vector_store = FAISS.load_local(os.path.join(current_dir, 'vector_stores','woonplaats_vector_store'), embeddings)

    elif dictionary_with_decomposed_question['SCOPE'] == 'Gemeente':
        list_with_selected_classes.append('wbk:Gemeente')
        vector_store = FAISS.load_local(os.path.join(current_dir, 'vector_stores', 'gemeente_vector_store'), embeddings)

    elif dictionary_with_decomposed_question['SCOPE'] == 'Provincie':
        list_with_selected_classes.append('sor:Provincie')
        vector_store = FAISS.load_local(os.path.join(current_dir, 'vector_stores','provincie_vector_store'), embeddings)

    elif dictionary_with_decomposed_question['SCOPE'] == 'Nederland':
        vector_store = FAISS.load_local(os.path.join(current_dir, 'vector_stores','land_vector_store'), embeddings)

    elif dictionary_with_decomposed_question['SCOPE'] == 'Straatnaam':
        vector_store = FAISS.load_local(os.path.join(current_dir, 'vector_stores','straat_vector_store'), embeddings)
        list_with_selected_classes.append('sor:Woonplaats')
        list_with_selected_classes.append('sor:OpenbareRuimte')

    if meta_data_filtering_for_few_shot_examples == False:
        vector_store = FAISS.load_local(os.path.join(current_dir, 'vector_stores','combined_vector_store'), embeddings)

    vector_store.embedding_function = vector_store.embedding_function[0]

    # You can use the following functions for different example selection mechanisms:
    # SemanticSimilarityExampleSelector
    # MaxMarginalRelevanceExampleSelector
    example_selector = None
    if example_retrieval_style == 'max_marginal_similarity':
        example_selector = MaxMarginalRelevanceExampleSelector(
            vectorstore=vector_store,
            k=number_of_few_shot_examples
        )
    elif example_retrieval_style == 'semantic_similarity':
        example_selector = SemanticSimilarityExampleSelector(
            vectorstore=vector_store,
            k=number_of_few_shot_examples
        )
    # print(example_selector.vectorstore.index.ntotal)

    selected_examples = None
    if retrieve_examples_with_question_without_location == True:
        search_query = {"query": dictionary_with_decomposed_question['QUESTION-NO-ADDRESS']}
        selected_examples = example_selector.select_examples(search_query)
    else:
        search_query = {"query": dictionary_with_decomposed_question['ORIGINAL-QUESTION']}
        selected_examples = example_selector.select_examples(search_query)

    print(selected_examples)
    example_prompt = PromptTemplate(
        input_variables=["QUESTION", "SPARQL"],
        template="Question: {QUESTION}\nQuery: {SPARQL}",
    )
    full_example_prompt = FewShotPromptTemplate(
        # We provide an ExampleSelector instead of examples.
        examples=selected_examples,
        example_prompt=example_prompt,
        suffix="Question: {question}\nQuery:",
        input_variables=["question"],
    )
    print(selected_examples)
    if verbose == True:
        print(full_example_prompt.format(question=dictionary_with_decomposed_question['ORIGINAL-QUESTION']))
        print()

        print('Condensing ontology based on relevant classes and relations..')
        print()
    # we create this object which stores a lot of information about our ontology and also a graph network representation
    network_ontology_store = network_and_ontology_store(k_shortest_routes)
    if condense_ontology == True:
        if naive_schema_linking_without_graph_traversal == False:
            ontology_string_object_part, ontology_string_datatype_part = network_ontology_store.condense_ontology_based_on_retrieved_schema_items(
                list_with_selected_classes, list_with_selected_relations)
        else:
            ontology_string_object_part, ontology_string_datatype_part = network_ontology_store.naive_ontology_selection(
                list_with_selected_classes, list_with_selected_relations)
    else:
        ontology_string_object_part, ontology_string_datatype_part = network_ontology_store.retrieve_full_ontology()

    if verbose == True:
        print('Processing API call to location server to retrieve location identifier..')
        print()
    targeted_search_suggest_url = location_and_granularity_object.get_suggest_url(
        dictionary_with_decomposed_question['SCOPE'])
    response_location_server = None
    # depending on the scope we need to format the search URL in a different way
    if dictionary_with_decomposed_question['SCOPE'] == 'Straatnaam':
        # if the scope/granularity is about streets, we format our location such that we have %20 instead of space
        # this way we can change the search url to make it search for street & Woonplaats at the same time.
        # otherwise we cannot identify a street and woonplaats at the same time
        response_location_server = location_and_granularity_object.retrieve_location_data(
            dictionary_with_decomposed_question['LOCATION'].replace(" ", "%20"),
            targeted_search_suggest_url)
    elif dictionary_with_decomposed_question['SCOPE'] == 'Nederland':
        reponse_location_server = None
    else:
        # if we don't consider streets we can directly format the single location into the URL that performs the search
        response_location_server = location_and_granularity_object.retrieve_location_data(
            dictionary_with_decomposed_question['LOCATION'], targeted_search_suggest_url)

    # We create instance triples which which specify the desired location.
    # Note that by adding such triples we explain to the model where in the ontology structure this data resides
    # This instance data represents the location that is mentioned in the question.
    # NOTE: I also manually create some ontology triples here.
    # Later we check if these ontology triples are in our condensed ontology for completeness. (this is not crucial)
    # This is basically the ontology triples (often datatype edge) that precisely represents
    # the same part of the instance triple.
    # Due to the mechanism of the graph traversal algoritm we might get the entire route except the
    # endpoint which points to the datatype. So it make sure it is added for completeness.
    instance_prompt_string = ""
    corresponding_datatype_ontology_triples_list = []
    corresponding_object_ontology_triples_list = []
    location_to_return = None
    if dictionary_with_decomposed_question['SCOPE'] == 'Woonplaats':
        location_to_return = response_location_server["woonplaatsnaam"]
        instance_triple_1 = """sor:Woonplaats skos:prefLabel "%s"@nl""" % (response_location_server["woonplaatsnaam"])

        instance_prompt_string = """
        You are provided with a triple of the form: (class relation instancedata)
        You should interpret this as: there are instances of sor:Woonplaats that are that have the prefLabel
        mentioned in the triple. 
        
        ALWAYS USE THIS PREFLABEL IN YOUR QUERY!
        \n
        """
        instance_prompt_string = instance_prompt_string + instance_triple_1
        corresponding_datatype_ontology_triples_list.append('sor:Woonplaats skos:prefLabel rdf:langString')

    elif dictionary_with_decomposed_question['SCOPE'] == 'Gemeente':
        location_to_return = response_location_server["gemeentecode"]
        instance_triple_1 = """wbk:Gemeente sdo0:identifier "%s" """ % ('GM' + response_location_server["gemeentecode"])
        instance_prompt_string = """
        You are provided with a triple of the form: (class relation instancedata)
        You should interpret this as: there are instances of wbk:Gemeente that have this identifier, and this
        identifier corresponds to the gemeente %s . 
        
        ALWAYS USE THIS CODE IN YOUR QUERY!
        \n
        """ % (dictionary_with_decomposed_question['LOCATION'])
        instance_prompt_string = instance_prompt_string + instance_triple_1
        corresponding_datatype_ontology_triples_list.append('wbk:Gemeente sdo0:identifier xsd:string')

    elif dictionary_with_decomposed_question['SCOPE'] == 'Provincie':
        location_to_return = response_location_server["provinciecode"].replace("PV", "00")
        instance_triple_1 = """sor:Gemeente geo:sfWithin provincie:%s""" % (
            response_location_server["provinciecode"].replace("PV", "00"))
        instance_prompt_string = """
        You are provided with a triple of the form: (class relation instancedata)
        You should interpret this as: there are instances of sor:Gemeente that are located in
        the province %s .
        ALWAYS USE THE PROVINCE CODE IN YOUR QUERY!
         \n
        """ % (dictionary_with_decomposed_question['LOCATION'])
        instance_prompt_string = instance_prompt_string + instance_triple_1
        corresponding_object_ontology_triples_list.append('sor:Gemeente geo:sfWithin sor:Provincie')

    elif dictionary_with_decomposed_question['SCOPE'] == 'Nederland':
        location_to_return = 'Nederland'
        pass
    elif dictionary_with_decomposed_question['SCOPE'] == 'Straatnaam':
        instance_triple_1 = """sor:Woonplaats skos:prefLabel "%s"@nl\n""" % (response_location_server["woonplaatsnaam"])
        instance_triple_2 = """sor:OpenbareRuimte skos:prefLabel "%s"@nl""" % (response_location_server["straatnaam"])
        instance_prompt_string = """
        You are provided with some triples of the form: (class relation instancedata)
        The first triple you should interpret as: there are instances of sor:Woonplaats that are that have the prefLabel
        mentioned in the triple.
        The second triple you should interpret as: there are instance of sor:OpenbareRuimte that have the prefLab
        as mentioned in the triple. These are typically streets. \n
        """
        instance_prompt_string = instance_prompt_string + instance_triple_1 + instance_triple_2
        corresponding_datatype_ontology_triples_list.append('sor:Woonplaats skos:prefLabel rdf:langString')
        corresponding_datatype_ontology_triples_list.append('sor:OpenbareRuimte skos:prefLabel rdf:langString')

    # we check if the corresponding datatype ontology triples are in our condensed ontology, if not we add it
    for trip in corresponding_datatype_ontology_triples_list:
        if trip not in ontology_string_datatype_part:
            ontology_string_datatype_part = ontology_string_datatype_part + '\n' + trip
    # we check if the corresponding object ontology triples are in our condensed ontology, if not we add it
    for trip in corresponding_object_ontology_triples_list:
        if trip not in ontology_string_object_part:
            ontology_string_datatype_part = ontology_string_datatype_part + '\n' + trip

    system_string = """
    You should act as a knowledge graph professional with excellent expertise in writing SPARQL queries.
    You will be provided with a schema of a knowledge graph in the geographical domain where the schema items
    are in Dutch. 

    You will be given a natural language question. Using the provided schema, and the examples provided to you,
    you should write SPARQL that best matches the natural language question.


    1)
    The section called "INSTANCE DATA REGARDING LOCATIONS" has triples which show position of certain instance
    data in the ontology. This instance data is always about geographical locations. this way you will be able to generate
    SPARQL regarding certain geographical locations.
    If the question is about the entire netherlands this place will be empty since you don't have
    to link it to certain locations.

    2)
    The section called "OBJECT/CLASS NAVIGATION PATH ONTOLOGY" contains lines where we see how classes are related to eachother
    through properties in the following format:
    class1 property1 class2
    
    -We define relationships between classes by specifying how they are linked to each other through properties.
    -In our ontology, the first class mentioned in a relationship represents the starting point, 
    while the second class represents the destination. This distinction is crucial for understanding the directionality of relationships.
    -By default, relationships are interpreted from left to right, with the first class acting as the subject and the second class as the object. 
    However, when using the SPARQL operator (^), relationships can be traversed from right to left, reversing the directionality.
    We provide examples demonstrating how to use the directionality in SPARQL.
    
    -When you apply the RELATION 'property1' to an instance of the CLASS 'class1', you take path towards
    and instance of CLASS 'class2'. SO IT TELLS YOU WHERE A RELATION PATH LEADS TO!
    
    -THESE PATHS ONLY GO IN ONE DIRECTION!
    IF YOU WANT TO NAVIGATE THE OTHER WAY AROUND USE THE "^" OPERATOR IN SPARQL.
    
    -The FIRST and THIRD words in such lines are CLASSES. USE THEM AS SUCH! DO NOT USE THEM AS RELATIONS!
    The SECOND word in such lines are RELATIONS. USE ONLY THOSE AS RELATIONS!

    -ONLY USE THE NAVIGATION PATHS THAT ARE DESCRIBED! THIS IS VERY IMPORTANT!
    YOU CANNOT SKIP INTERMEDIATE PATHS!
    
    -ALWAYS PRIORITIZE CLASSES, RELATIONS, AND NAVIGATION PATHS WITH THE "sor:" PREFIX IF THESE ARE AVAILABLE! 
    THESE SHOULD BE PREFERRED OVER 'bag:' and "kad:" IF POSSIBLE!
    
    -ALWAYS PRIORITIZE CLASSES, RELATIONS, AND NAVIGATION PATHS WITH THE "kad:" PREFIX over "bag:" 
    EXAMPLE: USE kad:Ligplaats  INSTEAD OF bag:Ligplaats !!
    
    WE GIVE SOME EXAMPLE, OBSERVE HOW WE ONLY USE EXISTING ROUTES:
    
    ------------------
    Example 1:
    
    ONTOLOGY:
    sor:Gebouw geo:sfWithin wbk:Buurt
    wbk:Buurt geo:sfWithin wbk:Wijk
    wbk:Wijk geo:sfWithin wbk:Gemeente
    sor:Gemeente owl:sameAs wbk:Gemeente
    sor:Gemeente geo:sfWithin sor:Provincie
    
    TASK:
    You want to link instances of  'sor:Gebouw' to 'sor:Provincie'
    
    PATH IN SPARQL:
    ?gebouw a sor:Gebouw ;
        geo:sfWithin/geo:sfWithin/geo:sfWithin/^owl:sameAs/geo:sfWithin ?provincie .
    ?gemeente a sor:Provincie.
    
    WRONG PATH IN SPARQL, NEVER DO THIS ( HERE YOU USE CLASSES AS RELATION. THIS IS INVALID ):
    ?gebouw a sor:Gebouw ;
        geo:sfWithin/wbk:Buurt/geo:sfWithin/wbk:Wijk/geo:sfWithin/wbk:Gemeente/^owl:sameAs/sor:Gemeente/geo:sfWithin/sor:Provincie ?provincie .
    -------------------------------------
    -------------------------------------
    Example 2:
    
    ONTOLOGY:
    sor:Verblijfsobject sor:maaktDeelUitVan sor:Gebouw
    sor:Verblijfsobject sor:hoofdadres sor:Nummeraanduiding
    sor:Nummeraanduiding sor:ligtAan sor:OpenbareRuimte
    sor:OpenbareRuimte sor:ligtIn sor:Woonplaats
    
    TASK:
    you want to link isntances of 'sor:Gebouw' to 'sor:Woonplaats':
    
    PATH IN SPARQL:
    ?gebouw a sor:Gebouw ;
        ^sor:maaktDeelUitVan/sor:hoofdadres/sor:ligtAan/sor:ligtIn ?woonplaats .
    ?woonplaats a sor:Woonplaats.
    
    -------------------------------------
    Example 3:
    
    ONTOLOGY:
    sor:Perceel geo:sfWithin sor:Gemeente
    sor:Gemeente geo:sfWithin sor:Provincie
    bag:Perceel geo:sfWithin bag:Gemeente
    bag:Gemeente geo:sfWithin bag:Provincie
    
    TASK
    you want to link percelen to provinces.
    Remember we put priority on sor: paths.
    
    PATH IN SPARQL:
    ?perceel a sor:Perceel ;
        geo:sfWithin/geo:sfWithin ?provincie
    ?provincie a sor:Provincie.
    ---------------------------------------


    WHEN USING CLASSES SPECIFICALLY DEFINE THE CLASSES THAT YOU USE, FOR EXAMPLE, LIKE THIS:
    ?variable a sor:Gebouw


    3)
    The section called "DATATYPE ONTOLOGY" contains lines where we see how classes are related to certain datatypes
    of instance data through properties using the following format:
    class1 property2 xsd:PositiveInteger
    
    Use this information: For example, take special care of variables of type xsd:gYear. 
    Ensure that any comparisons or operations involving ?bouwjaar are appropriate for this data type. 
    Additionally, consider any necessary adjustments to the query to accommodate the specific characteristics of xsd:gYear.
    A good option is to use the YEAR() function to extract the actual year.
    

    !!!!!
    DO NOT USE SCHEMA ITEMS/NAVIGATION PATHS THAT ARE: NOT IN THE ONTOLOGY I PROVIDED YOU WITH OR NOT IN THE EXAMPLES
    !!!!!
    
    USE COMPLETE RELATION PATH, DO NOT SKIP RELATIONS IN THE PATH!!
    
    When performing COUNT() operations always include DISTINCT INSIDE OF IT: COUNT( DISTINCT )
    
    If you're filtering based on the count of values in a HAVING clause, 
    make sure to use the aggregate function directly in the HAVING clause rather than referencing an aliased variable.
    
    When constructing SPARQL queries, double-check the direction of properties in triple patterns to accurately navigate the relationships between entities. 
    Make sure to use the correct classes specified in the instance data.


    
    Only respond with a SPARQL query, do not put any other comments in your response.

    """

    partial_prompt = """
    INSTANCE DATA REGARDING LOCATIONS:
    %s

    OBJECT/CLASS NAVIGATION PATH ONTOLOGY:
    %s

    DATATYPE ONTOLOGY:
    %s

    QUESTION IN NATURAL LANGUAGE:
    %s
    """ % (instance_prompt_string, ontology_string_object_part, ontology_string_datatype_part, question)

    system_string = system_string + '\n' + partial_prompt
    if verbose == True:
        print(system_string)

        print('Processing API call to generate SPARQL..')
        print()
    # Here we combine the partial prompt with a prefix we put in a SystemMessage.
    # To use ChatModels in langchain we need the prompt in this format.
    messages = [
        SystemMessage(
            content=system_string
        ),
        # THIS NEEDS TO BE CHANGED STILL!
        HumanMessage(
            content=str(full_example_prompt.format(question=dictionary_with_decomposed_question['ORIGINAL-QUESTION']))
        )
    ]

    response = llm(messages)
    generated_query = response.content
    if verbose == True:
        print('Generated SPARQL')
        print(generated_query)

    return {'query': generated_query, 'granularity':dictionary_with_decomposed_question['SCOPE'], 'location':location_to_return }





def semantic_parsing_workflow_few_shot_NO_ONTOLOGY(question,
                                                   number_of_few_shot_examples=5,
                                                   meta_data_filtering_for_few_shot_examples=True,
                                                   retrieve_examples_with_question_without_location=True,
                                                   example_retrieval_style='max_marginal_similarity'):
    '''
    This one is for the ablation WITHOUT an ontology.

    :param question:
    :param number_of_few_shot_examples:
    :param meta_data_filtering_for_few_shot_examples:
    :param retrieve_examples_with_question_without_location:
    :param example_retrieval_style:
    :return:
    '''
    # llm = AzureChatOpenAI(deployment_name="gpt-35-turbo", model_name="gpt-35-turbo", verbose=False, max_retries=2,
    #                       temperature=0, openai_api_version="2023-05-15")

    llm = AzureChatOpenAI(azure_deployment="gpt-4-32k", model_name="gpt-4-32k", verbose=False, max_retries=2,
                          temperature=0, openai_api_version="2023-05-15")

    print('Processing concurrent API call to extract location..')
    # We create an object which stores some useful functions we will use later
    location_and_granularity_obj = location_and_granularity_object()

    # we retrieve the prompt for span classification, type classification, and entity masking
    extract_location_prompt = location_and_granularity_obj.get_prompt_for_location_aggregation_and_questionWithoutLocation(
        question)

    loc_response = llm(extract_location_prompt)
    location_extractor_response = loc_response.content

    # now we define some functions which process the responses of the llm
    def process_location_response_and_return_dictionary(response, question):
        '''
        We take the LLM response and turn it into a dictionary for further processing.



        :param response: the LLM response we want to process
        :param question: the natural language question (we also put it in the dictionary.)
        :return: A dictionary with the following keys: LOCATION, SCOPE, QUESTION-NO-ADRESS, ORIGINAL-QUESTION
        '''
        info_dict = {}

        # Split the input string into lines
        lines = response.strip().split("\n")

        # Iterate through the lines and extract information
        for line in lines:
            key, value = line.split(": ", 1)  # Split each line at the first ": " occurrence
            key = key.strip()
            info_dict[key] = value
        info_dict['ORIGINAL-QUESTION'] = question

        return info_dict

    # Dictionary with decomposed question
    dictionary_with_decomposed_question = process_location_response_and_return_dictionary(location_extractor_response,
                                                                                          question)
    print(dictionary_with_decomposed_question)

    print('Retrieving few-shot examples based on granularity..')
    # We select the appropriate vector_store with training data based on the granularity
    # We also take the classes of the retrieved location and add it to our list of selected classes

    embeddings = AzureOpenAIEmbeddings(
        azure_deployment='text-embedding-ada-002',
        openai_api_version=os.environ['OPENAI_API_VERSION'],
        # chunk_size=1
    ),
    vector_store = None
    location_to_return = None

    if dictionary_with_decomposed_question['SCOPE'] == 'Woonplaats':

        os.path.join(current_dir, '/vector_stores/woonplaats_vector_store')
        vector_store = FAISS.load_local(os.path.join(current_dir, 'vector_stores', 'woonplaats_vector_store'),
                                        embeddings)

    elif dictionary_with_decomposed_question['SCOPE'] == 'Gemeente':

        vector_store = FAISS.load_local(os.path.join(current_dir, 'vector_stores', 'gemeente_vector_store'), embeddings)

    elif dictionary_with_decomposed_question['SCOPE'] == 'Provincie':

        vector_store = FAISS.load_local(os.path.join(current_dir, 'vector_stores', 'provincie_vector_store'),
                                        embeddings)

    elif dictionary_with_decomposed_question['SCOPE'] == 'Nederland':
        location_to_return = 'Nederland'
        vector_store = FAISS.load_local(os.path.join(current_dir, 'vector_stores', 'land_vector_store'), embeddings)

    elif dictionary_with_decomposed_question['SCOPE'] == 'Straatnaam':
        vector_store = FAISS.load_local(os.path.join(current_dir, 'vector_stores', 'straat_vector_store'), embeddings)


    if meta_data_filtering_for_few_shot_examples == False:
        vector_store = FAISS.load_local(os.path.join(current_dir, 'vector_stores','combined_vector_store'), embeddings)

    vector_store.embedding_function = vector_store.embedding_function[0]

    # You can use the following functions for different example selection mechanisms:
    # SemanticSimilarityExampleSelector
    # MaxMarginalRelevanceExampleSelector
    example_selector = None
    if example_retrieval_style == 'max_marginal_similarity':
        example_selector = MaxMarginalRelevanceExampleSelector(
            vectorstore=vector_store,
            k=number_of_few_shot_examples
        )
    elif example_retrieval_style == 'semantic_similarity':
        example_selector = SemanticSimilarityExampleSelector(
            vectorstore=vector_store,
            k=number_of_few_shot_examples
        )
    # print(example_selector.vectorstore.index.ntotal)

    selected_examples = None
    if retrieve_examples_with_question_without_location == True:
        search_query = {"query": dictionary_with_decomposed_question['QUESTION-NO-ADDRESS']}
        selected_examples = example_selector.select_examples(search_query)
    else:
        search_query = {"query": dictionary_with_decomposed_question['ORIGINAL-QUESTION']}
        selected_examples = example_selector.select_examples(search_query)

    example_prompt = PromptTemplate(
        input_variables=["QUESTION", "SPARQL"],
        template="Question: {QUESTION}\nQuery: {SPARQL}",
    )
    full_example_prompt = FewShotPromptTemplate(
        # We provide an ExampleSelector instead of examples.
        examples=selected_examples,
        example_prompt=example_prompt,
        suffix="Question: {question}\nQuery:",
        input_variables=["question"],
    )
    print(full_example_prompt.format(question=dictionary_with_decomposed_question['ORIGINAL-QUESTION']))
    print()

    print('Processing API call to location server to retrieve location identifier..')
    print()
    targeted_search_suggest_url = location_and_granularity_object.get_suggest_url(
        dictionary_with_decomposed_question['SCOPE'])
    response_location_server = None
    # depending on the scope we need to format the search URL in a different way
    if dictionary_with_decomposed_question['SCOPE'] == 'Straatnaam':
        # if the scope/granularity is about streets, we format our location such that we have %20 instead of space
        # this way we can change the search url to make it search for street & Woonplaats at the same time.
        # otherwise we cannot identify a street and woonplaats at the same time
        response_location_server = location_and_granularity_object.retrieve_location_data(
            dictionary_with_decomposed_question['LOCATION'].replace(" ", "%20"),
            targeted_search_suggest_url)
    elif dictionary_with_decomposed_question['SCOPE'] == 'Nederland':
        reponse_location_server = None
    else:
        # if we don't consider streets we can directly format the single location into the URL that performs the search
        response_location_server = location_and_granularity_object.retrieve_location_data(
            dictionary_with_decomposed_question['LOCATION'], targeted_search_suggest_url)

    messages = []
    system_string = ""
    location_to_return = None
    # we have an if-statement determining whether we want to include location information.
    if response_location_server is not None:

        if dictionary_with_decomposed_question['SCOPE'] == 'Provincie':
            location_to_return = response_location_server["provinciecode"].replace("PV", "00")
            system_string = """
            Please generate SPARQL output based on the natural language questions.
            Do not use filter on "Woonplaats".
            Only generate a SPARQL query for the last question.
            Only respond with a SPARQL query, no additional comments.
            
            When performing COUNT() operations always include DISTINCT INSIDE OF IT: COUNT( DISTINCT )
    
            If you're filtering based on the count of values in a HAVING clause, 
            make sure to use the aggregate function directly in the HAVING clause rather than referencing an aliased variable.

            All the queries that you generate contain a location code (with numbers or numbers with letters).
            Do not make up your own codes or descriptions. Always use the code or description that i give you in this prompt:
            %s
            Write a query where the location code is properly used in the following triple:
            "?gemeente geo:sfWithin provincie:location code."

            You are provided with some examples: 

            """ % ("provinciecode: " + response_location_server["provinciecode"].replace("PV", "00"))


        elif dictionary_with_decomposed_question['SCOPE'] == 'Gemeente':
            location_to_return = response_location_server["gemeentecode"]
            system_string = """
            Please generate SPARQL output based on the natural language questions.
            Do not use filter on "Woonplaats".
            Only generate a SPARQL query for the last question.
            Only respond with a SPARQL query, no additional comments.
            
            When performing COUNT() operations always include DISTINCT INSIDE OF IT: COUNT( DISTINCT )
    
            If you're filtering based on the count of values in a HAVING clause, 
            make sure to use the aggregate function directly in the HAVING clause rather than referencing an aliased variable.

            All the queries that you generate should contain a location code (with numbers or numbers with letters).
            Do not make up your own codes or descriptions. Always use the following location code in your query:
            %s
            Write a query where the location code is properly linked as a object to the "sdo0:identifier" predicate.

            You are provided with some examples: 

            """ % ("gemeentecode: " + 'GM' + response_location_server["gemeentecode"])




        elif dictionary_with_decomposed_question['SCOPE'] == 'Woonplaats':
            location_to_return = response_location_server["woonplaatsnaam"]
            system_string = """
            Please generate SPARQL output based on the natural language questions.
            Do not use filter on "Woonplaats".
            Only generate a SPARQL query for the last question.
            Only respond with a SPARQL query, no additional comments.
            
            When performing COUNT() operations always include DISTINCT INSIDE OF IT: COUNT( DISTINCT )
    
            If you're filtering based on the count of values in a HAVING clause, 
            make sure to use the aggregate function directly in the HAVING clause rather than referencing an aliased variable.

            All the queries that you generate contain a location description (it's name).
            Do not make up your own codes or descriptions. Always use the following code or description that i give you in this prompt:
            %s
            Make sure you use the location name properly in the following triple:
            "skos:prefLabel "location_name"@nl;"


            You are provided with some examples: 

            """ % ("woonplaatsnaam: " + response_location_server["woonplaatsnaam"])


        else:
            location_information_filtered = "woonplaatsnaam: " + response_location_server[
                "woonplaatsnaam"] + "\n" + "straatnaam: " + response_location_server["straatnaam"]

            system_string = """
            Please generate SPARQL output based on the natural language questions.
            Do not use filter on "Woonplaats".
            Do not use a filter on "huisnummer"
            Only generate a SPARQL query for the last question.
            Only respond with a SPARQL query, no additional comments.

            All the queries that you generate should contain a woonplaats description AND a streetname description.
            Do not make up your own descriptions. Always use the description that i give you in this prompt:
            %s
            Make sure you use the location name properly in the following triple linked to a woonplaats:
            "skos:prefLabel "location_name"@nl;"
            And make sure you use the streetname properly in the following triple linked to openbareruimte:
            "skos:prefLabel "Binnenhaven"@nl"


            You are provided with some examples: 

            """ % (location_information_filtered)

        # Here we combine the partial prompt with a prefix we put in a SystemMessage.
        # To use ChatModels in langchain we need the prompt in this format.
        messages = [
            SystemMessage(
                content=system_string
            ),
            # THIS NEEDS TO BE CHANGED STILL!
            HumanMessage(
                content=str(full_example_prompt.format(question=question))
            )
        ]
    else:
        location_to_return = 'Nederland'
        system_string = """
        Please generate SPARQL output based on the natural language questions.
        Only generate a SPARQL query for the last question.
        Only respond with a SPARQL query, do not put any other comments in your response.
        
        When performing COUNT() operations always include DISTINCT INSIDE OF IT: COUNT( DISTINCT )
    
        If you're filtering based on the count of values in a HAVING clause, 
        make sure to use the aggregate function directly in the HAVING clause rather than referencing an aliased variable.

        You are provided with some examples.
        Follow the examples closely. Do not deviate when not necessary.

        """

        # Here we combine the partial prompt with a prefix we put in a SystemMessage.
        # To use ChatModels in langchain we need the prompt in this format.
        messages = [
            SystemMessage(
                content=system_string
            ),
            # THIS NEEDS TO BE CHANGED STILL!
            HumanMessage(
                content=str(full_example_prompt.format(question=question))
            )
        ]

    print("Prompt sent to OpenAI:")
    for i in messages:
        print(i.content)
    print("\n")

    response = llm(messages)
    generated_query = response.content
    print("These are the unrevised results:")
    print(generated_query)

    return {'query': generated_query, 'granularity': dictionary_with_decomposed_question['SCOPE'],
            'location': location_to_return}

def only_get_condensed_ontologies(question, k_shortest_routes):
    # llm = AzureChatOpenAI(deployment_name="gpt-35-turbo", model_name="gpt-35-turbo", verbose=False, max_retries=2,
    #                       temperature=0, openai_api_version="2023-05-15")

    llm = AzureChatOpenAI(azure_deployment="gpt-4-32k", model_name="gpt-4-32k", verbose=False, max_retries=2,
                          temperature=0, openai_api_version="2023-05-15")

    print('Processing concurrent API calls to select relevant schema items and extract location..')
    print()
    # We create an object which stores some useful functions we will use later
    location_and_granularity_obj = location_and_granularity_object()

    # we retrieve the 3 prompt we want to execute concurrently
    retrieve_class_prompt, retrieve_relation_prompt = get_prompt_for_selecting_relevant_schema_items_based_on_natural_language_question(
        question)
    extract_location_prompt = location_and_granularity_obj.get_prompt_for_location_aggregation_and_questionWithoutLocation(
        question)

    # we process the 3 prompts concurrently
    class_response, relation_response, location_extractor_response = concurrent_executor(
        [retrieve_class_prompt, retrieve_relation_prompt, extract_location_prompt], llm=llm)

    # now we define some functions which process the responses of the llm
    def process_schema_response(list_as_string):
        '''
        We give it an LLM response which is a list formatted as a string.
        We convert it to an actual list object in python for further processing.
        :param list_as_string: the list as a string
        :return: an actual list retrieved from the string representation
        '''
        response_as_list = ast.literal_eval(list_as_string)
        return response_as_list

    def process_location_response_and_return_dictionary(response, question):
        '''
        We take the LLM response and turn it into a dictionary for further processing.



        :param response: the LLM response we want to process
        :param question: the natural language question (we also put it in the dictionary.)
        :return: A dictionary with the following keys: LOCATION, SCOPE, QUESTION-NO-ADRESS, ORIGINAL-QUESTION
        '''
        info_dict = {}

        # Split the input string into lines
        lines = response.strip().split("\n")

        # Iterate through the lines and extract information
        for line in lines:
            key, value = line.split(": ", 1)  # Split each line at the first ": " occurrence
            key = key.strip()
            info_dict[key] = value
        info_dict['ORIGINAL-QUESTION'] = question

        return info_dict

    # We use our functions to actually transform the LLM response
    # List with selected classes
    list_with_selected_classes = process_schema_response(class_response)
    # List with selected relations
    list_with_selected_relations = process_schema_response(relation_response)
    # Dictionary with decomposed question
    dictionary_with_decomposed_question = process_location_response_and_return_dictionary(
        location_extractor_response,
        question)
    print('SELECTED CLASSES:')
    print(list_with_selected_classes)
    print('SELECTED RELATIONS')
    print(list_with_selected_relations)
    print(dictionary_with_decomposed_question)
    # We take the classes of the retrieved location and add it to our list of selected classes
    if dictionary_with_decomposed_question['SCOPE'] == 'Woonplaats':
        list_with_selected_classes.append('sor:Woonplaats')
    elif dictionary_with_decomposed_question['SCOPE'] == 'Gemeente':
        list_with_selected_classes.append('wbk:Gemeente')
    elif dictionary_with_decomposed_question['SCOPE'] == 'Provincie':
        list_with_selected_classes.append('sor:Provincie')
    elif dictionary_with_decomposed_question['SCOPE'] == 'Nederland':
        pass
    elif dictionary_with_decomposed_question['SCOPE'] == 'Straatnaam':
        list_with_selected_classes.append('sor:Woonplaats')
        list_with_selected_classes.append('sor:OpenbareRuimte')

    print('Condensing ontology based on relevant classes and relations..')
    print()
    # we create this object which stores a lot of information about our ontology and also a graph network representation
    network_ontology_store = network_and_ontology_store(k_shortest_routes)

    ontology_string_object_part, ontology_string_datatype_part = network_ontology_store.condense_ontology_based_on_retrieved_schema_items(
        list_with_selected_classes, list_with_selected_relations)

    ontology_string_object_part_naive, ontology_string_datatype_part_naive = network_ontology_store.naive_ontology_selection(
        list_with_selected_classes, list_with_selected_relations)

    size_condensed_ontology = (ontology_string_object_part.count('\n') + 1) + (ontology_string_datatype_part.count('\n') + 1)

    return ontology_string_object_part, ontology_string_datatype_part, ontology_string_object_part_naive, ontology_string_datatype_part_naive, size_condensed_ontology


def semantic_parsing_workflow_few_shot_INCLUDE_ONTOLOGY_FOR_IN_CONTEXT_EXAMPLES(question,
                                       k_shortest_routes,
                                       number_of_few_shot_examples=5,
                                       meta_data_filtering_for_few_shot_examples=True,
                                       retrieve_examples_with_question_without_location=True,
                                       example_retrieval_style='max_marginal_similarity',
                                       verbose=False,
                                       naive_schema_linking_without_graph_traversal = False):
    # llm = AzureChatOpenAI(deployment_name="gpt-35-turbo", model_name="gpt-35-turbo", verbose=False, max_retries=2,
    #                       temperature=0, openai_api_version="2023-05-15")

    llm = AzureChatOpenAI(azure_deployment="gpt-4-32k", model_name="gpt-4-32k", verbose=False, max_retries=2,
                          temperature=0, openai_api_version="2023-05-15")
    if verbose == True:
        print('Processing concurrent API calls to select relevant schema items and extract location..')
    # We create an object which stores some useful functions we will use later
    location_and_granularity_obj = location_and_granularity_object()

    # we retrieve the 3 prompt we want to execute concurrently
    retrieve_class_prompt, retrieve_relation_prompt = get_prompt_for_selecting_relevant_schema_items_based_on_natural_language_question(
        question)
    extract_location_prompt = location_and_granularity_obj.get_prompt_for_location_aggregation_and_questionWithoutLocation(
        question)

    # we process the 3 prompts concurrently
    class_response, relation_response, location_extractor_response = concurrent_executor(
        [retrieve_class_prompt, retrieve_relation_prompt, extract_location_prompt], llm=llm)

    # now we define some functions which process the responses of the llm
    def process_schema_response(list_as_string):
        '''
        We give it an LLM response which is a list formatted as a string.
        We convert it to an actual list object in python for further processing.
        :param list_as_string: the list as a string
        :return: an actual list retrieved from the string representation
        '''
        response_as_list = ast.literal_eval(list_as_string)
        return response_as_list

    def process_location_response_and_return_dictionary(response, question):
        '''
        We take the LLM response and turn it into a dictionary for further processing.



        :param response: the LLM response we want to process
        :param question: the natural language question (we also put it in the dictionary.)
        :return: A dictionary with the following keys: LOCATION, SCOPE, QUESTION-NO-ADRESS, ORIGINAL-QUESTION
        '''
        info_dict = {}

        # Split the input string into lines
        lines = response.strip().split("\n")

        # Iterate through the lines and extract information
        for line in lines:
            key, value = line.split(": ", 1)  # Split each line at the first ": " occurrence
            key = key.strip()
            info_dict[key] = value
        info_dict['ORIGINAL-QUESTION'] = question

        return info_dict

    # We use our functions to actually transform the LLM response
    # List with selected classes
    list_with_selected_classes = process_schema_response(class_response)
    # List with selected relations
    list_with_selected_relations = process_schema_response(relation_response)
    # Dictionary with decomposed question
    dictionary_with_decomposed_question = process_location_response_and_return_dictionary(location_extractor_response,
                                                                                          question)
    if verbose == True:
        print(dictionary_with_decomposed_question)

        print('SELECTED CLASSES:')
        print(list_with_selected_classes)
        print('SELECTED RELATIONS:')
        print(list_with_selected_relations)
        print()

        print('Retrieving few-shot examples based on granularity..')
    # We select the appropriate vector_store with training data based on the granularity
    # We also take the classes of the retrieved location and add it to our list of selected classes

    embeddings = AzureOpenAIEmbeddings(
        azure_deployment='text-embedding-ada-002',
        openai_api_version=os.environ['OPENAI_API_VERSION'],
        # chunk_size=1
    ),
    vector_store = None

    if dictionary_with_decomposed_question['SCOPE'] == 'Woonplaats':
        list_with_selected_classes.append('sor:Woonplaats')
        os.path.join(current_dir, '/vector_stores/woonplaats_vector_store')
        vector_store = FAISS.load_local(
            os.path.join(current_dir,  'vector_stores',  'woonplaats_vector_store'),
            embeddings)

    elif dictionary_with_decomposed_question['SCOPE'] == 'Gemeente':
        list_with_selected_classes.append('wbk:Gemeente')
        vector_store = FAISS.load_local(
            os.path.join(current_dir,  'vector_stores',  'gemeente_vector_store'), embeddings)

    elif dictionary_with_decomposed_question['SCOPE'] == 'Provincie':
        list_with_selected_classes.append('sor:Provincie')
        vector_store = FAISS.load_local(
            os.path.join(current_dir, 'vector_stores',  'provincie_vector_store'),
            embeddings)

    elif dictionary_with_decomposed_question['SCOPE'] == 'Nederland':
        vector_store = FAISS.load_local(
            os.path.join(current_dir,  'vector_stores', 'land_vector_store'), embeddings)

    if meta_data_filtering_for_few_shot_examples == False:
        vector_store = FAISS.load_local(
            os.path.join(current_dir, 'vector_stores',  'combined_vector_store'),
            embeddings)

    vector_store.embedding_function = vector_store.embedding_function[0]

    # You can use the following functions for different example selection mechanisms:
    # SemanticSimilarityExampleSelector
    # MaxMarginalRelevanceExampleSelector
    example_selector = None
    if example_retrieval_style == 'max_marginal_similarity':
        example_selector = MaxMarginalRelevanceExampleSelector(
            vectorstore=vector_store,
            k=number_of_few_shot_examples
        )
    elif example_retrieval_style == 'semantic_similarity':
        example_selector = SemanticSimilarityExampleSelector(
            vectorstore=vector_store,
            k=number_of_few_shot_examples
        )
    # print(example_selector.vectorstore.index.ntotal)

    selected_examples = None
    if retrieve_examples_with_question_without_location == True:
        search_query = {"query": dictionary_with_decomposed_question['QUESTION-NO-ADDRESS']}
        selected_examples = example_selector.select_examples(search_query)
    else:
        search_query = {"query": dictionary_with_decomposed_question['ORIGINAL-QUESTION']}
        selected_examples = example_selector.select_examples(search_query)

    if verbose == True:
        print('Condensing ontology based on relevant classes and relations..')
        print()

    # We condense the ontology for the questions asked during inference
    network_ontology_store = network_and_ontology_store(k_shortest_routes)
    if naive_schema_linking_without_graph_traversal == False:
        ontology_string_object_part_inference, ontology_string_datatype_part_inference = network_ontology_store.condense_ontology_based_on_retrieved_schema_items(
            list_with_selected_classes, list_with_selected_relations)
    else:
        ontology_string_object_part_inference, ontology_string_datatype_part_inference = network_ontology_store.naive_ontology_selection(
            list_with_selected_classes, list_with_selected_relations)



    for example in selected_examples:
        retrieve_class_prompt, retrieve_relation_prompt = get_prompt_for_selecting_relevant_schema_items_based_on_natural_language_question(
            example['QUESTION'])

        class_response, relation_response = concurrent_executor(
            [retrieve_class_prompt, retrieve_relation_prompt], llm=llm)
        # We use our functions to actually transform the LLM response
        # List with selected classes
        list_with_selected_classes = process_schema_response(class_response)
        # List with selected relations
        list_with_selected_relations = process_schema_response(relation_response)
        if naive_schema_linking_without_graph_traversal == False:
            ontology_string_object_part, ontology_string_datatype_part = network_ontology_store.condense_ontology_based_on_retrieved_schema_items(
                list_with_selected_classes, list_with_selected_relations)
        else:
            ontology_string_object_part, ontology_string_datatype_part = network_ontology_store.naive_ontology_selection(
                list_with_selected_classes, list_with_selected_relations)

        # we manually add these for completeness (although not strictly required)
        if 'wbk:Gemeente' in example['INSTANCE_DATA_REGARDING_LOCATION']:
            if 'wbk:Gemeente sdo0:identifier xsd:string' not in ontology_string_datatype_part:
                ontology_string_datatype_part = ontology_string_datatype_part + '\n' + 'wbk:Gemeente sdo0:identifier xsd:string'

        elif 'provincie' in example['INSTANCE_DATA_REGARDING_LOCATION']:
            if 'sor:Gemeente geo:sfWithin sor:Provincie' not in ontology_string_object_part:
                ontology_string_object_part = ontology_string_object_part + '\n' + 'sor:Gemeente geo:sfWithin sor:Provincie'
        elif 'sor:Woonplaats' in example['INSTANCE_DATA_REGARDING_LOCATION']:
            if 'sor:Woonplaats skos:prefLabel rdf:langString' not in ontology_string_datatype_part:
                ontology_string_datatype_part = ontology_string_datatype_part + '\n' + 'sor:Woonplaats skos:prefLabel rdf:langString'
        example['OBJECT/CLASS_NAVIGATION_PATH_ONTOLOGY'] = ontology_string_object_part
        example['DATATYPE_ONTOLOGY'] = ontology_string_datatype_part

    if verbose == True:
        print('Processing API call to location server to retrieve location identifier..')
        print()
    targeted_search_suggest_url = location_and_granularity_object.get_suggest_url(
        dictionary_with_decomposed_question['SCOPE'])
    response_location_server = None
    # depending on the scope we need to format the search URL in a different way
    if dictionary_with_decomposed_question['SCOPE'] == 'Straatnaam':
        # if the scope/granularity is about streets, we format our location such that we have %20 instead of space
        # this way we can change the search url to make it search for street & Woonplaats at the same time.
        # otherwise we cannot identify a street and woonplaats at the same time
        response_location_server = location_and_granularity_object.retrieve_location_data(
            dictionary_with_decomposed_question['LOCATION'].replace(" ", "%20"),
            targeted_search_suggest_url)
    elif dictionary_with_decomposed_question['SCOPE'] == 'Nederland':
        reponse_location_server = None
    else:
        # if we don't consider streets we can directly format the single location into the URL that performs the search
        response_location_server = location_and_granularity_object.retrieve_location_data(
            dictionary_with_decomposed_question['LOCATION'], targeted_search_suggest_url)

    # We create instance triples which which specify the desired location.
    # Note that by adding such triples we explain to the model where in the ontology structure this data resides
    # This instance data represents the location that is mentioned in the question.
    # NOTE: I also manually create some ontology triples here.
    # Later we check if these ontology triples are in our condensed ontology for completeness. (this is not crucial)
    # This is basically the ontology triples (often datatype edge) that precisely represents
    # the same part of the instance triple.
    # Due to the mechanism of the graph traversal algoritm we might get the entire route except the
    # endpoint which points to the datatype. So i make sure it is added for completeness.
    instance_prompt_string = ""
    corresponding_datatype_ontology_triples_list = []
    corresponding_object_ontology_triples_list = []
    location_to_return = None
    if dictionary_with_decomposed_question['SCOPE'] == 'Woonplaats':
        location_to_return = response_location_server["woonplaatsnaam"]
        instance_triple_1 = """sor:Woonplaats skos:prefLabel "%s"@nl""" % (response_location_server["woonplaatsnaam"])

        instance_prompt_string = """
        You are provided with a triple of the form: (class relation instancedata)
        You should interpret this as: there are instances of sor:Woonplaats that have the prefLabel
        mentioned in the triple. 

        ALWAYS USE THIS PREFLABEL IN YOUR QUERY!
        \n
        """
        instance_prompt_string = instance_prompt_string + instance_triple_1
        corresponding_datatype_ontology_triples_list.append('sor:Woonplaats skos:prefLabel rdf:langString')

    elif dictionary_with_decomposed_question['SCOPE'] == 'Gemeente':
        location_to_return = response_location_server["gemeentecode"]
        instance_triple_1 = """wbk:Gemeente sdo0:identifier "%s" """ % ('GM' + response_location_server["gemeentecode"])
        instance_prompt_string = """
        You are provided with a triple of the form: (class relation instancedata)
        You should interpret this as: there are instances of wbk:Gemeente that have this identifier, and this
        identifier corresponds to the gemeente %s . 

        ALWAYS USE THIS CODE IN YOUR QUERY!
        \n
        """ % (dictionary_with_decomposed_question['LOCATION'])
        instance_prompt_string = instance_prompt_string + instance_triple_1
        corresponding_datatype_ontology_triples_list.append('wbk:Gemeente sdo0:identifier xsd:string')

    elif dictionary_with_decomposed_question['SCOPE'] == 'Provincie':
        location_to_return = response_location_server["provinciecode"].replace("PV", "00")
        instance_triple_1 = """sor:Gemeente geo:sfWithin provincie:%s""" % (
            response_location_server["provinciecode"].replace("PV", "00"))
        instance_prompt_string = """
        You are provided with a triple of the form: (class relation instancedata)
        You should interpret this as: there are instances of sor:Gemeente that are located in
        the province %s .
        ALWAYS USE THE PROVINCE CODE IN YOUR QUERY!
         \n
        """ % (dictionary_with_decomposed_question['LOCATION'])
        instance_prompt_string = instance_prompt_string + instance_triple_1
        corresponding_datatype_ontology_triples_list.append('sor:Gemeente geo:sfWithin sor:Provincie')

    elif dictionary_with_decomposed_question['SCOPE'] == 'Nederland':
        location_to_return = 'Nederland'
        pass

    # we check if the corresponding datatype ontology triples are in our condensed ontology, if not we add it
    for trip in corresponding_datatype_ontology_triples_list:
        if trip not in ontology_string_datatype_part:
            ontology_string_datatype_part = ontology_string_datatype_part + '\n' + trip
    # we check if the corresponding object ontology triples are in our condensed ontology, if not we add it
    for trip in corresponding_object_ontology_triples_list:
        if trip not in ontology_string_object_part:
            ontology_string_datatype_part = ontology_string_datatype_part + '\n' + trip

    example_prompt = PromptTemplate(
        input_variables=["QUESTION", "INSTANCE_DATA_REGARDING_LOCATION", "SPARQL",
                         "OBJECT/CLASS_NAVIGATION_PATH_ONTOLOGY", "DATATYPE_ONTOLOGY"],
        template="QUESTION: \n{QUESTION}\nINSTANCE DATA REGARDING LOCATION: \n{INSTANCE_DATA_REGARDING_LOCATION}\nOBJECT/CLASS NAVIGATION PATH ONTOLOGY:\n{OBJECT/CLASS_NAVIGATION_PATH_ONTOLOGY}\nDATA TYPE ONTOLOGY:\n{DATATYPE_ONTOLOGY}SPARQL: \n{SPARQL}",
    )
    full_example_prompt = FewShotPromptTemplate(
        # We provide an ExampleSelector instead of examples.
        examples=selected_examples,
        example_prompt=example_prompt,
        suffix="QUESTION: \n{QUESTION}\nINSTANCE DATA REGARDING LOCATION: \n{INSTANCE_DATA_REGARDING_LOCATION}\nOBJECT/CLASS NAVIGATION PATH ONTOLOGY:\n{OBJECT_CLASS_NAVIGATION_PATH_ONTOLOGY}\nDATA TYPE ONTOLOGY:\n{DATATYPE_ONTOLOGY}\nSPARQL:",
        input_variables=["QUESTION", "INSTANCE_DATA_REGARDING_LOCATION",
                         "OBJECT_CLASS_NAVIGATION_PATH_ONTOLOGY", "DATATYPE_ONTOLOGY"]
    )

    print(selected_examples)

    print(full_example_prompt.format(QUESTION=question,
                                     INSTANCE_DATA_REGARDING_LOCATION=instance_prompt_string,
                                     OBJECT_CLASS_NAVIGATION_PATH_ONTOLOGY=ontology_string_object_part_inference,
                                     DATATYPE_ONTOLOGY=ontology_string_datatype_part_inference))
    print()

    system_string = """
    You should act as a knowledge graph professional with excellent expertise in writing SPARQL queries.
    You will be provided with a schema of a knowledge graph in the geographical domain where the schema items
    are in Dutch. 

    You will be given a natural language question. Using the provided schema, and the examples provided to you,
    you should write SPARQL that best matches the natural language question. For each examplse we provide a separate
    schema and instance data.


    1)
    The section called "INSTANCE DATA REGARDING LOCATIONS" has triples which show position of certain instance
    data in the ontology. This instance data is always about geographical locations. this way you will be able to generate
    SPARQL regarding certain geographical locations.
    If the question is about the entire netherlands this place will be empty since you don't have
    to link it to certain locations.

    2)
    The section called "OBJECT/CLASS NAVIGATION PATH ONTOLOGY" contains lines where we see how classes are related to eachother
    through properties in the following format:
    class1 property1 class2

    -We define relationships between classes by specifying how they are linked to each other through properties.
    -In our ontology, the first class mentioned in a relationship represents the starting point, 
    while the second class represents the destination. This distinction is crucial for understanding the directionality of relationships.
    -By default, relationships are interpreted from left to right, with the first class acting as the subject and the second class as the object. 
    However, when using the SPARQL operator (^), relationships can be traversed from right to left, reversing the directionality.
    We provide examples demonstrating how to use the directionality in SPARQL.

    -When you apply the RELATION 'property1' to an instance of the CLASS 'class1', you take path towards
    and instance of CLASS 'class2'. SO IT TELLS YOU WHERE A RELATION PATH LEADS TO!

    -THESE PATHS ONLY GO IN ONE DIRECTION!
    IF YOU WANT TO NAVIGATE THE OTHER WAY AROUND USE THE "^" OPERATOR IN SPARQL.

    -The FIRST and THIRD words in such lines are CLASSES. USE THEM AS SUCH! DO NOT USE THEM AS RELATIONS!
    The SECOND word in such lines are RELATIONS. USE ONLY THOSE AS RELATIONS!

    -ONLY USE THE NAVIGATION PATHS THAT ARE DESCRIBED! THIS IS VERY IMPORTANT!
    YOU CANNOT SKIP INTERMEDIATE PATHS!

    -ALWAYS PRIORITIZE CLASSES, RELATIONS, AND NAVIGATION PATHS WITH THE "sor:" PREFIX IF THESE ARE AVAILABLE! 
    THESE SHOULD BE PREFERRED OVER 'bag:' and "kad:" IF POSSIBLE!

    -ALWAYS PRIORITIZE CLASSES, RELATIONS, AND NAVIGATION PATHS WITH THE "kad:" PREFIX over "bag:" 
    EXAMPLE: USE kad:Ligplaats  INSTEAD OF bag:Ligplaats !!



    WHEN USING CLASSES SPECIFICALLY DEFINE THE CLASSES THAT YOU USE, FOR EXAMPLE, LIKE THIS:
    ?variable a sor:Gebouw


    3)
    The section called "DATATYPE ONTOLOGY" contains lines where we see how classes are related to certain datatypes
    of instance data through properties using the following format:
    class1 property2 xsd:PositiveInteger

    Use this information: For example, take special care of variables of type xsd:gYear. 
    Ensure that any comparisons or operations involving ?bouwjaar are appropriate for this data type. 
    Additionally, consider any necessary adjustments to the query to accommodate the specific characteristics of xsd:gYear.
    A good option is to use the YEAR() function to extract the actual year.


    !!!!!
    DO NOT USE SCHEMA ITEMS/NAVIGATION PATHS THAT ARE: NOT IN THE ONTOLOGY I PROVIDED YOU WITH OR NOT IN THE EXAMPLES
    !!!!!

    USE COMPLETE RELATION PATH, DO NOT SKIP RELATIONS IN THE PATH!!

    hen performing COUNT() operations always include DISTINCT INSIDE OF IT: COUNT( DISTINCT )

    If you're filtering based on the count of values in a HAVING clause, 
    make sure to use the aggregate function directly in the HAVING clause rather than referencing an aliased variable.

    When constructing SPARQL queries, double-check the direction of properties in triple patterns to accurately navigate the relationships between entities. 
    Make sure to use the correct classes specified in the instance data.



    Only respond with a SPARQL query, do not put any other comments in your response.

    """

    if verbose == True:
        print(system_string)

        print('Processing API call to generate SPARQL..')
        print()
    # Here we combine the partial prompt with a prefix we put in a SystemMessage.
    # To use ChatModels in langchain we need the prompt in this format.
    messages = [
        SystemMessage(
            content=system_string
        ),
        # THIS NEEDS TO BE CHANGED STILL!
        HumanMessage(
            content=str(full_example_prompt.format(QUESTION=question,
                                                   INSTANCE_DATA_REGARDING_LOCATION=instance_prompt_string,
                                                   OBJECT_CLASS_NAVIGATION_PATH_ONTOLOGY=ontology_string_object_part,
                                                   DATATYPE_ONTOLOGY=ontology_string_datatype_part))
        )
    ]

    response = llm(messages)
    generated_query = response.content
    if verbose == True:
        print('Generated SPARQL')
        print(generated_query)

    return {'query': generated_query, 'granularity': dictionary_with_decomposed_question['SCOPE'],
            'location': location_to_return}


'''
BELANGRIJK:
-waarom stond bag:type niet in dictionary_that_maps_property_to_neighboring_nodes?
voor nu check ik gewoon of het uberhaupt in die dictionary staat.
'''
semantic_parsing_workflow_few_shot(
    'Hoeveel percelen zijn er die niet bij een nummeraanduiding horen?',
    1
)

# semantic_parsing_workflow_few_shot_NO_ONTOLOGY('Geef mij de nieuwste brandweerkazerne in gemeente eindhoven',
#                                                    number_of_few_shot_examples=5,
#                                                    meta_data_filtering_for_few_shot_examples=True,
#                                                    retrieve_examples_with_question_without_location=True,
#                                                    example_retrieval_style='max_marginal_similarity')

# semantic_parsing_workflow('Hoeveel percelen zijn er in provincie Gelderland?', 7)


