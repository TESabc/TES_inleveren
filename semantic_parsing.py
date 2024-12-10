from langchain.schema import (
    SystemMessage,
    HumanMessage
)
from langchain.prompts.example_selector import (
    MaxMarginalRelevanceExampleSelector,
    SemanticSimilarityExampleSelector,
)
from langchain_openai import AzureOpenAIEmbeddings
import ast
from langchain_community.vectorstores import FAISS
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
import configparser
import requests
from typing import List, Dict, Any
import os
from langchain_openai import AzureChatOpenAI
from concurrent.futures import ThreadPoolExecutor
from shortest_path_algorithms.shortest_path import network_and_ontology_store
import warnings

warnings.filterwarnings("ignore")

'''
SELECT THE GPT DEPLOYMENT YOU WANT TO USE.

options you can choose: OPENAI_GPT_3.5_TURBO     OPENAI_GPT_4_32K    OPENAI_GPT_4_TURBO
'''

gpt_option = 'OPENAI_GPT_4_32K'

'''
In this section, we load the secrets.ini file to retrieve essential configuration details, 
including API URLs and OpenAI keys.

The following information is extracted:

1) API URLs for the Location Server: 
Used to obtain International Resource Identifiers (IRIs) for specific locations in the Netherlands.
2) API URL for the Kadaster Knowledge Graph (KKG): 
Enables sending SPARQL queries and retrieving the corresponding results.
3) OpenAI Configuration: 
Includes the API key, version, and endpoint required to access an Azure-hosted Large Language Model (LLM) deployment.
'''

config = configparser.ConfigParser()
current_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(current_dir, 'secrets.ini')
config.read(path)

locatie_url = config['KADASTER']['LOC_SEARCH_API']
lookup_url = config['KADASTER']['LOC_LOOKUP_API']
sparql_url = config['KADASTER']['KKG_API']
os.environ['AZURE_OPENAI_API_KEY'] = config[gpt_option]['AZURE_API_KEY']
os.environ['OPENAI_API_VERSION'] = config[gpt_option]['AZURE_API_VERSION']
os.environ['AZURE_OPENAI_ENDPOINT'] = config[gpt_option]['AZURE_API_BASE']

'''
In this section, we define auxiliary methods that support the functionality of the rest of the file.
'''


def get_nested_value(o: dict, path: list) -> any:
    """
    Retrieves the value from a nested dictionary using a specified sequence of keys.

    This function traverses a nested dictionary based on the order of keys provided in the `path` list.
    If any key in the sequence is not found or if an error occurs during traversal, the function returns `None`.

    Args:
        o (dict): The nested dictionary to traverse.
        path (list): A list of keys specifying the traversal path within the dictionary.

    Returns:
        any: The value located at the end of the specified path, or `None` if the path cannot be fully traversed.

    Example:
        nested_dict = {"a": {"b": {"c": 42}}}
        path = ["a", "b", "c"]
        result = get_nested_value(nested_dict, path)  # Returns 42

        path = ["a", "x", "c"]
        result = get_nested_value(nested_dict, path)  # Returns None
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
    Executes a SPARQL query against a specified SPARQL endpoint and retrieves the results.

    Parameters:
    ----------
    query : str
        The SPARQL query string to be executed. Ensure the query is valid and formatted correctly.

    url : str, optional
        The URL of the SPARQL endpoint where the query will be sent.

    user_agent_header : str, optional
        An optional User-Agent header to include in the HTTP request. This can be used to specify
        the client making the request, aiding in analytics or access control.

    internal : bool, optional
        A flag that determines the behavior when an unsuccessful HTTP response is received:
        - If `False` (default), returns an error message as a string for unsuccessful responses.
        - If `True`, returns `None` for unsuccessful responses, allowing internal handling.

    Returns:
    -------
    List[Dict[str, Any]]
        A list of dictionaries representing the query results. Each dictionary corresponds to a
        result row, with variable names as keys.

        The value associated with each key is itself a dictionary containing metadata about the
        binding, including:
        - `type`: Indicates the type of the value (e.g., `uri`, `literal`, or `bnode`).
        - `value`: Contains the actual value of the binding (e.g., a URI, literal text, or blank node ID).


        Example:
        [
            {"var1": {"type": "uri", "value": "http://example.org"}},
            {"var1": {"type": "literal", "value": "Sample Text"}}
        ]
    """
    headers = {
        'Accept': 'application/sparql-results+json',
        'Content-Type': 'application/sparql-query'
    }
    if user_agent_header is not None:
        headers['User-Agent'] = user_agent_header

    response = requests.get(url, headers=headers, params={'query': query.replace("```", ""), 'format': 'json'})

    if response.status_code != 200 and internal == False:
        return "That query failed. Perhaps you could try a different one?"
    else:
        if response.status_code != 200 and internal == True:
            return None

    results = get_nested_value(response.json(), ['results', 'bindings'])

    return results


def concurrent_executor(messages_list, llm):
    '''
    This method allows the concurrent processing of multiple prompts to an LLM, substantially boosting speed
    through the usage of concurrency.

    Parameters:
    ----------
    messages_list : list
        A list of prompts to be processed concurrently by the LLM.

    llm : object
        The LLM object used to process the messages.

    Returns:
    -------
    tuple
        A tuple of LLM responses, one for each input message, in the same order.
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


"""
In this section, we define several methods for tasks such as Schema Selection, 
Span Classification, Type Classification, Entity Masking, and Entity Linking.
"""


def get_prompts_for_selecting_relevant_schema_items_based_on_natural_language_question(natural_language_question,
                                                                                       ):
    """
    Generates prompts for retrieving relevant schema items (classes and properties) from KKG
    based on a natural language question.

    This method creates two distinct prompts:
    1. A prompt for identifying relevant **classes** in the KKG that are likely to be used in a SPARQL query
       corresponding to the given natural language question.
    2. A prompt for identifying relevant **properties** (relations) in the KG needed for the SPARQL query.

    These prompts guide the LLM (e.g., GPT) to select schema items from a provided ontology.

    Args:
        natural_language_question (str): A natural language question for which schema items need to be identified.

    Returns:
        tuple: A pair of message lists (`class_messages`, `relation_messages`), each containing prompts tailored
               for retrieving relevant classes and properties, respectively. These messages can be sent to an
               LLM for processing.
    """
    prompt_string_template_classes = """
    You will be given all the classes from 
    an ontology belonging to a knowledge
    graph along with some entities. You will 
    also be provided with a question
    in natural language. 


    Your job is to select the top classes 
    and entities that you expect to be
    needed to write a SPARQL query 
    corresponding to the natural language 
    question.

    ONLY SELECT CLASSES AND ENTITIES THAT
    ARE IN THE LIST I PROVIDE TO YOU.
    BE VERY CAREFUL TO NOT MAKE UP PREFIXES!
    THE INCLUDED PREFIXES ARE THE ONLY ONES 
    THAT ARE PERMITTED.

    ALWAYS RETRIEVE THE RELEVANT 'sor:'
    CLASSES AT LEAST! ALSO RETRIEVE 
    SCHEMA ITEMS WITH OTHER PREFIXES,
    BUT NEVER ONLY RETRIEVE 'bag:' SCHEMA
    ITEMS WHILE NOT ALSO SELECTING THE
    CORRESPONDING 'sor:' SCHEMA ITEMS! 
    FOR EXAMPLE, IF YOU SELECT 
    'bag:Woonplaats' ALWAYS ALSO SELECT
    'sor:Woonplaats'. BUT ONLY IF THE
    'sor:' VERSION IS IN THE LIST I 
    SUPPLIED TO YOU!

    Give your answer in a Python list
    format as following:
    ['schema_item1','schema_item2', 
    'schema_item3']
    Make sure to enclose the items 
    within the list with quotation marks 
    as I demonstrated. Do not include
    anything else in your answer.



    You are given some examples.


    CLASSES AND ENTITIES:
    %s
    --------------------------
    EXAMPLES:
    %s
    ---------------------------
    QUESTION IN NATURAL LANGUAGE:
    %s
    """

    prompt_string_template_properties = """
    You will be given all the properties from 
    an ontology belonging to a knowledge
    graph. You will also be provided with a 
    question in natural language.

    Your job is to select the top 
    properties that you expect to be
    needed to write a SPARQL query 
    corresponding to the natural 
    language question.

    ONLY SELECT PROPERTIES THAT ARE IN THE
    LIST I PROVIDE TO YOU. BE VERY CAREFUL
    TO NOT MAKE UP PREFIXES! THE INCLUDED 
    PREFIXES ARE THE ONLY ONES THAT ARE
    PERMITTED.

    ALWAYS AT LEAST RETRIEVE THE RELEVANT
    'sor:' PROPERTIES! ALSO RETRIEVE 
    SCHEMA ITEMS WITH OTHER PREFIXES,
    BUT NEVER ONLY RETRIEVE 'bag:' SCHEMA
    ITEMS WHILE NOT ALSO SELECTING THE
    CORRESPONDING 'sor:' SCHEMA ITEMS! 
    FOR EXAMPLE, IF YOU SELECT 
    'bag:Woonplaats' ALWAYS ALSO SELECT
    'sor:Woonplaats'. BUT ONLY IF THE
    'sor:' VERSION IS IN THE LIST I 
    SUPPLIED TO YOU!

    Give your answer in a Python list 
    format as following:
    ['schema_item1','schema_item2', 
    'schema_item3']
    Make sure to enclose the items 
    within the list with quotation marks 
    as I demonstrated. Do not include 
    anything else in your answer.



    You are given some examples.


    PROPERTIES:
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

    class_query = prompt_string_template_classes % (
        network_ontology_store.classes_without_comments_string, examples_class,
        natural_language_question)

    relation_query = prompt_string_template_properties % (
        network_ontology_store.relations_without_comments_string, examples_relation,
        natural_language_question)

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

    return class_messages, relation_messages


class location_and_granularity_object:
    """
    This class provides several methods to facilitate tasks such as Span Classification,
    Type Classification, Entity Masking, and Entity Linking.
    """

    def __init__(self):
        pass

    @staticmethod
    def get_prompt_for_location_aggregation_and_questionWithoutLocation(question):
        '''
        Constructs a prompt for simultaneous execution of Span Classification, Type Classification, and Entity Masking
        tasks based on a given question.

        The prompt guides the model to:
        1. Identify and extract the location mentioned in the question, correcting any spelling mistakes.
           If no location is found, it should default to "Nederland".
        2. Classify the location as one of the following categories: "Woonplaats", "Gemeente", "Provincie", or "Nederland".
           The classification depends on how the location is referenced in the question.
        3. Reformulate the question by removing the location, ensuring it is in correct Dutch, and eliminating any spelling mistakes.

        The resulting prompt will instruct the model to provide the following answers:
        - **LOCATION**: The extracted location.
        - **SCOPE**: The classification of the location (Woonplaats, Gemeente, Provincie, Nederland).
        - **QUESTION-NO-ADDRESS**: The reformulated question with the location removed.

        Examples of correct execution and the required answer format are provided in the prompt.

        Parameters:
        ----------
        question : str
            The input question to be processed for location extraction, classification, and reformulation.

        Returns:
        -------
        list
            A list containing a `HumanMessage` object with the formatted prompt for the model, ready to be used for further processing.
        '''

        prompt_string_template = """
        You are given a question.
        Your task is to determine whether the
        question refers to a municipality,
        province, woonplaats, or the entirety 
        of the Netherlands. You also need to 
        extract the location and reformulate 
        the question to exclude any mention
        of the location.
        Your tasks are as following:
        
        1) extract a location from the question
        and write it behind "LOCATION:" in 
        your answer. Remove any spelling 
        mistakes in the location. If no location
        is provided you should say that 
        the location is "Nederland".
        2) You should classify this location as 
        being either a "Woonplaats", "Gemeente",
        "Provincie", or "Nederland". Only 
        strictly use these classifications.
        Note that if something is a Gemeente or 
        Provincie the question will typically
        specify that it belongs to this category 
        by directly mentioning this in the
        question. If nothing is mentioned you 
        should classify it as a Woonplaats. Note 
        that if a question mentions a 
        "gemeentehuis", this does not mean the
        scope is gemeente! This has nothing to 
        do with scope.
        If the previously extracted location 
        is "Nederland," the scope should also
        be classified as "Nederland."
        Put the answer to this part behind 
        "SCOPE:" in your answer.
        3) You should reformulate the question 
        where you remove the location from the 
        question. Write this is Dutch!
        Put the answer to this part behind
        "QUESTION-NO-ADDRESS". Remove any 
        spelling mistakes that are in the 
        original question.
        
        
        You are provided with some examples of 
        correct execution of your task and also 
        the answer format that is required:
        
        
        ----------------------------------------
        Here are some examples:
        Question: Hoeveel brandweertorens zijn
        er in Ugchelen gebouwd na 1980?
        LOCATION: Ugchelen
        SCOPE: Woonplaats
        QUESTION-NO_ADDRESS: Hoeveel 
        brandweertorens zijn gebouwd na 1980?
        
        Question: Hoeveel kerken kan ik vinden
        in Groningen?
        LOCATION: Groningen
        SCOPE: Woonplaats
        QUESTION-NO_ADDRESS: Hoeveel kerken kan
        ik vinden?
        
        Question: Hoeveel ziekenhuizen staan er
        in provincie Groningen?
        LOCATION: Groningen
        SCOPE: Provincie
        QUESTION-NO-ADDRESS: Hoeveel ziekenhuizen
        staan er?
        
        Question: Geef mij alle gemeentehuizen in
        hellevoetsluis.
        LOCATION: Hellevoetsluis
        SCOPE: Woonplaats
        QUESTION-NO-ADDRESS: Geef mij alle 
        gemeentehuizen.
        
        Question: Hoeveel ziekenhuizen staan
        er in Groningen?
        LOCATION: Groningen
        SCOPE: Woonplaats
        QUESTION-NO-ADDRESS: Hoeveel 
        ziekenhuizen staan er?
        
        Question: Hoeveel ziekenhuizen 
        zijn er in Nederland?
        LOCATION: Nederland
        SCOPE: Nederland
        QUESTION-NO-ADDRESS: Hoeveel 
        ziekenhuizen zijn er?
        
        Question: Hoeveel gevangenissen 
        zijn er?
        LOCATION: Nederland
        SCOPE: Nederland
        QUESTION-NO-ADDRESS: Hoeveel 
        gevangenissen zijn er?
        
        Question: Hoeveel begraafplaatsen 
        zijn er in gemeente Utrecht?
        LOCATION: Utrecht
        SCOPE: Gemeente
        QUESTION-NO-ADDRESS: Hoeveel 
        begraafplaatsen zijn er?
        
        Question: Geef mij het grootste
        gemeentehuis in groningen.
        LOCATION: Groningen
        SCOPE: Woonplaats
        QUESTION-NO-ADDRESS: Geef mij het
        grootste gemeentehuis.
        
        
        -----------------------------------
        Question: {q}
        LOCATION:
        SCOPE: 
        QUESTION-NO-ADDRESS: 
        """
        messages = [
            HumanMessage(
                content=prompt_string_template.format(q=question)
            )

        ]
        return messages

    @staticmethod
    def get_suggest_url(granularity):
        '''
        Returns the location server API URL for performing a filtered search based on the specified granularity.

        This method generates a URL for the location server API based on the granularity level provided. The granularity
        determines the type of location to search for (e.g., Woonplaats, Gemeente, Provincie), and the method returns a
        corresponding URL with the appropriate filter applied. If an invalid granularity is provided, an exception is raised.

        Parameters:
        ----------
        granularity : str
            A string representing the level of location granularity. Valid values are:
            - "Woonplaats" (for cities)
            - "Gemeente" (for municipalities)
            - "Provincie" (for provinces)
            - "Nederland" (for the entire Netherlands)

        Returns:
        -------
        str or None
            The URL to query the location server API with the appropriate filter based on the granularity.
            If the granularity is "Nederland", `None` is returned.

        Raises:
        ------
        ValueError
            If an invalid granularity is provided.
        '''
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
        else:
            raise ValueError("Invalid granularity provided: {}".format(granularity))
        return location_server_url

    @staticmethod
    def retrieve_location_data(location, SUGGEST_URL):
        """
        Retrieves location data from the location server API based on the provided location and URL.

        This method sends a request to the location server API using the provided location and the pre-filtered
        location server URL (which includes the appropriate granularity). It retrieves the corresponding IRI (International
        Resource Identifier) and other location-related information from the API response.

        Parameters:
        ----------
        location : str
            The location search query for which the IRI needs to be identified.

        SUGGEST_URL : str
            The pre-configured URL for the location server API, which includes the necessary filters based on the granularity
            (e.g., Woonplaats, Gemeente, etc.). The URL should have a placeholder for the location (e.g., "{}").

        Returns:
        -------
        dict
            A dictionary containing the location data, including the IRI and other related information, extracted from the API response.
        """
        url = SUGGEST_URL.format(location)
        address = requests.get(url).json()['response']['docs'][0]

        return address

"""
In the next section we provide several methods for performing semantic parsing with the full configuration in our paper
and also for the various ablations.
"""

def semantic_parsing_without_in_context_examples(question, k_shortest_routes, verbose=False):
    """
    This method is for the ablation where we exclude in-context examples for semantic parsing entirely.

    It accepts a natural language question and generates the corresponding SPARQL query.

    This method performs all necessary steps for semantic parsing, including the selection of relevant schema items and
    ontology condensation. It also handles span classification,
    type classification, entity masking, and entity linking.

    Parameters:
    ----------
    question : str
        A natural language question for which the corresponding SPARQL query needs to be generated.

    k_shortest_routes : int
        The number of shortest routes to consider between relevant nodes for ontology condensation during
        the execution of the GTOC algorithm.

    verbose : boolean
        A boolean indicating whether (intermediate) results should be printed during execution.

    Returns:
    -------
    dict
        A dictionary containing:
        - 'query': The generated SPARQL query based on the input question and ontology schema.
        - 'granularity': The level of granularity for the location (e.g., Woonplaats, Gemeente).
        - 'location': The location identified from the question, based on the extracted geographical context.
    """
    # llm = AzureChatOpenAI(deployment_name="gpt-35-turbo", model_name="gpt-35-turbo", verbose=False, max_retries=2,
    #                       temperature=0, openai_api_version="2023-05-15")

    llm = AzureChatOpenAI(azure_deployment="gpt-4-32k", model_name="gpt-4-32k", verbose=False, max_retries=2,
                          temperature=0, openai_api_version="2023-05-15")

    if verbose:
        print('Processing concurrent API calls to select relevant schema items and extract location..')
        print()

    # We create an object which stores some useful functions we will use later
    location_and_granularity_obj = location_and_granularity_object()

    # we retrieve the 3 prompt we want to execute concurrently
    retrieve_class_prompt, retrieve_relation_prompt = get_prompts_for_selecting_relevant_schema_items_based_on_natural_language_question(
        question)
    extract_location_prompt = location_and_granularity_obj.get_prompt_for_location_aggregation_and_questionWithoutLocation(
        question)

    # we process the 3 prompts concurrently
    class_response, relation_response, location_extractor_response = concurrent_executor(
        [retrieve_class_prompt, retrieve_relation_prompt, extract_location_prompt], llm=llm)

    # now we define some functions which process the responses of the llm
    def process_schema_response(list_as_string):
        '''
        This method converts a string representation of a list into an actual Python list object.

        It accepts a string that represents a list (e.g., "['abc', 'bcd', 'cde']") and converts it into a list object for
        further processing.

        Parameters:
        ----------
        list_as_string : str
            A string that represents a list in Python syntax.

        Returns:
        -------
        list
            A Python list corresponding to the string representation.
        '''
        response_as_list = ast.literal_eval(list_as_string)
        return response_as_list

    def process_location_response_and_return_dictionary(response, question):
        '''
        This method processes the LLM response for span classification, type classification, and entity masking.
        It takes the LLM's response and extracts the relevant information into a dictionary.
        It also adds the original question to the dictionary for reference.

        Parameters:
        ----------
        response : str
            The LLM response for the span classification, type classification, and entity masking prompt.

        question : str
            The original natural language question.

        Returns:
        -------
        dict
            A dictionary with the following keys:
            - 'LOCATION': The location extracted from the response.
            - 'SCOPE': The scope information extracted from the response.
            - 'QUESTION-NO-ADDRESS': The natural language questions reformulated without the location entity (if applicable).
            - 'ORIGINAL-QUESTION': The original natural language question.
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
    # Dictionary with span classification, type classification, and entity masking results.
    dictionary_with_decomposed_question = process_location_response_and_return_dictionary(location_extractor_response,
                                                                                          question)

    if verbose:
        print('SELECTED CLASSES:')
        print(list_with_selected_classes)
        print('SELECTED RELATIONS')
        print(list_with_selected_relations)
        print(dictionary_with_decomposed_question)

    # We take the class of the retrieved location and add it to our list of selected classes.
    if dictionary_with_decomposed_question['SCOPE'] == 'Woonplaats':
        list_with_selected_classes.append('sor:Woonplaats')
    elif dictionary_with_decomposed_question['SCOPE'] == 'Gemeente':
        list_with_selected_classes.append('wbk:Gemeente')
    elif dictionary_with_decomposed_question['SCOPE'] == 'Provincie':
        list_with_selected_classes.append('sor:Provincie')
    elif dictionary_with_decomposed_question['SCOPE'] == 'Nederland':
        pass

    if verbose:
        print('Condensing ontology based on relevant classes and relations..')
        print()

    # We use GTOC to condense the ontology based on selected schema items.
    network_ontology_store = network_and_ontology_store(k_shortest_routes)
    ontology_string_object_part, ontology_string_datatype_part = network_ontology_store.condense_ontology_based_on_retrieved_schema_items(
        list_with_selected_classes, list_with_selected_relations)

    # We perform entity linking to obtain IRI for the location entity. If the scope is for the entire Netherlands,
    # no entity linking is needed.
    if verbose:
        print('Processing API call to location server to retrieve location identifier..')
        print()
    targeted_search_suggest_url = location_and_granularity_object.get_suggest_url(
        dictionary_with_decomposed_question['SCOPE'])
    response_location_server = None
    if dictionary_with_decomposed_question['SCOPE'] == 'Nederland':
        response_location_server = None
    else:
        response_location_server = location_and_granularity_object.retrieve_location_data(
            dictionary_with_decomposed_question['LOCATION'], targeted_search_suggest_url)

    # Instance triple is created to specify the desired location.
    # We also formulate the ontology triples associated with the location entity.
    # Additionally, a textual description of the location entity is generated to enhance the LLM's understanding.
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

    # we check if the corresponding datatype ontology triples are in our condensed ontology, if not we add them
    for trip in corresponding_datatype_ontology_triples_list:
        if trip not in ontology_string_datatype_part:
            ontology_string_datatype_part = ontology_string_datatype_part + '\n' + trip
    # we check if the corresponding object ontology triples are in our condensed ontology, if not we add them
    for trip in corresponding_object_ontology_triples_list:
        if trip not in ontology_string_object_part:
            ontology_string_datatype_part = ontology_string_datatype_part + '\n' + trip

    if verbose:
        print('Processing API call to generate SPARQL..')
        print()

    # Here we define the section of the semantic parsing prompt that outlines the general task and
    # explains the provided ontology format. This section also includes fixed examples to
    # illustrate and clarify the ontology structure.
    system_string = """
    You should act as a knowledge graph 
    professional with excellent expertise in 
    writing SPARQL queries. You will be 
    provided with a schema of a knowledge 
    graph in the geographical domain where
    the schema items are in Dutch. 
    
    You will be given a natural language 
    question. Using the provided schema, 
    and the examples provided to you, you 
    should write a SPARQL query that best
    matches the natural language question.
    
    1)
    The section called "INSTANCE DATA 
    REGARDING LOCATIONS" has triples 
    which show the position of certain 
    instance data in the ontology. This
    instance data is always about 
    geographical locations. This way you
    will be able to generate a SPARQL 
    query regarding certain geographical 
    locations. If the question is about 
    the entire Netherlands this place 
    will be empty since you don't have to 
    link it to certain locations.
    
    2)
    The section called "OBJECT/CLASS 
    NAVIGATION PATH ONTOLOGY" contains 
    lines where we see how classes are
    related to eachother through 
    properties in the following format:
    class1 property1 class2
    
    -We define relationships between 
    classes by specifying how they 
    are linked to each other through
    properties.
    -In our ontology, the first class 
    mentioned in a relationship 
    represents the starting point, 
    while the second class represents 
    the destination. This distinction 
    is crucial for understanding the 
    directionality of relationships.
    -By default, relationships are 
    interpreted from left to right, 
    with the first class acting as 
    the subject and the second class 
    as the object. 
    However, when using the SPARQL 
    operator (^), relationships can 
    be traversed from right to left, 
    reversing the directionality.
    We provide examples demonstrating
    how to use the directionality 
    in SPARQL queries.
    
    -When you apply the PROPERTY 
    'property1' to an instance of 
    the CLASS 'class1', you take a 
    path towards and instance of
    CLASS 'class2'. SO IT TELLS 
    YOU WHERE A PROPERTY PATH 
    LEADS TO!
    
    -THESE PATHS ONLY GO IN ONE
    DIRECTION! IF YOU WANT TO 
    NAVIGATE THE OTHER WAY
    AROUND USE THE "^" OPERATOR
    IN YOUR SPARQL QUERY.
    
    -The FIRST and THIRD words 
    in such lines are CLASSES. 
    USE THEM AS SUCH! DO NOT USE 
    THEM AS PROPERTIES! The SECOND 
    word in such lines are PROPERTIES. 
    USE ONLY THOSE AS PROPERTIES!
    
    -ONLY USE THE NAVIGATION PATHS
    THAT ARE DESCRIBED! THIS IS VERY
    IMPORTANT! YOU CANNOT SKIP 
    INTERMEDIATE PATHS!
    
    -ALWAYS PRIORITIZE CLASSES, 
    PROPERTIES, AND NAVIGATION 
    PATHS WITH THE "sor:" PREFIX 
    IF THESE ARE AVAILABLE! 
    THESE SHOULD BE PREFERRED 
    OVER 'bag:' and 'kad:' IF
    POSSIBLE!
    
    -ALWAYS PRIORITIZE CLASSES,
    PROPERTIES, AND NAVIGATION PATHS 
    WITH THE "kad:" PREFIX over "bag:" 
    EXAMPLE: USE kad:Ligplaats  
    INSTEAD OF bag:Ligplaats!!
    
    WE GIVE SOME EXAMPLES, OBSERVE
    HOW WE ONLY USE EXISTING ROUTES:
    
    ---------------------------------------
    Example 1:
    
    ONTOLOGY:
    sor:Gebouw geo:sfWithin wbk:Buurt
    wbk:Buurt geo:sfWithin wbk:Wijk
    wbk:Wijk geo:sfWithin wbk:Gemeente
    sor:Gemeente owl:sameAs wbk:Gemeente
    sor:Gemeente geo:sfWithin sor:Provincie
    
    TASK:
    You want to link instances of  
    'sor:Gebouw' to 'sor:Provincie'.
    
    PATH IN SPARQL:
    ?gebouw a sor:Gebouw ;
        geo:sfWithin/geo:sfWithin
        /geo:sfWithin/^owl:sameAs
        /geo:sfWithin ?provincie .
    ?provincie a sor:Provincie.
    
    WRONG PATH IN SPARQL, 
    NEVER DO THIS (HERE YOU USE CLASSES 
    AS PROPERTIES. THIS IS INVALID):
    ?gebouw a sor:Gebouw ;
        geo:sfWithin/wbk:Buurt/geo:sfWithin
        /wbk:Wijk/geo:sfWithin/wbk:Gemeente
        /^owl:sameAs/sor:Gemeente
        /geo:sfWithin/sor:Provincie 
        ?provincie.
    ---------------------------------------
    Example 2:
    
    ONTOLOGY:
    sor:Verblijfsobject sor:maaktDeelUitVan 
    sor:Gebouw
    sor:Verblijfsobject sor:hoofdadres 
    sor:Nummeraanduiding
    sor:Nummeraanduiding sor:ligtAan 
    sor:OpenbareRuimte
    sor:OpenbareRuimte sor:ligtIn 
    sor:Woonplaats
    
    TASK:
    You want to link instances of 
    'sor:Gebouw' to 'sor:Woonplaats'.
    
    PATH IN SPARQL:
    ?gebouw a sor:Gebouw ;
        ^sor:maaktDeelUitVan/sor:hoofdadres
        /sor:ligtAan/sor:ligtIn ?woonplaats.
    ?woonplaats a sor:Woonplaats.
    
    ---------------------------------------
    Example 3:
    
    ONTOLOGY:
    sor:Perceel geo:sfWithin sor:Gemeente
    sor:Gemeente geo:sfWithin sor:Provincie
    bag:Perceel geo:sfWithin bag:Gemeente
    bag:Gemeente geo:sfWithin bag:Provincie
    
    TASK
    You want to link percelen to provinces.
    Remember we put priority on 'sor:' paths.
    
    PATH IN SPARQL:
    ?perceel a sor:Perceel ;
        geo:sfWithin/geo:sfWithin ?provincie
    ?provincie a sor:Provincie.
    ---------------------------------------
    
    
    WHEN USING CLASSES SPECIFICALLY DEFINE 
    THE CLASSES THAT YOU USE, FOR EXAMPLE, 
    LIKE THIS:
    ?variable a sor:Gebouw
    
    
    3)
    The section called "DATATYPE ONTOLOGY" 
    contains lines where we see how classes
    are related to certain datatypes
    of instance data through properties 
    using the following format:
    class1 property2 xsd:PositiveInteger
    
    Use this information: For example, 
    take special care of variables of type
    xsd:gYear. Ensure that any comparisons 
    or operations involving ?bouwjaar are
    appropriate for this datatype. 
    Additionally, consider any necessary 
    adjustments to the query to accommodate 
    the specific characteristics of 
    xsd:gYear. A good option is to use the
    YEAR() function to extract the actual
    year.
    
    
    !!!!!
    DO NOT USE SCHEMA ITEMS/NAVIGATION 
    PATHS THAT ARE: NOT IN THE ONTOLOGY 
    I PROVIDED YOU WITH OR NOT IN THE 
    EXAMPLES
    !!!!!
    
    USE COMPLETE RELATION PATH, DO NOT
    SKIP RELATIONS IN THE PATH!!
    
    When performing COUNT() operations
    always include DISTINCT INSIDE OF 
    IT: COUNT( DISTINCT )
    
    If you're filtering based on the
    count of values in a HAVING clause, 
    make sure to use the aggregate 
    function directly in the HAVING 
    clause rather than referencing an 
    aliased variable.
    
    When constructing SPARQL queries, 
    double-check the direction of 
    properties in triple patterns to
    accurately navigate the relationships 
    between entities. Make sure to use 
    the correct classes specified in 
    the instance data.
    
    
    
    Only respond with a SPARQL query,
    do not put any other comments in 
    your response.
    """

    # Define the section of the semantic parsing prompt that includes the various ontology components.
    # This section of the prompt also includes the natural language question for which the LLM must parse an SPARQL
    # query.
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

    # We combine both parts of the prompt into a single prompt which will be processed by the LLM.
    messages = [
        SystemMessage(
            content=system_string
        ),
        HumanMessage(
            content=str(partial_prompt)
        )
    ]

    # Here we process the resulting prompt with the LLM and return relevant information.
    llm = AzureChatOpenAI(azure_deployment="gpt-4-32k", model_name="gpt-4-32k", verbose=False, max_retries=2,
                          temperature=0, openai_api_version="2023-05-15")
    response = llm(messages)
    generated_query = response.content

    if verbose:
        print('Generated SPARQL')
        print(generated_query)

    return {'query': generated_query, 'granularity': dictionary_with_decomposed_question['SCOPE'],
            'location': location_to_return}


def semantic_parsing_few_shot_with_ontology(question,
                                            k_shortest_routes,
                                            number_of_few_shot_examples=2,
                                            meta_data_filtering_for_few_shot_examples=True,
                                            retrieve_examples_with_question_without_location=False,
                                            example_retrieval_style='max_marginal_similarity',
                                            naive_schema_linking_without_graph_traversal=False,
                                            verbose=True,
                                            condense_ontology=True):
    """

    This method performs semantic parsing including both in-context examples and ontology information.

    It accepts a natural language question and generates the corresponding SPARQL query, while also condensing
    the ontology based on relevant classes and relations. Additionally, it processes the geographical location
    mentioned in the question and identifies its granularity (e.g., Woonplaats, Gemeente, Provincie).

    Parameters:
    ----------
    question : str
        A natural language question for which the corresponding SPARQL query needs to be generated.

    k_shortest_routes : int
        The number of shortest routes to consider between relevant nodes for ontology condensation during
        the execution of the GTOC algorithm.

    number_of_few_shot_examples : int
        The number of in-context examples that should be retrieved from the vector store.

    meta_data_filtering_for_few_shot_examples : boolean
        A boolean indicating whether to enable metadata-based pre-filtering for retrieving in-context examples.
        Metadata pre-filtering leverages the type classification results to perform a filtered search in
        the vector store during example retrieval.

    retrieve_examples_with_question_without_location : boolean
        A boolean indicating whether to use the question with the location entity masked
        when retrieving in-context examples.

    example_retrieval_style : string
        Specifies the strategy to use for selecting in-context examples during retrieval.
        "max_marginal_similarity": Applies the Maximal Marginal Relevance (MMR) procedure, as detailed in our paper.
        "semantic_similarity": Utilizes a purely similarity-based search through FAISS for retrieval.

    naive_schema_linking_without_graph_traversal : boolean
        Indicates whether to apply a naive schema condensation technique.
        This approach selects all ontology triples associated with relevant classes, properties,
        and value list entities without utilizing graph traversal algorithms.
        Note: This parameter takes effect only when the "condense_ontology" parameter is set to true.

    verbose : boolean
        A boolean indicating whether (intermediate) results should be printed during execution.

    condense_ontology : boolean
        Specifies whether to apply ontology condensation.
        If set to False, the entire ontology is included without any condensation.



    Returns:
    -------
    dict
        A dictionary containing:
        - 'query': The generated SPARQL query based on the input question and ontology schema.
        - 'granularity': The level of granularity for the location (e.g., Woonplaats, Gemeente).
        - 'location': The location identified from the question, based on the extracted geographical context.
    """
    # llm = AzureChatOpenAI(deployment_name="gpt-35-turbo", model_name="gpt-35-turbo", verbose=False, max_retries=2,
    #                       temperature=0, openai_api_version="2023-05-15")

    llm = AzureChatOpenAI(azure_deployment="gpt-4-32k", model_name="gpt-4-32k", verbose=False, max_retries=2,
                          temperature=0, openai_api_version="2023-05-15")
    if verbose == True:
        print('Processing concurrent API calls to select relevant schema items and extract location..')

    location_and_granularity_obj = location_and_granularity_object()
    # we retrieve the 3 prompt we want to execute concurrently
    retrieve_class_prompt, retrieve_relation_prompt = get_prompts_for_selecting_relevant_schema_items_based_on_natural_language_question(
        question)
    extract_location_prompt = location_and_granularity_obj.get_prompt_for_location_aggregation_and_questionWithoutLocation(
        question)

    # we process the 3 prompts concurrently
    class_response, relation_response, location_extractor_response = concurrent_executor(
        [retrieve_class_prompt, retrieve_relation_prompt, extract_location_prompt], llm=llm)

    # now we define some functions which process the responses of the llm
    def process_schema_response(list_as_string):
        '''
        This method converts a string representation of a list into an actual Python list object.

        It accepts a string that represents a list (e.g., "['abc', 'bcd', 'cde']") and converts it into a list object for
        further processing.

        Parameters:
        ----------
        list_as_string : str
            A string that represents a list in Python syntax.

        Returns:
        -------
        list
            A Python list corresponding to the string representation.
        '''
        response_as_list = ast.literal_eval(list_as_string)
        return response_as_list

    def process_location_response_and_return_dictionary(response, question):
        '''
        This method processes the LLM response for span classification, type classification, and entity masking.
        It takes the LLM's response and extracts the relevant information into a dictionary.
        It also adds the original question to the dictionary for reference.

        Parameters:
        ----------
        response : str
            The LLM response for the span classification, type classification, and entity masking prompt.

        question : str
            The original natural language question.

        Returns:
        -------
        dict
            A dictionary with the following keys:
            - 'LOCATION': The location extracted from the response.
            - 'SCOPE': The scope information extracted from the response.
            - 'QUESTION-NO-ADDRESS': The natural language questions reformulated without the location entity (if applicable).
            - 'ORIGINAL-QUESTION': The original natural language question.
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
    # Dictionary with span classification, type classification, and entity masking results.
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

    # We initialize a model to generate embeddings for the input question asked during inference.
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment='text-embedding-ada-002',
        openai_api_version=os.environ['OPENAI_API_VERSION'],
        # chunk_size=1
    ),

    # Based on the type classification results, load the appropriate vector store containing
    # in-context examples that match the specified granularity.
    # Based on the retrieved entity, we also include corresponding classes in our list of selected classes.
    vector_store = None
    if dictionary_with_decomposed_question['SCOPE'] == 'Woonplaats':
        list_with_selected_classes.append('sor:Woonplaats')
        os.path.join(current_dir, '/vector_stores/woonplaats_vector_store')
        vector_store = FAISS.load_local(os.path.join(current_dir, 'vector_stores', 'woonplaats_vector_store'),
                                        embeddings)

    elif dictionary_with_decomposed_question['SCOPE'] == 'Gemeente':
        list_with_selected_classes.append('wbk:Gemeente')
        vector_store = FAISS.load_local(os.path.join(current_dir, 'vector_stores', 'gemeente_vector_store'), embeddings)

    elif dictionary_with_decomposed_question['SCOPE'] == 'Provincie':
        list_with_selected_classes.append('sor:Provincie')
        vector_store = FAISS.load_local(os.path.join(current_dir, 'vector_stores', 'provincie_vector_store'),
                                        embeddings)

    elif dictionary_with_decomposed_question['SCOPE'] == 'Nederland':
        vector_store = FAISS.load_local(os.path.join(current_dir, 'vector_stores', 'land_vector_store'), embeddings)

    # If we specified that we do not want to apply meta-data pre-filtering, we load the vector store containing
    # all in-context examples.
    if meta_data_filtering_for_few_shot_examples == False:
        vector_store = FAISS.load_local(os.path.join(current_dir, 'vector_stores', 'combined_vector_store'), embeddings)

    vector_store.embedding_function = vector_store.embedding_function[0]

    # Here, we initiate our example selector using the chosen vector store with
    # the specified retrieval technique.
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

    # Here we select actual examples using our previously initialized example selector.
    # We use a different search query based on whether we want to apply entity masking.
    selected_examples = None
    if retrieve_examples_with_question_without_location == True:
        search_query = {"query": dictionary_with_decomposed_question['QUESTION-NO-ADDRESS']}
        selected_examples = example_selector.select_examples(search_query)
    else:
        search_query = {"query": dictionary_with_decomposed_question['ORIGINAL-QUESTION']}
        selected_examples = example_selector.select_examples(search_query)

    if verbose:
        print(selected_examples)

    # Template for how the in-context examples will be formatted inside of the prompt.
    example_prompt = PromptTemplate(
        input_variables=["QUESTION", "SPARQL"],
        template="Question: {QUESTION}\nQuery: {SPARQL}",
    )

    # Here we use the template for in-context examples to process all our selected in-context examples.
    full_example_prompt = FewShotPromptTemplate(
        # We provide an ExampleSelector instead of examples.
        examples=selected_examples,
        example_prompt=example_prompt,
        suffix="Question: {question}\nQuery:",
        input_variables=["question"],
    )

    if verbose == True:
        print(selected_examples)
        print(full_example_prompt.format(question=dictionary_with_decomposed_question['ORIGINAL-QUESTION']))
        print()

        print('Condensing ontology based on relevant classes and relations..')
        print()

    # we instantiate an ontology/GTOC object using the specified number of shortest routes between relevant nodes.
    network_ontology_store = network_and_ontology_store(k_shortest_routes)

    # We generate the corresponding ontology description based on the provided input parameters.
    if condense_ontology == True:
        if naive_schema_linking_without_graph_traversal == False:
            ontology_string_object_part, ontology_string_datatype_part = network_ontology_store.condense_ontology_based_on_retrieved_schema_items(
                list_with_selected_classes, list_with_selected_relations)
        else:
            ontology_string_object_part, ontology_string_datatype_part = network_ontology_store.naive_ontology_selection(
                list_with_selected_classes, list_with_selected_relations)
    else:
        ontology_string_object_part, ontology_string_datatype_part = network_ontology_store.retrieve_full_ontology()

    # We perform entity linking, unless the question pertains to the entire Netherlands.
    if verbose == True:
        print('Processing API call to location server to retrieve location identifier..')
        print()
    targeted_search_suggest_url = location_and_granularity_object.get_suggest_url(
        dictionary_with_decomposed_question['SCOPE'])
    response_location_server = None
    if dictionary_with_decomposed_question['SCOPE'] == 'Nederland':
        response_location_server = None
    else:
        response_location_server = location_and_granularity_object.retrieve_location_data(
            dictionary_with_decomposed_question['LOCATION'], targeted_search_suggest_url)

    # Instance triple is created to specify the desired location.
    # We also formulate the ontology triples associated with the location entity.
    # Additionally, a textual description of the location entity is generated to enhance the LLM's understanding.
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

    # we check if the corresponding datatype ontology triples are in our condensed ontology, if not we add it
    for trip in corresponding_datatype_ontology_triples_list:
        if trip not in ontology_string_datatype_part:
            ontology_string_datatype_part = ontology_string_datatype_part + '\n' + trip
    # we check if the corresponding object ontology triples are in our condensed ontology, if not we add it
    for trip in corresponding_object_ontology_triples_list:
        if trip not in ontology_string_object_part:
            ontology_string_datatype_part = ontology_string_datatype_part + '\n' + trip

    # Here we define the section of the semantic parsing prompt that outlines the general task and
    # explains the provided ontology format. This section also includes fixed examples to
    # illustrate and clarify the ontology structure.
    system_string = """
    You should act as a knowledge graph 
    professional with excellent expertise in 
    writing SPARQL queries. You will be 
    provided with a schema of a knowledge 
    graph in the geographical domain where
    the schema items are in Dutch. 
    
    You will be given a natural language 
    question. Using the provided schema, 
    and the examples provided to you, you 
    should write a SPARQL query that best
    matches the natural language question.
    
    1)
    The section called "INSTANCE DATA 
    REGARDING LOCATIONS" has triples 
    which show the position of certain 
    instance data in the ontology. This
    instance data is always about 
    geographical locations. This way you
    will be able to generate a SPARQL 
    query regarding certain geographical 
    locations. If the question is about 
    the entire Netherlands this place 
    will be empty since you don't have to 
    link it to certain locations.
    
    2)
    The section called "OBJECT/CLASS 
    NAVIGATION PATH ONTOLOGY" contains 
    lines where we see how classes are
    related to eachother through 
    properties in the following format:
    class1 property1 class2
    
    -We define relationships between 
    classes by specifying how they 
    are linked to each other through
    properties.
    -In our ontology, the first class 
    mentioned in a relationship 
    represents the starting point, 
    while the second class represents 
    the destination. This distinction 
    is crucial for understanding the 
    directionality of relationships.
    -By default, relationships are 
    interpreted from left to right, 
    with the first class acting as 
    the subject and the second class 
    as the object. 
    However, when using the SPARQL 
    operator (^), relationships can 
    be traversed from right to left, 
    reversing the directionality.
    We provide examples demonstrating
    how to use the directionality 
    in SPARQL queries.
    
    -When you apply the PROPERTY 
    'property1' to an instance of 
    the CLASS 'class1', you take a 
    path towards and instance of
    CLASS 'class2'. SO IT TELLS 
    YOU WHERE A PROPERTY PATH 
    LEADS TO!
    
    -THESE PATHS ONLY GO IN ONE
    DIRECTION! IF YOU WANT TO 
    NAVIGATE THE OTHER WAY
    AROUND USE THE "^" OPERATOR
    IN YOUR SPARQL QUERY.
    
    -The FIRST and THIRD words 
    in such lines are CLASSES. 
    USE THEM AS SUCH! DO NOT USE 
    THEM AS PROPERTIES! The SECOND 
    word in such lines are PROPERTIES. 
    USE ONLY THOSE AS PROPERTIES!
    
    -ONLY USE THE NAVIGATION PATHS
    THAT ARE DESCRIBED! THIS IS VERY
    IMPORTANT! YOU CANNOT SKIP 
    INTERMEDIATE PATHS!
    
    -ALWAYS PRIORITIZE CLASSES, 
    PROPERTIES, AND NAVIGATION 
    PATHS WITH THE "sor:" PREFIX 
    IF THESE ARE AVAILABLE! 
    THESE SHOULD BE PREFERRED 
    OVER 'bag:' and 'kad:' IF
    POSSIBLE!
    
    -ALWAYS PRIORITIZE CLASSES,
    PROPERTIES, AND NAVIGATION PATHS 
    WITH THE "kad:" PREFIX over "bag:" 
    EXAMPLE: USE kad:Ligplaats  
    INSTEAD OF bag:Ligplaats!!
    
    WE GIVE SOME EXAMPLES, OBSERVE
    HOW WE ONLY USE EXISTING ROUTES:
    
    ---------------------------------------
    Example 1:
    
    ONTOLOGY:
    sor:Gebouw geo:sfWithin wbk:Buurt
    wbk:Buurt geo:sfWithin wbk:Wijk
    wbk:Wijk geo:sfWithin wbk:Gemeente
    sor:Gemeente owl:sameAs wbk:Gemeente
    sor:Gemeente geo:sfWithin sor:Provincie
    
    TASK:
    You want to link instances of  
    'sor:Gebouw' to 'sor:Provincie'.
    
    PATH IN SPARQL:
    ?gebouw a sor:Gebouw ;
        geo:sfWithin/geo:sfWithin
        /geo:sfWithin/^owl:sameAs
        /geo:sfWithin ?provincie .
    ?provincie a sor:Provincie.
    
    WRONG PATH IN SPARQL, 
    NEVER DO THIS (HERE YOU USE CLASSES 
    AS PROPERTIES. THIS IS INVALID):
    ?gebouw a sor:Gebouw ;
        geo:sfWithin/wbk:Buurt/geo:sfWithin
        /wbk:Wijk/geo:sfWithin/wbk:Gemeente
        /^owl:sameAs/sor:Gemeente
        /geo:sfWithin/sor:Provincie 
        ?provincie.
    ---------------------------------------
    Example 2:
    
    ONTOLOGY:
    sor:Verblijfsobject sor:maaktDeelUitVan 
    sor:Gebouw
    sor:Verblijfsobject sor:hoofdadres 
    sor:Nummeraanduiding
    sor:Nummeraanduiding sor:ligtAan 
    sor:OpenbareRuimte
    sor:OpenbareRuimte sor:ligtIn 
    sor:Woonplaats
    
    TASK:
    You want to link instances of 
    'sor:Gebouw' to 'sor:Woonplaats'.
    
    PATH IN SPARQL:
    ?gebouw a sor:Gebouw ;
        ^sor:maaktDeelUitVan/sor:hoofdadres
        /sor:ligtAan/sor:ligtIn ?woonplaats.
    ?woonplaats a sor:Woonplaats.
    
    ---------------------------------------
    Example 3:
    
    ONTOLOGY:
    sor:Perceel geo:sfWithin sor:Gemeente
    sor:Gemeente geo:sfWithin sor:Provincie
    bag:Perceel geo:sfWithin bag:Gemeente
    bag:Gemeente geo:sfWithin bag:Provincie
    
    TASK
    You want to link percelen to provinces.
    Remember we put priority on 'sor:' paths.
    
    PATH IN SPARQL:
    ?perceel a sor:Perceel ;
        geo:sfWithin/geo:sfWithin ?provincie
    ?provincie a sor:Provincie.
    ---------------------------------------
    
    
    WHEN USING CLASSES SPECIFICALLY DEFINE 
    THE CLASSES THAT YOU USE, FOR EXAMPLE, 
    LIKE THIS:
    ?variable a sor:Gebouw
    
    
    3)
    The section called "DATATYPE ONTOLOGY" 
    contains lines where we see how classes
    are related to certain datatypes
    of instance data through properties 
    using the following format:
    class1 property2 xsd:PositiveInteger
    
    Use this information: For example, 
    take special care of variables of type
    xsd:gYear. Ensure that any comparisons 
    or operations involving ?bouwjaar are
    appropriate for this datatype. 
    Additionally, consider any necessary 
    adjustments to the query to accommodate 
    the specific characteristics of 
    xsd:gYear. A good option is to use the
    YEAR() function to extract the actual
    year.
    
    
    !!!!!
    DO NOT USE SCHEMA ITEMS/NAVIGATION 
    PATHS THAT ARE: NOT IN THE ONTOLOGY 
    I PROVIDED YOU WITH OR NOT IN THE 
    EXAMPLES
    !!!!!
    
    USE COMPLETE RELATION PATH, DO NOT
    SKIP RELATIONS IN THE PATH!!
    
    When performing COUNT() operations
    always include DISTINCT INSIDE OF 
    IT: COUNT( DISTINCT )
    
    If you're filtering based on the
    count of values in a HAVING clause, 
    make sure to use the aggregate 
    function directly in the HAVING 
    clause rather than referencing an 
    aliased variable.
    
    When constructing SPARQL queries, 
    double-check the direction of 
    properties in triple patterns to
    accurately navigate the relationships 
    between entities. Make sure to use 
    the correct classes specified in 
    the instance data.
    
    
    
    Only respond with a SPARQL query,
    do not put any other comments in 
    your response.

    """

    # Define the section of the semantic parsing prompt that includes the various ontology components.
    # This section of the prompt also includes the natural language question for which the LLM must parse an SPARQL
    # query.
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

    # Here we combine all parts of the prompt into a single prompt which can be passed to an LLM.
    messages = [
        SystemMessage(
            content=system_string
        ),
        HumanMessage(
            content=str(full_example_prompt.format(question=dictionary_with_decomposed_question['ORIGINAL-QUESTION']))
        )
    ]

    # Process the prompt with the LLM and return relevant results.
    response = llm(messages)
    generated_query = response.content
    if verbose == True:
        print('Generated SPARQL')
        print(generated_query)

    return {'query': generated_query, 'granularity': dictionary_with_decomposed_question['SCOPE'],
            'location': location_to_return}


def semantic_parsing_few_shot_NO_ONTOLOGY(question,
                                                   number_of_few_shot_examples=5,
                                                   meta_data_filtering_for_few_shot_examples=True,
                                                   retrieve_examples_with_question_without_location=True,
                                                   example_retrieval_style='max_marginal_similarity',
                                                   verbose=False):
    '''
    This method is used for the ablation where we exclude ontology information entirely but include in-context examples
    for semantic parsing.

    It accepts a natural language question and generates the corresponding SPARQL query. Although we do not include
    ontology information, we do include the location IRI to facilitate a fair comparison.

    Parameters:
    ----------
    question : str
        A natural language question for which the corresponding SPARQL query needs to be generated.

    number_of_few_shot_examples : int
        The number of in-context examples that should be retrieved from the vector store.

    meta_data_filtering_for_few_shot_examples : boolean
        A boolean indicating whether to enable metadata-based pre-filtering for retrieving in-context examples.
        Metadata pre-filtering leverages the type classification results to perform a filtered search in
        the vector store during example retrieval.

    retrieve_examples_with_question_without_location : boolean
        A boolean indicating whether to use the question with the location entity masked
        when retrieving in-context examples.

    example_retrieval_style : string
        Specifies the strategy to use for selecting in-context examples during retrieval.
        "max_marginal_similarity": Applies the Maximal Marginal Relevance (MMR) procedure, as detailed in our paper.
        "semantic_similarity": Utilizes a purely similarity-based search through FAISS for retrieval.

    verbose : boolean
        A boolean indicating whether (intermediate) results should be printed during execution.

    Returns:
    -------
    dict
        A dictionary containing:
        - 'query': The generated SPARQL query based on the input question and ontology schema.
        - 'granularity': The level of granularity for the location (e.g., Woonplaats, Gemeente).
        - 'location': The location identified from the question, based on the extracted geographical context.
    '''
    # llm = AzureChatOpenAI(deployment_name="gpt-35-turbo", model_name="gpt-35-turbo", verbose=False, max_retries=2,
    #                       temperature=0, openai_api_version="2023-05-15")

    llm = AzureChatOpenAI(azure_deployment="gpt-4-32k", model_name="gpt-4-32k", verbose=False, max_retries=2,
                          temperature=0, openai_api_version="2023-05-15")
    if verbose:
        print('Processing concurrent API call to extract location..')
    location_and_granularity_obj = location_and_granularity_object()

    # We retrieve the prompt for span classification, type classification, and entity masking,
    # which is then used to decode the result using an LLM.
    extract_location_prompt = location_and_granularity_obj.get_prompt_for_location_aggregation_and_questionWithoutLocation(
        question)
    loc_response = llm(extract_location_prompt)
    location_extractor_response = loc_response.content

    # now we define some functions which process the responses of the llm
    def process_location_response_and_return_dictionary(response, question):
        '''
        This method processes the LLM response for span classification, type classification, and entity masking.
        It takes the LLM's response and extracts the relevant information into a dictionary.
        It also adds the original question to the dictionary for reference.

        Parameters:
        ----------
        response : str
            The LLM response for the span classification, type classification, and entity masking prompt.

        question : str
            The original natural language question.

        Returns:
        -------
        dict
            A dictionary with the following keys:
            - 'LOCATION': The location extracted from the response.
            - 'SCOPE': The scope information extracted from the response.
            - 'QUESTION-NO-ADDRESS': The natural language questions reformulated without the location entity (if applicable).
            - 'ORIGINAL-QUESTION': The original natural language question.
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

    # Dictionary with span classification, type classification, and entity masking results.
    dictionary_with_decomposed_question = process_location_response_and_return_dictionary(location_extractor_response,
                                                                                          question)
    if verbose:
        print(dictionary_with_decomposed_question)

        print('Retrieving few-shot examples based on granularity..')

    # We initialize a model to generate embeddings for the input question asked during inference.
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment='text-embedding-ada-002',
        openai_api_version=os.environ['OPENAI_API_VERSION'],
        # chunk_size=1
    )

    # Based on the type classification results, load the appropriate vector store containing
    # in-context examples that match the specified granularity.
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

    # If we specified that we do not want to apply meta-data pre-filtering, we load the vector store containing
    # all in-context examples.
    if meta_data_filtering_for_few_shot_examples == False:
        vector_store = FAISS.load_local(os.path.join(current_dir, 'vector_stores', 'combined_vector_store'), embeddings)

    vector_store.embedding_function = vector_store.embedding_function[0]

    # Here, we initiate our example selector using the chosen vector store with
    # the specified retrieval technique.
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

    # Here we select actual examples using our previously initialized example selector.
    # We use a different search query based on whether we want to apply entity masking.
    selected_examples = None
    if retrieve_examples_with_question_without_location == True:
        search_query = {"query": dictionary_with_decomposed_question['QUESTION-NO-ADDRESS']}
        selected_examples = example_selector.select_examples(search_query)
    else:
        search_query = {"query": dictionary_with_decomposed_question['ORIGINAL-QUESTION']}
        selected_examples = example_selector.select_examples(search_query)

    # Template for how the in-context examples will be formatted inside of the prompt.
    example_prompt = PromptTemplate(
        input_variables=["QUESTION", "SPARQL"],
        template="Question: {QUESTION}\nQuery: {SPARQL}",
    )
    # Here we use the template for in-context examples to process all our selected in-context examples.
    full_example_prompt = FewShotPromptTemplate(
        # We provide an ExampleSelector instead of examples.
        examples=selected_examples,
        example_prompt=example_prompt,
        suffix="Question: {question}\nQuery:",
        input_variables=["question"],
    )

    if verbose:
        print(full_example_prompt.format(question=dictionary_with_decomposed_question['ORIGINAL-QUESTION']))
        print()

        print('Processing API call to location server to retrieve location identifier..')
        print()

    # We perform entity linking, unless the question pertains to the entire Netherlands.
    targeted_search_suggest_url = location_and_granularity_object.get_suggest_url(
        dictionary_with_decomposed_question['SCOPE'])
    response_location_server = None
    if dictionary_with_decomposed_question['SCOPE'] == 'Nederland':
        response_location_server = None
    else:
        response_location_server = location_and_granularity_object.retrieve_location_data(
            dictionary_with_decomposed_question['LOCATION'], targeted_search_suggest_url)

    messages = []
    system_string = ""
    location_to_return = None

    # Here we generate the prompt based on the granularity of the question.
    if response_location_server is not None:

        if dictionary_with_decomposed_question['SCOPE'] == 'Provincie':
            location_to_return = response_location_server["provinciecode"].replace("PV", "00")
            system_string = """
            You should act as a knowledge graph 
            professional with excellent expertise in 
            writing SPARQL queries.
            
            You will be given a natural language 
            question. Using the provided examples, 
            you should write a SPARQL query that best
            matches the natural language question.
            
            When performing COUNT() operations
            always include DISTINCT INSIDE OF 
            IT: COUNT( DISTINCT )
            
            If you're filtering based on the
            count of values in a HAVING clause, 
            make sure to use the aggregate 
            function directly in the HAVING 
            clause rather than referencing an 
            aliased variable.
            
            Only respond with a SPARQL query,
            do not put any other comments in 
            your response.

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
            You should act as a knowledge graph 
            professional with excellent expertise in 
            writing SPARQL queries.
            
            You will be given a natural language 
            question. Using the provided examples, 
            you should write a SPARQL query that best
            matches the natural language question.
            
            When performing COUNT() operations
            always include DISTINCT INSIDE OF 
            IT: COUNT( DISTINCT )
            
            If you're filtering based on the
            count of values in a HAVING clause, 
            make sure to use the aggregate 
            function directly in the HAVING 
            clause rather than referencing an 
            aliased variable.
            
            Only respond with a SPARQL query,
            do not put any other comments in 
            your response.

            All the queries that you generate should contain a location code (with numbers or numbers with letters).
            Do not make up your own codes or descriptions. Always use the following location code in your query:
            %s
            Write a query where the location code is properly linked as a object to the "sdo0:identifier" predicate.

            You are provided with some examples: 

            """ % ("gemeentecode: " + 'GM' + response_location_server["gemeentecode"])




        elif dictionary_with_decomposed_question['SCOPE'] == 'Woonplaats':
            location_to_return = response_location_server["woonplaatsnaam"]
            system_string = """
            You should act as a knowledge graph 
            professional with excellent expertise in 
            writing SPARQL queries.
            
            You will be given a natural language 
            question. Using the provided examples, 
            you should write a SPARQL query that best
            matches the natural language question.
            
            When performing COUNT() operations
            always include DISTINCT INSIDE OF 
            IT: COUNT( DISTINCT )
            
            If you're filtering based on the
            count of values in a HAVING clause, 
            make sure to use the aggregate 
            function directly in the HAVING 
            clause rather than referencing an 
            aliased variable.
            
            Only respond with a SPARQL query,
            do not put any other comments in 
            your response.

            All the queries that you generate contain a location description (it's name).
            Do not make up your own codes or descriptions. Always use the following code or description that i give you in this prompt:
            %s
            Make sure you use the location name properly in the following triple:
            "skos:prefLabel "location_name"@nl;"


            You are provided with some examples: 

            """ % ("woonplaatsnaam: " + response_location_server["woonplaatsnaam"])


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
            You should act as a knowledge graph 
            professional with excellent expertise in 
            writing SPARQL queries.
            
            You will be given a natural language 
            question. Using the provided examples, 
            you should write a SPARQL query that best
            matches the natural language question.
            
            When performing COUNT() operations
            always include DISTINCT INSIDE OF 
            IT: COUNT( DISTINCT )
            
            If you're filtering based on the
            count of values in a HAVING clause, 
            make sure to use the aggregate 
            function directly in the HAVING 
            clause rather than referencing an 
            aliased variable.
            
            Only respond with a SPARQL query,
            do not put any other comments in 
            your response.
            
            You are provided with some examples: 

        """

        # We combine all parts of the prompt into a single prompt.
        messages = [
            SystemMessage(
                content=system_string
            ),
            HumanMessage(
                content=str(full_example_prompt.format(question=question))
            )
        ]

    # Here we print the complete prompt.
    if verbose:
        print("Prompt sent to OpenAI:")
        for i in messages:
            print(i.content)
        print("\n")

    # We utilize the LLM to process the prompt and generate a corresponding SPARQL query.
    response = llm(messages)
    generated_query = response.content

    if verbose:
        print("Generated query:")
        print(generated_query)

    return {'query': generated_query, 'granularity': dictionary_with_decomposed_question['SCOPE'],
            'location': location_to_return}


def only_get_condensed_ontologies(question, k_shortest_routes, verbose=False):
    """
    Generates ontology condensation results without performing semantic parsing.

    This function focuses exclusively on returning condensed ontologies.
    It is particularly useful for evaluating and analyzing the performance
    and effectiveness of GTOC compared to naive ontology condensation.

    Args:
    question :str
        The natural language question for which condensed ontologies must be obtained.

    k_shortest_routes :int
        The number of shortest routes to consider between relevant nodes in
        the ontology network during condensation.

    verbose : boolean
        A boolean indicating whether (intermediate) results should be printed during execution.

    Returns:
        tuple: A tuple containing:
            - ontology_string_object_part (str): GTOC condensed ontology object vertex-edge pairs.
            - ontology_string_datatype_part (str): GTOC condensed ontology datatype vertex-edge pairs.
            - ontology_string_object_part_naive (str): Naively condensed ontology object vertex-edge pairs.
            - ontology_string_datatype_part_naive (str): Naively condensed ontology datatype vertex-edge pairs.
            - size_condensed_ontology (int): The total size of the GTOC condensed ontology in terms of
                                             amount of vertex-edge pairs.


    """
    # llm = AzureChatOpenAI(deployment_name="gpt-35-turbo", model_name="gpt-35-turbo", verbose=False, max_retries=2,
    #                       temperature=0, openai_api_version="2023-05-15")

    llm = AzureChatOpenAI(azure_deployment="gpt-4-32k", model_name="gpt-4-32k", verbose=False, max_retries=2,
                          temperature=0, openai_api_version="2023-05-15")

    if verbose:
        print('Processing concurrent API calls to select relevant schema items and extract location..')
        print()

    location_and_granularity_obj = location_and_granularity_object()
    # we retrieve the 3 prompt we want to execute concurrently
    retrieve_class_prompt, retrieve_relation_prompt = get_prompts_for_selecting_relevant_schema_items_based_on_natural_language_question(
        question)
    extract_location_prompt = location_and_granularity_obj.get_prompt_for_location_aggregation_and_questionWithoutLocation(
        question)

    # we process the 3 prompts concurrently
    class_response, relation_response, location_extractor_response = concurrent_executor(
        [retrieve_class_prompt, retrieve_relation_prompt, extract_location_prompt], llm=llm)

    # now we define some functions which process the responses of the llm
    def process_schema_response(list_as_string):
        '''
        This method converts a string representation of a list into an actual Python list object.

        It accepts a string that represents a list (e.g., "['abc', 'bcd', 'cde']") and converts it into a list object for
        further processing.

        Parameters:
        ----------
        list_as_string : str
            A string that represents a list in Python syntax.

        Returns:
        -------
        list
            A Python list corresponding to the string representation.
        '''
        response_as_list = ast.literal_eval(list_as_string)
        return response_as_list

    def process_location_response_and_return_dictionary(response, question):
        '''
        This method processes the LLM response for span classification, type classification, and entity masking.
        It takes the LLM's response and extracts the relevant information into a dictionary.
        It also adds the original question to the dictionary for reference.

        Parameters:
        ----------
        response : str
            The LLM response for the span classification, type classification, and entity masking prompt.

        question : str
            The original natural language question.

        Returns:
        -------
        dict
            A dictionary with the following keys:
            - 'LOCATION': The location extracted from the response.
            - 'SCOPE': The scope information extracted from the response.
            - 'QUESTION-NO-ADDRESS': The natural language questions reformulated without the location entity (if applicable).
            - 'ORIGINAL-QUESTION': The original natural language question.
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
    # Dictionary with span classification, type classification, and entity masking results.
    dictionary_with_decomposed_question = process_location_response_and_return_dictionary(
        location_extractor_response,
        question)

    if verbose:
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

    if verbose:
        print('Condensing ontology based on relevant classes and relations..')
        print()

    network_ontology_store = network_and_ontology_store(k_shortest_routes)

    ontology_string_object_part, ontology_string_datatype_part = network_ontology_store.condense_ontology_based_on_retrieved_schema_items(
        list_with_selected_classes, list_with_selected_relations)

    ontology_string_object_part_naive, ontology_string_datatype_part_naive = network_ontology_store.naive_ontology_selection(
        list_with_selected_classes, list_with_selected_relations)

    size_condensed_ontology = (ontology_string_object_part.count('\n') + 1) + (
            ontology_string_datatype_part.count('\n') + 1)

    return ontology_string_object_part, ontology_string_datatype_part, ontology_string_object_part_naive, ontology_string_datatype_part_naive, size_condensed_ontology


'''
In this section, you can implement the methods described above and run this file to observe the results directly. 

Ensure the "verbose" parameter is set to True to display the results in the console.
'''
semantic_parsing_few_shot_with_ontology('Hoeveel percelen zijn er die niet bij een nummeraanduiding horen?', 1)

# semantic_parsing_workflow_few_shot_NO_ONTOLOGY('Geef mij de nieuwste brandweerkazerne in gemeente eindhoven',
#                                                    number_of_few_shot_examples=5,
#                                                    meta_data_filtering_for_few_shot_examples=True,
#                                                    retrieve_examples_with_question_without_location=True,
#                                                    example_retrieval_style='max_marginal_similarity')

# semantic_parsing_workflow('Hoeveel percelen zijn er in provincie Gelderland?', 7)
