from langchain_community.vectorstores import FAISS
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
import configparser
import requests
from typing import List, Dict, Any
import os
import json
from langchain_openai import AzureChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langchain.prompts.example_selector import (
    MaxMarginalRelevanceExampleSelector,
    SemanticSimilarityExampleSelector,
)
import pickle
import dill
from langchain_openai import AzureOpenAIEmbeddings

'''
Here i set up some stuff with which we can pass prompt to OpenAI 
'''
gpt_option = 'OPENAI_GPT_3.5_TURBO'
# Put the absolute path to the .ini file here:
config = configparser.ConfigParser()
# Put the absolute path to the .ini file here:
path = r"../secrets.ini"
config.read(path)

locatie_url = config['KADASTER']['LOC_SEARCH_API']
lookup_url = config['KADASTER']['LOC_LOOKUP_API']
sparql_url = config['KADASTER']['KKG_API']
os.environ['AZURE_OPENAI_API_KEY'] = config[gpt_option]['AZURE_API_KEY']
# os.environ['AZURE_API_TYPE'] = config[gpt_option]['OPENAI_API_TYPE']
os.environ['OPENAI_API_VERSION'] = config[gpt_option]['AZURE_API_VERSION']
os.environ['AZURE_OPENAI_ENDPOINT'] = config[gpt_option]['AZURE_API_BASE']


# llm = AzureChatOpenAI(deployment_name="gpt-35-turbo", model_name="gpt-35-turbo", verbose=False,
#                           max_retries=2, temperature=0, openai_api_version="2023-05-15")
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


remove_location_prompt = """
You will be given an question. Your task is to remove the location from the question and return the question without location.
Not all questions contain locations, if the question does not contain a location do not change the question.
You are provided with some examples.

-------------------------------------------------------------------------------------------------
Here some examples are given:
QUESTION: geef mij de nieuwste toren in gemeente almelo jaar van 8
QUESTION-WITHOUT-LOCATION: geef mij de nieuwste toren jaar van 8

QUESTION: welk kerk in gemeente heeten heeft de kleinste tuin
QUESTION-WITHOUT-LOCATION: welke kerk heeft de kleinste tuin

QUESTION: welke bedehuisje heeft de kleinste tuin
QUESTION-WITHOUT-LOCATION: welke bedehuisje heeft de kleinste tuin

QUESTION: wat is de oudste paleizen in overijssel
QUESTION-WITHOUT-LOCATION: wat is de oudste paleizen

QUESTION: wat is de nieuwste huis lichttorens in de provincie kloosterhaar
QUESTION-WITHOUT-LOCATION: wat is de nieuwste huis lichttorens

QUESTION: ik wil weten wat de oppervlaktes van woningen zijn in provincie dedemsvaart
QUESTION-WITHOUT-LOCATION: ik wil weten wat de oppervlaktes van woningen zijn in provincie dedemsvaart

QUESTION: welk tankstations in de provincie enschede heeft het grootste oppervlakte
QUESTION-WITHOUT-LOCATION: welk tankstation heeft het grootste oppervlakte

QUESTION: welk tankstations in de provincie enschede heeft het grootste oppervlakte
QUESTION-WITHOUT-LOCATION: welk tankstation heeft het grootste oppervlakte

QUESTION: geef mij de grootste reddingboothuisje in nederland gebouwd in is 65
QUESTION-WITHOUT-LOCATION: geef mij de grootste reddingboothuisje gebouwd in is 65

QUESTION: geef mij de radar met het grootste perceelsoppervlakte in laan reggestraat almelo gelijk 29
QUESTION-WITHOUT-LOCATION: geef mij de radar met het grootste perceelsoppervlakte gelijk 29

QUESTION: waar kan ik de windmolens in laan hoolstraat 16 te teteringen vinden
QUESTION-WITHOUT-LOCATION: waar kan ik de windmolens vinden

QUESTION: ik wil de vuurtorens in laan oude twentseweg 1 luttenberg op de kaart zien
QUESTION-WITHOUT-LOCATION: ik wil de vuurtorens op de kaart zien

QUESTION: ik wil graag de crematorium bouwjaar voor 26 in straat grote huiswei 14 in teteringen
QUESTION-WITHOUT-LOCATION: ik wil graag de crematorium bouwjaar voor 26

QUESTION: geef me de windturbines in straat ommerkanaal oost in ommen gebouwd in van 2801
QUESTION-WITHOUT-LOCATION: geef me de windturbines gebouwd in van 2801

QUESTION: ik wil de kerken in laan herenstraat 108 a in slagharen zien gebouwd kleiner dan 903
QUESTION-WITHOUT-LOCATION: ik wil de kerken zien gebouwd kleiner dan 903

QUESTION: hoeveel luchtwachttorens tuin formaat van 1990 staan in laan poststraat te nieuwerkerk
QUESTION-WITHOUT-LOCATION: hoeveel luchtwachttorens tuin formaat van 1990

-------------------------------------------------------------------------------------------------
QUESTION: {vraag}
QUESTION-WITHOUT-LOCATION:

"""
current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)


def vector_store_creator():
    # We load our trainingdata
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, '../Training_data/saved_training_data/training_data_brt_filter.json')
    with open(path, 'r') as file:
        data = json.load(file)

    # We create empty lists which will store training data for different granularities.
    # We use this to create separate vector stores. This way we can apply meta-data filtering
    # when selecting examples.
    example_queries_woonplaats = []
    example_queries_gemeente = []
    example_queries_provincie = []
    example_queries_nederland = []
    # I additionally create a vectorstore where all of them are combined. This way we can perform
    # an abblation study later.
    example_queries_combined = []

    print(len(data))
    counter = 0
    for item in data:
        counter = counter + 1
        print(counter)
        answer_escaped = item['sparql_query'].replace('{', '{{')
        answer_escaped = answer_escaped.replace('}', '}}')
        question = item['question']
        #
        # try:
        #     messages = [
        #         # THIS NEEDS TO BE CHANGED STILL!
        #         HumanMessage(
        #             content=str(remove_location_prompt.format(vraag=question))
        #         )
        #     ]
        #     response = llm(messages)
        #     generated_response = response.content
        #
        #     question = generated_response
        # except ValueError as e:
        #     if "Azure has not provided the response due to a content filter being triggered" in str(e):
        #         print("Flagged by azure as inappropriate:")
        #         print(question)
        #         continue
        #     else:
        #         # Handle other ValueError cases
        #         print("Another ValueError occurred")
        # except Exception as e:
        #     # Handle other exceptions
        #     print(f"An error occurred: {str(e)}")

        location_entity_instance_data_prompt = item["instance_prompt_string"]
        dictionary_for_example = {"QUESTION": question,
                                  "INSTANCE_DATA_REGARDING_LOCATION": location_entity_instance_data_prompt,
                                  "SPARQL": answer_escaped}
        if item['granularity_key'] == 'woonplaats':
            example_queries_woonplaats.append(dictionary_for_example)
            example_queries_combined.append(dictionary_for_example)
        if item['granularity_key'] == 'gemeente':
            example_queries_gemeente.append(dictionary_for_example)
            example_queries_combined.append(dictionary_for_example)
        if item['granularity_key'] == 'provincie':
            example_queries_provincie.append(dictionary_for_example)
            example_queries_combined.append(dictionary_for_example)
        if item['granularity_key'] == 'land':
            example_queries_nederland.append(dictionary_for_example)
            example_queries_combined.append(dictionary_for_example)

    list_with_everything_that_we_will_save = [
        (example_queries_woonplaats, 'woonplaats'),
        (example_queries_gemeente, 'gemeente'),
        (example_queries_provincie, 'provincie'),
        (example_queries_nederland, 'land'),
        (example_queries_combined, 'combined')
    ]
    print(len(list_with_everything_that_we_will_save))
    counter = 0
    for query_granularity in list_with_everything_that_we_will_save:
        counter = counter + 1
        print(counter)
        example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
            # This is the list of examples available to select from.
            query_granularity[0],
            # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
            AzureOpenAIEmbeddings(
                azure_deployment='text-embedding-ada-002',
                openai_api_version=os.environ['OPENAI_API_VERSION'],
                # chunk_size=1
            ),
            # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
            FAISS,
            # This is the number of examples to produce.
            k=2,
            input_keys=['QUESTION']
        )

        # AzureOpenAIEmbeddings(
        #     #azure_deployment='text-embedding-ada-002',
        #     openai_api_key = os.environ['AZURE_OPENAI_API_KEY'],
        #     openai_api_version=os.environ['OPENAI_API_VERSION'],
        #     openai_api_type="azure",
        #     model = 'text-embedding-ada-002',
        #     #chunk_size=1)

        # # Save the instance to a file named "example_selector.pickle"
        # with open(query_granularity[1] + '_vector_store.pickle', 'wb') as f:
        #     dill.dump(example_selector, f, protocol=4)
        vector_store = example_selector.vectorstore
        vector_store.save_local(query_granularity[1] + '_vector_store')


vector_store_creator()
