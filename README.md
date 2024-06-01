# Large Language Models and Graph Traversal Algorithms for Text-to-SPARQL Generalization on Enterprise Knowledge Graphs
Code that enables SPARQL parsing through langchain on the Kadaster Knowledge Graph (KKG).

All software is written in Python 3 (https://www.python.org/) and makes use of the LangChain framework (https://www.langchain.com/) and NetworkX package (https://networkx.org/).

## Installation Instructions
To set up the environment with the required packages and versions, please refer to the requirements.txt file included in this repository. 
You can install all necessary packages by running the following command:

pip install -r requirements.txt

We use GPT-4 for semantic parsing, which requires a paid subscription to make API calls. To use the code in this repository, you will need either:

1. An OpenAI API Key: You can obtain this key by subscribing to OpenAI’s services.
2. Your Own Model Deployment on Azure: Once a model is deployed, you can extract the necessary authentication keys from the deployment.

Our repository supports integration with various Large Language Models (LLMs) through LangChain, including future releases. 
To set up the API key for your particular model API, modify the secrets.ini file and update the LangChain LLM object creation in semantic_parsing.py. For configuring models of your choice, consult the LangChain (https://python.langchain.com/v0.2/docs/introduction/) website for instructions.

Our secrets.ini file contains all API endpoints for the KKG as they are publicly available and free to use.
It includes the endpoint for sending SPARQL queries to be executed against the KKG, 
as well as the locationserver endpoint which can be used to retrieve IRIs of location entities.

Note that the vector stores for the 4 granularities (province, municipality, street & national) fit inside a github repository.
However, the combined vectorstore for all granularities together exceeds the file limit of github.
Therefore, it is excluded from the github repository.
However, all vectorstores can be replicated (including the combined vectorstore) when running the "create_training_data.py" file in
the Creating_training_data folder.


## Main Text-to-SPARQL File
In this section we discuss the "semantic_parsing.py" in the main directory which is used to perform semantic parsing. 

This file houses the primary function responsible for executing text-to-SPARQL operations on the Kadaster Knowledge Graph (KKG).
It also encompasses the other prompting-based modules:
- Schema selection.
- Span classification, type classification & entity masking.

This file offers various functions and parameters tailored for specific ablations.

Furthermore, this file orchestrates the execution of several other files within our repository during inference:
- Ontology condensation files are invoked.
- Examples are fetched from vector stores.
- The location server API is utilized to retrieve IRIs for location entities, 
leveraging the prompting-based span classification results.


Performing text-to-SPARQL with the optimal model configuration can be done with following code:
semantic_parsing_workflow_few_shot('question_in_natural_language',1)


## KKG Ontology
The ontology_kkg.txt file in the main repository folder provides the ontology of the Kadaster Knowledge Graph (KKG) extracted
from the SHACL shapes. 

It represents the ontology using all the edges in the graph network, presented in tuple format. 
Each tuple comprises three elements:
1. Source Node: The node from which the edge originates.
2. Edge: The property connecting the nodes.
3. Destination Node: The node to which the edge leads.

The file includes the following types of edges:
1. Datatype Edges: Represented as tuples in the format (class, property, datatype).
2. Object Edges: Represented as tuples in the format (class, property, class).
3. Value List Edges: Represented as tuples in the format (class, property, entity).

The directionality of edges in the ontology is read from left to right. 
For example, the object edge ('sor:Gebouw', 'geo:sfWithin', 'wbk:Buurt') indicates that applying the property 
geo:sfWithin to entities of the class sor:Gebouw results in the corresponding entity of the class wbk:Buurt.


## Directories
In this section we discuss all the different directories contained in our repository.

### automated_testing
This directory contains:
1. Our development set
2. Our test set
3. Scripts for hyperparameter tuning.
4. Scripts for testing
5. The parsed queries and evaluation metrics on our test set for the full model and all ablations.
6. The parsed queries and evaluation metrics on our development set during hyperparameter tuning.
7. The condensed ontologies for each question in the test set

- test_and_dev_set.py: This Python file includes two lists: one for the development set and one for the test set. 
 Both lists are populated with dictionaries representing the associated questions. Each dictionary contains (in this order):
1 The natural language question
2 The ground-truth SPARQL query
3 The generalization level of the query (i.i.d., compositional, or zero-shot)
4 The granularity of the location entity associated with the natural language question
5 The name of the location entity
6 A list of the complex SPARQL functionalities used (GROUP BY, HAVING, SUBQUERY, MULTIPLE VARIABLE, AVERAGE, or STRING function, if available)

- development_set.json and test_set.json: These files provide the same information as in test_and_dev_set.py but in JSON format.

- hyperparameter_tuning_dev_set: This folder contains the results of our grid search hyperparameter tuning on the development set. 
We vary the number of in-context examples from 1 to 10 and the number of routes between relevant nodes from 1 to 3. 
Each combination is stored in a separate JSON file, following the same format as development_set.json 
but with additional information for each question dictionary, including the generated SPARQL query and whether its results match the ground-truth query. 
At the end of each JSON file, there is a detailed execution accuracy summary for the entire development set, as well as specific breakdowns by 
1 Generalization levels (iid, compositional, zero-shot)
2 SPARQL operators
3 Queries with new schema items
4 Queries with a particular number of relations
5 Queries with both a specific number of relations and at least one novel schema item.

- performance_and_ablations_with_ontology_test_set: This folder mirrors the structure of hyperparameter_tuning_dev_set but applies it to the test set and includes results where the ontology is INCLUDED in the prompt. The file test_set_performance represents the full model, and the remaining JSON files cover various ablation scenarios.

- performance_and_ablations_without_ontology_test_set: Similar to performance_and_ablations_with_ontology_test_set, 
this folder contains parsed queries and evaluation metrics for configurations where the ontology is EXCLUDED from the prompt. Separate JSON files are provided for the full model configuration and for all ablation scenarios.

- saved_condensed_ontologies: This folder contains a JSON where for each question we saved the condensed ontologies produced by our GTOC algorithm. We used this to check whether the condensed ontologies contain all information (whether the ground-truth results can be obtained by formulating a SPARQL query with the condensed ontologies) and to measure the strength of the condensation (the ratio of the condensed ontology to the full ontology).

- testing_automated.py: This script contains various functions for performing automated evaluations and producing the JSON files with parsed queries and evaluation metrics for both the test, as well as the hyperparameter tuning on the development set.

### Training_data
This folder contains our created training data.

- Main_Training_data: This folder stores the primary training data used in my thesis and for production, 
located in training_data_brt_filter.json. Each question in this JSON file includes (in this order):
1 Question ID (qid)
2 The natural language question
3 The corresponding SPARQL query
4 Golden classes (ground-truth classes)
5 Golden relations (ground-truth relations)
(Note that the golden classes and golden relations are not diretly used in my thesis, but can be used
to train the cross-encoder neural network for schema retrieval that I replicated from TIARA for the KKG)
6 The geographical property considered
7 The granularity considered
8 Meta-data used during SPARQL query construction (e.g., the need for ordering, year filter options). This meta-data is produced in the "brt_filter_query.py" file in the "Creating_Training_data" folder.
9 A prompt string that describes to an LLM the IRI of the location entity associated with the question and 
the class it belongs to. (this can be used to include instance data for in-context examples for example, to enhance symmetry in the future)


- Train_dev_split_for_hyperparameter_tuning: This folder is not used in my thesis. 
It contains separate training and development sets, where the development set includes schema items not present in the 
training data. This is used to train the cross-encoder schema-retrieval neural network from TIARA which i replicated for the KKG.








### Creating_Training_data
This folder contains all files used for creating training data.

- create_training_data.py: Executes the main training data creation script, utilizing all other files in this folder.

- brt_filter_query: Generates natural language questions and corresponding SPARQL queries using templates. Called by create_training_data.py.

- addresses.json: Contains the locations used in the training data.

- templates folder: Contains all YAML files with natural language questions and their attributes.
1. Production_templates: Contains templates used in my thesis for creating training data.
2. train_dev_split_for_hyperparameter_tuning: Not important for my thesis. 
Contains a train/dev split of YAML templates. The dev split includes schema items not in the training data 
and is used to train the cross-encoder neural network for schema retrieval from TIARA, which I replicated for KKG 
(not incorporated in my thesis).

- dependencies.py: Auxiliary file used by the main files in this folder.

- functions.py: Auxiliary file used by the main files in this folder.

- query.py: Auxiliary file used by the main files in this folder.

- query_components.py: Auxiliary file used by the main files in this folder.

- tokenizer.py: Auxiliary file used by the main files in this folder.



### vector_stores
This folder contains five subfolders, each holding FAISS vector stores. 
There are four subfolders for different geographical granularities (used for meta-data prefiltering) 
and one subfolder where all granularities are combined. These vector stores are created from the training data.

- create_vector_store.py: Generates embeddings and creates the FAISS vector stores.

### ontology_tools_kkg
network_creation_and_shacl_ontology_retriever.py: This Python script uses various SPARQL queries to deduce object edges, 
datatype edges, and value list edges to understand the ontology structure of the KKG. 
It then creates a graph network object in NetworkX, facilitating the application of graph traversal 
algorithms on the KKG ontology. The script pre-computes the shortest routes for all unordered pairs in the 
ontology network, storing only valid routes determined through the heuristic described in my thesis. 
These pre-computed routes enable fast ontology condensation during inference.

- saved_network_and_ontology: This folder stores the NetworkX graph network object and various other information 
obtained by "network_creation_and_shacl_ontology_retriever.py". Key files include: 
1 all_edges_dictionary.picke Stores all edges in the graph network.
2 classes.pickle Stores all classes.
3 classes_without_comments.pickle Stores a single string with all classes, which can be directly formatted inside a prompt.
4 prefix_string Stores a single string containing all prefix definitions for SPARQL queries. 
5 When executing a SPARQL query against the KKG, include this prefix string so the query engine recognizes 
all prefixes used in the query.

- dictionaries_that_map_pairs_of_nodes_to_ontolology_items: This folder stores the precomputed graph traversal algorithm results. 
It contains dictionaries where the keys are pairs of nodes in the full ontology graph network, 
and the values are all the ontology triples associated with the shortest routes between them. 
We have a seperate dictionary for shortest routes ranging from 1 to 10.

### shortest_path_algorithms
This folder contains a Python file that loads the NetworkX graph network and other information stored in pickle files by "network_creation_and_shacl_ontology_retriever.py" in the "ontology_tools_kkg folder". This includes the classes and properties string used in the prompt for schema retrieval, and the prefix string appended to SPARQL queries before execution.

- shortest_path.py: Performs ontology condensation using the precomputed results, 
taking relevant classes and properties as input. This file can retrieve various types of information about the 
KKG during inference such as the prefix string, class string etc.

### dense_retrievers_schema
This folder contains a replication of the cross-encoder neural networks for schema retrieval from TIARA, 
including a complete hyperparameter tuning pipeline. Please note that this is not used in my thesis.

### utils
This folder contains auxiliary files used by other Python scripts in this repository in an object-oriented manner.

### Dataloader
This folder contains auxiliary files used by other Python scripts in this repository in an object-oriented manner.
























