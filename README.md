# Graph Traversal Algorithms and In-Context Learning for Enhanced Generalization in Text-to-SPARQL
This repository contains the official implementation of the research paper **“Graph Traversal Algorithms and In-Context Learning for Enhanced Generalization in Text-to-SPARQL”**. The code is designed to support the findings and methodologies presented in the paper. It enables semantic parsing of SPARQL queries on Kadaster Knowledge Graph (KKG).

The software is implemented in **Python 3** (https://www.python.org/) and utilizes the **LangChain framework** (https://www.langchain.com/). It also incorporates the **NetworkX package** (https://networkx.org/) for graph-based computations. Additionally, the system leverages **GPT-4-32k** (https://openai.com/index/gpt-4-research/).

## Installation Instructions
To set up the environment with the required dependencies, please use the `requirements.txt` file provided in this repository. You can install all necessary packages and their specified versions by running the following command:

```
pip install -r requirements.txt
```
This will ensure that all required libraries and dependencies are installed for the proper functioning of the code.

Since we use **GPT-4-32k**, which requires a paid subscription to make API calls, you will need one of the following to run the code in this repository:

1. **An OpenAI API Key**: You can obtain an API key by subscribing to OpenAI's services.
2. **Your Own Model Deployment on Azure**: If you have deployed the model on Azure, you can extract the necessary authentication keys from your deployment.

Our repository supports integration with various Large Language Models (LLMs) through LangChain, including future releases. To set up the API key for your chosen model, modify the `secrets.ini` file and update the LangChain LLM object creation in `semantic_parsing.py`. 
For configuration details specific to your model, refer to the official LangChain documentation: [LangChain Documentation](https://python.langchain.com/v0.2/docs/introduction/).

The `secrets.ini` file contains the API endpoints for Kadaster Knowledge Graph (KKG), which are publicly available and free to use. 
It includes the endpoint for sending SPARQL queries to be executed against KKG, as well as the **LocationServer** endpoint, which can be used to retrieve IRIs of location entities.


Please note that the vector stores for the four granularities (province, municipality, street, and national) fit within the file size limits of this GitHub repository. However, the combined vector store for all granularities exceeds GitHub's file size limit and is therefore excluded from the repository.
That said, all vector stores, including the combined vector store, can be replicated by running the `create_vector_store.py` script located in the `vector_stores` folder.

## Main Text-to-SPARQL File
This section describes the `semantic_parsing.py` file located in the main directory, which is responsible for performing semantic parsing.

The file contains the primary function for executing text-to-SPARQL operations on Kadaster Knowledge Graph (KKG). It also includes various prompting-based modules, such as:

- Schema Selection
- Span Classification, Type Classification, and Entity Masking

In addition, the file provides several functions and parameters designed for specific ablations.

Moreover, the `semantic_parsing.py` file orchestrates the execution of multiple other components within the repository during inference, including:

- Invocation of ontology condensation files
- Fetching of examples from vector stores
- Utilization of the Location Server API to retrieve IRIs for location entities, leveraging the results of the prompting-based span classification.



## KKG Ontology

The `ontology_kkg.txt` file, located in the main repository folder, contains the ontology of Kadaster Knowledge Graph (KKG), extracted from the SHACL shapes.

It represents the ontology as vertex-edge pairs, presented in tuple format. Each tuple consists of three elements:

1. **Source Node**: The node from which the edge originates.
2. **Edge**: The property connecting the nodes.
3. **Destination Node**: The node to which the edge leads.

The file includes the following types of vertex-edge pairs:

1. **Datatype Vertex-Edge Pairs**: Represented as tuples in the format `(class, property, datatype)`.
2. **Object Vertex-Edge Pairs**: Represented as tuples in the format `(class, property, class)`.
3. **Value List Vertex-Edge Pairs**: Represented as tuples in the format `(class, property, entity)`.

The directionality of the edges in the ontology is interpreted from left to right. For example, the object edge `('sor:Gebouw', 'geo:sfWithin', 'wbk:Buurt')` indicates that applying the property `geo:sfWithin` to instances of the class `sor:Gebouw` results in instances of the class `wbk:Buurt`.


## Directories
In this section we discuss all the different directories contained in our repository.

### system_evaluation
1. The `evaluate_system.py` file is an automated script for performing hyperparameter tuning on the development set, evaluating the test set (including ablations), and saving the results of both the hyperparameter tuning and the test set evaluations.

2. The `test_and_dev_set.py` file contains two lists: one for the development set and one for the test set. Each list is populated with dictionaries representing the associated questions. Each dictionary contains the following fields:
   - The natural language question
   - The ground-truth SPARQL query
   - The generalization level of the query (i.i.d., compositional, or zero-shot)
   - The granularity of the location entity associated with the natural language question
   - The name of the location entity
   - A list of the complex SPARQL functionalities used (e.g., `GROUP BY`, `HAVING`, `SUBQUERY`, `MULTIPLE VARIABLE`, `AVERAGE`, or `STRING` functions, if applicable)

3. The `development_set.json` and `test_set.json` files contain the same information as in `test_and_dev_set.py`, but are provided in JSON format.

4. The `development_set.json` sub-folder contains the results of our grid search hyperparameter tuning on the development set. In this process, we vary the number of in-context examples (from 1 to 10) and the number of routes between relevant nodes (from 1 to 3). Each combination is stored in a separate JSON file, following the same format as `development_set.json`, but with additional information for each question dictionary. This includes the generated SPARQL query and a comparison of the query results with the ground-truth query.
At the end of each JSON file, a detailed execution accuracy summary is provided. Execution accuracy is reported for the entire development set as well as for specific subsets, including:
   - Generalization levels (i.i.d., compositional, zero-shot)
   - SPARQL operators used
   - Queries with new schema items
   - Queries with a specific number of relations
   - Queries with both a particular number of relations and at least one novel schema item

5. The `performance_and_ablations_with_ontology_test_set` sub-folder mirrors the structure of `hyperparameter_tuning_dev_set`, but is applied to the test set and includes results where the ontology is **included** in the prompt. The `test_set_performance` file represents the performance of the full model, while the remaining JSON files correspond to various ablation scenarios.

6. The `performance_and_ablations_without_ontology_test_set` sub-folder mirrors the structure of the `performance_and_ablations_with_ontology_test_set` sub-folder, but contains parsed queries and evaluation metrics for configurations where the ontology is **excluded** from the prompt. Separate JSON files are provided for the full model configuration and for all ablation scenarios.

7. The `saved_condensed_ontologies` sub-folder contains a JSON file that stores the condensed ontologies produced by our GTOC algorithm and the naive condensation approach. These condensed ontologies are used to verify whether they retain all necessary information (i.e., whether ground-truth results can be obtained by formulating a SPARQL query with the condensed ontologies) and to assess the strength of the condensation, defined as the ratio of the condensed ontology to the full ontology.


### training_data
This directory contains the 24,000 training examples that we created. The training data is provided in JSON format within the `training_data_brt_filter.json` file.

Each question in this JSON file includes the following fields:

1. **Question ID (qid)**: A unique identifier for the question.
2. **Natural language question**: The question phrased in natural language.
3. **Corresponding SPARQL query**: The SPARQL query that answers the question.
4. **Golden classes**: The ground-truth classes associated with the question.
5. **Golden relations**: The ground-truth relations required to construct the SPARQL query.
6. **Geographical property**: The geographical property considered in the query.
7. **Granularity**: The level of granularity associated with the location entity in the query.
8. **Meta-data**: Additional information used during SPARQL query construction (e.g., the need for ordering, year filter options). This meta-data is produced by the `brt_filter_query.py` script located in the `create_training_data` directory.
9. **Prompt string**: A string provided to a Large Language Model (LLM) that describes the IRI of the location entity associated with the question and its class. This prompt can be utilized, for example, to include instance data for in-context examples, enabling enhanced symmetry for future research.

### create_training_data
This directory contains all files necessary for generating the training data.

The `create_training_data.py` file is the primary script for generating the training data. Running this script produces and saves the training data in the training_data folder. The create_training_data.py script depends on the following files:

1. **`brt_filter_query.py`**: Implements the core logic for generating natural language questions and their corresponding SPARQL queries.
2. **`addresses.json`**: Provides location data used in the training examples.
3. **`templates/brt_filter_query.yml`**: Contains YAML templates for creating natural language questions.
4. Additional support files that facilitate the functionality of the main scripts:
   - `dependencies.py`
   - `functions.py`
   - `query.py`
   - `query_components.py`
   - `tokenizer.py`

### vector_stores
This directory is structured to include five sub-folders, each containing a FAISS vector store:

1. **Geographical Granularities**: Four subfolders are provided, corresponding to distinct geographical granularities, and are utilized for meta-data prefiltering.
2. **Combined Vector Store**: The subfolder for the combined vector store, which consolidates all granularities, is excluded from the repository due to GitHub's file size limitations.

All vector stores, including the combined one, can be recreated by running the `create_vector_store.py` script. This script processes the JSON file containing the complete training data, generates embeddings, and produces the FAISS vector stores for each granularity as well as the combined store.


### precompute_shortest_paths
The `network_creation_and_shacl_ontology_retriever.py` script in this directory performs the following tasks: 

- Executes SPARQL queries to extract the ontology defined in the SHACL shapes of Kadaster Knowledge Graph (KKG).
- Constructs a NetworkX graph network based on the extracted ontology.
- Implements the heuristic described in our research for detecting invalid routes within the ontology graph network.
- Precomputes valid paths between all unordered pairs of nodes within the graph network.
- Saves the precomputed valid routes, ontology information, and the graph network.



### ontology_condensation_inference
The `ontology_condensation_inference.py` file in this directory defines an object responsible for loading all data generated and saved by the `network_creation_and_shacl_ontology_retriever.py` script. 

This object offers the following functionalities:

1. **Apply GTOC**: Implement the Graph Traversal Ontology Condenser (GTOC) approach for ontology condensation, as described in our research paper.
2. **Apply the Naive Ontology Condensation Approach**: Execute the naive ontology condensation method, also outlined in our research.
3. **Access Ontology Information**: Retrieve various details related to the ontology.
4. **Access the NetworkX Ontology Graph**: Interact with the previously constructed NetworkX ontology graph.

























