import os
import configparser
import requests
from typing import List, Dict, Any
import networkx as nx
import pickle
from itertools import combinations
import itertools

"""
This file performs the following main tasks:
1) Extracts the SHACL ontology from the Kadaster Knowledge Graph using SPARQL queries.
2) Builds a graph network in NetworkX from the extracted SHACL ontology.
3) Pre-computes the shortest routes between nodes and saves them.
"""

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
# Put the absolute path to the .ini file here:
path = r"../secrets.ini"
config.read(path)

locatie_url = config['KADASTER']['LOC_SEARCH_API']
lookup_url = config['KADASTER']['LOC_LOOKUP_API']
sparql_url = config['KADASTER']['KKG_API']

os.environ['OPENAI_API_KEY'] = config['OPENAI_GPT_3.5_TURBO']['AZURE_API_KEY']
os.environ['OPENAI_API_TYPE'] = config['OPENAI_GPT_3.5_TURBO']['AZURE_API_TYPE']
os.environ['OPENAI_API_VERSION'] = config['OPENAI_GPT_3.5_TURBO']['AZURE_API_VERSION']
os.environ['OPENAI_API_BASE'] = config['OPENAI_GPT_3.5_TURBO']['AZURE_API_BASE']

'''
This dictionary defines commonly used URI prefixes along with their corresponding shorthand notations.
These prefixes are used to simplify references to entities in various ontologies, 
enabling concise and readable SPARQL queries.
'''

full_prefix_dictionary = {

    'http://schema.org/': 'sdo0:',
    'http://www.w3.org/2002/07/owl#': 'owl:',
    'http://www.w3.org/2000/01/rdf-schema#': 'rdfs:',
    'http://www.w3.org/1999/02/22-rdf-syntax-ns#': 'rdf:',

    'http://www.opengis.net/ont/geosparql#': 'geo:',
    'https://data.kkg.kadaster.nl/sor/model/def/': 'sor:',
    'https://data.kkg.kadaster.nl/sor/model/con/': 'sor-con:',
    'https://data.kkg.kadaster.nl/kad/model/def/': 'kad:',
    'https://data.kkg.kadaster.nl/kad/model/con/': 'kad-con:',
    'http://brt.basisregistraties.overheid.nl/def/top10nl#': 'brt:',
    'http://bag.basisregistraties.overheid.nl/def/bag#': 'bag:',
    'http://www.w3.org/2001/XMLSchema#': 'xsd:',
    'http://www.w3.org/2004/02/skos/core#': 'skos:',
    'https://data.labs.kadaster.nl/cbs/wbk/vocab/': 'wbk:',
    'http://purl.org/linked-data/cube#': 'cube:',
    'https://data.kkg.kadaster.nl/nen3610/model/def/': 'nen3610:',

    'http://www.w3.org/ns/shacl#': 'sh:',
    'http://www.w3.org/2006/time#': 'time:',
    'http://www.opengis.net/def/function/geosparql/': 'geof:',
    'http://www.opengis.net/def/uom/OGC/1.0/': 'uom:',
    'https://linkeddata.cultureelerfgoed.nl/def/ceo#': 'ceo:',
    'http://www.wikidata.org/prop/': 'p:',
    'http://www.wikidata.org/prop/statement/': 'ps:',
    'https://data.kkg.kadaster.nl/id/provincie/': 'provincie:',
    'http://www.openlinksw.com/schemas/bif#': 'bif:',
    'https://api.labs.kadaster.nl/datasets/brt/top10nl/services/default/sparql': 'top10nl:',
    'https://brt.basisregistraties.overheid.nl/top10nl2/id/hoofdverkeersgebruik/snelverkeer': 'snelverkeer:',

    # Some extra prefixes. Double check whether actually needed.
    'http://purl.org/dc/terms/': 'dct:',
    'http://www.w3.org/ns/prov#': 'prov:',
    'http://bp4mc2.org/def/mim#': 'mim:',
    'http://purl.org/dc/elements/1.1/': 'dcelements:',
    'http://www.opengis.net/ont/gml#': 'gml:'
}

'''
In this section, we define auxiliary methods that support the functionality of the rest of the file.
'''

def create_prefix_start_for_SPARQL_queries(prefix_dict):
    """
    Generates a string of PREFIX declarations for use in SPARQL queries.

    This method takes a dictionary of namespace URIs and their corresponding shorthand
    prefixes, then constructs the appropriate PREFIX statements that can be prepended to
    a SPARQL query. These PREFIXes allow for using shorthand notations instead of full URIs
    when writing SPARQL queries.

    Args:
        prefix_dict (dict): A dictionary where keys are full namespace URIs and values
                             are the corresponding shorthand prefixes to be used in the query.

    Returns:
        str: A string of PREFIX declarations to be included at the beginning of a SPARQL query.

    Example:
        prefix_dict = {
            'http://schema.org/': 'sdo0:',
            'http://www.w3.org/2002/07/owl#': 'owl:'
        }
        result = create_prefix_start_for_SPARQL_queries(prefix_dict)
        print(result)
        # Output:
        # PREFIX sdo0: <http://schema.org/>
        # PREFIX owl: <http://www.w3.org/2002/07/owl#>
    """
    prefix_string = ""
    for key, value in prefix_dict.items():
        prefix_string = prefix_string + "PREFIX %s <%s>\n" % (value, key)
    return prefix_string

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
        the client making the request.

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


def get_items_in_list_format_from_sparql_query(query, variable_name):
    """
    Executes a SPARQL query and converts the resulting JSON response into a Python list.

    This function is particularly useful when querying RDF triple stores, as it extracts the values
    of a specified variable from the query results and formats them as a list.

    Args:
        query (str): The SPARQL query to execute.
        variable_name (str): The name of the variable in the SPARQL query whose values should be extracted.

    Returns:
        list: A list of extracted values corresponding to the specified variable from the SPARQL query results.
        str: An error message if the query execution fails.
    """
    result = run_sparql(query)
    if result == "That query failed. Perhaps you could try a different one?":
        return result
    result_list = []
    for dict in result:
        result_list.append(get_nested_value(dict, [variable_name, 'value']))
    return result_list


class SHACL_ontology_retriever:
    """
    A utility class for extracting and analyzing the SHACL ontology from the Kadaster Knowledge Graph.

    Key Features:
    - Provides methods to retrieve and process the SHACL ontology.
    - Identifies paths of length 3 within the ontology that are absent in the underlying instance data.
      This functionality supports the heuristic outlined in our research paper for discarding invalid paths.
    """
    def __init__(self):
        # list with classes
        self.classes = self.get_classes()
        # list with properties
        self.properties = self.get_properties()

        # list with tuples representing edges
        self.object_edges = self.get_object_edges()
        # list with tuples representing edge
        self.object_edges_with_restrictions = self.get_object_edges_with_restrictions()
        # list with tuples representing edges
        self.datatype_edges = self.get_datatype_edges()
        # list with tuples representing subclass edges
        self.subclass_structure = self.get_subclass_structure()

        # A dictionary that maps full names to prefixes
        # This helps to make International Resource Identifiers more readable
        self.prefix_dictionary = full_prefix_dictionary

        # a string which contains all the prefixes in the correct format to paste
        # at the start of a SPARQL query
        self.prefix_string = create_prefix_start_for_SPARQL_queries(full_prefix_dictionary)

        self.replace_prefixes()

        self.fix_class_list_such_that_it_only_contains_classes_in_the_network_graph()

        # We format all classes and relations (without comments) into one big string.
        # This can be used to let a Large Language Model decide which classes are relevant
        # basd on natural language.
        self.classes_without_comments_string = self.create_classes_without_comments_string()
        self.relations_without_comments_string = self.create_relations_without_comments_string()

        dictionary_that_maps_property_to_neighboring_nodes, dictionary_that_maps_property_to_triples = self.create_dictionary_that_maps_property_to_neighboring_nodes()
        self.dictionary_that_maps_property_to_neighboring_nodes = dictionary_that_maps_property_to_neighboring_nodes
        self.dictionary_that_maps_property_to_neighboring_triples = dictionary_that_maps_property_to_triples

        # we store invalid paths
        self.invalid_paths = self.get_invalid_paths_in_object_edges()

    def get_classes(self):
        """
        Retrieves a list of classes defined in the SHACL NodeShapes from the Kadaster Knowledge Graph.

        Returns:
            list: A list of class IRIs extracted from the SHACL ontology.
        """
        query_to_retrieve_classes_from_shacl_shapes = """
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX sh: <http://www.w3.org/ns/shacl#>

        SELECT DISTINCT ?class
        WHERE {
        ?shape a sh:NodeShape ;
            sh:targetClass ?class .
        }
        """
        classes = get_items_in_list_format_from_sparql_query(query_to_retrieve_classes_from_shacl_shapes, 'class')

        return classes

    def fix_class_list_such_that_it_only_contains_classes_in_the_network_graph(self):
        '''
        Resolves data quality issues by ensuring the class list only includes classes connected in the ontology network graph.

        This is done by intersecting the retrieved classes with class nodes present in object edges, datatype edges,
        and subclass structures to exclude disconnected classes.
        '''

        classes_in_ontology_network_graph = set()

        for tuple in self.object_edges:
            classes_in_ontology_network_graph.add(tuple[0])
            classes_in_ontology_network_graph.add(tuple[2])

        for tuple in self.object_edges_with_restrictions:
            classes_in_ontology_network_graph.add(tuple[0])
            classes_in_ontology_network_graph.add(tuple[2])

        for tuple in self.datatype_edges:
            classes_in_ontology_network_graph.add(tuple[0])

        for tuple in self.subclass_structure:
            classes_in_ontology_network_graph.add(tuple[0])
            classes_in_ontology_network_graph.add(tuple[2])
        all_classes_set = set(self.classes)
        classes_in_class_list_but_not_in_network = all_classes_set.difference(classes_in_ontology_network_graph)

        self.classes = list(set(self.classes).difference(classes_in_class_list_but_not_in_network))

    def get_properties(self):
        """
        Retrieves all properties (relations) defined in the SHACL shapes from the Kadaster Knowledge Graph.

        Executes a SPARQL query to extract unique properties and returns them as a list.

        Returns:
            list: A list of properties (relations) extracted from the SHACL shapes.
        """
        query_to_retrieve_relations_from_shacl_shapes = """
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX sh: <http://www.w3.org/ns/shacl#>
        SELECT DISTINCT ?property
        WHERE {
        ?shape a sh:NodeShape ;
            sh:targetClass ?originclass ;
            sh:property ?prop .

        ?prop sh:path ?property ;
        }
        """

        relations = get_items_in_list_format_from_sparql_query(query_to_retrieve_relations_from_shacl_shapes,
                                                               'property')

        return relations

    def get_object_edges(self):
        """
        Retrieves object edges in the ontology, representing relationships between classes as triples:
        (class1, relation, class2).

        Constructs a SPARQL query to extract connections where a property links an origin class to a target class.

        The query includes the following filter to exclude edges with value-list restrictions:

            FILTER NOT EXISTS {
                ?prop sh:in ?list .
            }

        For edges where the target class has specific value restrictions, a separate query is used to traverse
        the restriction list and add an edge for each valid element in the
        function "get_object_edges_with_restrictions()".

        Additionally, this function manually adds edges for integrating data from the Central Bureau of
        Statistics (CBS), enhancing compatibility with Kadaster's knowledge graph and enabling queries about
        provinces and municipalities.

        Returns:
            list: A list of tuples, where each tuple represents an edge in the format
                  (origin_class, property, target_class).
        """

        get_object_edges_query = """
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX sh: <http://www.w3.org/ns/shacl#>

        SELECT DISTINCT ?originclass ?property ?targetclass
        WHERE {
        ?shape a sh:NodeShape ;
            sh:targetClass ?originclass ;
            sh:property ?prop .
        ?prop sh:path ?property ;
            sh:class ?targetclass .

        FILTER NOT EXISTS {
        ?prop sh:in ?list .
        }
        }
        """
        result = run_sparql(get_object_edges_query)

        def get_one_variable_from_sparql_result_in_list(result, variable_name):
            result_list = []
            for dict in result:
                result_list.append(get_nested_value(dict, [variable_name, 'value']))
            return result_list

        origin_class_list = get_one_variable_from_sparql_result_in_list(result, 'originclass')
        property_list = get_one_variable_from_sparql_result_in_list(result, 'property')
        target_class_list = get_one_variable_from_sparql_result_in_list(result, 'targetclass')

        edge_list = []
        for origin_cls, prop, target_cls in zip(origin_class_list, property_list, target_class_list):
            edge_list.append((origin_cls, prop, target_cls))

        # Manually add object edges that are not included in the SHACL ontology.
        # These edges originate from the Central Bureau of Statistics (CBS) knowledge graph.
        # Although CBS data is not maintained by Kadaster, it can be integrated with the Kadaster knowledge graph.
        # This integration aligns with the principles of the semantic web and linked data.
        # Adding these edges enables querying about provinces and municipalities within the combined graph.
        edge_list.append(('sor:Gebouw', 'geo:sfWithin', 'wbk:Buurt'))
        edge_list.append(('wbk:Buurt', 'geo:sfWithin', 'wbk:Wijk'))
        edge_list.append(('wbk:Wijk', 'geo:sfWithin', 'wbk:Gemeente'))
        edge_list.append(('sor:Gemeente', 'owl:sameAs', 'wbk:Gemeente'))

        return edge_list

    def get_invalid_paths_in_object_edges(self):
        """
        Identifies invalid paths of length 3 in the ontology that do not exist in the underlying instance data.

        This method examines the ontology to identify sequences of three connected classes (paths of length 3).
        It verifies each path's presence in the underlying instance data by executing SPARQL queries.
        If a path is absent in the instance data, it is marked as invalid. This functionality is crucial for
        implementing the heuristic described in the research paper to discard invalid paths.

        The process includes:
        - Enumerating all possible paths of three connected classes using object edges.
        - Validating each path by checking its existence in the instance data through SPARQL queries.
        - Returning a list of invalid paths that do not exist in the instance data.

        Returns:
            list: A list containing lists of invalid paths, where each path is represented as a list of three classes
                  [class1, class2, class3]. Note paths can be viewed from both left-to-right and right-to-left,
                  To ensure consistency and ease of comparison, paths are standardized by ordering class1 and class3 in
                  such that class1 is always lexicographically smaller than class3. The key constraint is that the
                  middle class (class2) always remains fixed in its position.
        """

        from itertools import combinations

        def generate_unordered_pairs(tuples_list):
            pairs = []
            for pair in combinations(tuples_list, 2):
                pairs.append(pair)
            return pairs

        unordered_pairs_of_object_edges = generate_unordered_pairs(self.object_edges)
        unorder_pairs_of_object_edges_that_have_one_matching_class = []
        for pair in unordered_pairs_of_object_edges:
            classes_in_first_edge = set()
            classes_in_second_edge = set()

            classes_in_first_edge.add(pair[0][0])
            classes_in_first_edge.add(pair[0][2])

            classes_in_second_edge.add(pair[1][0])
            classes_in_second_edge.add(pair[1][2])

            intersection_set = classes_in_first_edge.intersection(classes_in_second_edge)

            if len(intersection_set) == 1:
                unorder_pairs_of_object_edges_that_have_one_matching_class.append(pair)
        dictionary_3_class_paths = dict()
        for pair in unorder_pairs_of_object_edges_that_have_one_matching_class:
            classes_in_first_edge = set()
            classes_in_second_edge = set()
            classes_in_first_edge.add(pair[0][0])
            classes_in_first_edge.add(pair[0][2])

            classes_in_second_edge.add(pair[1][0])
            classes_in_second_edge.add(pair[1][2])

            union_classes = classes_in_first_edge.union(classes_in_second_edge)

            # This is the class that matches
            intersection_set = classes_in_first_edge.intersection(classes_in_second_edge)

            # these are the remaining classes in the path
            # we sort them in order to create an ordered tuple representing the path
            remaining_classes = sorted(union_classes.difference(intersection_set))
            sorted_class_path_as_tuple = (remaining_classes[0], intersection_set.pop(), remaining_classes[1])

            # Now we create a dictionary
            # In this dictionary we store the pairs of object nodes that belong to a class path of length 3
            # By default we set fact_level_path to False.
            # Later we loop through the object node pairs and try to find a path in the fact_level_structure
            # by using SPARQL queries. Only if one of the object edge pairs is found in the fact_level structure do we set
            # it to True.
            if sorted_class_path_as_tuple in dictionary_3_class_paths:

                dictionary_3_class_paths[sorted_class_path_as_tuple]['pairs'].append(pair)

            else:
                dictionary_3_class_paths[sorted_class_path_as_tuple] = dict()
                dictionary_3_class_paths[sorted_class_path_as_tuple]['pairs'] = []
                dictionary_3_class_paths[sorted_class_path_as_tuple]['pairs'].append(pair)
                dictionary_3_class_paths[sorted_class_path_as_tuple]['fact_level_path'] = False

        for three_class_path in dictionary_3_class_paths:
            # print(three_class_path)
            for object_edge_pair in dictionary_3_class_paths[three_class_path]['pairs']:
                # print(object_edge_pair)
                fact_level_check_query = """
                SELECT DISTINCT (count(?class) as ?count)
                WHERE {
                ?class a %s;
                    %s/a %s;
                    %s/a %s.
                }

                """

                # We determine in which direction we should apply the property/relation in the query
                list_with_relation_class_tuples = []
                for edge in object_edge_pair:
                    new_class = None
                    relation_and_direction = None
                    if edge[0] != three_class_path[1]:
                        new_class = edge[0]
                        relation_and_direction = '^' + edge[1]
                    else:
                        new_class = edge[2]
                        relation_and_direction = edge[1]

                    list_with_relation_class_tuples.append((relation_and_direction, new_class))

                fact_level_check_query = fact_level_check_query % (
                three_class_path[1], list_with_relation_class_tuples[0][0],
                list_with_relation_class_tuples[0][1], list_with_relation_class_tuples[1][0],
                list_with_relation_class_tuples[1][1])

                result_fact_level_count = run_sparql(
                    self.prefix_string + '\n' + fact_level_check_query)

                result_fact_level_count = result_fact_level_count[0]['count']['value']
                if int(result_fact_level_count) > 0:
                    dictionary_3_class_paths[three_class_path]['fact_level_path'] = True

        list_with_invalid_paths = []

        for key, value in dictionary_3_class_paths.items():

            if value['fact_level_path'] == False:
                list_with_invalid_paths.append(list(key))
        return list_with_invalid_paths

    def get_object_edges_with_restrictions(self):
        """
        Generates object edges for every value-list entity contained in the SHACL ontology.

        This function constructs vertex-edge pairs of the format:
            (class, property, value-list-entity)

        For each value-list entity associated with a class, a corresponding vertex-edge pair is created.

        The SPARQL query utilizes the following pattern:
            sh:in ?list .
            ?list rdf:rest*/rdf:first ?value .

        This pattern efficiently traverses all value-list entities in the SHACL ontology using SPARQL,
        allowing us to directly retrieve all value list entities associated with a class.

        Additionally, the retrieved value-list entities are appended to the class list (self.classes),
        ensuring they can seamlessly be included in the prompt used for simultaneously selecting both
        classes and value-list entities.

        Returns:
            list: A list of tuples representing edges, where each tuple is in the format
                  (origin_class, property, value_list_entity).
        """

        get_object_edges_query = """
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX sh: <http://www.w3.org/ns/shacl#>

        SELECT DISTINCT ?originclass ?property ?value
        WHERE {
        ?shape a sh:NodeShape ;
            sh:targetClass ?originclass ;
            sh:property ?prop .

        ?prop sh:path ?property ;
            sh:class ?targetclass ;
                       sh:in ?list .

        ?list rdf:rest*/rdf:first ?value .
        }
        """
        result = run_sparql(get_object_edges_query)

        def get_one_variable_from_sparql_result_in_list(result, variable_name):
            result_list = []
            for dict in result:
                result_list.append(get_nested_value(dict, [variable_name, 'value']))
            return result_list

        origin_class_list = get_one_variable_from_sparql_result_in_list(result, 'originclass')
        property_list = get_one_variable_from_sparql_result_in_list(result, 'property')
        target_list = get_one_variable_from_sparql_result_in_list(result, 'value')

        edge_list = []
        for origin_cls, prop, target in zip(origin_class_list, property_list, target_list):
            edge_list.append((origin_cls, prop, target))
            self.classes.append(target)

        return edge_list

    def get_datatype_edges(self):
        '''
        Retrieves all datatype vertex-edge pairs in the ontology, representing relationships between classes and
        literal types (such as strings or integers) through properties.

        The resulting triples describe edges in a graph with the format: (class1, property, datatype), where the
        final node is not another class, but the datatype associated with the literals.

        This function constructs a SPARQL query to extract the relevant class-property-datatype connections.

        :return: A list of tuples, where each tuple represents an edge in the format
                 (origin_class, property, datatype).
        '''

        get_datatype_edges_query = """
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX sh: <http://www.w3.org/ns/shacl#>

        SELECT DISTINCT ?originclass ?property ?datatype
        WHERE {
        ?shape a sh:NodeShape ;
            sh:targetClass ?originclass ;
            sh:property ?prop .

        ?prop sh:path ?property ;
            sh:datatype ?datatype .
        }
        """
        result = run_sparql(get_datatype_edges_query)

        def get_one_variable_from_sparql_result_in_list(result, variable_name):
            result_list = []
            for dict in result:
                result_list.append(get_nested_value(dict, [variable_name, 'value']))
            return result_list

        origin_class_list = get_one_variable_from_sparql_result_in_list(result, 'originclass')
        property_list = get_one_variable_from_sparql_result_in_list(result, 'property')
        datatype_list = get_one_variable_from_sparql_result_in_list(result, 'datatype')

        edge_list = []
        for origin_cls, prop, dat_type in zip(origin_class_list, property_list, datatype_list):
            edge_list.append((origin_cls, prop, dat_type))

        return edge_list

    def get_subclass_structure(self):
        """
        Retrieves the subclass-superclass relationships for each class in the ontology, constructing triples that
        describe the class hierarchy (subclass, "rdfs:subClassOf", superclass).

        This method iterates through all classes in the ontology and retrieves their superclasses using SPARQL queries,
        then creates triples that represent the subclass/superclass structure.

        :return: A list of triples, each representing a subclass-superclass relationship in the format
                 (subclass, "rdfs:subClassOf", superclass).
        """
        subclass_structure_set = set()

        def add_triples(subclass, superclass_list):
            for sup_cls in superclass_list:
                subclass_structure_set.add((subclass, "rdfs:subClassOf", sup_cls))

        for cls in self.classes:
            query_to_get_superclass = """
            PREFIX kad: <https://data.kkg.kadaster.nl/kad/model/def/>
            PREFIX sor: <https://data.kkg.kadaster.nl/sor/model/def/>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

            SELECT DISTINCT ?superclass
            WHERE {
                %s rdfs:subClassOf ?superclass .
            }
            """ % (cls)

            superclass_list = get_items_in_list_format_from_sparql_query(query_to_get_superclass, 'superclass')
            if superclass_list != "That query failed. Perhaps you could try a different one?":
                for sup_cls in superclass_list:
                    add_triples(cls, sup_cls)

        return list(subclass_structure_set)

    def replace_prefixes(self):
        """
        Replaces full URI prefixes with their shorthand notation in all ontology-related objects.

        This method updates the following elements:
        - Classes (`self.classes`)
        - Properties (`self.properties`)
        - Object edges (`self.object_edges`)
        - Datatype edges (`self.datatype_edges`)
        - Object edges with restrictions (`self.object_edges_with_restrictions`)
        - Subclass structure (`self.subclass_structure`)

        It iterates through each element and replaces full URIs with their corresponding shorthand prefixes using
        the provided `full_prefix_dictionary`.

        This ensures consistency and readability by simplifying the URIs throughout the ontology data.
        """
        def replace_substrings_with_prefixes_in_list(input_list, full_prefix_dictionary=full_prefix_dictionary):
            # Iterate through each string in the list
            for i in range(len(input_list)):
                # Iterate through each key-value pair in the prefix dictionary
                for full_prefix, prefix in full_prefix_dictionary.items():
                    # Replace the substring with the corresponding prefix

                    input_list[i] = input_list[i].replace(full_prefix, prefix)
            return input_list

        def replace_substrings_with_prefixes_in_string(input_string, full_prefix_dictionary=full_prefix_dictionary):

            # Iterate through each key-value pair in the prefix dictionary
            for full_prefix, prefix in full_prefix_dictionary.items():
                # Replace the substring with the corresponding prefix

                input_string = input_string.replace(full_prefix, prefix)
            return input_string

        self.classes = replace_substrings_with_prefixes_in_list(self.classes)
        self.properties = replace_substrings_with_prefixes_in_list(self.properties)

        def replace_substring_with_prefixes_in_edge_list(edge_list):
            for i, tuple in enumerate(edge_list):
                element_1 = replace_substrings_with_prefixes_in_string(tuple[0])
                element_2 = replace_substrings_with_prefixes_in_string(tuple[1])
                element_3 = replace_substrings_with_prefixes_in_string(tuple[2])
                new_tuple = (element_1, element_2, element_3)

                edge_list[i] = new_tuple

            return edge_list

        self.object_edges = replace_substring_with_prefixes_in_edge_list(self.object_edges)
        self.datatype_edges = replace_substring_with_prefixes_in_edge_list(self.datatype_edges)
        self.object_edges_with_restrictions = replace_substring_with_prefixes_in_edge_list(
            self.object_edges_with_restrictions)
        self.subclass_structure = replace_substring_with_prefixes_in_edge_list(self.subclass_structure)

    def create_classes_without_comments_string(self):
        """
        Generates a string containing all classes and value-list entities, each on a new line.

        This function iterates through the list of classes (`self.classes`) and concatenates them into a single
        string, with each class or value-list entity separated by a newline. This string can be used later to
        facilitate the selection of relevant classes and value-list entities in prompts.

        :return: A string where each class or value-list entity is listed on a new line.
        """
        class_string = ""
        for class_ in self.classes:
            class_string += "%s\n" % class_
        return class_string

    def create_relations_without_comments_string(self):
        """
        Generates a string containing all properties (relations), each on a new line.

        This function iterates through the list of properties (`self.properties`) and concatenates them into a
        single string, with each property (relation) separated by a newline. This string can be used later to
        facilitate the selection of relevant properties in prompts.

        :return: A string where each property (relation) is listed on a new line.
        """
        relation_string = ""
        for relation in self.properties:
            relation_string += "%s\n" % relation
        return relation_string

    def create_dictionary_that_maps_property_to_neighboring_nodes(self):
        """
        This function processes the object edges (`self.object_edges`), object edges with restrictions
        (`self.object_edges_with_restrictions`), and datatype edges (`self.datatype_edges`) to construct two
        dictionaries:

        1. `dictionary_set`: A dictionary where each property is a key, and the corresponding value is a set of all
           nodes that are connected to it.
        2. `dictionary_triple_string`: A dictionary where each property is a key, and the corresponding value is a set
           of all vertex-edge pairs (as strings in the format "element1 property element2") that involve that property.

        These dictionaries help in mapping properties to the nodes they are related to and the vertex-edge relationships
        they participate in.

        Additionally, the dictionaries are saved as pickle files for later use:
        - `dictionary_set` is saved in:
          "precompute_shortest_paths/saved_data/dictionary_that_maps_property_to_neighboring_nodes"
        - `dictionary_triple_string` is saved in:
          "precompute_shortest_paths/saved_data/dictionary_that_maps_property_to_all_ontology_triples_as_string"

        :return:
            tuple: A tuple containing two dictionaries:
                - `dictionary_set`: Maps properties to sets of neighboring nodes.
                - `dictionary_triple_string`: Maps properties to sets of vertex-edge pairs as strings.
        """

        dictionary_set = dict()
        dictionary_triple_string = dict()

        def process_edge_type(edge_type):
            for edge in edge_type:
                if edge[1] in dictionary_set:
                    dictionary_set[edge[1]].add(edge[0])
                    dictionary_set[edge[1]].add(edge[2])
                    triple_string = "%s %s %s" % (edge[0], edge[1], edge[2])
                    dictionary_triple_string[edge[1]].add(triple_string)
                else:
                    dictionary_set[edge[1]] = set()
                    dictionary_set[edge[1]].add(edge[0])
                    dictionary_set[edge[1]].add(edge[2])
                    triple_string = "%s %s %s" % (edge[0], edge[1], edge[2])
                    dictionary_triple_string[edge[1]] = set()
                    dictionary_triple_string[edge[1]].add(triple_string)

        process_edge_type(self.object_edges)
        process_edge_type(self.object_edges_with_restrictions)
        process_edge_type(self.datatype_edges)

        main_path = './saved_data/'
        # we save dictionary_that_maps_property_to_neighboring_nodes
        with open(main_path + 'dictionary_that_maps_property_to_neighboring_nodes', 'wb') as f:
            pickle.dump(dictionary_set, f)

        # we save dictionary_that_maps_property_to_all_ontology_triples_as_string
        with open(main_path + 'dictionary_that_maps_property_to_all_ontology_triples_as_string', 'wb') as f:
            pickle.dump(dictionary_triple_string, f)

        return dictionary_set, dictionary_triple_string


class graph_network_creator:
    def __init__(self):
        self.shacl_ontology = SHACL_ontology_retriever()
        self.graph_network = nx.DiGraph()
        self.edge_labels = dict()
        self.create_network_graph()
        self.dictionary_that_maps_all_pairs_to_the_necessary_ontology_items_in_top_k_shortest_routes = self.create_dictionary_that_maps_all_pairs_to_the_necessary_ontology_items_in_top_k_shortest_routes()
        self.save_created_network_graph_and_ontology_tools()

    def create_network_graph(self):
        """
        This function formulates the ontology as a directed graph network using the `networkx` package.

        The graph models the following types of relationships:

        1. **Object Vertex-Edge Pairs**: Modeled as **bi-directional edges**.
           In SPARQL, the reverse property operator (`^`) allows traversal of properties in the opposite direction.
           By including edges in both directions, the graph supports scenarios where properties are defined in only one direction, which is a common occurrence in real-world ontologies.

        2. **Datatype Vertex-Edge Pairs**: For each datatype vertex-edge pair in the ontology, a **directed edge is added pointing to the datatype**.
           This ensures that nonsensical paths like:
           "building -> buildingyear -> xsd:PositiveInteger -> average_temperature -> Netherlands"
           are prevented.

        3. **Value-List-Entity Vertex-Edge Pairs**: Similar to datatype edges,
           only **edges pointing towards the value-list entity** are modeled.
           This is because the value-list entities in the KKG SHACL ontology are not logically suited to serve as
           intermediate nodes in graph paths.

        4. **Subclass Relationships**: Directed edges are added in **both directions** for subclass relationships.
           This could be useful for traversing subclass hierarchies in either direction when constructing SPARQL queries.

        :return:
            This method returns nothing, but it instantiates two instance attributes:
            1) self.graph_network:
               networkx.DiGraph: A directed graph object representing the ontology, with nodes and edges reflecting object
               relations, datatype constraints, value-list entities, and subclass hierarchies.
            2) self.edge_labels:
               Dictionary: A mapping where the keys are SORTED tuples representing adjacent nodes in the graph network
               (e.g., ('node1', 'node2')), and the values are lists of ontology triples (formatted as strings, for
               example: "node1 property node2") that describe the relationships enabling traversal between these nodes.
               Since we do not store properties in the NetworkX `DiGraph` , this dictionary provides a way to
               capture and reference the specific ontology properties used for traversing from one node to another.
               IMPORTANT: the tuple keys are SORTED.

        """

        for object_edge in self.shacl_ontology.object_edges:
            # we add edges going both direction to our network
            self.graph_network.add_edge(object_edge[0], object_edge[2], label="none")
            self.graph_network.add_edge(object_edge[2], object_edge[0], label="none")


            node_key = tuple(sorted([object_edge[0], object_edge[2]]))

            # we create a string from the edge_tuple.
            edge_string = "%s %s %s" % (object_edge[0], object_edge[1], object_edge[2])

            if node_key in self.edge_labels:
                # Key exists, append the item to the list associated with that key
                self.edge_labels[node_key].append(edge_string)
            else:
                # Key doesn't exist, create a new key-value pair with an empty list as the value
                self.edge_labels[node_key] = [edge_string]

        for datatype_edge in self.shacl_ontology.datatype_edges:
            # we add edges going both direction to our network
            self.graph_network.add_edge(datatype_edge[0], datatype_edge[2], label="none")

            node_key = tuple(sorted([datatype_edge[0], datatype_edge[2]]))

            # we create a string from the edge_tuple.
            edge_string = "%s %s %s" % (datatype_edge[0], datatype_edge[1], datatype_edge[2])

            if node_key in self.edge_labels:
                # Key exists, append the item to the list associated with that key
                self.edge_labels[node_key].append(edge_string)
            else:
                # Key doesn't exist, create a new key-value pair with an empty list as the value
                self.edge_labels[node_key] = [edge_string]

        for subclass_edge in self.shacl_ontology.subclass_structure:
            self.graph_network.add_edge(subclass_edge[0], subclass_edge[2], label="subclass")
            self.graph_network.add_edge(subclass_edge[2], subclass_edge[0], label="superclass")

            node_key = tuple(sorted([subclass_edge[0], subclass_edge[2]]))

            # we create a string from the edge_tuple.
            edge_string = "%s %s %s" % (subclass_edge[0], subclass_edge[1], subclass_edge[2])

            if node_key in self.edge_labels:
                # Key exists, append the item to the list associated with that key
                self.edge_labels[node_key].append(edge_string)
            else:
                # Key doesn't exist, create a new key-value pair with an empty list as the value
                self.edge_labels[node_key] = [edge_string]

        for subclass_edge_with_retrictions in self.shacl_ontology.object_edges_with_restrictions:
            self.graph_network.add_edge(subclass_edge_with_retrictions[0], subclass_edge_with_retrictions[2],
                                        label="none")


            node_key = tuple(sorted([subclass_edge_with_retrictions[0], subclass_edge_with_retrictions[2]]))

            # we create a string from the edge_tuple.
            edge_string = "%s %s %s" % (
            subclass_edge_with_retrictions[0], subclass_edge_with_retrictions[1], subclass_edge_with_retrictions[2])

            if node_key in self.edge_labels:
                # Key exists, append the item to the list associated with that key
                self.edge_labels[node_key].append(edge_string)
            else:
                # Key doesn't exist, create a new key-value pair with an empty list as the value
                self.edge_labels[node_key] = [edge_string]

    def get_40_shortest_simple__valid_routes_between_nodes(self, source, target):
        """
        Finds the shortest simple paths between a source node and a target node in the ontology graph, filtering out
        invalid paths based on predefined invalid patterns.

        This method first uses NetworkX's `shortest_simple_paths` function to generate up to the 40 shortest simple
        routes between the source and target nodes.

        It then filters these paths using a list of invalid sequences of three classes (stored in
        `self.shacl_ontology.invalid_paths`). A path is discarded if it contains any of the invalid sequences, either
        in their original order or reversed (since the sequence is considered invalid in both directions).

        :param source:
            The starting node for path search.
        :param target:
            The target node for path search.

        :return:
            list: A list of valid paths (each path represented as a list of nodes) between the source and target nodes.
            If no paths exist or all paths are invalid, an empty list is returned.
        """

        if nx.has_path(self.graph_network, source, target):
            all_routes = []
            shortest_paths_generator = nx.shortest_simple_paths(self.graph_network, source=source, target=target)

            # Take the first 40 paths from the generator
            first_40_paths = list(itertools.islice(shortest_paths_generator, 40))

            for path in first_40_paths:

                # WE FILTER INVALID PATHS HERE
                def is_contained_in_order(a, b):
                    for i in range(len(a) - len(b) + 1):
                        if a[i:i + len(b)] == b:
                            return True
                    return False

                valid_path = True
                for invalid_path in self.shacl_ontology.invalid_paths:
                    if is_contained_in_order(path, invalid_path):
                        valid_path = False

                for invalid_path in self.shacl_ontology.invalid_paths:
                    if is_contained_in_order(path, invalid_path[::-1]):
                        valid_path = False

                if valid_path == True:
                    all_routes.append(path)
        else:
            all_routes = []

        return all_routes

    def get_all_unique_ontology_items_for_top_k_simple_routes_between_nodes(self, k, top_k_shortest_routes):
        """
        Retrieves all unique ontology items (triples) associated with the top-k shortest simple routes between two nodes
        in the ontology graph.

        This method takes a list of shortest routes, selects the top-k shortest paths, and for each path,
        it extracts all the unique consecutive node pairs. For each pair, it retrieves the associated ontology triples
        from `self.edge_labels` and returns a list of all unique ontology items (in the form of triple strings) that
        belong to these top-k shortest routes.

        The parameter `top_k_shortest_routes` allows for varying the number of shortest routes used, which is needed
        when varying the amount of routes used between relevant nodes during GTOC for ontology condensation.

        :param k:
            The number of top shortest routes to consider from the provided list of paths.
        :param top_k_shortest_routes:
            A list of shortest routes, where each route is a list of consecutive nodes representing a valid path.

        :return:
            list: A list of unique ontology items (triples) in the format
            ["element1 property1 element2", "element2 property2 element3"],
            corresponding to the top-k shortest routes between the source and target nodes.
        """
        sorted_pairs = set()
        ontology_items_belonging_to_pairs = set()

        for route in top_k_shortest_routes[:k]:
            # Iterate through the list, considering consecutive pairs
            for i in range(len(route) - 1):
                # Get the current and next elements
                current_element = route[i]
                next_element = route[i + 1]

                # Sort the pair of elements
                sorted_pair = tuple(sorted([current_element, next_element]))

                # Add the sorted pair to the list
                sorted_pairs.add(sorted_pair)

        for pair in sorted_pairs:
            ontology_items_belonging_to_pair = self.edge_labels[pair]
            for ontology_item in ontology_items_belonging_to_pair:
                ontology_items_belonging_to_pairs.add(ontology_item)

        return list(ontology_items_belonging_to_pairs)

    def find_all_relevant_pairs(self):
        """
        This method identifies all pairs of nodes in the ontology network to be considered for precomputing
        shortest paths. Note: Paths originating from datatype nodes are excluded because such paths do not exist.
        In the ontology, edges only point towards datatype nodes, not away from them.

        It returns a list of tuples, where each tuple represents a pair of nodes for which the shortest path
        will be precomputed.
        """

        # we get all datatype nodes and the remaining edges
        datatype_nodes = []
        other_nodes = []

        # Iterate over nodes in the graph
        for node in self.graph_network.nodes():
            # Check if the node starts with 'xsd:'
            if node.startswith(('xsd:', 'rdf:', 'sor-con:', 'kad-con:')):
                # If it does, add it to the datatype_nodes list
                datatype_nodes.append(node)
            else:
                # If it doesn't, add it to the other_nodes list
                other_nodes.append(node)

        # get all unordered pairs in other_nodes list.
        # Generate all unordered pairs
        unordered_pairs = list(combinations(other_nodes, 2))

        # Sort the pairs alphabetically
        sorted_pairs = sorted([tuple(sorted(pair)) for pair in unordered_pairs])

        # get all pairs between other_nodes and datatype_nodes
        remaining_pairs = []
        for oth_node in other_nodes:
            for dat_node in datatype_nodes:
                pair = (oth_node, dat_node)
                remaining_pairs.append(pair)

        combined_pairs = sorted_pairs + remaining_pairs
        return combined_pairs

    def create_dictionary_that_maps_all_pairs_to_the_necessary_ontology_items_in_top_k_shortest_routes(self):
        """
        Creates and saves multiple dictionaries that map pairs of nodes to their corresponding ontology items
        in the top-k shortest routes.

        This method generates the following dictionaries:
        1. A dictionary that maps each pair of nodes (as a tuple) to the valid routes between them. This dictionary
           is saved as 'all_routes_dict'.
        2. Ten additional dictionaries, where each dictionary corresponds to the k-th shortest route (for k=1 to 10)
           between node pairs. Each dictionary maps node pairs to a list of ontology triples
           (e.g., "element1 property element2") associated with the respective k-th shortest routes.
           These dictionaries are saved as
           'dictionary_that_maps_all_pairs_to_the_necessary_ontology_items_in_top_k_shortest_routes'.

        All data is saved in the
        'precompute_shortest_routes/saved_data/precomputed_shortest_routes/'
        directory.

        Returns:
            None
        """
        all_routes_dictionary = dict()

        all_pairs = self.find_all_relevant_pairs()
        print('total_amount_of_pairs_we_need_to_compute')
        print(len(all_pairs))
        counter = 0
        for pair in all_pairs:
            counter = counter + 1
            print(counter)
            top_k_shortest_routes = self.get_40_shortest_simple__valid_routes_between_nodes(pair[0], pair[1])

            all_routes_dictionary[tuple(sorted(pair))] = top_k_shortest_routes

        main_path = './saved_data/precomputed_shortest_routes/'
        # we save dictionary_that_maps_all_pairs_to_the_necessary_ontology_items_in_top_k_shortest_routes
        path = 'all_routes_dict'
        with open(main_path + path, 'wb') as f:
            pickle.dump(all_routes_dictionary, f)

        # We create and save 10 dictionaries. Dictionary k contains as key the ontology triples belonging
        # to the k shortest routes between the nodes that are used as keys in the dictionaries
        for k in range(1, 11):
            k_routes_dict = dict()
            for key, value in all_routes_dictionary.items():
                unique_ontology_items_needed_for_top_k_shortest_routes = self.get_all_unique_ontology_items_for_top_k_simple_routes_between_nodes(
                    k, value)
                k_routes_dict[key] = unique_ontology_items_needed_for_top_k_shortest_routes
            main_path = './saved_data/precomputed_shortest_routes/'
            # we save dictionary_that_maps_all_pairs_to_the_necessary_ontology_items_in_top_k_shortest_routes
            path = 'dictionary_that_maps_all_pairs_to_the_necessary_ontology_items_in_top_%s_shortest_routes' % (str(k))
            with open(main_path + path, 'wb') as f:
                pickle.dump(k_routes_dict, f)

        return None

    def save_created_network_graph_and_ontology_tools(self):
        """
        Saves the gathered data and computed results from this file.

        This method is responsible for persisting the information that has been collected and any calculations
        performed within this file, ensuring that the data is stored for future use or analysis.
        """
        main_path = './saved_data/'

        # first we save the network
        nx.write_graphml(self.graph_network, main_path + "network_graph.graphml")

        # next we save the dictionary containing all the ontology triples belonging to edges in the network graph
        with open(main_path + 'ontology_triples_matched_with_network_in_dict.pickle', 'wb') as f:
            pickle.dump(self.edge_labels, f)

        # we save the list with classes
        with open(main_path + 'classes.pickle', 'wb') as f:
            pickle.dump(list(set(self.shacl_ontology.classes)), f)

        # we save the list with relations
        with open(main_path + 'relations.pickle', 'wb') as f:
            pickle.dump(list(set(self.shacl_ontology.properties)), f)

        # we save our prefix dictionary
        with open(main_path + 'prefix_dictionary.pickle', 'wb') as f:
            pickle.dump(self.shacl_ontology.prefix_dictionary, f)

        # we save our prefix string
        with open(main_path + 'prefix_string.pickle', 'wb') as f:
            pickle.dump(self.shacl_ontology.prefix_string, f)

        # we save all edges
        all_edges = dict()
        all_edges['datatype_edges'] = self.shacl_ontology.datatype_edges
        all_edges['object_edges'] = self.shacl_ontology.object_edges
        all_edges['object_edges_with_restrictions'] = self.shacl_ontology.object_edges_with_restrictions
        all_edges['subclass_edges'] = self.shacl_ontology.subclass_structure
        with open(main_path + 'all_edges_dictionary.pickle', 'wb') as f:
            pickle.dump(all_edges, f)

        # We save the class and relation strings (without comments)
        with open(main_path + 'classes_without_comments_string.pickle', 'wb') as f:
            pickle.dump(self.shacl_ontology.classes_without_comments_string, f)

        with open(main_path + 'relations_without_comments_string.pickle', 'wb') as f:
            pickle.dump(self.shacl_ontology.relations_without_comments_string, f)

        # we save dictionary_that_maps_property_to_neighboring_nodes
        with open(main_path + 'dictionary_that_maps_property_to_neighboring_nodes', 'wb') as f:
            pickle.dump(self.shacl_ontology.dictionary_that_maps_property_to_neighboring_nodes, f)


graph = graph_network_creator()


