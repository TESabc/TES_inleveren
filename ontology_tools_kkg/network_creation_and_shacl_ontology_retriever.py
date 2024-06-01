import os
import configparser
import requests
from typing import List, Dict, Any
import networkx as nx
import networkx as nx
import matplotlib.pyplot as plt
import pickle
from itertools import combinations
import itertools

'''
Here i set up access to the SPARQL endpoint of the KKG.
I also set up the keys needed for access to Azure hosted OpenAI models.
I also define functions that help us get results.
'''
config = configparser.ConfigParser()
# Put the absolute path to the .ini file here:
path = r"../secrets.ini"
config.read(path)

locatie_url = config['KADASTER']['LOC_SEARCH_API']
lookup_url = config['KADASTER']['LOC_LOOKUP_API']
sparql_url = config['KADASTER']['KKG_API']

os.environ['OPENAI_API_KEY'] = '3751bf8a4de94df8acd78a60213d688e'
os.environ['OPENAI_API_TYPE'] = 'azure'
os.environ['OPENAI_API_VERSION'] = '2022-12-01'
os.environ['OPENAI_API_BASE'] = 'https://kadasterdst.openai.azure.com/'

full_prefix_dictionary = {

    # Those we did not include before
    'http://schema.org/': 'sdo0:',
    'http://www.w3.org/2002/07/owl#': 'owl:',
    'http://www.w3.org/2000/01/rdf-schema#': 'rdfs:',
    'http://www.w3.org/1999/02/22-rdf-syntax-ns#': 'rdf:',

    # These are from our old file
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

    # This one looked like it was added later:
    'http://www.w3.org/ns/shacl#': 'sh:',

    # There are still a couple of prefices in the shacl ontology file..
    # Did not add for now

    # I retrieved this from the file of Wim
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

    # based on the relation list i also added these prefixes:
    # SOME OF THESE MIGHT BE WORTHLESS ( BETTER TO REMOVE RELATION ALLTOGETHER. )
    # you can check that by rerunning while commenting out these prefixes
    'http://purl.org/dc/terms/': 'dct:',
    'http://www.w3.org/ns/prov#': 'prov:',
    'http://bp4mc2.org/def/mim#': 'mim:',
    'http://purl.org/dc/elements/1.1/': 'dcelements:',
    'http://www.opengis.net/ont/gml#': 'gml:'
}


def create_prefix_start_for_SPARQL_queries(prefix_dict):
    prefix_string = ""
    for key, value in prefix_dict.items():
        prefix_string = prefix_string + "PREFIX %s <%s>\n" % (value, key)
    return prefix_string


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


def get_items_in_list_format_from_sparql_query(query, variable_name):
    '''
    When performing api calls on triple stores that store RDF data the results will in JSON.
    Here we convert it to a list.

    :param query: the sparql query needed to get the desired schema items
    :param schema_type: should be either cls (for classes) or rel (for relations) depending on the sparql query used
    :return: a list containing either all classes or all relations
    '''
    result = run_sparql(query)
    if result == "That query failed. Perhaps you could try a different one?":
        return result
    result_list = []
    for dict in result:
        result_list.append(get_nested_value(dict, [variable_name, 'value']))
    return result_list


class SHACL_ontology_retriever:
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
        There seem to be some data quality issues. Some classes are defined, but do not seem to be
        connected to other parts in the ontology.

        So we fix this by taking an interesection between the retrieved classes and the relevant nodes
        that will be added to our ontology network graph.
        :return:
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
        '''
        With this function we retrieve all combinations of
        class1 relation class2

        We obtain triples which show how classes are related to eachother through properties
        Basically a triple describes an edge in a graph as following: node edge node

        If there are additional restrictions on the right-hand edge defined by the SHACL shapes,
        we exclude edges like that in this function.

        The get_object_edges_with_restictions() function will handle the case where there are restrictions.

        :return: returns a list with tuples, where each tuple represents an edge
        '''

        # Note that the following part in the query:

        # FILTER NOT EXISTS {
        # ?propshp sh:in ?lijst .
        # }.

        # Makes sure we do not obtain object edges where there are additional restrictions on the right-side edge.
        # When there are additional restrictions on the right-side edge where that part can only take
        # specific values, we use a separate query traversing the restrictions list in order to get
        # to add an edge for each of the elements in the list.
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

        # I manually add some object edges that are not in the SHACL ontology.
        # This part of the knowledge graph is actually from the Central Bureau of Statistics (CBS),
        # it is not maintained by Kadaster, but it can be combined with our own knowledge graph.
        # such linking is often done within semantic web / linked data.
        # We add this in order to be able to ask questions about provinces & municipalities also.
        edge_list.append(('sor:Gebouw', 'geo:sfWithin', 'wbk:Buurt'))
        edge_list.append(('wbk:Buurt', 'geo:sfWithin', 'wbk:Wijk'))
        edge_list.append(('wbk:Wijk', 'geo:sfWithin', 'wbk:Gemeente'))
        edge_list.append(('sor:Gemeente', 'owl:sameAs', 'wbk:Gemeente'))
        # This one we already have in our knowledge graph:
        # edge_list.append(('sor:Gemeente', 'geo:sfWithin', 'sor:Provincie'))

        return edge_list

    def get_invalid_paths_in_object_edges(self):
        '''
        This function analyzes all paths that constitute 3 classes through the object edges.

        We perform SPARQL queries to verify whether the paths implied by the ontology actually exist in the fact-level
        structure. 75% of these 3-class paths do not exist in the fact-level structure.

        It is critical to use these invalid (sub)paths as restrictions when determining paths between nodes
        in the ontology graph.

        :return: list containing lists of invalid paths along 3 classes
        '''

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
        '''
        With this function we retrieve all combinations of
        class1 relation class2

        We obtain triples which show how classes are related to eachother through properties
        Basically a triple describes an edge in a graph as following: node edge node

        :return: returns a list with tuples, where each tuple represents an edge
        '''

        # Note that the following part in the query:

        #     sh:in ?list .

        # ?list rdf:rest*/rdf:first ?value .

        # we use smart SPARQL to traverse all the different types that are in the restriction.
        # We create edges for each of the restrictions. (If there are 8 restrictions we get 8 edges)
        #
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

            # We add our target to our list of classes.
            # This is helpful since our ontology retriever should retrieve these nodes,
            # because they allow us to directly write SPARQL
            self.classes.append(target)

        return edge_list

    def get_datatype_edges(self):
        '''
        With this function we retrieve all combinations of
        class1 relation data

        We obtain triples which show how classes are related to literals (data values such as strings or integers)
        through properties.

        Basically a triple describes an edge in a graph as following: node edge node
        In this case the last node is not a class but a piece of data.

        :return: returns a list with tuples, where each tuple represents an edge
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
        '''
        For each class we get its superclasses. We create triples describing subclass/superclass structure.
        :return: a list with subclass/superclass structure triples
        '''
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
        class_string = ""
        for class_ in self.classes:
            class_string += "%s\n" % class_
        return class_string

    def create_relations_without_comments_string(self):
        relation_string = ""
        for relation in self.properties:
            relation_string += "%s\n" % relation
        return relation_string

    def create_dictionary_that_maps_property_to_neighboring_nodes(self):

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

        main_path = './saved_network_and_ontology/'
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
        '''
        We formulate the ontology as a graph network.

        We formulate the network as a directed graph because we want to have certain restrictions at certain nodes
        which allow to model an ontology as a graph network.


        :return: a graph network object from the networkx package
        '''

        '''
        For every object_edge in the ontology we add a directed edge going BOTH ways!
        This is because in SPARQL it is possible to use a property defined to go in one direction,
        on the other direction by switching subject & object.
        this is because in knowledge graphs sometimes the ontology can be incomplete where properties are
        defined in one direction but not in the other.
        '''
        for object_edge in self.shacl_ontology.object_edges:
            # we add edges going both direction to our network
            self.graph_network.add_edge(object_edge[0], object_edge[2], label="none")
            self.graph_network.add_edge(object_edge[2], object_edge[0], label="none")

            '''
            We first sort the nodes on alphabetic order.

            We use these sorted nodes as a key in our edge_label dictionary where the corresponding values
            will be all ontology triples between these two nodes (going in both directions).

            When we have a condensed version of our network (found through shortest routes), we will have a collection 
            of edges in this condensed version of the network.

            If there is an edge in this condensed version of the network, we do not care which direction the edge
            points at. We would like the have all ontology triples existing in the ontology structure between
            those two nodes (regardless of which point these ontology triples point at).

            By sorting the nodes and using them as key in our edge_label dictionary we have as
            corresponding value a list with all ontology triples between these two nodes going in both directions. 
            This will later be helpful for reconstructing an ontology based on a condensed network.
            '''
            node_key = tuple(sorted([object_edge[0], object_edge[2]]))

            # we create a string from the edge_tuple.
            edge_string = "%s %s %s" % (object_edge[0], object_edge[1], object_edge[2])

            if node_key in self.edge_labels:
                # Key exists, append the item to the list associated with that key
                self.edge_labels[node_key].append(edge_string)
            else:
                # Key doesn't exist, create a new key-value pair with an empty list as the value
                self.edge_labels[node_key] = [edge_string]

        '''
        For every datatype_edge in the ontology we add a directed edge ONLY pointing to the datatype.
        Otherwise, the network would allow such a path:
        building -> buildingyear -> xsd:PositiveInteger -> average_temperate -> Netherlands.
        Such paths do not make sense. We want to model such datatype nodes as endpoints in our graph, one way
        to accomplish this is by creating a directed edge only pointing towards the datatype.
        '''
        for datatype_edge in self.shacl_ontology.datatype_edges:
            # we add edges going both direction to our network
            self.graph_network.add_edge(datatype_edge[0], datatype_edge[2], label="none")

            '''
            We first sort the nodes on alphabetic order.

            We use these sorted nodes as a key in our edge_label dictionary where the corresponding values
            will be all ontology triples between these two nodes (going in both directions).

            When we have a condensed version of our network (found through shortest routes), we will have a collection 
            of edges in this condensed version of the network.

            If there is an edge in this condensed version of the network, we do not care which direction the edge
            points at. We would like the have all ontology triples existing in the ontology structure between
            those two nodes (regardless of which point these ontology triples point at).

            By sorting the nodes and using them as key in our edge_label dictionary we have as
            corresponding value a list with all ontology triples between these two nodes going in both directions. 
            This will later be helpful for reconstructing an ontology based on a condensed network.
            '''
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

    def get_top_k_shortest_simple_routs_between_nodes(self, source, target, top_k=10):
        '''

        :param source: source node
        :param target: target node
        :param top_k: the top k shortest paths you want to retrive (for example 5 or 10)
        :return: list with top k shortest paths
        '''

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
        '''

        :param top_k_shortest_routes: list containing lists that describe shortet routes
        :return: list containing all unique ontology items belonging to the top_k shortest routes
        '''
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

    def find_all_relevant_pairs(self, k=5):
        '''
        For each valid pairs we will pre-compute the top-k shortest routes, this way we have a high probability that
        the best route is among the top k. We choose k=5 for now because it seemed to work very well for a couple
        of routes that were tested.

        In general we only care about unordered pairs. This is because I added edges both ways for almost all nodes.

        The exception is that we only have a one-way edge from the class to the datatype in the datatype_edges.
        Therefore, from those datatype edges we cannot reach any of the other edges. However, we can potentially reach
        these datatype edges from the other edges.

        Therefore, my approach will be to get all unordered pairs in the set of all edges EXCLUDING datatype edges.
        These top-k shortest routes will be stored in a dictionary where we first sort the edges on alphabetic order.

        Next, I find top-k shortest routes from the non-datatype nodes to the datatype nodes.


        :return: list with tuples (sorther on alphabetial order) which describe all pairs
        '''

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
        all_routes_dictionary = dict()

        all_pairs = self.find_all_relevant_pairs()
        print('total_amount_of_pairs_we_need_to_compute')
        print(len(all_pairs))
        counter = 0
        for pair in all_pairs:
            counter = counter + 1
            print(counter)
            top_k_shortest_routes = self.get_top_k_shortest_simple_routs_between_nodes(pair[0], pair[1])

            all_routes_dictionary[tuple(sorted(pair))] = top_k_shortest_routes

        main_path = './saved_network_and_ontology/dictionaries_that_map_pairs_of_nodes_to_ontology_items/'
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
            main_path = './saved_network_and_ontology/dictionaries_that_map_pairs_of_nodes_to_ontology_items/'
            # we save dictionary_that_maps_all_pairs_to_the_necessary_ontology_items_in_top_k_shortest_routes
            path = 'dictionary_that_maps_all_pairs_to_the_necessary_ontology_items_in_top_%s_shortest_routes' % (str(k))
            with open(main_path + path, 'wb') as f:
                pickle.dump(k_routes_dict, f)

        return None

    def save_created_network_graph_and_ontology_tools(self):
        main_path = './saved_network_and_ontology/'

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


