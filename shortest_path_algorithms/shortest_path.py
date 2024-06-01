import networkx as nx
import pickle
import matplotlib.pyplot as plt
import os
from itertools import combinations

class network_and_ontology_store:
    def __init__(self, k_shortest_routes):
        '''

        :param k_shortest_routes: The number of shortest routes you want to consider between nodes
        '''
        current_dir = os.path.dirname(os.path.abspath(__file__))
        main_path = os.path.join(current_dir, '../ontology_tools_kkg/saved_network_and_ontology/')
        self.network = nx.read_graphml(main_path + "network_graph.graphml")

        # the keys of this dictionary contains all SORTED adjecent nodes in the network. ( 1 edge away from eachother)
        # The corresponding values are lists with therein the corresponding ontology triples as single strings in the
        # list.
        # (So including the relation)
        #
        with open(main_path + 'ontology_triples_matched_with_network_in_dict.pickle', 'rb') as f:
            edge_labels = pickle.load(f)

        self.ontology_triples_belonging_to_edge = edge_labels

        # list with all classes
        with open(main_path + 'classes.pickle', 'rb') as f:
            classes = pickle.load(f)

        self.classes = classes

        # list with all relations
        with open(main_path + 'relations.pickle', 'rb') as f:
            relations = pickle.load(f)

        self.relations = relations

        # Load prefix dictionary
        # dictionaries where keys are full URL's and the value is the corresponding PREFIX
        with open(main_path + 'prefix_dictionary.pickle', 'rb') as f:
            prefix_dictionary = pickle.load(f)

        self.prefix_dictionary = prefix_dictionary

        # Load prefix string
        # all the possible prefixes in the format in which it is inserted in a SPARQL query.
        # We can append this to each SPARQL that is generated to make sure no PREFIX is missing.
        with open(main_path + 'prefix_string.pickle', 'rb') as f:
            prefix_string = pickle.load(f)

        self.prefix_string = prefix_string.replace('cbs:', 'wbk:')

        # NOTE THAT THIS DICTIONARY CONTAINS THE FOLLOWING KEYS:
        # datatype_edges
        # object_edges
        # object_edges_with_restrictions
        # subclass_edges

        # The corresponding values are: lists with tuples. each tuple contains 3 strings in the format:
        # ('class', 'relation', 'class/datatype')
        with open(main_path + 'all_edges_dictionary.pickle', 'rb') as f:
            all_edges = pickle.load(f)

        self.all_edges_dictionary = all_edges

        # Load classes without comments string (It is now in 1 string object instead of list, so we can put in prompt)
        with open(main_path + 'classes_without_comments_string.pickle', 'rb') as f:
            classes_without_comments_string = pickle.load(f)
        self.classes_without_comments_string = classes_without_comments_string


        # Load relations without comments string (It is now in 1 string object instead of list, so we can put in prompt)
        with open(main_path + 'relations_without_comments_string.pickle', 'rb') as f:
            relations_without_comments_string = pickle.load(f)

        self.relations_without_comments_string = relations_without_comments_string




        # load dictionary that we will use to find neighboring nodes of relevant relations we retrieved.
        # The keys are all the relations in our ontology.
        # the corresponding keys are SETS containing all nodes that neighbor that relation.
        # We can use these nodes to apply graph traversal algorithms
        with open(main_path + 'dictionary_that_maps_property_to_neighboring_nodes', 'rb') as f:
            loaded_data = pickle.load(f)

        self.dictionary_that_maps_property_to_neighboring_nodes = loaded_data

        # I create a set of central nodes that we always wish to have in the condensed ontology
        self.central_nodes = ['sor:Perceel', 'sor:Verblijfsobject', 'sor:Gebouw']

        main_path = os.path.join(current_dir, '../ontology_tools_kkg/saved_network_and_ontology/dictionaries_that_map_pairs_of_nodes_to_ontology_items/')
        # load pre-calculated unique relevant ontology items extracted from top-k shortest paths for
        # ALL PAIRS of nodes
        path = 'dictionary_that_maps_all_pairs_to_the_necessary_ontology_items_in_top_%s_shortest_routes' % (str(k_shortest_routes))
        with open(main_path + path, 'rb') as f:
            loaded_data = pickle.load(f)
        self.dictionary_that_maps_all_pairs_to_the_necessary_ontology_items_in_top_k_shortest_routes = loaded_data

        main_path = os.path.join(current_dir, '../ontology_tools_kkg/saved_network_and_ontology/')

        with open(main_path + 'dictionary_that_maps_property_to_all_ontology_triples_as_string', 'rb') as f:
            loaded_data = pickle.load(f)
        self.dictionary_that_maps_property_to_all_ontology_triples_as_string = loaded_data




    def print_classes(self):
        print(self.classes)

    def print_relations(self):
        print(self.relations)

    def print_datatype_edges(self):
        for edge in self.all_edges_dictionary['datatype_edges']:
            print(edge)

    def print_object_edges(self):
        for edge in self.all_edges_dictionary['object_edges']:
            print(edge)

    def print_object_edges_with_restrictions(self):
        for edge in self.all_edges_dictionary['object_edges_with_restrictions']:
            print(edge)

    def print_subclass_edges(self):
        for edge in self.all_edges_dictionary['subclass_edges']:
            print(edge)

    def get_classes(self):
        return self.classes

    def get_relations(self):
        return self.relations

    def print_all_edges_in_graph(self):
        for edge in self.network.edges:
            print(edge)
    def is_valid_path(self, path):
        '''
        This function provides a restriction we can put on paths.
        This one does not allow a subclass path to be taken whenever a superclass path has already been taken.
        And it also does not allow a subclass path to be taken whenever a subclass path has already be taken.
        :param path:
        :return:
        '''
        subclass_was_present = False
        superclass_was_present = False
        for i in range(len(path) - 1):
            if 'subclass' in self.network[path[i]][path[i + 1]]['label']:
                subclass_was_present = True
            if 'superclass' in self.network[path[i]][path[i + 1]]['label']:
                superclass_was_present = True

        if superclass_was_present == True and subclass_was_present == True:
            return False
        else:
            return True

    def get_shortest_route_between_nodes(self, source, target):

        shortest_path = nx.shortest_path(self.network, source=source, target=target)

        print(shortest_path)

    def get_all_simple_routs_between_nodes(self, source, target, satisfy_subclass_restiction=True):
        all_routes = nx.all_simple_paths(self.network, source=source, target=target)

        def path_length(G, path):
            return len(path)
        sorted_paths = sorted(all_routes, key=lambda path: path_length(self.network, path))
        count =0
        if satisfy_subclass_restiction == True:

            for route in sorted_paths:
                if self.is_valid_path(route):
                    print(route)
        else:
            for route in sorted_paths:
                if count <= 10:
                    count = count +1
                    print(route)
        print(count)

    def get_all_shortest_path_with_intermediate_node_constraint(self, source, target, node_in_between):
        shortest_path = None
        shortest_length = float('inf')

        def constraint_func(path, node_in_between=node_in_between):
            if node_in_between in path:
                return True
            else:
                return False

        # Iterate over all simple paths from source to target
        for path in nx.all_simple_paths(self.network, source=source, target=target):
            # Check if the constraint is satisfied for the current path
            if constraint_func(path):
                # If the path is shorter than the current shortest path, update shortest path
                if len(path) < shortest_length:
                    shortest_path = path
                    shortest_length = len(path)

        return shortest_path

    def get_subgraph_intersection_between_nodes(self, source, target):
        def get_full_reachable_subgraph(graph, central_node):
            """
            Get the subgraph of all nodes reachable from the specified node.

            Parameters:
            - graph: NetworkX graph
            - central_node: Node for which the subgraph is to be obtained

            Returns:
            - reachable_subgraph: NetworkX subgraph
            """
            reachable_nodes = nx.descendants(graph, central_node)
            reachable_nodes.add(central_node)  # Include the central node itself
            reachable_subgraph = graph.subgraph(reachable_nodes)
            return reachable_subgraph

        # Get the full reachable subgraph
        reachable_subgraph_1 = get_full_reachable_subgraph(self.network, source)
        reachable_subgraph_2 = get_full_reachable_subgraph(self.network, target)

        intersection_graph = nx.intersection(reachable_subgraph_1, reachable_subgraph_2)

        node_labels = {node: str(node) for node in intersection_graph.nodes()}

        pos = nx.spring_layout(intersection_graph)
        nx.draw(intersection_graph, pos, node_size=700, node_color="salmon", labels=node_labels)
        plt.title("Intersection Subgraph with Node Names")
        plt.show()

        # pos = nx.spring_layout(intersection_graph)
        # #nx.draw(G, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=10, font_color="black")
        # nx.draw(intersection_graph, pos, node_size=700, node_color="salmon", label="Full Reachable Subgraph")
        # plt.show()

    def condense_ontology_based_on_retrieved_schema_items(self, retrieved_classes, retrieved_relations, include_central_part = False, sort = True):
        triples_set = set()

        #First we determine all the relevant nodes
        nodes_obtained_from_retrieved_classes = retrieved_classes
        nodes_obtain_from_retrieved_relations = set()
        for rel in retrieved_relations:

            neighbor_nodes = self.dictionary_that_maps_property_to_neighboring_nodes.get(rel, [])

            if rel in self.dictionary_that_maps_property_to_all_ontology_triples_as_string:
                for triple in self.dictionary_that_maps_property_to_all_ontology_triples_as_string[rel]:
                    triples_set.add(triple)


            # we add the nodes surrounding the relation to our node set to apply graph traversal algorithms
            for node in neighbor_nodes:
                nodes_obtain_from_retrieved_relations.add(node)
        nodes_obtain_from_retrieved_relations = list(nodes_obtain_from_retrieved_relations)

        if include_central_part:

            nodes_we_always_include = self.central_nodes
        else:
            nodes_we_always_include = []

        all_nodes = nodes_obtained_from_retrieved_classes + nodes_obtain_from_retrieved_relations + nodes_we_always_include
        all_nodes = list(set(all_nodes))
        # Generate all unordered pairs
        unordered_pairs = list(combinations(all_nodes, 2))

        # Sort the pairs alphabetically
        sorted_pairs = sorted([tuple(sorted(pair)) for pair in unordered_pairs])


        for pair in sorted_pairs:
            # simple check, making sure we do not try to find paths between two datatype edges
            if 'xsd:' not in pair[0] and 'xsd:' not in pair[1]:
                if tuple(sorted(pair)) in self.dictionary_that_maps_all_pairs_to_the_necessary_ontology_items_in_top_k_shortest_routes:
                    for ontology_triple in self.dictionary_that_maps_all_pairs_to_the_necessary_ontology_items_in_top_k_shortest_routes[pair]:
                        triples_set.add(ontology_triple)


        ontology_string_object_part = ""
        ontology_string_datatype_part = ""

        # we remove some triples which should not be in the ontology (gebruiksdoel should only be defined on sor:Verblijfsobject)
        triples_to_remove = {'sor:Gebouwzone sor:gebruiksdoel sor-con:bijeenkomstfunctie',
                             'sor:Gebouwzone sor:gebruiksdoel sor-con:celfunctie',
                             'sor:Gebouwzone sor:gebruiksdoel sor-con:gezondheidsfunctie',
                             'sor:Gebouwzone sor:gebruiksdoel sor-con:industriefunctie',
                             'sor:Gebouwzone sor:gebruiksdoel sor-con:kantoorfunctie',
                             'sor:Gebouwzone sor:gebruiksdoel sor-con:logiesfunctie',
                             'sor:Gebouwzone sor:gebruiksdoel sor-con:onderwijsfunctie',
                             'sor:Gebouwzone sor:gebruiksdoel sor-con:sportfunctie',
                             'sor:Gebouwzone sor:gebruiksdoel sor-con:winkelfunctie',
                             'sor:Gebouwzone sor:gebruiksdoel sor-con:woonfunctie'}
        triples_set = triples_set.difference(triples_to_remove)
        triples_set = sorted(triples_set)

        for ontology_triple in triples_set:
            # simple check, making sure we do not try to find paths between two datatype edges
            if 'xsd:' in ontology_triple:
                ontology_string_datatype_part += "%s\n" % ontology_triple
            else:
                ontology_string_object_part += "%s\n" % ontology_triple
        # print('ONTOLOGY SIZE:')
        # print(len(triples_set))

        return ontology_string_object_part, ontology_string_datatype_part

    def naive_ontology_selection(self, retrieved_classes, retrieved_relations):
        '''
        This is a naive method of retrieving a condensed ontology. We simply select all "triples" associated
        with the retrieved classes and relations.

        We do not apply graph traversal algorithms to determine intermediate ontology triples for which
        the question in natural language might not contain linguistic cues for.

        :param retrieved_classes:
        :param retrieved_relations:
        :return:
        '''
        triples_set = set()

        # First we select all ontology triples that include the retrieved relations
        for rel in retrieved_relations:
            if rel in self.dictionary_that_maps_property_to_all_ontology_triples_as_string:
                for triple in self.dictionary_that_maps_property_to_all_ontology_triples_as_string[rel]:
                    triples_set.add(triple)

        # Next we select the ontology triples that include the retrieved classes
        for cls in retrieved_classes:
            for edge in self.all_edges_dictionary['datatype_edges']:
                if cls == edge[0]:
                    triples_set.add("%s %s %s" % (edge[0], edge[1], edge[2]))

            for edge in self.all_edges_dictionary['object_edges']:
                if cls == edge[0] or cls == edge[2]:
                    triples_set.add("%s %s %s" % (edge[0], edge[1], edge[2]))

            for edge in self.all_edges_dictionary['object_edges_with_restrictions']:
                if cls == edge[0] or cls == edge[2]:
                    triples_set.add("%s %s %s" % (edge[0], edge[1], edge[2]))

        ontology_string_object_part = ""
        ontology_string_datatype_part = ""

        # we remove some triples which should not be in the ontology (gebruiksdoel should only be defined on sor:Verblijfsobject)
        triples_to_remove = {'sor:Gebouwzone sor:gebruiksdoel sor-con:bijeenkomstfunctie',
                             'sor:Gebouwzone sor:gebruiksdoel sor-con:celfunctie',
                             'sor:Gebouwzone sor:gebruiksdoel sor-con:gezondheidsfunctie',
                             'sor:Gebouwzone sor:gebruiksdoel sor-con:industriefunctie',
                             'sor:Gebouwzone sor:gebruiksdoel sor-con:kantoorfunctie',
                             'sor:Gebouwzone sor:gebruiksdoel sor-con:logiesfunctie',
                             'sor:Gebouwzone sor:gebruiksdoel sor-con:onderwijsfunctie',
                             'sor:Gebouwzone sor:gebruiksdoel sor-con:sportfunctie',
                             'sor:Gebouwzone sor:gebruiksdoel sor-con:winkelfunctie',
                             'sor:Gebouwzone sor:gebruiksdoel sor-con:woonfunctie'}
        triples_set = triples_set.difference(triples_to_remove)
        triples_set = sorted(triples_set)

        for ontology_triple in triples_set:
            # simple check, making sure we do not try to find paths between two datatype edges
            if 'xsd:' in ontology_triple:
                ontology_string_datatype_part += "%s\n" % ontology_triple
            else:
                ontology_string_object_part += "%s\n" % ontology_triple
        # print('ONTOLOGY SIZE:')
        # print(len(triples_set))

        return ontology_string_object_part, ontology_string_datatype_part

    def retrieve_full_ontology(self):
        triples_set = set()

        for edge in self.all_edges_dictionary['datatype_edges']:

            triples_set.add("%s %s %s" % (edge[0], edge[1], edge[2]))

        for edge in self.all_edges_dictionary['object_edges']:

            triples_set.add("%s %s %s" % (edge[0], edge[1], edge[2]))

        for edge in self.all_edges_dictionary['object_edges_with_restrictions']:

            triples_set.add("%s %s %s" % (edge[0], edge[1], edge[2]))

        ontology_string_object_part = ""
        ontology_string_datatype_part = ""

        # we remove some triples which should not be in the ontology (gebruiksdoel should only be defined on sor:Verblijfsobject)
        triples_to_remove = {'sor:Gebouwzone sor:gebruiksdoel sor-con:bijeenkomstfunctie',
                             'sor:Gebouwzone sor:gebruiksdoel sor-con:celfunctie',
                             'sor:Gebouwzone sor:gebruiksdoel sor-con:gezondheidsfunctie',
                             'sor:Gebouwzone sor:gebruiksdoel sor-con:industriefunctie',
                             'sor:Gebouwzone sor:gebruiksdoel sor-con:kantoorfunctie',
                             'sor:Gebouwzone sor:gebruiksdoel sor-con:logiesfunctie',
                             'sor:Gebouwzone sor:gebruiksdoel sor-con:onderwijsfunctie',
                             'sor:Gebouwzone sor:gebruiksdoel sor-con:sportfunctie',
                             'sor:Gebouwzone sor:gebruiksdoel sor-con:winkelfunctie',
                             'sor:Gebouwzone sor:gebruiksdoel sor-con:woonfunctie'}
        triples_set = triples_set.difference(triples_to_remove)
        triples_set = sorted(triples_set)

        for ontology_triple in triples_set:
            # simple check, making sure we do not try to find paths between two datatype edges
            if 'xsd:' in ontology_triple:
                ontology_string_datatype_part += "%s\n" % ontology_triple
            else:
                ontology_string_object_part += "%s\n" % ontology_triple
        # print('ONTOLOGY SIZE:')
        # print(len(triples_set))

        return ontology_string_object_part, ontology_string_datatype_part


    def condense_ontology_based_on_retrieved_schema_items_TURTLE(self, retrieved_classes, retrieved_relations, include_central_part = False, sort = True):
        triples_set = set()

        #First we determine all the relevant nodes
        nodes_obtained_from_retrieved_classes = retrieved_classes
        nodes_obtain_from_retrieved_relations = set()
        for rel in retrieved_relations:

            neighbor_nodes = self.dictionary_that_maps_property_to_neighboring_nodes.get(rel, [])

            if rel in self.dictionary_that_maps_property_to_all_ontology_triples_as_string:
                for triple in self.dictionary_that_maps_property_to_all_ontology_triples_as_string[rel]:
                    triples_set.add(triple)

            # we add the nodes surrounding the relation to our node set to apply graph traversal algorithms
            for node in neighbor_nodes:
                nodes_obtain_from_retrieved_relations.add(node)
        nodes_obtain_from_retrieved_relations = list(nodes_obtain_from_retrieved_relations)

        if include_central_part:

            nodes_we_always_include = self.central_nodes
        else:
            nodes_we_always_include = []

        all_nodes = nodes_obtained_from_retrieved_classes + nodes_obtain_from_retrieved_relations + nodes_we_always_include
        all_nodes = list(set(all_nodes))
        # Generate all unordered pairs
        unordered_pairs = list(combinations(all_nodes, 2))

        # Sort the pairs alphabetically
        sorted_pairs = sorted([tuple(sorted(pair)) for pair in unordered_pairs])


        for pair in sorted_pairs:
            # simple check, making sure we do not try to find paths between two datatype edges
            if 'xsd:' not in pair[0] and 'xsd:' not in pair[1]:
                if tuple(sorted(pair)) in self.dictionary_that_maps_all_pairs_to_the_necessary_ontology_items_in_top_k_shortest_routes:
                    for ontology_triple in self.dictionary_that_maps_all_pairs_to_the_necessary_ontology_items_in_top_k_shortest_routes[pair]:
                        triples_set.add(ontology_triple)
        ontology_string_object_part = ""
        ontology_string_datatype_part = ""

        #we remove some triples which should not be in the ontology (gebruiksdoel should only be defined on sor:Verblijfsobject)
        triples_to_remove = {'sor:Gebouwzone sor:gebruiksdoel sor-con:bijeenkomstfunctie',
                             'sor:Gebouwzone sor:gebruiksdoel sor-con:celfunctie',
                             'sor:Gebouwzone sor:gebruiksdoel sor-con:gezondheidsfunctie',
                             'sor:Gebouwzone sor:gebruiksdoel sor-con:industriefunctie',
                             'sor:Gebouwzone sor:gebruiksdoel sor-con:kantoorfunctie',
                             'sor:Gebouwzone sor:gebruiksdoel sor-con:logiesfunctie',
                             'sor:Gebouwzone sor:gebruiksdoel sor-con:onderwijsfunctie',
                             'sor:Gebouwzone sor:gebruiksdoel sor-con:sportfunctie',
                             'sor:Gebouwzone sor:gebruiksdoel sor-con:winkelfunctie',
                             'sor:Gebouwzone sor:gebruiksdoel sor-con:woonfunctie'}
        triples_set = triples_set.difference(triples_to_remove)
        triples_set = sorted(triples_set)


        for ontology_triple in triples_set:
            # simple check, making sure we do not try to find paths between two datatype edges
            if 'xsd:' in ontology_triple:
                ontology_string_datatype_part += "%s\n" % ontology_triple
            else:
                ontology_string_object_part += "%s\n" % ontology_triple

        class_set = set()
        property_set = set()
        for ontology_triple in triples_set:
            list_ontology_triple = ontology_triple.split()

            if 'xsd:' not in list_ontology_triple[0] and 'rdf:' not in list_ontology_triple[0]:
                class_set.add(list_ontology_triple[0])
            if 'xsd:' not in list_ontology_triple[2] and 'rdf:' not in list_ontology_triple[2]:
                class_set.add(list_ontology_triple[2])

            property_set.add(list_ontology_triple[1])

        class_triples_string = ""
        property_triples_string = ""
        for cls in class_set:
            string_to_add = "%s rdf:type rdf:Class" % (cls)
            class_triples_string += "%s\n" % (string_to_add)
        for prop in property_set:
            string_to_add = "%s rdf:type rdf:Property" % (prop)
            property_triples_string += "%s\n" % (string_to_add)

        print('ONTOLOGY NUMBER:')
        print(len(class_set) + len(property_set) + len(triples_set))
        return ontology_string_object_part, ontology_string_datatype_part, class_triples_string, property_triples_string








def get_full_reachable_subgraph(graph, central_node):
    """
    Get the subgraph of all nodes reachable from the specified node.

    Parameters:
    - graph: NetworkX graph
    - central_node: Node for which the subgraph is to be obtained

    Returns:
    - reachable_subgraph: NetworkX subgraph
    """
    reachable_nodes = nx.descendants(graph, central_node)
    reachable_nodes.add(central_node)  # Include the central node itself
    reachable_subgraph = graph.subgraph(reachable_nodes)
    return reachable_subgraph





store = network_and_ontology_store(1)
print(len(store.all_edges_dictionary['datatype_edges']) + len(store.all_edges_dictionary['object_edges']) + len(store.all_edges_dictionary['object_edges_with_restrictions']))
for i in store.all_edges_dictionary['datatype_edges']:
    print(i)
for i in store.all_edges_dictionary['object_edges']:
    print(i)
for i in store.all_edges_dictionary['object_edges_with_restrictions']:
    print(i)





# current_dir = os.path.dirname(os.path.abspath(__file__))
# main_path = os.path.join(current_dir, '../ontology_tools_kkg/saved_network_and_ontology/')
#
# for edge in store.all_edges_dictionary['datatype_edges']:
#     if edge[1] in dict_to_save:
#         string_to_add = "%s %s %s" % (edge[0], edge[1], edge[2])
#         dict_to_save[edge[1]].add(string_to_add)
#     else:
#         string_to_add = "%s %s %s" % (edge[0], edge[1], edge[2])
#         dict_to_save[edge[1]] = set()
#         dict_to_save[edge[1]].add(string_to_add)
#
# for edge in store.all_edges_dictionary['object_edges']:
#     if edge[1] in dict_to_save:
#         string_to_add = "%s %s %s" % (edge[0], edge[1], edge[2])
#         dict_to_save[edge[1]].add(string_to_add)
#     else:
#         string_to_add = "%s %s %s" % (edge[0], edge[1], edge[2])
#         dict_to_save[edge[1]] = set()
#         dict_to_save[edge[1]].add(string_to_add)
# for edge in store.all_edges_dictionary['object_edges_with_restrictions']:
#     if edge[1] in dict_to_save:
#         string_to_add = "%s %s %s" % (edge[0], edge[1], edge[2])
#         dict_to_save[edge[1]].add(string_to_add)
#     else:
#         string_to_add = "%s %s %s" % (edge[0], edge[1], edge[2])
#         dict_to_save[edge[1]] = set()
#         dict_to_save[edge[1]].add(string_to_add)
#
# with open(main_path + 'dictionary_that_maps_property_to_all_ontology_triples_as_string', 'wb') as f:
#     pickle.dump(dict_to_save, f)

# store = network_and_ontology_store(7)
#
#
# print('OBJECT EDGES')
# store.print_object_edges()
# print()
# print('DATATYPE EDGES')
# store.print_datatype_edges()
# print()
# print('OBJECT EDGES WITH RESTRICTIONS')
# store.print_object_edges_with_restrictions()
# print()
# print('SUBCLASS STRUCTURE')
# store.print_subclass_edges()
# Get the full reachable subgraph


# obj = network_and_ontology_store(7)
# src = 'sor:Gebouw'
# tar = 'sor:Gemeente'
# paths = sorted(nx.all_simple_paths(obj.network, source=src, target=tar))
# # Sort paths based on length
# sorted_paths = sorted(paths, key=lambda x: len(x))
# # Extract the first 10 routes
# first_10_routes = sorted_paths[:10]
#
# # Print the first 10 routes
# for route in first_10_routes:
#     print(route)
# print()
# print()
# print('OTHER APPROACH')
#
import itertools


# # Compute the shortest simple paths
# shortest_paths_generator = nx.shortest_simple_paths(obj.network, source=src, target=tar)
#
# # Take the first 10 paths from the generator
# first_10_paths = list(itertools.islice(shortest_paths_generator, 10))
#
# # Print the first 10 paths
# for path in first_10_paths:
#     print(path)
#
# obj.print_datatype_edges()
# obj.print_object_edges()
# obj.print_object_edges_with_restrictions()

#
# reachable_subgraph_1 = get_full_reachable_subgraph(obj.network, "sor:Gebouw")
# reachable_subgraph_2 = get_full_reachable_subgraph(obj.network, "wbk:Buurt")
# reachable_subgraph_3 = get_full_reachable_subgraph(obj.network, "wbk:Wijk")
# reachable_subgraph_4 = get_full_reachable_subgraph(obj.network, "wbk:Gemeente")
# reachable_subgraph_5 = get_full_reachable_subgraph(obj.network, "sor:Gemeente")
# reachable_subgraph_6 = get_full_reachable_subgraph(obj.network, "sor:Provincie")
# reachable_subgraph_7 = get_full_reachable_subgraph(obj.network, "sor:Verblijfsobject")
# reachable_subgraph_8 = get_full_reachable_subgraph(obj.network, "sor:Nummeraanduiding")
# reachable_subgraph_9 = get_full_reachable_subgraph(obj.network, "kad:Standplaats")
# reachable_subgraph_10 = get_full_reachable_subgraph(obj.network, "kad:Ligplaats")
# reachable_subgraph_11 = get_full_reachable_subgraph(obj.network, "sor:OpenbareRuimte")
# reachable_subgraph_12 = get_full_reachable_subgraph(obj.network, "sor:Woonplaats")
# reachable_subgraph_13 = get_full_reachable_subgraph(obj.network, "sor:Perceel")
# all_subgraphs = [reachable_subgraph_1,reachable_subgraph_2,reachable_subgraph_3,reachable_subgraph_4,
#  reachable_subgraph_5,reachable_subgraph_6,reachable_subgraph_7,reachable_subgraph_8,
#  reachable_subgraph_9,reachable_subgraph_10,reachable_subgraph_11,reachable_subgraph_12, reachable_subgraph_13]
#
#
# # Compute the union of all subgraphs
# union_subgraph = nx.compose_all(all_subgraphs)
# print(len(union_subgraph.nodes()))
#
# nodes_not_in_union = set(obj.network.nodes()) - set(union_subgraph.nodes())
# print(nodes_not_in_union)























































# store.print_classes()
# store.print_relations()
#store.print_object_edges()
# store.get_subgraph_intersection_between_nodes(source= "sor:Verblijfsobject", target = 'sor:Perceel' )
#store.print_all_edges_in_graph()
#store.get_all_shortest_path_with_intermediate_node_constraint(source="sor:Verblijfsobject", target='sor:Perceel',
                                                            #  node_in_between='sor:Nummeraanduiding')
# store.get_shortest_route_between_nodes(source='sor:Verblijfsobject', target='sor:Perceel')
# store.get_all_simple_routs_between_nodes(source='sor:Verblijfsobject', target='sor:Perceel')

# '''
# This shortest path algorithm works fine.
# '''
#
# # Create a directed graph
# G = nx.DiGraph()
#
# # Add edges with labels (adding a second label between two nodes just overwrite the previous one)
# G.add_edge('Gebouw', 'Bouwjaar', label='heeftBouwjaar')
# G.add_edge('Gebouw', 'Woonplaats', label='ligtIn')
# G.add_edge('Woonplaats', 'Provincie', label='ligtIn')
# G.add_edge('Woonplaats', 'InwonerAantal', label='heeftInwonerAantal')
# G.add_edge('Provincie', 'Bestuurder', label='bestuur')
#
# # Perform shortest path calculation
# source_node = 'Gebouw'
# target_node = 'Bestuurder'
# shortest_path = nx.shortest_path(G, source=source_node, target=target_node)
#
# # Retrieve edge labels for the path
# edge_labels = {(shortest_path[i], shortest_path[i + 1]): G[shortest_path[i]][shortest_path[i + 1]]['label']
#                for i in range(len(shortest_path) - 1)}
#
# # Print the result
# print("Shortest path:", shortest_path)
# print("Edge labels:", edge_labels)
#
#
#
# '''
# Note that it might be most sensible to model the ontology as an undirected graph.
# Then we keep track of the relationships in between them in a dictionary. This allows for many different
# relations between classes that we store in the dictionary
# '''
# G = nx.Graph()
#
# # Add edges with labels (adding a second label between two nodes just overwrite the previous one)
# G.add_edge('Gebouw', 'Bouwjaar')
# G.add_edge('Gebouw', 'Woonplaats')
# G.add_edge('Woonplaats', 'Provincie')
# G.add_edge('Woonplaats', 'InwonerAantal')
# G.add_edge('Provincie', 'Bestuurder')
#
# # Perform shortest path calculation
# source_node = 'Bestuurder'
# target_node = 'Gebouw'
# shortest_path = nx.shortest_path(G, source=source_node, target=target_node)
#
# print(shortest_path)
# # Retrieve edge labels for the path
# #edge_labels = {(shortest_path[i], shortest_path[i + 1]): G[shortest_path[i]][shortest_path[i + 1]]['label']
# #               for i in range(len(shortest_path) - 1)}
#
# # Print the result
# #print("Shortest path:", shortest_path)
# #print("Edge labels:", edge_labels)
#
# '''
# Now note i will add a seperate part to the graph that is not connected to the previous part.
# '''
#
# G.add_edge('Kadaster', 'Employee')
# G.add_edge('Kadaster', 'Manager')
# G.add_edge('Employee', 'Employeetype')
# G.add_edge('Employee', 'Salary')
#
# source_node = 'Bestuurder'
# target_node = 'Kadaster'
# #shortest_path = nx.shortest_path(G, source=source_node, target=target_node)
#
# #print(shortest_path)
#
# '''
# Now we create functions for finding subgraphs and test them out with our disconnected graph
# '''
#
# import matplotlib.pyplot as plt
#
# def get_full_reachable_subgraph(graph, central_node):
#     """
#     Get the subgraph of all nodes reachable from the specified node.
#
#     Parameters:
#     - graph: NetworkX graph
#     - central_node: Node for which the subgraph is to be obtained
#
#     Returns:
#     - reachable_subgraph: NetworkX subgraph
#     """
#     reachable_nodes = nx.descendants(graph, central_node)
#     reachable_nodes.add(central_node)  # Include the central node itself
#     reachable_subgraph = graph.subgraph(reachable_nodes)
#     return reachable_subgraph
#
#
#
# # Get the full reachable subgraph
# reachable_subgraph = get_full_reachable_subgraph(G, "Kadaster")
# print(reachable_subgraph.edges)
#
# # Visualization (optional)
# pos = nx.spring_layout(G)
# nx.draw(G, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=10, font_color="black")
# nx.draw(reachable_subgraph, pos, node_size=700, node_color="salmon", label="Full Reachable Subgraph")
# #plt.show()
#
# # WEES VOORZICHTIG MET DE EDGES!
# # WE HEBBEN EEN UNDIRECTED GRAPH! DUS DE VOLGORDE KAN OOK ANDERS ZIJN DAN TOEGEVOEGD!
# # WANNEER WE DE ONTOLOGY OPHALEN WIL JE BEIDE KANTEN UITPROBEREN OM ONTOLOGY BESCHRIJVENDE TRIPLES TE VINDEN
#
# '''
# Here we describe how to get UNIONS of subgraphs.
# This can be nice to check which parts of the ontology are connected to our central part, since my method will aim
# to answer questions only about that for now..
#
# Note that an INTERSECTION could be useful to determine whether classes are related to our central classes. However,
# this is less interesting if simply do a UNION on the subgraphs around our central classes. Then we already only focus
# on parts of the ontology that actually are connected.
# '''
#
# # Create an example graph
#
# subgraph_1 = get_full_reachable_subgraph(G, "Kadaster")
# subgraph_2 = get_full_reachable_subgraph(G, "Gebouw")
# # Take the union of the three subgraphs
# union_subgraph = nx.union(subgraph_1, subgraph_2)
# print(subgraph_1.edges)
# print(subgraph_2.edges)
# print(union_subgraph.edges)
