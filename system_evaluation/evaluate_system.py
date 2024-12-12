'''
Novel functions: GROUP BY, HAVING, AVERAGE, SUBQUERY, MULTIPLE VARIABLES, STRING

'''
from semantic_parsing import semantic_parsing_few_shot_with_ontology, \
    semantic_parsing_few_shot_NO_ONTOLOGY, semantic_parsing_without_in_context_examples, only_get_condensed_ontologies
from ontology_condensation_inference.ontology_condensation_inference import network_and_ontology_store
from semantic_parsing import run_sparql
import time
import openai

import json

"""
In the following section, we define several functions designed to evaluate results 
and extract key characteristics from datasets.
"""


def get_statistics_of_dataset(dataset):
    """
    Retrieves key statistics and characteristics of the given dataset.

    This method processes a dataset (loaded from JSON files), computing various statistics that describe the
    characteristics of the data. These statistics are returned as a dictionary, with each key representing a
    specific statistic and the corresponding value providing the associated count.
    Additionally, the statistics are printed to the console for immediate review.

    The returned dictionary includes the following keys and their respective values:
    - 'iid_count': Number of IID generalization questions
    - 'compositional_count': Number of compositional generalization questions
    - 'zeroshot_count': Number of zero-shot generalization questions
    - 'groupby_count': Number of zero-shot questions containing the GROUP BY operator
    - 'having_count': Number of zero-shot questions containing the HAVING operator
    - 'average_count': Number of zero-shot questions containing the AVERAGE operator
    - 'subquery_count': Number of zero-shot questions containing SUBQUERY
    - 'multiplevariables_count': Number of zero-shot questions containing MULTIPLE VARIABLE RETRIEVAL
    - 'string_count': Number of zero-shot questions containing STRING FUNCTIONS
    - 'newschema_count': Number of zero-shot questions containing new schema items
    - 'onerelation_count': Number of zero-shot questions requiring 1 relation
    - 'tworelation_count': Number of zero-shot questions requiring 2 relations
    - 'threerelation_count': Number of zero-shot questions requiring 3 relations
    - 'fourrelation_count': Number of zero-shot questions requiring 4 relations
    - 'fiverelation_count': Number of zero-shot questions requiring 5 relations
    - 'sixrelation_count': Number of zero-shot questions requiring 6 relations
    - 'new_schema_onerelation_count': Number of zero-shot questions with new schema items and requiring 1 relation
    - 'new_schema_tworelation_count': Number of zero-shot questions with new schema items and requiring 2 relations
    - 'new_schema_threerelation_count': Number of zero-shot questions with new schema items and requiring 3 relations
    - 'new_schema_fourrelation_count': Number of zero-shot questions with new schema items and requiring 4 relations
    - 'new_schema_fiverelation_count': Number of zero-shot questions with new schema items and requiring 5 relations
    - 'new_schema_sixrelation_count': Number of zero-shot questions with new schema items and requiring 6 relations

    Args:
        dataset (list of dicts): A dataset loaded from a JSON file, represented as a list of dictionaries.
                                  Each dictionary corresponds to a question in the dataset.

    Returns:
        dict: A dictionary containing the statistics described above, with each key-value pair providing
              a specific statistic about the dataset.


    """
    dict_to_return = {}
    print('Total number of questions in the dataset:')
    print(len(dataset))
    print('Number of iid generalization questions:')
    counter = 0
    for question_dict in dataset:
        if question_dict['generalization_level'] == 'iid':
            counter = counter + 1
    print(counter)
    dict_to_return['iid_count'] = counter
    print('Number of compositional generalization questions:')
    counter = 0
    for question_dict in dataset:
        if question_dict['generalization_level'] == 'compositional':
            counter = counter + 1
    print(counter)
    dict_to_return['compositional_count'] = counter
    print('Number of zero-shot generalization questions:')
    counter = 0
    for question_dict in dataset:
        if question_dict['generalization_level'] == 'zeroshot':
            counter = counter + 1
    print(counter)
    dict_to_return['zeroshot_count'] = counter
    print('Number of zero-shot questions containing GROUP BY:')
    counter = 0
    for question_dict in dataset:
        if 'GROUP BY' in question_dict['novel_functions']:
            counter = counter + 1
    print(counter)
    dict_to_return['groupby_count'] = counter
    print('Number of zero-shot questions containing HAVING:')
    counter = 0
    for question_dict in dataset:
        if 'HAVING' in question_dict['novel_functions']:
            counter = counter + 1
    print(counter)
    dict_to_return['having_count'] = counter
    print('Number of zero-shot questions containing AVERAGE')
    counter = 0
    for question_dict in dataset:
        if 'AVERAGE' in question_dict['novel_functions']:
            counter = counter + 1
    print(counter)
    dict_to_return['average_count'] = counter
    print('Number of zero-shot questions containing SUBQUERY')
    counter = 0
    for question_dict in dataset:
        if 'SUBQUERY' in question_dict['novel_functions']:
            counter = counter + 1
    print(counter)
    dict_to_return['subquery_count'] = counter
    print('Number of zero-shot questions containing MULTIPLE VARIABLE RETRIEVAL')
    counter = 0
    for question_dict in dataset:
        if 'MULTIPLE VARIABLES' in question_dict['novel_functions']:
            counter = counter + 1
    print(counter)
    dict_to_return['multiplevariables_count'] = counter
    print('Number of zero-shot questions containing STRING FUNCTIONS')
    counter = 0
    for question_dict in dataset:
        if 'STRING' in question_dict['novel_functions']:
            counter = counter + 1
    print(counter)
    dict_to_return['string_count'] = counter
    print('Number of zero-shot questions containing new schema items')
    counter = 0
    for question_dict in dataset:
        if 'new_schema_item' in question_dict.keys():
            if question_dict['new_schema_item'] == True:
                counter = counter + 1
    print(counter)
    dict_to_return['newschema_count'] = counter
    print('Number of zero-shot questions requiring 1 relation')
    counter = 0
    for question_dict in dataset:
        if 'number_of_relations' in question_dict.keys():
            if question_dict['number_of_relations'] == 1:
                counter = counter + 1
    print(counter)
    dict_to_return['onerelation_count'] = counter
    print('Number of zero-shot questions requiring 2 relation')
    counter = 0
    for question_dict in dataset:
        if 'number_of_relations' in question_dict.keys():
            if question_dict['number_of_relations'] == 2:
                counter = counter + 1
    print(counter)
    dict_to_return['tworelation_count'] = counter
    print('Number of zero-shot questions requiring 3 relation')
    counter = 0
    for question_dict in dataset:
        if 'number_of_relations' in question_dict.keys():
            if question_dict['number_of_relations'] == 3:
                counter = counter + 1
    print(counter)
    dict_to_return['threerelation_count'] = counter
    print('Number of zero-shot questions requiring 4 relation')
    counter = 0
    for question_dict in dataset:
        if 'number_of_relations' in question_dict.keys():
            if question_dict['number_of_relations'] == 4:
                counter = counter + 1
    print(counter)
    dict_to_return['fourrelation_count'] = counter
    print('Number of zero-shot questions requiring 5 relation')
    counter = 0
    for question_dict in dataset:
        if 'number_of_relations' in question_dict.keys():
            if question_dict['number_of_relations'] == 5:
                counter = counter + 1
    print(counter)
    dict_to_return['fiverelation_count'] = counter
    print('Number of zero-shot questions requiring 6 relation')
    counter = 0
    for question_dict in dataset:
        if 'number_of_relations' in question_dict.keys():
            if question_dict['number_of_relations'] == 6:
                counter = counter + 1
    print(counter)
    dict_to_return['sixrelation_count'] = counter

    print('Number of zero-shot questions with new schema item requiring 1 relation')
    counter = 0
    for question_dict in dataset:
        if 'new_schema_item' in question_dict.keys():
            if question_dict['new_schema_item'] == True:
                if 'number_of_relations' in question_dict.keys():
                    if question_dict['number_of_relations'] == 1:
                        counter = counter + 1
    print(counter)
    dict_to_return['new_schema_onerelation_count'] = counter
    print('Number of zero-shot questions with new schema item requiring 2 relation')
    counter = 0
    for question_dict in dataset:
        if 'new_schema_item' in question_dict.keys():
            if question_dict['new_schema_item'] == True:
                if 'number_of_relations' in question_dict.keys():
                    if question_dict['number_of_relations'] == 2:
                        counter = counter + 1
    print(counter)
    dict_to_return['new_schema_tworelation_count'] = counter
    print('Number of zero-shot questions with new schema item requiring 3 relation')
    counter = 0
    for question_dict in dataset:
        if 'new_schema_item' in question_dict.keys():
            if question_dict['new_schema_item'] == True:
                if 'number_of_relations' in question_dict.keys():
                    if question_dict['number_of_relations'] == 3:
                        counter = counter + 1
    print(counter)
    dict_to_return['new_schema_threerelation_count'] = counter
    print('Number of zero-shot questions with new schema item requiring 4 relation')
    counter = 0
    for question_dict in dataset:
        if 'new_schema_item' in question_dict.keys():
            if question_dict['new_schema_item'] == True:
                if 'number_of_relations' in question_dict.keys():
                    if question_dict['number_of_relations'] == 4:
                        counter = counter + 1
    print(counter)
    dict_to_return['new_schema_fourrelation_count'] = counter
    print('Number of zero-shot questions with new schema item requiring 5 relation')
    counter = 0
    for question_dict in dataset:
        if 'new_schema_item' in question_dict.keys():
            if question_dict['new_schema_item'] == True:
                if 'number_of_relations' in question_dict.keys():
                    if question_dict['number_of_relations'] == 5:
                        counter = counter + 1
    print(counter)
    dict_to_return['new_schema_fiverelation_count'] = counter
    print('Number of zero-shot questions with new schema item requiring 6 relation')
    counter = 0
    for question_dict in dataset:
        if 'new_schema_item' in question_dict.keys():
            if question_dict['new_schema_item'] == True:
                if 'number_of_relations' in question_dict.keys():
                    if question_dict['number_of_relations'] == 6:
                        counter = counter + 1
    print(counter)
    dict_to_return['new_schema_sixrelation_count'] = counter

    return dict_to_return


# Retrieve a string containing all the necessary prefixes used for the KKG.
# These prefixes must be appended to a SPARQL query to ensure the SPARQL processor
# can correctly interpret terms like "sor:", among others.
prefix_string = network_and_ontology_store(7).prefix_string


def check_whether_sparql_results_are_the_same(query_1, query_2):
    """
    Compares the results of two SPARQL queries executed against the KKG to determine if the retrieved results are
    identical. The ordering of columns is allowed to be different in the two result sets.
    This method is needed when calculating execution accuracy.

    The method returns True if the results from both queries match,
    and False if there are any discrepancies.

    If either of the queries returns an error or an empty result, the method prints an error message and returns False.

    Args:
        query_1 (str): The first SPARQL query to be executed.
        query_2 (str): The second SPARQL query to be executed.

    Returns:
        bool: True if the results of both queries are identical in terms of columns, rows, and data; False otherwise.

    Prints:
        - Error message if either query fails or returns empty results.

    """
    # We retrieve the JSON results from the triple store belonging to the queries
    result_1 = run_sparql(prefix_string + '\n' + query_1)
    result_2 = run_sparql(prefix_string + '\n' + query_2)

    # If training data returns an error we print there is a fatal error
    if result_1 == 'That query failed. Perhaps you could try a different one?':
        print('TRAINING DATA WRONG')
        return False
    elif result_1 == []:
        print('TRAINING DATA WRONG')
        return False
    elif result_2 == 'That query failed. Perhaps you could try a different one?':
        return False
    elif result_2 == []:
        return False

    # First we check whether the two results have the same amount of columns.
    # If they do not, we return False
    are_same = True

    print(result_1)
    print(result_2)
    if len(result_1[0]) != len(result_2[0]):
        return False

    # Next we check whether they have the same amount of rows.
    # If they do not, we return False
    if len(result_1) != len(result_2):
        return False

    # Now we check whether the rows contain the same information.
    # Even if the information is in a different order, we classify the results as being the same
    for row_a, row_b in zip(result_1, result_2):
        row_a = list(row_a.values())
        row_b = list(row_b.values())
        set1 = {frozenset(d.items()) for d in row_a}
        set2 = {frozenset(d.items()) for d in row_b}

        if set1 != set2:
            are_same = False

    return are_same


def test_data_set_with_ontology(dataset='dev',  # you can also set this to test
                                number_of_in_context_examples=5,
                                number_of_routes=1,
                                apply_meta_data_filtering=True,
                                diversity_reranking_of_in_context_examples=True,
                                mask_location_in_question=True,
                                naive_schema_linking_without_graph_traversal=False,
                                print_detailed_performance=False,
                                ontology_condensation=True
                                ):
    """
    This method takes a dataset and parses SPARQL queries for the questions contained in it.
    The original dataset is a list of dictionaries, where each dictionary represents a specific question.
    To each of the dictionaries we add two new key-value pairs:
    1) 'generated_query' : the query generated by the system
    2) 'results_of_generated_and_golden_query_match' : A boolean indicating whether the results of the generated
    query and benchmark query match with eachother.

    Semantic parsing is executed while including in-context examples and ontology information during semantic parsing.
    To the original dataset, a final dictionary is appended containing a detailed performance summary.

    Args:
        dataset (str): The dataset to test, either 'dev' (default) or 'test'.
        number_of_in_context_examples (int): The number of in-context examples to use during evaluation.
        number_of_routes (int): The number of routes between relevant nodes in GTOC.
        apply_meta_data_filtering (bool): Whether to apply meta-data pre-filtering during example retrieval.
        diversity_reranking_of_in_context_examples (bool): Whether to apply MMR during example retrieval.
        mask_location_in_question (bool): Whether to apply entity masking during example retrieval.
        naive_schema_linking_without_graph_traversal (bool): If True, apply naive schema linking without graph traversal.
        print_detailed_performance (bool): If True, print detailed performance metrics for the evaluation.
        ontology_condensation (bool): Whether to apply schema condensation or include the entire schema during evaluation.

    Returns:
        list: The method returns the original dataset, now updated to include parsed queries for each question.
        Additionally, a dictionary containing a detailed performance summary is appended as the final element of the
        dataset. The appended performance summary dictionary contains the following key-value pairs:
            - 'overall_execution_accuracy': Execution accuracy on the entire dataset.
            - 'entity linking accuracy': Accuracy of entity linking during reasoning.
            - 'iid': Execution accuracy on i.i.d. generalization questions.
            - 'compositional': Execution accuracy on compositional generalization questions.
            - 'zeroshot': Execution accuracy on zero-shot generalization questions.
            - 'group by': Execution accuracy on zero-shot questions with the GROUP BY operator.
            - 'having': Execution accuracy on zero-shot questions with the HAVING operator.
            - 'average': Execution accuracy on zero-shot questions with the AVERAGE operator.
            - 'subquery': Execution accuracy on zero-shot questions with subqueries.
            - 'multiple variables': Execution accuracy on zero-shot questions requiring multiple variable retrieval.
            - 'string': Execution accuracy on zero-shot questions involving string functions.
            - 'new schema item': Execution accuracy on zero-shot questions with new schema items.
            - 'one relation': Execution accuracy on zero-shot questions requiring one relation.
            - 'two relation': Execution accuracy on zero-shot questions requiring two relations.
            - 'three relation': Execution accuracy on zero-shot questions requiring three relations.
            - 'four relation': Execution accuracy on zero-shot questions requiring four relations.
            - 'five relation': Execution accuracy on zero-shot questions requiring five relations.
            - 'six relation': Execution accuracy on zero-shot questions requiring six relations.
            - 'new schema & one relation': Execution accuracy on questions with new schema items requiring one relation.
            - 'new schema & two relation': Execution accuracy on questions with new schema items requiring two relations.
            - 'new schema & three relation': Execution accuracy on questions with new schema items requiring three relations.
            - 'new schema & four relation': Execution accuracy on questions with new schema items requiring four relations.
            - 'new schema & five relation': Execution accuracy on questions with new schema items requiring five relations.
            - 'new schema & six relation': Execution accuracy on questions with new schema items requiring six relations.

    """
    if dataset == 'dev':

        # Load development_set from JSON
        file_path = "development_set.json"
        with open(file_path, "r") as json_file:
            dataset = json.load(json_file)

    elif dataset == 'test':

        # Load test_set from JSON
        file_path = "test_set.json"
        with open(file_path, "r") as json_file:
            dataset = json.load(json_file)

    if diversity_reranking_of_in_context_examples == True:
        rerank_style = 'max_marginal_similarity'
    else:
        rerank_style = 'semantic_similarity'

    correct_execution_count = 0
    correct_entity_linking_count = 0
    iid_correct_execution_count = 0
    compositional_correct_execution_count = 0
    zeroshot_correct_execution_count = 0
    group_by_correct_execution_count = 0
    having_correct_execution_count = 0
    average_correct_execution_count = 0
    subquery_correct_execution_count = 0
    multiple_variables_correct_execution_count = 0
    string_correct_execution_count = 0
    new_schema_correct_execution_count = 0
    one_relation_correct_execution_count = 0
    two_relation_correct_execution_count = 0
    three_relation_correct_execution_count = 0
    four_relation_correct_execution_count = 0
    five_relation_correct_execution_count = 0
    six_relation_correct_execution_count = 0

    new_schema_one_relation_correct_execution_count = 0
    new_schema_two_relation_correct_execution_count = 0
    new_schema_three_relation_correct_execution_count = 0
    new_schema_four_relation_correct_execution_count = 0
    new_schema_five_relation_correct_execution_count = 0
    new_schema_six_relation_correct_execution_count = 0

    question_processed_count = 0
    for question_dict in dataset:

        print('Amount of questions that need to be processed:')
        print(len(dataset))
        question_processed_count = question_processed_count + 1
        print("Processing question %s" % (question_processed_count))

        # First we check whether the execution accuracy
        answer_dict = None
        try:
            answer_dict = semantic_parsing_few_shot_with_ontology(question_dict['question'],
                                                                  k_shortest_routes=number_of_routes,
                                                                  number_of_few_shot_examples=number_of_in_context_examples,
                                                                  meta_data_filtering_for_few_shot_examples=apply_meta_data_filtering,
                                                                  retrieve_examples_with_question_without_location=mask_location_in_question,
                                                                  example_retrieval_style=rerank_style,
                                                                  naive_schema_linking_without_graph_traversal=naive_schema_linking_without_graph_traversal,
                                                                  verbose=False,
                                                                  condense_ontology=ontology_condensation)
        except openai.RateLimitError as err:
            # Handle rate limit error
            print("Rate limit exceeded. Retrying after 1 minute...")
            time.sleep(60)  # Wait for 1 minute
            # Retry the code block
            try:
                answer_dict = semantic_parsing_few_shot_with_ontology(question_dict['question'],
                                                                      k_shortest_routes=number_of_routes,
                                                                      number_of_few_shot_examples=number_of_in_context_examples,
                                                                      meta_data_filtering_for_few_shot_examples=apply_meta_data_filtering,
                                                                      retrieve_examples_with_question_without_location=mask_location_in_question,
                                                                      example_retrieval_style=rerank_style,
                                                                      naive_schema_linking_without_graph_traversal=naive_schema_linking_without_graph_traversal,
                                                                      verbose=False,
                                                                      condense_ontology=ontology_condensation)
            except openai.RateLimitError as err:
                # If rate limit error occurs again, handle it or raise it
                print("Rate limit exceeded even after retrying. Please check your usage.")
                raise err

        golden_query = question_dict['sparql']
        generated_query = answer_dict['query']
        question_dict['generated_query'] = generated_query

        execution_accuracy_boolean = check_whether_sparql_results_are_the_same(golden_query, generated_query)
        question_dict['results_of_generated_and_golden_query_match'] = execution_accuracy_boolean

        if execution_accuracy_boolean == True:
            correct_execution_count = correct_execution_count + 1

            if question_dict['generalization_level'] == 'iid':
                iid_correct_execution_count = iid_correct_execution_count + 1

            if question_dict['generalization_level'] == 'compositional':
                compositional_correct_execution_count = compositional_correct_execution_count + 1

            if question_dict['generalization_level'] == 'zeroshot':
                zeroshot_correct_execution_count = zeroshot_correct_execution_count + 1

            if 'GROUP BY' in question_dict['novel_functions']:
                group_by_correct_execution_count = group_by_correct_execution_count + 1

            if 'HAVING' in question_dict['novel_functions']:
                having_correct_execution_count = having_correct_execution_count + 1

            if 'AVERAGE' in question_dict['novel_functions']:
                average_correct_execution_count = average_correct_execution_count + 1

            if 'SUBQUERY' in question_dict['novel_functions']:
                subquery_correct_execution_count = subquery_correct_execution_count + 1

            if 'MULTIPLE VARIABLES' in question_dict['novel_functions']:
                multiple_variables_correct_execution_count = multiple_variables_correct_execution_count + 1

            if 'STRING' in question_dict['novel_functions']:
                string_correct_execution_count = string_correct_execution_count + 1

            if 'new_schema_item' in question_dict.keys():
                if question_dict['new_schema_item'] == True:
                    new_schema_correct_execution_count = new_schema_correct_execution_count + 1

            if 'number_of_relations' in question_dict.keys():
                if question_dict['number_of_relations'] == 1:
                    one_relation_correct_execution_count = one_relation_correct_execution_count + 1
                    if 'new_schema_item' in question_dict.keys():
                        if question_dict['new_schema_item'] == True:
                            new_schema_one_relation_correct_execution_count = new_schema_one_relation_correct_execution_count + 1

            if 'number_of_relations' in question_dict.keys():
                if question_dict['number_of_relations'] == 2:
                    two_relation_correct_execution_count = two_relation_correct_execution_count + 1
                    if 'new_schema_item' in question_dict.keys():
                        if question_dict['new_schema_item'] == True:
                            new_schema_two_relation_correct_execution_count = new_schema_two_relation_correct_execution_count + 1

            if 'number_of_relations' in question_dict.keys():
                if question_dict['number_of_relations'] == 3:
                    three_relation_correct_execution_count = three_relation_correct_execution_count + 1
                    if 'new_schema_item' in question_dict.keys():
                        if question_dict['new_schema_item'] == True:
                            new_schema_three_relation_correct_execution_count = new_schema_three_relation_correct_execution_count + 1

            if 'number_of_relations' in question_dict.keys():
                if question_dict['number_of_relations'] == 4:
                    four_relation_correct_execution_count = four_relation_correct_execution_count + 1
                    if 'new_schema_item' in question_dict.keys():
                        if question_dict['new_schema_item'] == True:
                            new_schema_four_relation_correct_execution_count = new_schema_four_relation_correct_execution_count + 1

            if 'number_of_relations' in question_dict.keys():
                if question_dict['number_of_relations'] == 5:
                    five_relation_correct_execution_count = five_relation_correct_execution_count + 1
                    if 'new_schema_item' in question_dict.keys():
                        if question_dict['new_schema_item'] == True:
                            new_schema_five_relation_correct_execution_count = new_schema_five_relation_correct_execution_count + 1

            if 'number_of_relations' in question_dict.keys():
                if question_dict['number_of_relations'] == 6:
                    six_relation_correct_execution_count = six_relation_correct_execution_count + 1
                    if 'new_schema_item' in question_dict.keys():
                        if question_dict['new_schema_item'] == True:
                            new_schema_six_relation_correct_execution_count = new_schema_six_relation_correct_execution_count + 1

        generated_granularity = answer_dict['granularity']
        generated_location = answer_dict['location']
        golden_granularity = question_dict['granularity']
        golden_location = question_dict['location']

        if generated_granularity == golden_granularity:
            if generated_location == golden_location:
                correct_entity_linking_count = correct_entity_linking_count + 1

            else:
                'entity linking went wrong'
                print(question_dict)
                print(generated_granularity)
                print(generated_location)

    print('Execution accuracy:')
    print(correct_execution_count / len(dataset))
    print('Entity linking accuracy:')
    print(correct_entity_linking_count / len(dataset))

    counts = get_statistics_of_dataset(dataset)
    performance_dict = {
        'overall_execution_accuracy': correct_execution_count / len(dataset),
        'entity linking accuracy': correct_entity_linking_count / len(dataset),
        'iid': iid_correct_execution_count / counts['iid_count'] if counts['iid_count'] != 0 else 'not available',
        'compositional': compositional_correct_execution_count / counts['compositional_count'] if counts[
                                                                                                      'compositional_count'] != 0 else 'not available',
        'zeroshot': zeroshot_correct_execution_count / counts['zeroshot_count'] if counts[
                                                                                       'zeroshot_count'] != 0 else 'not available',
        'group by': group_by_correct_execution_count / counts['groupby_count'] if counts[
                                                                                      'groupby_count'] != 0 else 'not available',
        'having': having_correct_execution_count / counts['having_count'] if counts[
                                                                                 'having_count'] != 0 else 'not available',
        'average': average_correct_execution_count / counts['average_count'] if counts[
                                                                                    'average_count'] != 0 else 'not available',
        'subquery': subquery_correct_execution_count / counts['subquery_count'] if counts[
                                                                                       'subquery_count'] != 0 else 'not available',
        'multiple variables': multiple_variables_correct_execution_count / counts['multiplevariables_count'] if counts[
                                                                                                                    'multiplevariables_count'] != 0 else 'not available',
        'string': string_correct_execution_count / counts['string_count'] if counts[
                                                                                 'string_count'] != 0 else 'not available',
        'new schema item': new_schema_correct_execution_count / counts['newschema_count'] if counts[
                                                                                                 'newschema_count'] != 0 else 'not available',
        'one relation': one_relation_correct_execution_count / counts['onerelation_count'] if counts[
                                                                                                  'onerelation_count'] != 0 else 'not available',
        'two relation': two_relation_correct_execution_count / counts['tworelation_count'] if counts[
                                                                                                  'tworelation_count'] != 0 else 'not available',
        'three relation': three_relation_correct_execution_count / counts['threerelation_count'] if counts[
                                                                                                        'threerelation_count'] != 0 else 'not available',
        'four relation': four_relation_correct_execution_count / counts['fourrelation_count'] if counts[
                                                                                                     'fourrelation_count'] != 0 else 'not available',
        'five relation': five_relation_correct_execution_count / counts['fiverelation_count'] if counts[
                                                                                                     'fiverelation_count'] != 0 else 'not available',
        'six relation': six_relation_correct_execution_count / counts['sixrelation_count'] if counts[
                                                                                                  'sixrelation_count'] != 0 else 'not available',
        'new schema & one relation': new_schema_one_relation_correct_execution_count / counts[
            'new_schema_onerelation_count'] if
        counts[
            'new_schema_onerelation_count'] != 0 else 'not available',
        'new schema & two relation': new_schema_two_relation_correct_execution_count / counts[
            'new_schema_tworelation_count'] if
        counts[
            'new_schema_tworelation_count'] != 0 else 'not available',
        'new schema & three relation': new_schema_three_relation_correct_execution_count / counts[
            'new_schema_threerelation_count'] if counts[
                                                     'new_schema_threerelation_count'] != 0 else 'not available',
        'new schema & four relation': new_schema_four_relation_correct_execution_count / counts[
            'new_schema_fourrelation_count'] if
        counts[
            'new_schema_fourrelation_count'] != 0 else 'not available',
        'new schema & five relation': new_schema_five_relation_correct_execution_count / counts[
            'new_schema_fiverelation_count'] if
        counts[
            'new_schema_fiverelation_count'] != 0 else 'not available',
        'new schema & six relation': new_schema_six_relation_correct_execution_count / counts[
            'new_schema_sixrelation_count'] if
        counts[
            'new_schema_sixrelation_count'] != 0 else 'not available'
    }

    dataset.append(performance_dict)

    if print_detailed_performance:
        for key, value in performance_dict.items():
            print(key)
            print(value)
    return dataset


def test_data_set_no_ontology(dataset='dev',  # you can also set this to test
                              number_of_in_context_examples=5,
                              apply_meta_data_filtering=True,
                              diversity_reranking_of_in_context_examples=True,
                              mask_location_in_question=True,
                              print_detailed_performance=False
                              ):
    """
    This method takes a dataset and parses SPARQL queries for the questions contained in it.
    The original dataset is a list of dictionaries, where each dictionary represents a specific question.
    To each of the dictionaries we add two new key-value pairs:
    1) 'generated_query' : the query generated by the system
    2) 'results_of_generated_and_golden_query_match' : A boolean indicating whether the results of the generated
    query and benchmark query match with eachother.

    Semantic parsing is executed while including in-context examples but excluding ontology information during
    semantic parsing. To the original dataset, a final dictionary is appended containing a detailed performance summary.

    Args:
        dataset (str): The dataset to test, either 'dev' (default) or 'test'.
        number_of_in_context_examples (int): The number of in-context examples to use during evaluation.
        apply_meta_data_filtering (bool): Whether to apply meta-data pre-filtering during example retrieval.
        diversity_reranking_of_in_context_examples (bool): Whether to apply MMR during example retrieval.
        mask_location_in_question (bool): Whether to apply entity masking during example retrieval.
        print_detailed_performance (bool): If True, print detailed performance metrics for the evaluation.

    Returns:
        list: The method returns the original dataset, now updated to include parsed queries for each question.
        Additionally, a dictionary containing a detailed performance summary is appended as the final element of the
        dataset. The appended performance summary dictionary contains the following key-value pairs:
            - 'overall_execution_accuracy': Execution accuracy on the entire dataset.
            - 'entity linking accuracy': Accuracy of entity linking during reasoning.
            - 'iid': Execution accuracy on i.i.d. generalization questions.
            - 'compositional': Execution accuracy on compositional generalization questions.
            - 'zeroshot': Execution accuracy on zero-shot generalization questions.
            - 'group by': Execution accuracy on zero-shot questions with the GROUP BY operator.
            - 'having': Execution accuracy on zero-shot questions with the HAVING operator.
            - 'average': Execution accuracy on zero-shot questions with the AVERAGE operator.
            - 'subquery': Execution accuracy on zero-shot questions with subqueries.
            - 'multiple variables': Execution accuracy on zero-shot questions requiring multiple variable retrieval.
            - 'string': Execution accuracy on zero-shot questions involving string functions.
            - 'new schema item': Execution accuracy on zero-shot questions with new schema items.
            - 'one relation': Execution accuracy on zero-shot questions requiring one relation.
            - 'two relation': Execution accuracy on zero-shot questions requiring two relations.
            - 'three relation': Execution accuracy on zero-shot questions requiring three relations.
            - 'four relation': Execution accuracy on zero-shot questions requiring four relations.
            - 'five relation': Execution accuracy on zero-shot questions requiring five relations.
            - 'six relation': Execution accuracy on zero-shot questions requiring six relations.
            - 'new schema & one relation': Execution accuracy on questions with new schema items requiring one relation.
            - 'new schema & two relation': Execution accuracy on questions with new schema items requiring two relations.
            - 'new schema & three relation': Execution accuracy on questions with new schema items requiring three relations.
            - 'new schema & four relation': Execution accuracy on questions with new schema items requiring four relations.
            - 'new schema & five relation': Execution accuracy on questions with new schema items requiring five relations.
            - 'new schema & six relation': Execution accuracy on questions with new schema items requiring six relations.

    """
    if dataset == 'dev':

        # Load development_set from JSON
        file_path = "development_set.json"
        with open(file_path, "r") as json_file:
            dataset = json.load(json_file)

    elif dataset == 'test':

        # Load test_set from JSON
        file_path = "test_set.json"
        with open(file_path, "r") as json_file:
            dataset = json.load(json_file)

    if diversity_reranking_of_in_context_examples == True:
        rerank_style = 'max_marginal_similarity'
    else:
        rerank_style = 'semantic_similarity'

    correct_execution_count = 0
    correct_entity_linking_count = 0
    iid_correct_execution_count = 0
    compositional_correct_execution_count = 0
    zeroshot_correct_execution_count = 0
    group_by_correct_execution_count = 0
    having_correct_execution_count = 0
    average_correct_execution_count = 0
    subquery_correct_execution_count = 0
    multiple_variables_correct_execution_count = 0
    string_correct_execution_count = 0
    new_schema_correct_execution_count = 0
    one_relation_correct_execution_count = 0
    two_relation_correct_execution_count = 0
    three_relation_correct_execution_count = 0
    four_relation_correct_execution_count = 0
    five_relation_correct_execution_count = 0
    six_relation_correct_execution_count = 0

    new_schema_one_relation_correct_execution_count = 0
    new_schema_two_relation_correct_execution_count = 0
    new_schema_three_relation_correct_execution_count = 0
    new_schema_four_relation_correct_execution_count = 0
    new_schema_five_relation_correct_execution_count = 0
    new_schema_six_relation_correct_execution_count = 0

    question_processed_count = 0
    for question_dict in dataset:
        print('Amount of questions that need to be processed:')
        print(len(dataset))
        question_processed_count = question_processed_count + 1
        print("Processing question %s" % (question_processed_count))

        # First we check whether the execution accuracy
        answer_dict = None
        try:
            answer_dict = semantic_parsing_few_shot_NO_ONTOLOGY(question_dict['question'],
                                                                number_of_few_shot_examples=number_of_in_context_examples,
                                                                meta_data_filtering_for_few_shot_examples=apply_meta_data_filtering,
                                                                retrieve_examples_with_question_without_location=mask_location_in_question,
                                                                example_retrieval_style=rerank_style)


        except openai.RateLimitError as err:
            # Handle rate limit error
            print("Rate limit exceeded. Retrying after 1 minute...")
            time.sleep(60)  # Wait for 1 minute
            # Retry the code block
            try:
                answer_dict = semantic_parsing_few_shot_NO_ONTOLOGY(question_dict['question'],
                                                                    number_of_few_shot_examples=number_of_in_context_examples,
                                                                    meta_data_filtering_for_few_shot_examples=apply_meta_data_filtering,
                                                                    retrieve_examples_with_question_without_location=mask_location_in_question,
                                                                    example_retrieval_style=rerank_style)
            except openai.RateLimitError as err:
                # If rate limit error occurs again, handle it or raise it
                print("Rate limit exceeded even after retrying. Please check your usage.")
                raise err

        golden_query = question_dict['sparql']
        generated_query = answer_dict['query']
        question_dict['generated_query'] = generated_query

        execution_accuracy_boolean = check_whether_sparql_results_are_the_same(golden_query, generated_query)
        question_dict['results_of_generated_and_golden_query_match'] = execution_accuracy_boolean

        if execution_accuracy_boolean == True:
            correct_execution_count = correct_execution_count + 1

            if question_dict['generalization_level'] == 'iid':
                iid_correct_execution_count = iid_correct_execution_count + 1

            if question_dict['generalization_level'] == 'compositional':
                compositional_correct_execution_count = compositional_correct_execution_count + 1

            if question_dict['generalization_level'] == 'zeroshot':
                zeroshot_correct_execution_count = zeroshot_correct_execution_count + 1

            if 'GROUP BY' in question_dict['novel_functions']:
                group_by_correct_execution_count = group_by_correct_execution_count + 1

            if 'HAVING' in question_dict['novel_functions']:
                having_correct_execution_count = having_correct_execution_count + 1

            if 'AVERAGE' in question_dict['novel_functions']:
                average_correct_execution_count = average_correct_execution_count + 1

            if 'SUBQUERY' in question_dict['novel_functions']:
                subquery_correct_execution_count = subquery_correct_execution_count + 1

            if 'MULTIPLE VARIABLES' in question_dict['novel_functions']:
                multiple_variables_correct_execution_count = multiple_variables_correct_execution_count + 1

            if 'STRING' in question_dict['novel_functions']:
                string_correct_execution_count = string_correct_execution_count + 1

            if 'new_schema_item' in question_dict.keys():
                if question_dict['new_schema_item'] == True:
                    new_schema_correct_execution_count = new_schema_correct_execution_count + 1

            if 'number_of_relations' in question_dict.keys():
                if question_dict['number_of_relations'] == 1:
                    one_relation_correct_execution_count = one_relation_correct_execution_count + 1
                    if 'new_schema_item' in question_dict.keys():
                        if question_dict['new_schema_item'] == True:
                            new_schema_one_relation_correct_execution_count = new_schema_one_relation_correct_execution_count + 1

            if 'number_of_relations' in question_dict.keys():
                if question_dict['number_of_relations'] == 2:
                    two_relation_correct_execution_count = two_relation_correct_execution_count + 1
                    if 'new_schema_item' in question_dict.keys():
                        if question_dict['new_schema_item'] == True:
                            new_schema_two_relation_correct_execution_count = new_schema_two_relation_correct_execution_count + 1

            if 'number_of_relations' in question_dict.keys():
                if question_dict['number_of_relations'] == 3:
                    three_relation_correct_execution_count = three_relation_correct_execution_count + 1
                    if 'new_schema_item' in question_dict.keys():
                        if question_dict['new_schema_item'] == True:
                            new_schema_three_relation_correct_execution_count = new_schema_three_relation_correct_execution_count + 1

            if 'number_of_relations' in question_dict.keys():
                if question_dict['number_of_relations'] == 4:
                    four_relation_correct_execution_count = four_relation_correct_execution_count + 1
                    if 'new_schema_item' in question_dict.keys():
                        if question_dict['new_schema_item'] == True:
                            new_schema_four_relation_correct_execution_count = new_schema_four_relation_correct_execution_count + 1

            if 'number_of_relations' in question_dict.keys():
                if question_dict['number_of_relations'] == 5:
                    five_relation_correct_execution_count = five_relation_correct_execution_count + 1
                    if 'new_schema_item' in question_dict.keys():
                        if question_dict['new_schema_item'] == True:
                            new_schema_five_relation_correct_execution_count = new_schema_five_relation_correct_execution_count + 1

            if 'number_of_relations' in question_dict.keys():
                if question_dict['number_of_relations'] == 6:
                    six_relation_correct_execution_count = six_relation_correct_execution_count + 1
                    if 'new_schema_item' in question_dict.keys():
                        if question_dict['new_schema_item'] == True:
                            new_schema_six_relation_correct_execution_count = new_schema_six_relation_correct_execution_count + 1

        generated_granularity = answer_dict['granularity']
        generated_location = answer_dict['location']
        golden_granularity = question_dict['granularity']
        golden_location = question_dict['location']

        if generated_granularity == golden_granularity:
            if generated_location == golden_location:
                correct_entity_linking_count = correct_entity_linking_count + 1

            else:
                'entity linking went wrong'
                print(question_dict)
                print(generated_granularity)
                print(generated_location)

    print('Execution accuracy:')
    print(correct_execution_count / len(dataset))
    print('Entity linking accuracy:')
    print(correct_entity_linking_count / len(dataset))

    counts = get_statistics_of_dataset(dataset)
    performance_dict = {
        'overall_execution_accuracy': correct_execution_count / len(dataset),
        'entity linking accuracy': correct_entity_linking_count / len(dataset),
        'iid': iid_correct_execution_count / counts['iid_count'] if counts['iid_count'] != 0 else 'not available',
        'compositional': compositional_correct_execution_count / counts['compositional_count'] if counts[
                                                                                                      'compositional_count'] != 0 else 'not available',
        'zeroshot': zeroshot_correct_execution_count / counts['zeroshot_count'] if counts[
                                                                                       'zeroshot_count'] != 0 else 'not available',
        'group by': group_by_correct_execution_count / counts['groupby_count'] if counts[
                                                                                      'groupby_count'] != 0 else 'not available',
        'having': having_correct_execution_count / counts['having_count'] if counts[
                                                                                 'having_count'] != 0 else 'not available',
        'average': average_correct_execution_count / counts['average_count'] if counts[
                                                                                    'average_count'] != 0 else 'not available',
        'subquery': subquery_correct_execution_count / counts['subquery_count'] if counts[
                                                                                       'subquery_count'] != 0 else 'not available',
        'multiple variables': multiple_variables_correct_execution_count / counts['multiplevariables_count'] if
        counts[
            'multiplevariables_count'] != 0 else 'not available',
        'string': string_correct_execution_count / counts['string_count'] if counts[
                                                                                 'string_count'] != 0 else 'not available',
        'new schema item': new_schema_correct_execution_count / counts['newschema_count'] if counts[
                                                                                                 'newschema_count'] != 0 else 'not available',
        'one relation': one_relation_correct_execution_count / counts['onerelation_count'] if counts[
                                                                                                  'onerelation_count'] != 0 else 'not available',
        'two relation': two_relation_correct_execution_count / counts['tworelation_count'] if counts[
                                                                                                  'tworelation_count'] != 0 else 'not available',
        'three relation': three_relation_correct_execution_count / counts['threerelation_count'] if counts[
                                                                                                        'threerelation_count'] != 0 else 'not available',
        'four relation': four_relation_correct_execution_count / counts['fourrelation_count'] if counts[
                                                                                                     'fourrelation_count'] != 0 else 'not available',
        'five relation': five_relation_correct_execution_count / counts['fiverelation_count'] if counts[
                                                                                                     'fiverelation_count'] != 0 else 'not available',
        'six relation': six_relation_correct_execution_count / counts['sixrelation_count'] if counts[
                                                                                                  'sixrelation_count'] != 0 else 'not available',
        'new schema & one relation': new_schema_one_relation_correct_execution_count / counts[
            'new_schema_onerelation_count'] if counts[
                                                   'new_schema_onerelation_count'] != 0 else 'not available',
        'new schema & two relation': new_schema_two_relation_correct_execution_count / counts[
            'new_schema_tworelation_count'] if counts[
                                                   'new_schema_tworelation_count'] != 0 else 'not available',
        'new schema & three relation': new_schema_three_relation_correct_execution_count / counts[
            'new_schema_threerelation_count'] if counts[
                                                     'new_schema_threerelation_count'] != 0 else 'not available',
        'new schema & four relation': new_schema_four_relation_correct_execution_count / counts[
            'new_schema_fourrelation_count'] if counts[
                                                    'new_schema_fourrelation_count'] != 0 else 'not available',
        'new schema & five relation': new_schema_five_relation_correct_execution_count / counts[
            'new_schema_fiverelation_count'] if counts[
                                                    'new_schema_fiverelation_count'] != 0 else 'not available',
        'new schema & six relation': new_schema_six_relation_correct_execution_count / counts[
            'new_schema_sixrelation_count'] if counts[
                                                   'new_schema_sixrelation_count'] != 0 else 'not available'
    }

    dataset.append(performance_dict)

    if print_detailed_performance:
        for key, value in performance_dict.items():
            print(key)
            print(value)
    return dataset


def test_data_no_in_context_examples(dataset='dev',  # you can also set this to test
                                     number_of_routes=1,

                                     print_detailed_performance=False
                                     ):
    """
    This method takes a dataset and parses SPARQL queries for the questions contained in it.
    The original dataset is a list of dictionaries, where each dictionary represents a specific question.
    To each of the dictionaries we add two new key-value pairs:
    1) 'generated_query' : the query generated by the system
    2) 'results_of_generated_and_golden_query_match' : A boolean indicating whether the results of the generated
    query and benchmark query match with eachother.

    Semantic parsing is executed while excluding in-context examples but including ontology information during
    semantic parsing. To the original dataset, a final dictionary is appended containing a detailed performance summary.



    Args:
        dataset (str): The dataset to test, either 'dev' (default) or 'test'.
        number_of_routes (int): The number of routes between relevant nodes in GTOC.
        print_detailed_performance (bool): If True, print detailed performance metrics for the evaluation.

    Returns:
        list: The method returns the original dataset, now updated to include parsed queries for each question.
        Additionally, a dictionary containing a detailed performance summary is appended as the final element of the
        dataset. The appended performance summary dictionary contains the following key-value pairs:
            - 'overall_execution_accuracy': Execution accuracy on the entire dataset.
            - 'entity linking accuracy': Accuracy of entity linking during reasoning.
            - 'iid': Execution accuracy on i.i.d. generalization questions.
            - 'compositional': Execution accuracy on compositional generalization questions.
            - 'zeroshot': Execution accuracy on zero-shot generalization questions.
            - 'group by': Execution accuracy on zero-shot questions with the GROUP BY operator.
            - 'having': Execution accuracy on zero-shot questions with the HAVING operator.
            - 'average': Execution accuracy on zero-shot questions with the AVERAGE operator.
            - 'subquery': Execution accuracy on zero-shot questions with subqueries.
            - 'multiple variables': Execution accuracy on zero-shot questions requiring multiple variable retrieval.
            - 'string': Execution accuracy on zero-shot questions involving string functions.
            - 'new schema item': Execution accuracy on zero-shot questions with new schema items.
            - 'one relation': Execution accuracy on zero-shot questions requiring one relation.
            - 'two relation': Execution accuracy on zero-shot questions requiring two relations.
            - 'three relation': Execution accuracy on zero-shot questions requiring three relations.
            - 'four relation': Execution accuracy on zero-shot questions requiring four relations.
            - 'five relation': Execution accuracy on zero-shot questions requiring five relations.
            - 'six relation': Execution accuracy on zero-shot questions requiring six relations.
            - 'new schema & one relation': Execution accuracy on questions with new schema items requiring one relation.
            - 'new schema & two relation': Execution accuracy on questions with new schema items requiring two relations.
            - 'new schema & three relation': Execution accuracy on questions with new schema items requiring three relations.
            - 'new schema & four relation': Execution accuracy on questions with new schema items requiring four relations.
            - 'new schema & five relation': Execution accuracy on questions with new schema items requiring five relations.
            - 'new schema & six relation': Execution accuracy on questions with new schema items requiring six relations.

    """
    if dataset == 'dev':

        # Load development_set from JSON
        file_path = "development_set.json"
        with open(file_path, "r") as json_file:
            dataset = json.load(json_file)

    elif dataset == 'test':

        # Load test_set from JSON
        file_path = "test_set.json"
        with open(file_path, "r") as json_file:
            dataset = json.load(json_file)

    correct_execution_count = 0
    correct_entity_linking_count = 0
    iid_correct_execution_count = 0
    compositional_correct_execution_count = 0
    zeroshot_correct_execution_count = 0
    group_by_correct_execution_count = 0
    having_correct_execution_count = 0
    average_correct_execution_count = 0
    subquery_correct_execution_count = 0
    multiple_variables_correct_execution_count = 0
    string_correct_execution_count = 0
    new_schema_correct_execution_count = 0
    one_relation_correct_execution_count = 0
    two_relation_correct_execution_count = 0
    three_relation_correct_execution_count = 0
    four_relation_correct_execution_count = 0
    five_relation_correct_execution_count = 0
    six_relation_correct_execution_count = 0

    new_schema_one_relation_correct_execution_count = 0
    new_schema_two_relation_correct_execution_count = 0
    new_schema_three_relation_correct_execution_count = 0
    new_schema_four_relation_correct_execution_count = 0
    new_schema_five_relation_correct_execution_count = 0
    new_schema_six_relation_correct_execution_count = 0

    question_processed_count = 0
    for question_dict in dataset:
        print('Amount of questions that need to be processed:')
        print(len(dataset))
        question_processed_count = question_processed_count + 1
        print("Processing question %s" % (question_processed_count))

        # First we check whether the execution accuracy
        answer_dict = None
        try:
            answer_dict = semantic_parsing_without_in_context_examples(question_dict['question'],
                                                                       k_shortest_routes=number_of_routes)
        except openai.RateLimitError as err:
            # Handle rate limit error
            print("Rate limit exceeded. Retrying after 1 minute...")
            time.sleep(60)  # Wait for 1 minute
            # Retry the code block
            try:
                answer_dict = semantic_parsing_without_in_context_examples(question_dict['question'],
                                                                           k_shortest_routes=number_of_routes)
            except openai.RateLimitError as err:
                # If rate limit error occurs again, handle it or raise it
                print("Rate limit exceeded even after retrying. Please check your usage.")
                raise err

        golden_query = question_dict['sparql']
        generated_query = answer_dict['query']
        question_dict['generated_query'] = generated_query

        execution_accuracy_boolean = check_whether_sparql_results_are_the_same(golden_query, generated_query)
        question_dict['results_of_generated_and_golden_query_match'] = execution_accuracy_boolean

        if execution_accuracy_boolean == True:
            correct_execution_count = correct_execution_count + 1

            if question_dict['generalization_level'] == 'iid':
                iid_correct_execution_count = iid_correct_execution_count + 1

            if question_dict['generalization_level'] == 'compositional':
                compositional_correct_execution_count = compositional_correct_execution_count + 1

            if question_dict['generalization_level'] == 'zeroshot':
                zeroshot_correct_execution_count = zeroshot_correct_execution_count + 1

            if 'GROUP BY' in question_dict['novel_functions']:
                group_by_correct_execution_count = group_by_correct_execution_count + 1

            if 'HAVING' in question_dict['novel_functions']:
                having_correct_execution_count = having_correct_execution_count + 1

            if 'AVERAGE' in question_dict['novel_functions']:
                average_correct_execution_count = average_correct_execution_count + 1

            if 'SUBQUERY' in question_dict['novel_functions']:
                subquery_correct_execution_count = subquery_correct_execution_count + 1

            if 'MULTIPLE VARIABLES' in question_dict['novel_functions']:
                multiple_variables_correct_execution_count = multiple_variables_correct_execution_count + 1

            if 'STRING' in question_dict['novel_functions']:
                string_correct_execution_count = string_correct_execution_count + 1

            if 'new_schema_item' in question_dict.keys():
                if question_dict['new_schema_item'] == True:
                    new_schema_correct_execution_count = new_schema_correct_execution_count + 1

            if 'number_of_relations' in question_dict.keys():
                if question_dict['number_of_relations'] == 1:
                    one_relation_correct_execution_count = one_relation_correct_execution_count + 1
                    if 'new_schema_item' in question_dict.keys():
                        if question_dict['new_schema_item'] == True:
                            new_schema_one_relation_correct_execution_count = new_schema_one_relation_correct_execution_count + 1

            if 'number_of_relations' in question_dict.keys():
                if question_dict['number_of_relations'] == 2:
                    two_relation_correct_execution_count = two_relation_correct_execution_count + 1
                    if 'new_schema_item' in question_dict.keys():
                        if question_dict['new_schema_item'] == True:
                            new_schema_two_relation_correct_execution_count = new_schema_two_relation_correct_execution_count + 1

            if 'number_of_relations' in question_dict.keys():
                if question_dict['number_of_relations'] == 3:
                    three_relation_correct_execution_count = three_relation_correct_execution_count + 1
                    if 'new_schema_item' in question_dict.keys():
                        if question_dict['new_schema_item'] == True:
                            new_schema_three_relation_correct_execution_count = new_schema_three_relation_correct_execution_count + 1

            if 'number_of_relations' in question_dict.keys():
                if question_dict['number_of_relations'] == 4:
                    four_relation_correct_execution_count = four_relation_correct_execution_count + 1
                    if 'new_schema_item' in question_dict.keys():
                        if question_dict['new_schema_item'] == True:
                            new_schema_four_relation_correct_execution_count = new_schema_four_relation_correct_execution_count + 1

            if 'number_of_relations' in question_dict.keys():
                if question_dict['number_of_relations'] == 5:
                    five_relation_correct_execution_count = five_relation_correct_execution_count + 1
                    if 'new_schema_item' in question_dict.keys():
                        if question_dict['new_schema_item'] == True:
                            new_schema_five_relation_correct_execution_count = new_schema_five_relation_correct_execution_count + 1

            if 'number_of_relations' in question_dict.keys():
                if question_dict['number_of_relations'] == 6:
                    six_relation_correct_execution_count = six_relation_correct_execution_count + 1
                    if 'new_schema_item' in question_dict.keys():
                        if question_dict['new_schema_item'] == True:
                            new_schema_six_relation_correct_execution_count = new_schema_six_relation_correct_execution_count + 1

        generated_granularity = answer_dict['granularity']
        generated_location = answer_dict['location']
        golden_granularity = question_dict['granularity']
        golden_location = question_dict['location']

        if generated_granularity == golden_granularity:
            if generated_location == golden_location:
                correct_entity_linking_count = correct_entity_linking_count + 1

            else:
                'entity linking went wrong'
                print(question_dict)
                print(generated_granularity)
                print(generated_location)

    print('Execution accuracy:')
    print(correct_execution_count / len(dataset))
    print('Entity linking accuracy:')
    print(correct_entity_linking_count / len(dataset))

    counts = get_statistics_of_dataset(dataset)
    performance_dict = {
        'overall_execution_accuracy': correct_execution_count / len(dataset),
        'entity linking accuracy': correct_entity_linking_count / len(dataset),
        'iid': iid_correct_execution_count / counts['iid_count'] if counts['iid_count'] != 0 else 'not available',
        'compositional': compositional_correct_execution_count / counts['compositional_count'] if counts[
                                                                                                      'compositional_count'] != 0 else 'not available',
        'zeroshot': zeroshot_correct_execution_count / counts['zeroshot_count'] if counts[
                                                                                       'zeroshot_count'] != 0 else 'not available',
        'group by': group_by_correct_execution_count / counts['groupby_count'] if counts[
                                                                                      'groupby_count'] != 0 else 'not available',
        'having': having_correct_execution_count / counts['having_count'] if counts[
                                                                                 'having_count'] != 0 else 'not available',
        'average': average_correct_execution_count / counts['average_count'] if counts[
                                                                                    'average_count'] != 0 else 'not available',
        'subquery': subquery_correct_execution_count / counts['subquery_count'] if counts[
                                                                                       'subquery_count'] != 0 else 'not available',
        'multiple variables': multiple_variables_correct_execution_count / counts['multiplevariables_count'] if
        counts[
            'multiplevariables_count'] != 0 else 'not available',
        'string': string_correct_execution_count / counts['string_count'] if counts[
                                                                                 'string_count'] != 0 else 'not available',
        'new schema item': new_schema_correct_execution_count / counts['newschema_count'] if counts[
                                                                                                 'newschema_count'] != 0 else 'not available',
        'one relation': one_relation_correct_execution_count / counts['onerelation_count'] if counts[
                                                                                                  'onerelation_count'] != 0 else 'not available',
        'two relation': two_relation_correct_execution_count / counts['tworelation_count'] if counts[
                                                                                                  'tworelation_count'] != 0 else 'not available',
        'three relation': three_relation_correct_execution_count / counts['threerelation_count'] if counts[
                                                                                                        'threerelation_count'] != 0 else 'not available',
        'four relation': four_relation_correct_execution_count / counts['fourrelation_count'] if counts[
                                                                                                     'fourrelation_count'] != 0 else 'not available',
        'five relation': five_relation_correct_execution_count / counts['fiverelation_count'] if counts[
                                                                                                     'fiverelation_count'] != 0 else 'not available',
        'six relation': six_relation_correct_execution_count / counts['sixrelation_count'] if counts[
                                                                                                  'sixrelation_count'] != 0 else 'not available',
        'new schema & one relation': new_schema_one_relation_correct_execution_count / counts[
            'new_schema_onerelation_count'] if counts[
                                                   'new_schema_onerelation_count'] != 0 else 'not available',
        'new schema & two relation': new_schema_two_relation_correct_execution_count / counts[
            'new_schema_tworelation_count'] if counts[
                                                   'new_schema_tworelation_count'] != 0 else 'not available',
        'new schema & three relation': new_schema_three_relation_correct_execution_count / counts[
            'new_schema_threerelation_count'] if counts[
                                                     'new_schema_threerelation_count'] != 0 else 'not available',
        'new schema & four relation': new_schema_four_relation_correct_execution_count / counts[
            'new_schema_fourrelation_count'] if counts[
                                                    'new_schema_fourrelation_count'] != 0 else 'not available',
        'new schema & five relation': new_schema_five_relation_correct_execution_count / counts[
            'new_schema_fiverelation_count'] if counts[
                                                    'new_schema_fiverelation_count'] != 0 else 'not available',
        'new schema & six relation': new_schema_six_relation_correct_execution_count / counts[
            'new_schema_sixrelation_count'] if counts[
                                                   'new_schema_sixrelation_count'] != 0 else 'not available'
    }
    dataset.append(performance_dict)

    if print_detailed_performance:
        for key, value in performance_dict.items():
            print(key)
            print(value)
    return dataset


def perform_hyperparameter_tuning_with_GRID_SEARCH():
    """
    This method conducts hyperparameter tuning using grid search on the development dataset.

    During the grid search, two parameters are varied:
    1. The number of in-context examples, ranging from 1 to 10.
    2. The number of routes between relevant nodes during ontology condensation using GTOC, ranging from 1 to 3.

    For each combination of hyperparameters, a JSON file is generated and saved in the directory
    "system_evaluation/hyperparameter_tuning_dev_set/".

    Each JSON file contains the original dataset, updated to include parsed queries for each question and a boolean
    indicating whether the parsed query and benchmark query retrieve the same results,
    with a detailed performance summary dictionary appended at the end.
    This performance summary dictionary includes the following key-value pairs:

        - 'overall_execution_accuracy': Execution accuracy on the entire dataset.
        - 'entity linking accuracy': Accuracy of entity linking during reasoning.
        - 'iid': Execution accuracy on i.i.d. generalization questions.
        - 'compositional': Execution accuracy on compositional generalization questions.
        - 'zeroshot': Execution accuracy on zero-shot generalization questions.
        - 'group by': Execution accuracy on zero-shot questions with the GROUP BY operator.
        - 'having': Execution accuracy on zero-shot questions with the HAVING operator.
        - 'average': Execution accuracy on zero-shot questions with the AVERAGE operator.
        - 'subquery': Execution accuracy on zero-shot questions with subqueries.
        - 'multiple variables': Execution accuracy on zero-shot questions requiring multiple variable retrieval.
        - 'string': Execution accuracy on zero-shot questions involving string functions.
        - 'new schema item': Execution accuracy on zero-shot questions with new schema items.
        - 'one relation': Execution accuracy on zero-shot questions requiring one relation.
        - 'two relation': Execution accuracy on zero-shot questions requiring two relations.
        - 'three relation': Execution accuracy on zero-shot questions requiring three relations.
        - 'four relation': Execution accuracy on zero-shot questions requiring four relations.
        - 'five relation': Execution accuracy on zero-shot questions requiring five relations.
        - 'six relation': Execution accuracy on zero-shot questions requiring six relations.
        - 'new schema & one relation': Execution accuracy on questions with new schema items requiring one relation.
        - 'new schema & two relation': Execution accuracy on questions with new schema items requiring two relations.
        - 'new schema & three relation': Execution accuracy on questions with new schema items requiring three relations.
        - 'new schema & four relation': Execution accuracy on questions with new schema items requiring four relations.
        - 'new schema & five relation': Execution accuracy on questions with new schema items requiring five relations.
        - 'new schema & six relation': Execution accuracy on questions with new schema items requiring six relations.



    """

    optimal_number_of_examples = 1
    optimal_number_of_routes = 1
    optimal_execution_accuracy = 0
    for number_of_examples in range(1, 11):
        for number_of_routes in range(1, 4):
            results = test_data_set_with_ontology(dataset='dev',
                                                  number_of_in_context_examples=number_of_examples,
                                                  number_of_routes=number_of_routes,
                                                  apply_meta_data_filtering=True,
                                                  diversity_reranking_of_in_context_examples=True,
                                                  mask_location_in_question=True,
                                                  print_detailed_performance=False
                                                  )

            results_performance_dict = results[-1]

            if results_performance_dict['overall_execution_accuracy'] > optimal_execution_accuracy:
                optimal_number_of_routes = number_of_routes
                optimal_number_of_examples = number_of_examples

            path = r"./hyperparameter_tuning_dev_set/%s_example_%s_routes.json" % (number_of_examples, number_of_routes)

            # Write the data_list to a JSON file
            with open(path, "w") as json_file:
                json.dump(results, json_file, indent=4)

    optimal_hyper_parameters = {optimal_number_of_examples: optimal_number_of_examples,
                                optimal_number_of_routes: optimal_number_of_routes}

    path = r"./hyperparameter_tuning_dev_set/optimal_hyper_parameters.json"

    # Write the data_list to a JSON file
    with open(path, "w") as json_file:
        json.dump(optimal_hyper_parameters, json_file, indent=4)


def analyze_test_set_performance_and_perform_ablations(number_of_in_context_examples=2,
                                                       number_of_routes=1):
    """
    Once the optimal hyperparameters are determined, they can be passed to this function to perform evaluations
    and ablations on the test set for the case where ontology information is INCLUDED during semantic parsing.

    For the performance of the full model and ablations a JSON file is generated and saved in the directory
    "system_evaluation/performance_and_ablations_with_ontology_test_set/".

    Each JSON file contains the original dataset, updated to include parsed queries for each question and a boolean
    indicating whether the parsed query and benchmark query retrieve the same results,
    with a detailed performance summary dictionary appended at the end.
    This performance summary dictionary includes the following key-value pairs:

        - 'overall_execution_accuracy': Execution accuracy on the entire dataset.
        - 'entity linking accuracy': Accuracy of entity linking during reasoning.
        - 'iid': Execution accuracy on i.i.d. generalization questions.
        - 'compositional': Execution accuracy on compositional generalization questions.
        - 'zeroshot': Execution accuracy on zero-shot generalization questions.
        - 'group by': Execution accuracy on zero-shot questions with the GROUP BY operator.
        - 'having': Execution accuracy on zero-shot questions with the HAVING operator.
        - 'average': Execution accuracy on zero-shot questions with the AVERAGE operator.
        - 'subquery': Execution accuracy on zero-shot questions with subqueries.
        - 'multiple variables': Execution accuracy on zero-shot questions requiring multiple variable retrieval.
        - 'string': Execution accuracy on zero-shot questions involving string functions.
        - 'new schema item': Execution accuracy on zero-shot questions with new schema items.
        - 'one relation': Execution accuracy on zero-shot questions requiring one relation.
        - 'two relation': Execution accuracy on zero-shot questions requiring two relations.
        - 'three relation': Execution accuracy on zero-shot questions requiring three relations.
        - 'four relation': Execution accuracy on zero-shot questions requiring four relations.
        - 'five relation': Execution accuracy on zero-shot questions requiring five relations.
        - 'six relation': Execution accuracy on zero-shot questions requiring six relations.
        - 'new schema & one relation': Execution accuracy on questions with new schema items requiring one relation.
        - 'new schema & two relation': Execution accuracy on questions with new schema items requiring two relations.
        - 'new schema & three relation': Execution accuracy on questions with new schema items requiring three relations.
        - 'new schema & four relation': Execution accuracy on questions with new schema items requiring four relations.
        - 'new schema & five relation': Execution accuracy on questions with new schema items requiring five relations.
        - 'new schema & six relation': Execution accuracy on questions with new schema items requiring six relations.

    Args:
        number_of_in_context_examples (int, optional):
            The number of in-context examples for the model to use during semantic parsing.
            Defaults to 2.
        number_of_routes (int, optional):
            The number of routes between relevant nodes during GTOC ontology condensation. Defaults to 1.
    """

    test_set_performance = test_data_set_with_ontology(dataset='test',
                                                       number_of_in_context_examples=number_of_in_context_examples,
                                                       number_of_routes=number_of_routes,
                                                       apply_meta_data_filtering=True,
                                                       diversity_reranking_of_in_context_examples=True,
                                                       mask_location_in_question=True,
                                                       print_detailed_performance=False
                                                       )

    no_meta_data_filtering_ablation = test_data_set_with_ontology(dataset='test',
                                                                  number_of_in_context_examples=number_of_in_context_examples,
                                                                  number_of_routes=number_of_routes,
                                                                  apply_meta_data_filtering=False,
                                                                  diversity_reranking_of_in_context_examples=True,
                                                                  mask_location_in_question=True,
                                                                  print_detailed_performance=False
                                                                  )

    no_diversity_reranking_ablation = test_data_set_with_ontology(dataset='test',
                                                                  number_of_in_context_examples=number_of_in_context_examples,
                                                                  number_of_routes=number_of_routes,
                                                                  apply_meta_data_filtering=True,
                                                                  diversity_reranking_of_in_context_examples=False,
                                                                  mask_location_in_question=True,
                                                                  print_detailed_performance=False
                                                                  )

    no_entity_masking_ablation = test_data_set_with_ontology(dataset='test',
                                                             number_of_in_context_examples=number_of_in_context_examples,
                                                             number_of_routes=number_of_routes,
                                                             apply_meta_data_filtering=True,
                                                             diversity_reranking_of_in_context_examples=True,
                                                             mask_location_in_question=False,
                                                             print_detailed_performance=False
                                                             )

    naive_ontology_selection = test_data_set_with_ontology(dataset='test',
                                                           number_of_in_context_examples=2,
                                                           number_of_routes=1,
                                                           apply_meta_data_filtering=True,
                                                           diversity_reranking_of_in_context_examples=True,
                                                           mask_location_in_question=True,
                                                           naive_schema_linking_without_graph_traversal=True,
                                                           print_detailed_performance=False
                                                           )

    full_ontology_without_condensation = test_data_set_with_ontology(dataset='test',
                                                                     number_of_in_context_examples=2,
                                                                     number_of_routes=1,
                                                                     apply_meta_data_filtering=True,
                                                                     diversity_reranking_of_in_context_examples=True,
                                                                     mask_location_in_question=True,
                                                                     print_detailed_performance=False,
                                                                     ontology_condensation=False
                                                                     )

    no_in_context_examples = test_data_no_in_context_examples(dataset='test',  # you can also set this to test
                                                              number_of_routes=1,

                                                              print_detailed_performance=False
                                                              )

    path = r"./performance_and_ablations_with_ontology_test_set/test_set_performance.json"

    with open(path, "w") as json_file:
        json.dump(test_set_performance, json_file, indent=4)

    path = r"./performance_and_ablations_with_ontology_test_set/no_meta_data_filtering_ablation.json"

    with open(path, "w") as json_file:
        json.dump(no_meta_data_filtering_ablation, json_file, indent=4)

    path = r"./performance_and_ablations_with_ontology_test_set/no_diversity_reranking_ablation.json"

    with open(path, "w") as json_file:
        json.dump(no_diversity_reranking_ablation, json_file, indent=4)

    path = r"./performance_and_ablations_with_ontology_test_set/no_entity_masking_ablation.json"

    with open(path, "w") as json_file:
        json.dump(no_entity_masking_ablation, json_file, indent=4)

    path = r"./performance_and_ablations_with_ontology_test_set/naive_ontology_selection.json"

    with open(path, "w") as json_file:
        json.dump(naive_ontology_selection, json_file, indent=4)

    path = r"./performance_and_ablations_with_ontology_test_set/no_in_context_examples.json"

    with open(path, "w") as json_file:
        json.dump(no_in_context_examples, json_file, indent=4)

    path = r"./performance_and_ablations_with_ontology_test_set/full_ontology_without_condensation.json"

    with open(path, "w") as json_file:
        json.dump(full_ontology_without_condensation, json_file, indent=4)


def analyze_test_set_performance_and_perform_ablations_WITHOUT_ONTOLOGY(number_of_in_context_examples=2):
    """
    Once the optimal hyperparameters are determined, they can be passed to this function to perform evaluations
    and ablations on the test set for the case where ontology information is EXCLUDED during semantic parsing.

    For the performance of the full model and ablations a JSON file is generated and saved in the directory
    "system_evaluation/performance_and_ablations_without_ontology_test_set/".

    Each JSON file contains the original dataset, updated to include parsed queries for each question and a boolean
    indicating whether the parsed query and benchmark query retrieve the same results,
    with a detailed performance summary dictionary appended at the end.
    This performance summary dictionary includes the following key-value pairs:

        - 'overall_execution_accuracy': Execution accuracy on the entire dataset.
        - 'entity linking accuracy': Accuracy of entity linking during reasoning.
        - 'iid': Execution accuracy on i.i.d. generalization questions.
        - 'compositional': Execution accuracy on compositional generalization questions.
        - 'zeroshot': Execution accuracy on zero-shot generalization questions.
        - 'group by': Execution accuracy on zero-shot questions with the GROUP BY operator.
        - 'having': Execution accuracy on zero-shot questions with the HAVING operator.
        - 'average': Execution accuracy on zero-shot questions with the AVERAGE operator.
        - 'subquery': Execution accuracy on zero-shot questions with subqueries.
        - 'multiple variables': Execution accuracy on zero-shot questions requiring multiple variable retrieval.
        - 'string': Execution accuracy on zero-shot questions involving string functions.
        - 'new schema item': Execution accuracy on zero-shot questions with new schema items.
        - 'one relation': Execution accuracy on zero-shot questions requiring one relation.
        - 'two relation': Execution accuracy on zero-shot questions requiring two relations.
        - 'three relation': Execution accuracy on zero-shot questions requiring three relations.
        - 'four relation': Execution accuracy on zero-shot questions requiring four relations.
        - 'five relation': Execution accuracy on zero-shot questions requiring five relations.
        - 'six relation': Execution accuracy on zero-shot questions requiring six relations.
        - 'new schema & one relation': Execution accuracy on questions with new schema items requiring one relation.
        - 'new schema & two relation': Execution accuracy on questions with new schema items requiring two relations.
        - 'new schema & three relation': Execution accuracy on questions with new schema items requiring three relations.
        - 'new schema & four relation': Execution accuracy on questions with new schema items requiring four relations.
        - 'new schema & five relation': Execution accuracy on questions with new schema items requiring five relations.
        - 'new schema & six relation': Execution accuracy on questions with new schema items requiring six relations.

    Args:
        number_of_in_context_examples (int, optional):
            The number of in-context examples for the model to use during semantic parsing.
            Defaults to 2.
        number_of_routes (int, optional):
            The number of routes between relevant nodes during GTOC ontology condensation. Defaults to 1.
    """
    test_set_performance = test_data_set_no_ontology(dataset='test',  # you can also set this to test
                                                     number_of_in_context_examples=number_of_in_context_examples,
                                                     apply_meta_data_filtering=True,
                                                     diversity_reranking_of_in_context_examples=True,
                                                     mask_location_in_question=True,
                                                     print_detailed_performance=False
                                                     )
    no_meta_data_filtering_ablation = test_data_set_no_ontology(dataset='test',  # you can also set this to test
                                                                number_of_in_context_examples=number_of_in_context_examples,
                                                                apply_meta_data_filtering=False,
                                                                diversity_reranking_of_in_context_examples=True,
                                                                mask_location_in_question=True,
                                                                print_detailed_performance=False
                                                                )

    no_diversity_reranking_ablation = test_data_set_no_ontology(dataset='test',  # you can also set this to test
                                                                number_of_in_context_examples=number_of_in_context_examples,
                                                                apply_meta_data_filtering=True,
                                                                diversity_reranking_of_in_context_examples=False,
                                                                mask_location_in_question=True,
                                                                print_detailed_performance=False
                                                                )

    no_entity_masking_ablation = test_data_set_no_ontology(dataset='test',  # you can also set this to test
                                                           number_of_in_context_examples=number_of_in_context_examples,
                                                           apply_meta_data_filtering=True,
                                                           diversity_reranking_of_in_context_examples=True,
                                                           mask_location_in_question=False,
                                                           print_detailed_performance=False
                                                           )

    path = r"./performance_and_ablations_without_ontology_test_set/test_set_performance.json"

    with open(path, "w") as json_file:
        json.dump(test_set_performance, json_file, indent=4)

    path = r"./performance_and_ablations_without_ontology_test_set/no_meta_data_filtering_ablation.json"

    with open(path, "w") as json_file:
        json.dump(no_meta_data_filtering_ablation, json_file, indent=4)

    path = r"./performance_and_ablations_without_ontology_test_set/no_diversity_reranking_ablation.json"

    with open(path, "w") as json_file:
        json.dump(no_diversity_reranking_ablation, json_file, indent=4)

    path = r"./performance_and_ablations_without_ontology_test_set/no_entity_masking_ablation.json"

    with open(path, "w") as json_file:
        json.dump(no_entity_masking_ablation, json_file, indent=4)


def test_ontology_condensation(number_of_routes=1):
    """
    Save condensed ontologies for analysis and inspection.

    This function loads a dataset, and adds to each question dictionary the following key-value pairs:

        - `ontology_string_object_part`: All object vertex-edge pairs in the ontology
          condensed through GTOC.
        - `ontology_string_datatype_part`: All datatype vertex-edge pairs in the ontology
          condensed through GTOC.
        - `ontology_string_object_part_naive`: All object-type vertex-edge pairs in the
          ontology condensed through the naive approach.
        - `ontology_string_datatype_part_naive`: All datatype vertex-edge pairs in the
          ontology condensed through the naive approach.
        - `size_condensed_ontology`: The total number of vertex-edge pairs in the ontology
          condensed through GTOC.


    The dataset where we included the above keys-value pairs in each question dictionary, is saved in
    "system_evaluation/ontology_condensation_results/no_meta_data_filtering_ablation_ontology_condensation_results.json"

    Args:
    number_of_routes (int, optional):
        The number of routes between relevant nodes to consider during ontology condensation through GTOC.
        Defaults to 1.

    """
    file_path = r"./performance_and_ablations_with_ontology_test_set/no_meta_data_filtering_ablation.json"
    with open(file_path, "r") as json_file:
        dataset = json.load(json_file)

    question_processed_count = 0
    for question_dict in dataset:
        if 'question' in question_dict.keys():
            print('Amount of questions that need to be processed:')
            print(len(dataset))
            question_processed_count = question_processed_count + 1
            print("Processing question %s" % (question_processed_count))


            try:
                ontology_string_object_part, ontology_string_datatype_part, ontology_string_object_part_naive, ontology_string_datatype_part_naive, size_condensed_ontology = only_get_condensed_ontologies(
                    question_dict['question'],
                    k_shortest_routes=number_of_routes)
            except openai.RateLimitError as err:
                # Handle rate limit error
                print("Rate limit exceeded. Retrying after 1 minute...")
                time.sleep(60)  # Wait for 1 minute
                # Retry the code block
                try:
                    ontology_string_object_part, ontology_string_datatype_part, ontology_string_object_part_naive, ontology_string_datatype_part_naive, size_condensed_ontology = only_get_condensed_ontologies(
                        question_dict['question'],
                        k_shortest_routes=number_of_routes)
                except openai.RateLimitError as err:
                    # If rate limit error occurs again, handle it or raise it
                    print("Rate limit exceeded even after retrying. Please check your usage.")
                    raise err

        question_dict['ontology_string_object_part'] = ontology_string_object_part
        question_dict['ontology_string_datatype_part'] = ontology_string_datatype_part
        question_dict['ontology_string_object_part_naive'] = ontology_string_object_part_naive
        question_dict['ontology_string_datatype_part_naive'] = ontology_string_datatype_part_naive
        question_dict['size_condensed_ontology'] = size_condensed_ontology
    file_path = r"./ontology_condensation_results/no_meta_data_filtering_ablation_ontology_condensation_results.json"
    with open(file_path, "w") as json_file:
        json.dump(dataset, json_file, indent=4)


"""
To perform hyperparameter tuning on the development set and save the results in JSON format, 
uncomment the following line:
"""
# perform_hyperparameter_tuning_with_GRID_SEARCH()

"""
To evaluate the test set performance (including all ablations) and save the results in JSON format, 
uncomment the relevant sections below. 

**Note:** Before running these functions, ensure the optimal hyperparameter configuration 
is manually included in the function calls. This can be done after performing hyperparameter 
tuning on the development set and extracting the optimal parameters from the generated JSON results.
"""

# analyze_test_set_performance_and_perform_ablations(number_of_in_context_examples=2,
#                                                    number_of_routes=1)
# analyze_test_set_performance_and_perform_ablations_WITHOUT_ONTOLOGY(number_of_in_context_examples=2)

"""
To save ontology condensation results for further analysis, uncomment the following line:
"""
# test_ontology_condensation(number_of_routes=1)
