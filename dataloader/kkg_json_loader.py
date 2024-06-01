# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import re



class KadasterJsonLoader():

    def __init__(self, file_path: str):
        with open(file_path, 'r', encoding='UTF-8') as file:
            f = json.load(file)
            self.data = f
            self.len = len(self.data)
            self.file_path = file_path

        self.question_id_to_idx_dict = dict()  # question_id: str -> idx: int
        self.build_question_id_to_idx_dict()

    # We simply build a dictionary that maps a qid to an idx
    def build_question_id_to_idx_dict(self):
        for idx in range(0, self.len):
            # quite straightforward function that simply gets qid from index
            question_id = self.get_question_id_by_idx(idx)
            # We simply populate the dictionary here
            self.question_id_to_idx_dict[str(question_id)] = idx

    def get_question_id_by_idx(self, idx, format='int'):
        qid = self.data[idx]['qid']
        if format == 'str':
            return str(qid)
        return qid


    def get_idx_by_question_id(self, question_id):
        """
        Get the index in the json file by the question id
        :param question_id: the GrailQA question id
        :return: if the question is in the file, return the index, otherwise -1
        """
        question_id = str(question_id)
        if self.question_id_to_idx_dict is None or len(self.question_id_to_idx_dict) == 0:
            self.build_question_id_to_idx_dict()
        return self.question_id_to_idx_dict.get(question_id, -1)


    def get_sparql_by_idx(self, idx):
        return self.data[idx]['sparql_query']

    def get_question_by_idx(self, idx):
        return self.data[idx]['question']

    def get_len(self):
        return self.len



    def get_golden_class_by_idx(self, idx):
        return self.data[idx]['golden_classes']


    def get_golden_relation_by_idx(self, idx):
        return self.data[idx]['golden_relations']

    def get_property_key_by_idx(self, idx):
        return self.data[idx]['property_key']

    def get_granularity_key_by_idx(self, idx):
        return self.data[idx]['granularity_key']

    def get_answer_kwargs_by_idx(self, idx):
        return self.data[idx]['info_dictionary']

    def get_dataset_split(self):
        if 'train' in self.file_path:
            return 'train'
        elif 'dev' in self.file_path or 'val' in self.file_path:
            return 'dev'
        elif 'test' in self.file_path:
            return 'test'
        return ''
