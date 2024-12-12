import random
from enum import Flag, auto
from create_training_data.query_components import ADDRESS_QUESTION


class Dependency(Flag):
    ADDRESS = auto()


def get_dependency_text(dependency):
    if dependency == Dependency.ADDRESS:
        return ADDRESS_QUESTION
    else:
        return None