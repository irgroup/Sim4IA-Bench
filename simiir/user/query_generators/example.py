from simiir.user.query_generators.base import BaseQueryGenerator
from ifind.common.utils import get_given_queries

import random


class QuestionGenerator_A1(BaseQueryGenerator):
    """
    Takes the first given query, and turns it into a question.
    """

    def __init__(self, stopword_file, query_file, user, background_file=[]):
        super().__init__(stopword_file, background_file=[])
        self.__query_filename = query_file
        self.__user = user

    def generate_query_list(self, user_context) -> list[tuple[str, int]]:
        return_queries: list[tuple[str, int]] = get_given_queries(
            self.__query_filename, self.__user, user_context.topic.id, task_a2=False
        )

        new_query: tuple[str, int] = (self._create_question(return_queries[-1][0]), 0)

        return_queries.append(new_query)

        return return_queries

    def _create_question(self, query: str) -> str:
        prefixes: list[str] = [
            "What is",
            "How to",
            "How would I",
            "Why is",
            "Where is",
        ]
        chosen_prefix: str = random.choice(prefixes)
        question_query: str = f"{chosen_prefix} {query.strip()}?"
        return question_query


class QuestionGenerator_A2(BaseQueryGenerator):
    """
    Takes the first given query, and turns it into a question.
    """

    def __init__(self, stopword_file, query_file, user, background_file=[]):
        super().__init__(stopword_file, background_file=[])
        self.__query_filename = query_file
        self.__user = user

    def generate_query_list(self, user_context) -> list[tuple[str, int]]:
        return_queries: list[tuple[str, int]] = get_given_queries(
            self.__query_filename, self.__user, user_context.topic.id, task_a2=True
        )

        new_query: tuple[str, int] = (self._create_question(return_queries[-1][0]), 0)

        return_queries.append(new_query)

        return return_queries

    def _create_question(self, query: str) -> str:
        prefixes: list[str] = [
            "What is",
            "How to",
            "How would I",
            "Why is",
            "Where is",
        ]
        chosen_prefix: str = random.choice(prefixes)
        question_query: str = f"{chosen_prefix} {query.strip()}?"
        return question_query
