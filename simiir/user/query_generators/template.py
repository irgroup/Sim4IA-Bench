from simiir.user.query_generators.base import BaseQueryGenerator


class YourQueryGenerator(BaseQueryGenerator):
    """
    Briefly describe the core idea of your query generator!
    """

    def __init__(self, stopword_file, query_file, user, background_file=[]):
        super().__init__(stopword_file, background_file=[])
        self.__query_filename = query_file
        self.__user = user

    def generate_query_list(self, user_context) -> list[tuple[str, int]]:
        # This method is the heart of your query generator.
        # Add your main query generator logic here!

        topic = user_context.topic

        # Make sure to return a list. Every list entry should be a tuple, consisting of at least 2 elements, where the first one is your query as a string.
        # You can utilize the second value in the nested tuple to order the queries. This value will not be relevant once the list is returned.
        return [("your query 1", 1), ("your query 2", 2)]

    def _help_function(self):
        # Make sure to structure your code using internal help functions.
        pass
