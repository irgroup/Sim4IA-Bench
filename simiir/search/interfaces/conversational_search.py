from simiir.search.interfaces.base import ConversationalBaseInterface
import json
import os

class FileBasedConversationalInterface(ConversationalBaseInterface):
    """
    A conversational interface that retrieves responses from a pre-defined dictionary of responses.
    """

    def __init__(self, path_to_predefined_responses=None):
        """
        Initialize the interface with a dictionary of responses.
        :param response_dict: A dictionary where keys are topic IDs and values are lists of responses.
        """
        super(FileBasedConversationalInterface, self).__init__()
        
        with open(path_to_predefined_responses, "r") as f:
            self._response_dict = json.load(f)
       


    def issue_utterance(self, utterance):

        """
        Stores the issued utterance and prepares to retrieve a response.
        :param utterance: The user's natural language input.
        """
        self._last_query = utterance
       
        for response in self._response_dict:
            if self._last_query in self._response_dict:
                
                return self._response_dict[self._last_query]

      
        self._last_response = ""

        return self._last_response
