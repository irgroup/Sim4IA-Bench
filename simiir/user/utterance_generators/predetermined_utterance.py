import abc
import logging
import random
from itertools import combinations
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from simiir.user.utils.langchain_wrapper import LangChainWrapper
import random
import string

from simiir.user.utterance_generators.base import BaseUtteranceGenerator
import json

log = logging.getLogger('utterance_generators.base_generator')

class PredeterminedAndLLMGeneratedUtteranceGenerator(BaseUtteranceGenerator):
    """
    The base Utterance generator class.
    You can use this to inherit from to make your own query generator
    """
    def __init__(self, stopword_file,prompt_file,n=10,provider='ollama',model='mistral',temperature=1.0,verbose=False, background_file=[], path_to_predefined_utterances=None):
        super(BaseUtteranceGenerator, self).__init__()
       
        with open(path_to_predefined_utterances, 'r') as f:
            self._utterance_dict = json.load(f)
        
        prompt_template = ""
        with open(prompt_file,'r') as prompt:
            prompt_template = prompt.read()
        self._template = prompt_template + """\n\n{format_instructions}\n"""
        log.debug(self._template)
        
        self._result_schema = [ResponseSchema(
            name="queries",
            description=f"Generate an array of 10 possible utterance for the given previous utterances ordered by likelihood of success regarding matching the newly developed search intent of a user.",
            type="list"
            )]
        
        self._output_parser = StructuredOutputParser.from_response_schemas(self._result_schema)

        format_instructions = self._output_parser.get_format_instructions()
        log.debug(f'init(): {format_instructions}')
        self._prompt = PromptTemplate(
            template=self._template,
            input_variables=["prev_utterance", "prev_responses"],
            partial_variables={"format_instructions": format_instructions})
                
        self._llm = LangChainWrapper(self._prompt, provider, model, temperature, verbose)

        
    def update_model(self, user_context):
        """
        Enables the model of query/topic to be updated, based on the search context
        The update model based on the documents etc in the search context (i.e. memory of the user)

        :param  user_context: user_contexts.user_context object
        :return: returns True is topic model is updated.
        """
        return True
        
    
    def get_next_utterance(self, user_context):
        
        topic = user_context.topic
            
        issued_utterance_list = user_context.get_issued_utterances()
        
        utterances = self._utterance_dict[str(topic.id)][:-1]
        self.initial_predetermined_uttterance_amount = len(utterances)

        candidate_utterance = None
        
        for utterance in utterances:
            if utterance not in issued_utterance_list:
                candidate_utterance = utterance
                user_context.add_issued_utterance(candidate_utterance)
                return candidate_utterance
            else:
                continue
        
        previous_utterances_str = "\n".join(issued_utterance_list)
        
        print("=== Previous Utterances ===")
        print(issued_utterance_list)

        previous_responses_str = "\n\n".join([response.content for response in user_context.get_all_examined_responses()])
        print("=== Previous Responses ===")
        print([response.content for response in user_context.get_all_examined_responses()])

        rendered_prompt = self._prompt.format(
            prev_utterance=previous_utterances_str,
            prev_responses=previous_responses_str
        )
        print("=== Prompt ===")
        print(rendered_prompt)

        out = self._llm.generate_response(
            self._output_parser,
            {'prev_utterance': "\n" + previous_utterances_str, 'prev_responses': "\n" + previous_responses_str},
            self._result_schema
        )

        combined_utterances = out["queries"]
        self._utterance_dict[str(topic.id)].extend(combined_utterances)
        return combined_utterances[0]

   