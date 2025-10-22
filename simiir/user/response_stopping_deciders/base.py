import abc
import random
from simiir.user.loggers import Actions

class BaseResponseDecisionMaker(object):
    """
    
    """
    def __init__(self, user_context, logger):
        self._user_context = user_context
        self._logger = logger
        #self._utterance_count = 0
    
    
    def decide(self):
        #if not hasattr(self, '_utterance_count'):
        #    self._utterance_count = 0

        if len(self._user_context._issued_utterances) <= self._user_context.initial_predetermined_uttterance_amount + 10: 
            #print("decide utterance count", self._utterance_count)
            #self._utterance_count += 1
            return Actions.UTTERANCE
        else:
            #print("decide stop", self._utterance_count)
            return Actions.STOP
