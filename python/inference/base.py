import numpy as np
from abc import ABC, abstractmethod

class InferenceEngineBase(ABC):
    @abstractmethod
    def preprocess(self, img: np.ndarray, target_size: int):
        pass
    
    #@abstractmethod
    #def postprocess(self):
    #    pass