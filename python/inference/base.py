import numpy as np
from abc import ABC, abstractmethod

class InferenceEngineBase(ABC):
    @abstractmethod
    def preprocess(self, img: np.ndarray, target_size: int):
        pass
    
    @abstractmethod
    def infer(self, input_data: np.ndarray, verbose: bool = False, save_output: bool = False, conf_threshold: float = 0.5) -> list[dict]:
        pass