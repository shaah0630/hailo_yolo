import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum

class ModelInputSize(IntEnum):
    YOLO26 = 640

@dataclass(slots=True)
class InferenceTimeStats:
  """Runtime statistics for different stages of inference"""
  # All in seconds
  preprocess: float = 0.0
  forward: float = 0.0
  postprocess: float = 0.0
  resize_to_orig_size: float = 0.0

  @property
  def total(self) -> float:
    return self.preprocess + self.forward + self.postprocess + self.resize_to_orig_size

class InferenceEngineBase(ABC):
    @abstractmethod
    def preprocess(self, img: np.ndarray, target_size: int, normalize: bool = False, verbose: bool = False, debug: bool = False) -> tuple[np.ndarray, float, int, int, float]:
        pass
    
    @abstractmethod
    def forward(self, input_data: np.ndarray, verbose: bool = False, debug: bool = False) -> tuple[list[dict], float]:
        pass

    @abstractmethod
    def postprocess(self, dequantized_results: list, conf_threshold: float = 0.5, verbose: bool = False, debug: bool = False) -> tuple[list[dict], float]:
        pass