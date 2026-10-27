from abc import ABC, abstractmethod
import numpy as np
import casadi as ca


class DynamicalSystem(ABC):
    
    @abstractmethod
    def state_dot(
        self,
        current_state: np.ndarray,
        current_input: np.ndarray,
        current_disturbance: np.ndarray
    ) -> np.ndarray:
        """Time derivative of the state vector"""
        pass

    @abstractmethod
    def state_next(
        self,
        current_state: np.ndarray,
        current_input: np.ndarray,
        current_disturbance: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """Integrate to get next state"""
        pass

    @abstractmethod
    def linearize(
        self,
        nominal_state: np.ndarray,
        nominal_input: np.ndarray,
        nominal_disturbance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Linearize the system around a nominal trajectory"""
        pass

    @abstractmethod
    def get_output(
        self,
        current_state: np.ndarray,
        current_noise: np.ndarray
    ) -> np.ndarray:
        """Get the output measurement vector"""
        pass
