from abc import ABC, abstractmethod
import math
from typing import List


class FactorModel(ABC):
    @abstractmethod
    def get_value(self, step: int):
        pass

    @abstractmethod
    def state_dict(self):
        pass

    @abstractmethod
    def load_state_dict(self, state: dict[any, any]):
        pass


class Constant(FactorModel):
    def __init__(self, value: float):
        self.value = value

    def get_value(self, step: int):
        return self.value

    def state_dict(self):
        return {}

    def load_state_dict(self, _):
        pass


class LinearInterpolate(FactorModel):
    def __init__(self, key_steps: List[int], key_values: List[int]):
        self.key_points: list[tuple[int, float]] = zip(key_steps, key_values)
        self.key_points.sort(key=lambda pair: pair[0])

    def get_value(self, step: int):
        i = 0
        while i < len(self.key_points) and self.key_points[i][0] < step:
            i += 1

        if i == 0:
            return self.key_points[0][1]

        if i == len(self.key_points):
            return self.key_points[-1][1]

        weight = (step - self.key_points[i - 1][1]) / (
            self.key_points[i][1] + self.key_points[i - 1][1]
        )
        return (
            self.key_points[i - 1][1] * (1.0 - weight) + self.key_points[i][1] * weight
        )

    def state_dict(self):
        return {}

    def load_state_dict(self, _):
        pass


class Exponential(FactorModel):
    def __init__(self, start_value: float, limit_value: float, half_life: float):
        self.factor = start_value - limit_value
        self.offset = limit_value
        self.alpha = math.log(2) / half_life

    def get_value(self, step: int):
        return self.offset + self.factor * math.exp(-self.alpha * step)

    def state_dict(self):
        return {}

    def load_state_dict(self, _):
        pass
