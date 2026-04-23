from abc import ABC, abstractmethod
import math


class FactorFn(ABC):
    @abstractmethod
    def get_value(self, step: int):
        pass

    @abstractmethod
    def adjust_steps(self, factor: float):
        pass

    @abstractmethod
    def state_dict(self):
        pass

    @abstractmethod
    def load_state_dict(self, state: dict[any, any]):
        pass


class Constant(FactorFn):
    def __init__(self, value: float):
        self.value = value

    def get_value(self, step: int):
        return self.value

    def adjust_steps(self, factor: float):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, _):
        pass


class LinearInterpolate(FactorFn):
    def __init__(self, key_steps: list[int], key_values: list[int]):
        self.key_points: list[tuple[int, float]] = list(zip(key_steps, key_values))
        self.key_points.sort(key=lambda pair: pair[0])

    def get_value(self, step: int):
        i = 0
        while i < len(self.key_points) and self.key_points[i][0] < step:
            i += 1

        if i == 0:
            return self.key_points[0][1]

        if i == len(self.key_points):
            return self.key_points[-1][1]

        weight = (step - self.key_points[i - 1][0]) / (
            self.key_points[i][0] - self.key_points[i - 1][0]
        )

        return (
            self.key_points[i - 1][1] * (1.0 - weight) + self.key_points[i][1] * weight
        )

    def adjust_steps(self, factor: float):
        self.key_points = [(int(s * factor), v) for s, v in self.key_points]

    def state_dict(self):
        return {}

    def load_state_dict(self, _):
        pass


class Quadratic(FactorFn):
    def __init__(self, one_step: int):
        self.one_step = one_step

    def get_value(self, step: int):
        return (step / self.one_step) * (step / self.one_step)

    def adjust_steps(self, factor: float):
        self.one_step = int(self.one_step * factor)

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass


class SquareRoot(FactorFn):
    def __init__(self, one_step: int):
        self.one_step = one_step

    def get_value(self, step: int):
        return math.sqrt(step / self.one_step)

    def adjust_steps(self, factor: float):
        self.one_step = int(self.one_step * factor)

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass


class Exponential(FactorFn):
    @staticmethod
    def one_at_step_exponential(start_value: float, one_step: int):
        return Exponential(
            start_value=start_value,
            limit_value=0,
            half_life=one_step / math.log2(start_value),
        )

    def __init__(self, start_value: float, limit_value: float, half_life: float):
        self.factor = start_value - limit_value
        self.offset = limit_value
        self.alpha = math.log(2) / half_life

    def get_value(self, step: int):
        return self.offset + self.factor * math.exp(-self.alpha * step)

    def adjust_steps(self, factor: float):
        self.alpha /= factor

    def state_dict(self):
        return {}

    def load_state_dict(self, _):
        pass
