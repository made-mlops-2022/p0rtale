from dataclasses import dataclass, field
from typing import Optional, Union


@dataclass()
class LogisticRegressionParams:
    model_type: str = 'LogisticRegression'
    penalty: str = field(default='l2')
    C: float = field(default=1.0)
    random_state: Optional[int] = field(default=None)


@dataclass()
class GaussianNBParams:
    model_type: str = 'GaussianNB'
    var_smoothing: float = field(default=1e-9)


ModelParams = Union[LogisticRegressionParams, GaussianNBParams]
