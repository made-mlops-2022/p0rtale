import numpy as np
import pandas as pd

from faker.providers import BaseProvider
from faker.providers import DynamicProvider
from faker import Faker


def generate_data():
    fake = Faker()
    Faker.seed(42)
    np.random.seed(42)

    class OldpeakProvider(BaseProvider):
        def oldpeak(self) -> float:
            return np.random.normal(1.0, 1.166)

    feature_providers = [
        DynamicProvider(provider_name="age", elements=list(range(20, 80, 1))),
        DynamicProvider(provider_name="sex", elements=[0, 1]),
        DynamicProvider(provider_name="cp", elements=[0, 1, 2, 3]),
        DynamicProvider(provider_name="trestbps", elements=list(range(90, 200, 1))),
        DynamicProvider(provider_name="chol", elements=list(range(126, 564, 1))),
        DynamicProvider(provider_name="fbs", elements=[0, 1]),
        DynamicProvider(provider_name="restecg", elements=[0, 1, 2]),
        DynamicProvider(provider_name="thalach", elements=list(range(70, 200, 1))),
        DynamicProvider(provider_name="exang", elements=[0, 1]),
        OldpeakProvider,
        DynamicProvider(provider_name="slope", elements=[0, 1, 2]),
        DynamicProvider(provider_name="ca", elements=[0, 1, 2, 3]),
        DynamicProvider(provider_name="thal", elements=[0, 1, 2]),
        DynamicProvider(provider_name="condition", elements=[0, 1]),
    ]

    for provider in feature_providers:
        fake.add_provider(provider)

    data = pd.DataFrame()
    object_number = np.random.randint(50, 200)
    data["age"] = [fake.age() for _ in range(object_number)]
    data["sex"] = [fake.sex() for _ in range(object_number)]
    data["cp"] = [fake.cp() for _ in range(object_number)]
    data["trestbps"] = [fake.trestbps() for _ in range(object_number)]
    data["chol"] = [fake.chol() for _ in range(object_number)]
    data["fbs"] = [fake.fbs() for _ in range(object_number)]
    data["restecg"] = [fake.restecg() for _ in range(object_number)]
    data["thalach"] = [fake.thalach() for _ in range(object_number)]
    data["exang"] = [fake.exang() for _ in range(object_number)]
    data["oldpeak"] = [fake.oldpeak() for _ in range(object_number)]
    data["slope"] = [fake.slope() for _ in range(object_number)]
    data["ca"] = [fake.ca() for _ in range(object_number)]
    data["thal"] = [fake.thal() for _ in range(object_number)]
    data["condition"] = [fake.condition() for _ in range(object_number)]

    return data
