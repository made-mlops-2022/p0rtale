from typing import Literal
from pydantic import BaseModel, validator


class HeartDiseaseData(BaseModel):
    age: int
    sex: Literal[0, 1]
    cp: Literal[0, 1, 2, 3]
    trestbps: int
    chol: int
    fbs: Literal[0, 1]
    restecg: Literal[0, 1, 2]
    thalach: int
    exang: Literal[0, 1]
    oldpeak: float
    slope: Literal[0, 1, 2]
    ca: Literal[0, 1, 2, 3]
    thal: Literal[0, 1, 2]

    @validator("age")
    def age_limits(cls, v):
        if not 20 <= v <= 90:
            raise ValueError("must be between 20 and 90")
        return v

    @validator("trestbps")
    def trestbps_limits(cls, v):
        if not 80 <= v <= 220:
            raise ValueError("must be between 80 and 220")
        return v

    @validator("chol")
    def chol_limits(cls, v):
        if not 100 <= v <= 600:
            raise ValueError("must be between 100 and 600")
        return v

    @validator("thalach")
    def thalach_limits(cls, v):
        if not 60 <= v <= 220:
            raise ValueError("must be between 60 and 220")
        return v

    @validator("oldpeak")
    def oldpeak_limits(cls, v):
        if not 0.0 <= v <= 6.5:
            raise ValueError("must be between 0.0 and 6.5")
        return v
