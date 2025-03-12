from sqlmodel import Field, SQLModel, Relationship
from typing import List, Dict, Literal

from .enumerated import SupportModelFrameworks, SupportedModelTasks, SupportedPlatforms, MetricCategory, platform_url_mapping

############## DEVICE

# Base, Public, Create
class DeviceBase(SQLModel):
    fingerprint: str  = Field(index=True, unique=True)

# Table
class Device(DeviceBase, table=True):
    id: int | None = Field(default=None, primary_key=True)

    tests: List["Run"] = Relationship(back_populates="device")

############## MODEL & DATASET

# Base, Public, Create
# class ResourceBase(SQLModel):
#     path: str = Field(index=True, unique=True)
#     platform: SupportedPlatforms = Field(index=True)

# # Table
# class Resource(ResourceBase, table=True):
#     id: int | None = Field(default=None, primary_key=True)

#     models: List["Model"] = Relationship(back_populates="resource")
#     datasets: List["Dataset"] = Relationship(back_populates="resource")

############## MODEL

# Base, Public
class ModelBase(SQLModel):
    framework: SupportModelFrameworks = Field(index=True)
    task: SupportedModelTasks = Field(index=True)
    path: str = Field(index=True, unique=True)
    platform: SupportedPlatforms = Field(index=True)

    @property
    def platform_url(self) -> str:
        """Returns the URL associated with the platform."""
        
        return platform_url_mapping.get(self.platform, "")

# Table
class Model(ModelBase, table=True):
    id: int | None = Field(default=None, primary_key=True)
    # resource_id: int = Field(foreign_key="resource.id", index=True, unique=True)

    # resource: Resource = Relationship(back_populates="models")
    tests: List["Run"] = Relationship(back_populates="model")    


############## DATASET

class DatasetBase(SQLModel):
    path: str = Field(index=True, unique=True)
    platform: SupportedPlatforms = Field(index=True)

    @property
    def platform_url(self) -> str:
        """Returns the URL associated with the platform."""
        
        return platform_url_mapping.get(self.platform, "")

# Base, Public, Table
class Dataset(DatasetBase, table=True):
    id: int | None = Field(default=None, primary_key=True)
    # resource_id: int = Field(foreign_key="resource.id", index=True, unique=True)

    # resource: Resource = Relationship(back_populates="datasets")
    tests: List["Run"] = Relationship(back_populates="dataset")

################ RUN

# Base, Public, Table
class Run(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    device_id: int = Field(foreign_key="device.id")
    model_id: int = Field(foreign_key="model.id")
    dataset_id: int = Field(foreign_key="dataset.id")

    device: Device = Relationship(back_populates="tests")
    model: Model = Relationship(back_populates="tests")
    dataset: Dataset = Relationship(back_populates="tests")
    metrics: List["Metric"] = Relationship(back_populates="run")

# ############## METRIC

# Base, Public, Create
class MetricBase(SQLModel):
    name : str = Field(index=True)
    value : float = Field(index=True)

# Table
class Metric(MetricBase, table=True):
    id: int | None = Field(default=None, primary_key=True)
    category : MetricCategory = Field(index=True)
    run_id : int = Field(foreign_key="run.id")

    run : Run = Relationship(back_populates="metrics")