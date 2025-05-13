from datetime import datetime

from sqlmodel import Field, Relationship, SQLModel, UniqueConstraint

from .enumerated import (
    MetricCategory,
    SupportedModelTasks,
    SupportedPlatforms,
    SupportModelFrameworks,
    platform_url_mapping,
)

############## DEVICE


class DeviceBase(SQLModel):
    """Base SQL class representing a device."""

    sys_name: str = Field(index=True)
    kernel_version: str = Field(index=True)


class Device(DeviceBase, table=True):
    """Class representing device SQL table."""

    id: int | None = Field(default=None, primary_key=True)

    tests: list["Run"] = Relationship(back_populates="device")

    __table_args__ = (
        UniqueConstraint("sys_name", "kernel_version", name="fingerprint"),
        # If you want a composite index that's not unique
        # Index("idx_sys_kernel", "sys_name", "kernel_version"),
    )


############## MODEL


class ModelBase(SQLModel):
    """Base SQL class representing a model."""

    framework: SupportModelFrameworks = Field(index=True)
    task: SupportedModelTasks = Field(index=True)
    path: str = Field(index=True, unique=True)
    platform: SupportedPlatforms = Field(index=True)

    @property
    def platform_url(self) -> str:
        """Returns the URL associated with the platform."""
        return platform_url_mapping.get(self.platform, "")


class Model(ModelBase, table=True):
    """Class representing model SQL table."""

    id: int | None = Field(default=None, primary_key=True)

    tests: list["Run"] = Relationship(back_populates="model")


############## DATASET


class DatasetBase(SQLModel):
    """Base SQL class representing a dataset."""

    path: str = Field(index=True, unique=True)
    platform: SupportedPlatforms = Field(index=True)

    @property
    def platform_url(self) -> str:
        """Returns the URL associated with the platform."""
        return platform_url_mapping.get(self.platform, "")


class Dataset(DatasetBase, table=True):
    """Class representing dataset SQL table."""

    id: int | None = Field(default=None, primary_key=True)

    tests: list["Run"] = Relationship(back_populates="dataset")


################ RUN


class Run(SQLModel, table=True):
    """Class representing run SQL table."""

    id: int | None = Field(default=None, primary_key=True)
    created_datetime: datetime = Field(default_factory=lambda: datetime.now())
    device_id: int = Field(foreign_key="device.id")
    model_id: int = Field(foreign_key="model.id")
    dataset_id: int = Field(foreign_key="dataset.id")

    device: Device = Relationship(back_populates="tests")
    model: Model = Relationship(back_populates="tests")
    dataset: Dataset = Relationship(back_populates="tests")
    metrics: list["Metric"] = Relationship(back_populates="run")


# ############## METRIC


class MetricBase(SQLModel):
    """Class representing metric SQL table."""

    name: str = Field(index=True)
    value: float = Field(index=True)
    unit: str | None = None


class Metric(MetricBase, table=True):
    """Class representing metric SQL table."""

    id: int | None = Field(default=None, primary_key=True)
    category: MetricCategory = Field(index=True)
    run_id: int = Field(foreign_key="run.id")

    run: Run = Relationship(back_populates="metrics")
