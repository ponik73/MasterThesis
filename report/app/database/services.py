from collections.abc import Generator
from typing import Any

from sqlmodel import Session, SQLModel, create_engine, select

from .models import Device, Run

engine = create_engine("sqlite:///database.db", connect_args={"check_same_thread": False})


def create_db_and_tables() -> None:
    """Initialize database."""
    SQLModel.metadata.create_all(engine)


def get_session() -> Generator[Session, Any, None]:
    """Get database session engine."""
    with Session(engine) as session:
        yield session


def insert_single_into_db(db_object: SQLModel, db: Session) -> SQLModel:
    """Insert single object into database.

    :param db_object: database object
    :param db: database dependency injection
    :return: database object
    """
    db.add(db_object)
    db.commit()
    db.refresh(db_object)
    return db_object


def insert_multiple_into_db(db_object: list[SQLModel], db: Session) -> None:
    """Insert multiple objects into database.

    :param db_object: list of database objects
    :param db: database dependency injection
    """
    db.add_all(db_object)
    db.commit()


def get_single_by_all_properties(
    db_object_class: type[SQLModel], object_data: SQLModel, db: Session
) -> SQLModel | None:
    """Get first object matching the given properties from database."""
    query = select(db_object_class)

    for name, value in object_data.model_dump().items():
        query = query.where(getattr(db_object_class, name) == value)

    return db.exec(query).first()


def get_single_by_id(db_object_class: type[SQLModel], identifier: int, db: Session) -> SQLModel:
    """Get object from database with corresponding identifier."""
    query = select(db_object_class).where(db_object_class.id == identifier)
    return db.exec(query).first()


def get_multiple_by_ids(db_object_class: type[SQLModel], identifiers: list[int], db: Session) -> list[SQLModel]:
    """Get multiple object from database with corresponding identifiers."""
    query = select(db_object_class).where(db_object_class.id.in_(identifiers))
    return db.exec(query).all()


def get_or_create_record(object_data: SQLModel, db_object_class: type[SQLModel], db: Session) -> SQLModel:
    """Get first object matching the given properties from database. Insert into database if object does not exists."""
    db_object = get_single_by_all_properties(db_object_class=db_object_class, object_data=object_data, db=db)

    if not db_object:
        db_object = insert_single_into_db(db=db, db_object=db_object_class.model_validate(object_data))

    return db_object


def get_all_objects_by_properties(
    db_object_class: type[SQLModel],
    db: Session,
    properties: dict[str, Any] | None = None,
) -> list[SQLModel]:
    """Get all objects from database that share given properties."""
    query = select(db_object_class)

    if properties is None:
        return db.exec(query).all()

    for name, value in properties.items():
        query = query.where(getattr(db_object_class, name) == value)

    return db.exec(query).all()


def get_runs_with_sys_name(db: Session, sys_name: str) -> list[Run]:
    """Get list of runs that are associated with given `sys_name`."""
    device_ids_subquery = select(Device.id).where(Device.sys_name == sys_name)
    query = select(Run).where(Run.device_id.in_(device_ids_subquery))
    return db.exec(query).all()


def get_latest_run(
    db: Session, model_id: int, sys_name: str, kernel_version: str, dataset_id: int | None = None
) -> Run | None:
    """Get latest run of a given model (and a given dataset) that was performed on a given device."""
    device_id_subquery = select(Device.id)
    device_id_subquery = device_id_subquery.where(Device.sys_name == sys_name)
    device_id_subquery = device_id_subquery.where(Device.kernel_version == kernel_version)

    query = select(Run)
    query = query.where(Run.device_id.in_(device_id_subquery))
    query = query.where(Run.model_id == model_id)
    if dataset_id is not None:
        query = query.where(Run.dataset_id == dataset_id)
    query = query.order_by(Run.created_datetime.desc())
    return db.exec(query).first()
