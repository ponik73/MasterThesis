from typing import Type, List, Dict, Any
from sqlmodel import Session, SQLModel, select, create_engine

from .models import Device, Run
from .enumerated import SupportedPlatforms

engine = create_engine(
    "sqlite:///database.db",
    connect_args={
        "check_same_thread": False
    }
)
def create_db_and_tables():
    SQLModel.metadata.create_all(engine)
def get_session():
    with Session(engine) as session:
        yield session

def insert_single_into_db(db_object: SQLModel, db: Session):
    db.add(db_object)
    db.commit()
    db.refresh(db_object)
    return db_object


def insert_multiple_into_db(db_object: List[SQLModel], db: Session):
    db.add_all(db_object)
    db.commit()


def get_single_by_all_properties(db_object_class: Type[SQLModel], object_data: SQLModel, db: Session):
    query = select(db_object_class)
    
    for name, value in object_data.model_dump().items():
        query = query.where(getattr(db_object_class, name) == value)
    
    return db.exec(query).first()

def get_single_by_id(db_object_class: Type[SQLModel], identifier: int, db: Session) -> SQLModel:    
    query = select(db_object_class).where(db_object_class.id == identifier)
    return db.exec(query).first()
    

def get_multiple_by_ids(db_object_class: Type[SQLModel], identifiers: List[int], db: Session) -> List[SQLModel]:
    query = select(db_object_class).where(db_object_class.id.in_(identifiers))
    return db.exec(query).all()

def get_or_create_record(object_data: SQLModel, db_object_class: Type[SQLModel], db: Session):
    db_object = get_single_by_all_properties(
        db_object_class=db_object_class,
        object_data=object_data,
        db=db
    )

    if not db_object:
        db_object = insert_single_into_db(
            db=db,
            db_object=db_object_class.model_validate(object_data)
        )

    return db_object

def get_all_objects_by_properties(
        db_object_class: Type[SQLModel],
        db: Session,
        properties: Dict[str, Any] | None = None,
) -> List[SQLModel]:
    query = select(db_object_class)

    if properties is None:
        return db.exec(query).all()
    
    for name, value in properties.items():
        query = query.where(getattr(db_object_class, name) == value)

    return db.exec(query).all()

def get_runs_with_sys_name(db: Session, sys_name: str) -> List[Run]:
    device_ids_subquery = select(Device.id).where(Device.sys_name == sys_name)
    query = select(Run).where(Run.device_id.in_(device_ids_subquery))
    return db.exec(query).all()

def get_latest_run(
    db: Session,
    model_id: int,
    sys_name: str,
    kernel_version: str,
    dataset_id: int | None = None
) -> Run:
    device_id_subquery = select(Device.id)
    device_id_subquery = device_id_subquery.where(Device.sys_name == sys_name)
    device_id_subquery = device_id_subquery.where(Device.kernel_version == kernel_version)

    query = select(Run)
    query = query.where(Run.device_id.in_(device_id_subquery))
    query = query.where(Run.model_id == model_id)
    if dataset_id is not None:
        query = query.where(Run.dataset_id == dataset_id)
    query = query.order_by(Run.created_datetime)
    return db.exec(query).first()