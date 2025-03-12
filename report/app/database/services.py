from typing import Type, List
from sqlmodel import Session, SQLModel, select


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
