from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import RelationshipProperty, class_mapper, relationship
from sqlalchemy.sql import func

from api.schemas import EgBaseModel

Base = declarative_base()

# By convention, we reserve the String type for Enum defined in the schema scope.


def is_relationship(model, attribute_name):
    mapper = class_mapper(model)
    property = mapper.attrs.get(attribute_name)
    return isinstance(property, RelationshipProperty)


def create_object_from_dict(db, model, data):
    """Create a SQL object from a nested dictionary with relationship"""
    if isinstance(data, Base):
        return data

    # Separate relationships from simple attributes
    relationships = {}
    attributes = {}
    for k, v in data.items():
        if hasattr(model, k):
            if is_relationship(model, k):
                relationships[k] = v
            else:
                attributes[k] = v

    # Create the main object
    obj = model(**attributes)

    # Handle relationships
    for rel_name, rel_data in relationships.items():
        rel_model = getattr(model, rel_name).property.mapper.class_
        if isinstance(rel_data, Base):  # Relationship is already a db model
            setattr(obj, rel_name, rel_data)
        elif isinstance(rel_data, list):  # Handle one-to-many or many-to-many
            related_objs = [create_object_from_dict(db, rel_model, item) for item in rel_data]
            setattr(obj, rel_name, related_objs)
        elif isinstance(rel_data, (dict, EgBaseModel)):  # Handle one-to-one or many-to-one
            if isinstance(rel_data, EgBaseModel):
                rel_data = vars(rel_data)
            related_obj = create_object_from_dict(db, rel_model, rel_data)
            setattr(obj, rel_name, related_obj)
        else:
            raise NotImplementedError(
                f"Unknonw type for relationtioship {rel_name}:{type(rel_data)}"
            )

    return obj


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True)
    name = Column(Text, unique=True)
    df = Column(JSON)
    has_query = Column(Boolean)
    has_output = Column(Boolean)
    has_output_true = Column(Boolean)
    size = Column(Integer)


class Model(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True)
    name = Column(Text)
    base_url = Column(Text)
    api_key = Column(Text)
    prompt_system = Column(Text)
    # prompt_template = Column(Text) # rag, composition, multiagents ?
    sampling_params = Column(JSON)  # dict
    extra_kw = Column(JSON)  # dict


class Result(Base):
    __tablename__ = "results"

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, server_default=func.now())
    metric_name = Column(String)
    num_try = Column(Integer, default=0)
    num_success = Column(Integer, default=0)

    # One
    experiment_id = Column(Integer, ForeignKey("experiments.id"))
    experiment = relationship("Experiment", back_populates="results")
    # Many
    observation_table = relationship("ObservationTable", back_populates="result")


class ObservationTable(Base):
    __tablename__ = "observation_table"
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, server_default=func.now())
    score = Column(Float)
    observation = Column(JSON)
    num_line = Column(Integer)
    error_msg = Column(String)

    # One
    result_id = Column(Integer, ForeignKey("results.id"))
    result = relationship("Result", back_populates="observation_table")


class Answer(Base):
    __tablename__ = "answers"
    id = Column(Integer, primary_key=True)
    answer = Column(Text)
    num_line = Column(Integer)
    error_msg = Column(String)

    # One
    experiment_id = Column(Integer, ForeignKey("experiments.id"))
    experiment = relationship("Experiment", back_populates="answers")


class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True)
    name = Column(Text)
    readme = Column(Text)
    is_archived = Column(Boolean, default=False)  # do not allow user to remove without IAM.
    created_at = Column(DateTime, server_default=func.now())
    metrics = Column(JSON)  # asked metrics,  list of enum
    experiment_status = Column(String)
    num_try = Column(Integer, default=0)
    num_success = Column(Integer, default=0)

    # One
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    dataset = relationship("Dataset")
    model_id = Column(Integer, ForeignKey("models.id"))
    model = relationship("Model")
    experiment_set_id = Column(Integer, ForeignKey("experiment_sets.id"))
    experiment_set = relationship("ExperimentSet", back_populates="experiments")
    # Many
    results = relationship("Result", back_populates="experiment")  # len == #metrics
    answers = relationship("Answer", back_populates="experiment")  # len == #dataset.df

    __table_args__ = (
        UniqueConstraint("experiment_set_id", "name", name="_expset_name_unique_constraint"),
    )


class ExperimentSet(Base):
    __tablename__ = "experiment_sets"

    id = Column(Integer, primary_key=True)
    name = Column(Text, unique=True)
    readme = Column(Text)
    created_at = Column(DateTime, server_default=func.now())
    is_archived = Column(Boolean, default=False)  # do not allow user to remove without IAM.

    # Many
    experiments = relationship("Experiment", back_populates="experiment_set")
