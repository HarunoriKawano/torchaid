import pathlib
from typing import Type
from pydantic import BaseModel

__all__ = ["json_to_instance"]

def json_to_instance(path: str, structure: Type[BaseModel]):
    """Loads a JSON file and deserialises it into a Pydantic model instance.

    Args:
        path (str): File system path to the JSON file.
        structure (Type[BaseModel]): The Pydantic model class to deserialise into.
            The JSON schema must be compatible with this model.

    Returns:
        BaseModel: A validated instance of ``structure`` populated with the
            data from the JSON file.
    """
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    json_string = p.read_text()
    instance = structure.model_validate_json(json_string)
    return instance
