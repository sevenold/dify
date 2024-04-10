from enum import Enum


class Field(Enum):
    CONTENT_KEY = "page_content"
    METADATA_KEY = "metadata"
    GROUP_KEY = "group_id"
    VECTOR = "vector"
    DATA_TYPE = "data_type"
    TEXT_KEY = "text"
    PRIMARY_KEY = "id"
