from pymilvus import MilvusClient
from pydantic import BaseModel, PrivateAttr


class MilvusStore(BaseModel):
    milvus_client: MilvusClient = PrivateAttr()

