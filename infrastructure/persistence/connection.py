from __future__ import annotations

from dataclasses import dataclass

import psycopg
from psycopg import sql
from psycopg.rows import dict_row


@dataclass(frozen=True)
class PostgresConnectionConfig:
    """描述 PostgreSQL 连接与 schema 配置。"""

    host: str
    port: int
    dbname: str
    user: str
    password: str | None
    schema: str
    sslmode: str
    connect_timeout: int

    def conninfo(self) -> str:
        """组装 psycopg 连接字符串。"""
        parts = [
            f"host={self.host}",
            f"port={self.port}",
            f"dbname={self.dbname}",
            f"user={self.user}",
            f"sslmode={self.sslmode}",
            f"connect_timeout={self.connect_timeout}",
        ]
        if self.password:
            parts.append(f"password={self.password}")
        return " ".join(parts)


class PostgresConnectionFactory:
    """统一生成 PostgreSQL 连接和 schema 表引用。"""

    def __init__(self, config: PostgresConnectionConfig) -> None:
        self._config = config

    @property
    def schema(self) -> str:
        return self._config.schema

    def connect(self):
        """创建带字典行工厂的 PostgreSQL 连接。"""
        return psycopg.connect(self._config.conninfo(), row_factory=dict_row, autocommit=False)

    def table(self, name: str):
        """生成带 schema 的表名引用。"""
        return sql.SQL("{}.{}").format(sql.Identifier(self._config.schema), sql.Identifier(name))
