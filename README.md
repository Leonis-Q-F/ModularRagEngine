# ModularRagEngine

面向 Python 后端系统的分层 RAG 引擎。

## 项目介绍

`ModularRagEngine` 提供统一的文档接入、分块、索引构建、检索与上下文组装能力。  
当前默认链路采用 `api -> application -> domain <- infrastructure` 四层结构，并把默认依赖装配、namespace 解析、持久化仓储拆分、搜索中间模型收敛到了更稳定的工程边界。

<p align="center">
  <img src="./images/setup.png" width="980" alt="ModularRagEngine workflow" />
</p>

## 主要特性

- 对外入口稳定：继续通过 `RAGEngine` 暴露统一 API。
- namespace 语义收敛：`namespace_id` 与 `namespace_key` 的解析、校验、创建策略统一。
- ingest 语义固定：文档写入后默认完成分块、向量化与索引同步，结果立即可检索。
- 存储层职责拆分：PostgreSQL 层拆成连接工厂、建表引导、mapper、repository，再由 `DocumentStore` 门面聚合。
- 搜索类型边界更清晰：过滤条件、重排结果不再靠裸字典和魔法 key 传递。
- 运行方式标准化：支持 `pip install -e .`、`python -m ModularRagEngine.retrieval_eval` 与 `rag-retrieval-eval`。

## 安装

建议先以可编辑模式安装：

```bash
python -m pip install -e .[dev]
```

如果你已经手工安装了运行依赖，也可以使用：

```bash
python -m pip install -e . --no-deps
```

## 运行前准备

默认真实链路依赖以下服务：

- PostgreSQL
- Milvus
- embedding 服务

按 `.env.example` 填写 PostgreSQL、Milvus、embedding、OCR 等配置。

## 快速开始

### 1. 创建引擎

```python
from ModularRagEngine import RAGEngine

engine = RAGEngine()
```

### 2. 写入文档

```python
from ModularRagEngine.application.dto import IngestDocumentsRequest, InputDocument

engine.ingest_documents(
    IngestDocumentsRequest(
        namespace_key="legal-case-001",
        documents=[
            InputDocument(
                external_doc_id="doc-001",
                file_name="case.md",
                file_type="md",
                parsed_md_content=(
                    "# Case Background\n\n"
                    "The dispute focuses on liquidated damages.\n\n"
                    "## Judgment\n\n"
                    "The court supports the plaintiff's claim."
                ),
            )
        ],
    )
)
```

### 3. 执行检索

```python
from ModularRagEngine.application.dto import SearchRequest

result = engine.search(
    SearchRequest(
        namespace_key="legal-case-001",
        query="liquidated damages",
        top_k_recall=8,
        top_k_rerank=5,
        top_k_context=3,
        filters={"language": "en"},
    )
)

print(result.llm_context)
```

### 4. 重建索引

```python
from ModularRagEngine.application.dto import RebuildIndexRequest

engine.rebuild_index(
    RebuildIndexRequest(
        namespace_key="legal-case-001",
        retrieval_text_policy="content_only",
    )
)
```

## namespace 约定

- 仅传 `namespace_key`：允许创建或复用 namespace。
- 仅传 `namespace_id`：只解析既有 namespace，不会隐式创建新 key。
- 同时传 `namespace_id` 与 `namespace_key`：必须指向同一个 namespace，否则报错。

## 对外 API

- `RAGEngine.ingest_files()`
- `RAGEngine.ingest_documents()`
- `RAGEngine.rebuild_index()`
- `RAGEngine.search()`

请求与响应模型位于 `application/dto.py`。

## 检索评测

真实链路评测模块保留在包内，可通过以下方式运行：

```bash
python -m ModularRagEngine.retrieval_eval \
  --doc /absolute/path/project_runtime_guide.md \
  --ks 1,3,5
```

或者：

```bash
rag-retrieval-eval \
  --doc /absolute/path/project_runtime_guide.md \
  --ks 1,3,5
```

默认会自动清理评测产生的 namespace 与 collection；如需保留现场，可追加 `--keep-artifacts`。

## 目录结构

```text
ModularRagEngine/
├── api/                        # 对外 Facade
├── application/                # 应用编排、namespace 解析、搜索中间模型
├── domain/                     # 领域模型、值对象、异常、常量、端口
├── infrastructure/             # 基础设施实现
├── infrastructure/persistence/ # PostgreSQL 连接、建表、mapper、repository
├── tests/                      # 回归测试与工程化整改测试
├── composition.py              # 默认依赖装配入口
├── retrieval_eval.py           # 包内评测模块
├── pyproject.toml              # 项目元数据与工具链配置
└── __init__.py                 # 包根导出
```

## 开发验证

当前仓库至少应通过：

```bash
python -m compileall -q .
pytest -q
```

## 许可证

按仓库后续约定补充。
