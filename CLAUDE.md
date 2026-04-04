# ModularRagEngine

## 目录骨架

```text
ModularRagEngine/
├── api/                        # 对外 Facade，只保留统一入口
├── application/                # 用例编排、namespace 解析、搜索中间模型
├── domain/                     # 实体、值对象、常量、异常、端口
├── infrastructure/             # 真实基础设施实现
├── infrastructure/persistence/ # PostgreSQL 连接、建表、mapper、repository
├── tests/                      # 行为回归与工程化整改测试
├── images/                     # README 展示资源
├── data/                       # 示例文档与研发资料
├── composition.py              # 默认依赖装配入口
├── retrieval_eval.py           # 包内真实检索评测模块
├── pyproject.toml              # 项目元数据、打包与工具链配置
├── config.py                   # 运行时配置入口，统一读取 `.env`
├── DATA_MODELS.md              # 当前主链路与 backup 历史模型的结构总览
├── README.md                   # 面向使用者的快速说明
└── __init__.py                 # 包根导出
```

## 文件职责

- `api/engine.py`：暴露 `RAGEngine` 门面，只转发请求并委托应用服务。
- `composition.py`：组装默认 `document_store / vector_store / loader / chunker / embedding / reranker` 依赖。
- `DATA_MODELS.md`：集中说明生产模型与历史模型的字段结构、关系骨架与迁移语义。
- `application/dto.py`：定义对外 DTO，并收紧请求边界校验。
- `application/namespace_resolver.py`：统一 `namespace_id / namespace_key` 的创建、查询与冲突校验。
- `application/ingest_service.py`：负责文档接入、切分落库与同步索引，确保 ingest 后即可检索。
- `application/index_service.py`：负责索引构建、激活切换、检索文本投影和向量写入。
- `application/search_service.py`：负责召回、重排、父块回填与上下文组装。
- `application/search_models.py`：定义搜索链路内部中间模型，替代裸字典。
- `domain/constants.py`：统一语言、索引状态、检索文本策略等常量与枚举。
- `domain/value_objects.py`：定义 `NamespaceReference`、`ResolvedNamespace`、`SearchFilters` 等值对象。
- `infrastructure/document_store.py`：PostgreSQL 门面，聚合 namespace/content/index 子仓储。
- `infrastructure/persistence/connection.py`：管理 PostgreSQL 连接配置与表引用。
- `infrastructure/persistence/bootstrap.py`：负责 schema、表与索引自举。
- `infrastructure/persistence/mappers.py`：负责数据库行到领域实体的转换。
- `infrastructure/persistence/repositories.py`：拆分 namespace、内容、索引仓储职责。
- `infrastructure/milvus_store.py`：Milvus 检索适配器，过滤条件与原始命中结果在此收敛。
- `infrastructure/embedding_service.py`：embedding 适配层，维度改为懒解析，不在构造期远程探测。
- `retrieval_eval.py`：真实链路评测入口，要求通过已安装包或 `python -m` 运行。
- `tests/test_engine_regularization.py`：验证 namespace 解析、请求校验、策略透传与延迟维度行为。

## 依赖边界

- `api -> application -> domain`
- `application -> domain.ports`
- `composition -> application + infrastructure`
- `infrastructure -> domain`
- `infrastructure/document_store.py -> infrastructure/persistence/*`

边界原则：

- `api` 只做 Facade，不写业务规则，也不直接 new 真实基础设施。
- `application` 只负责编排和契约收敛，不持有底层连接细节。
- `composition.py` 是唯一默认依赖装配入口。
- `infrastructure` 只实现端口，不反向依赖 `api`。
- `persistence` 只解决 PostgreSQL 技术细节，不承载应用层规则。

## 当前架构决策

- PostgreSQL 是元数据与索引账本的单一真相源。
- Milvus 只负责向量与稀疏检索，不承担业务状态。
- `namespace_id` 与 `namespace_key` 的解析由 `NamespaceResolver` 统一收口。
- `SearchFilters` 与 `RankedEntry` 用于替换弱类型字典边界。
- `EmbeddingService` 不在构造函数内触发远程向量探测。
- 评测链路保留真实 `ingest -> index -> search -> context assembly` 数据流，不引入内存模拟。
- `backup/` 仅作历史参考，不参与生产运行。

## 开发约定

- 代码命名使用英文；注释、文档、日志文案使用中文。
- 构造函数不做昂贵外部调用；副作用通过显式方法或懒加载触发。
- 公共函数保持单一职责，避免在应用层继续传递裸 `dict[str, Any]` 语义。
- 涉及架构变更时，同步更新此文件，避免系统记忆断裂。
- 统一通过 `pyproject.toml` 管理打包、测试与静态检查入口。

## 变更记录

- `2026-04-02`：新增 `retrieval_eval.py`，为真实检索链路补齐量化评测入口。
- `2026-04-03`：`MarkdownChunker` 内联迁移旧版切分算法，并新增切分回归测试。
- `2026-04-03`：引入 `composition.py`、`NamespaceResolver`、`SearchFilters`、`RankedEntry`，收敛请求契约与装配边界。
- `2026-04-03`：拆分 PostgreSQL 持久化层为连接工厂、建表引导、mapper、repository，并保留 `DocumentStore` 门面。
- `2026-04-03`：新增 `pyproject.toml`，去除测试和评测脚本中的路径注入，统一包运行方式。
- `2026-04-04`：新增 `DATA_MODELS.md`，汇总当前生产链路与 `backup/` 历史模型的结构定义。
