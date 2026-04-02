class RagChunker:
    """ 整个工具的顶层类"""
    def __init__(self):
        # 数据处理加载
        from core.dataloader import DocumentLoader
        self.loader = DocumentLoader()

        # 分块工具加载
        from core.chunk import MarkdownStructureSplitter
        self.splitter = MarkdownStructureSplitter()


    def run(self,file_paths: str | list[str], use_ocr: bool = False):
        source_documents = self.loader.load(file_paths, use_ocr)
        chunks = self.splitter.split_document(source_documents)
        return chunks
