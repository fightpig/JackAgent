import re
from pathlib import Path
from typing import Tuple, List, Union, Optional, Any, Dict

import pandas as pd
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.schema import Document, RUN_KEY
from langchain.tools.base import BaseTool
from langchain_community.callbacks import get_openai_callback
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.word_document import UnstructuredWordDocumentLoader
from langchain_community.tools.file_management import FileSearchTool, ReadFileTool, WriteFileTool, ListDirectoryTool
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.callbacks import CallbackManager
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.output_parsers.base import T
from langchain_core.tools import StructuredTool, tool
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agent import BaseAgent
from color_print import ColorPrint
from models import get_zhipu_chat_llm, get_zhipu_embeddings
from utils import exec_python_code


class JsonOutputParser(BaseOutputParser):

    def parse(self, text: str) -> T:
        pass


class PythonOutputParser(BaseOutputParser):

    def parse(self, text: str) -> T:
        pass


def get_sheet_names(filename: str) -> str:
    """获取Excel中所有sheet名字"""
    excel_file = pd.ExcelFile(filename)
    return excel_file.sheet_names


def get_column_names(filename: str) -> str:
    """获取Excel中第一个sheet的列名"""
    df = pd.read_excel(filename, sheet_name=0)
    return '\n'.join(df.columns.to_list())


@tool('InspectExcel')
def get_excel_meta_description(filename: str, n: int = 3) -> str:
    """
    描述指定excel的内容，返回它所有的sheet名、第一个sheet的列名，以及第一个sheet的列名和前n行数据，n默认为3
    """

    sheet_names = get_sheet_names(filename)
    column_names = get_column_names(filename)

    # 读取 Excel 文件的第一个工作表
    df = pd.read_excel(filename, sheet_name=0)  # sheet_name=0 表示第一个工作表
    lines = '\n'.join(df.head(n).to_string(index=False, header=True).split('\n'))

    description = (
        f"这是Excel '{filename}' 文件内容的简短描述："
        f"\n"
        f"1. 共有 {sheet_names} 这{len(sheet_names)}个Sheet列表"
        f"\n\n"
        f"2. 第一个Sheet的列名：{column_names}"
        f"\n\n"
        f"3. 第一个Sheet的列名和前{n}行数据：\n{lines}"
    )
    return description


def remove_markdown_tag(lines: str) -> str:
    lines = lines.strip().split('\n')
    if lines and lines[0].strip().startswith('```'):
        del lines[0]
    if lines and lines[-1].strip().startswith('```'):
        del lines[-1]
    return '\n'.join(lines)


def extract_markdown_txt(text: str, tag='python') -> Union[List[str], str]:
    txt_blocks = re.findall(rf'```{tag}\n(.*?)\n```', text, re.DOTALL)
    if len(txt_blocks) > 0:
        return [remove_markdown_tag(txt) for txt in txt_blocks]
    return txt_blocks


def extract_python_code(text: str) -> Union[List[str], str]:
    return extract_markdown_txt(text, tag='python')


def extract_json_code(text: str) -> Union[List[str], str]:
    return extract_markdown_txt(text, tag='json')


class DocumentRetriever(BaseTool):
    llm: BaseChatModel
    embeddings: Embeddings
    name: str = "DocumentRetriever"
    description: str = "此工具可以读取PDF/WORD/TXT文档内容，然后根据 问题 检索出你需要的 答案。"
    chunk_size: int = 300
    chunk_overlap: int = 50
    k: int = 1

    def load_docs(self, filename: str) -> List[Document]:
        suffix = Path(filename).suffix
        match suffix:
            case '.pdf':
                loader = PyPDFLoader(filename)
            case '.docx' | '.doc':
                loader = UnstructuredWordDocumentLoader(filename)
            case '.txt' | '.text':
                loader = TextLoader(filename)
            case _:
                raise NotImplementedError(f"File extension {suffix} not supported.")
        return loader.load_and_split()

    def _run(self, filename: str, query: str):

        raw_docs = self.load_docs(filename)
        if len(raw_docs) == 0:
            return {'query': query, 'result': f'文档{filename}内容为空'}

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            add_start_index=True,
        )

        documents = splitter.split_documents(raw_docs)
        if documents is None or len(documents) == 0:
            return {'query': query, 'result': f"无法读取文档{filename}内容"}

        db = FAISS.from_documents(documents, self.embeddings)
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            # "stuff": 这种方法通常在处理需要大量上下文信息的问题时很有用，因为它允许语言模型直接访问检索到的文档内容。
            #          然而，这可能会导致输入长度过长，超出某些模型的处理能力，并且可能会引入噪声，因为所有检索到的内容都被视为相关。
            # "map": 这种方法在处理多个相关文档时很有用，因为它可以分别考虑每个文档与查询的关系。它适用于需要精确匹配和评估每个文档相关性的场景。
            # "map_rerank": 当你需要根据模型生成的答案质量来调整最终答案时，这种方法很有用。它可以帮助提高系统的整体回答质量，但可能会增加计算成本。
            # "refine": 这种方法适用于需要快速生成初步答案，然后使用更复杂的模型进行细化的场景。它可以减少对计算资源的需求，同时仍然提供高质量的答案。
            # "combine": 当你想要结合多个模型的优点时，这种方法很有用。它可以提高系统的鲁棒性和准确性，但需要更多的计算资源和管理复杂性。
            chain_type="stuff",
            retriever=db.as_retriever(
                search_kwargs={
                    'k': self.k,
                    # 'score_threshold': 0.8,
                },
            ),  # 检索器
        )
        # qa_chain.return_source_documents = True
        # response = qa_chain.invoke({'query': query})
        response = qa_chain.run({'query': query})
        return response


class AutoExcelAnalyser(BaseAgent):
    def __init__(self,
                 llm: BaseChatModel,
                 base_prompt_config_path: str,
                 color_print: ColorPrint,
                 short_chat_memory_obj: Optional[BaseChatMessageHistory] = ChatMessageHistory(),
                 long_chat_vdb_obj: Optional[VectorStoreRetriever] = None,
                 docs_vdb_obj: Optional[VectorStoreRetriever] = None,
                 max_send_tokens: int = 4500,
                 max_short_chat_tokens: int = 800,
                 max_long_chat_tokens: int = 400,
                 max_docs_tokens: int = 1200,
                 max_reason_cn: int = 3,
                 ):
        super().__init__(llm=llm,
                         base_prompt_config_path=base_prompt_config_path,
                         output_parser=None,
                         short_chat_memory_obj=short_chat_memory_obj,
                         long_chat_vdb_obj=long_chat_vdb_obj,
                         docs_vdb_obj=docs_vdb_obj,
                         max_send_tokens=max_send_tokens,
                         max_short_chat_tokens=max_short_chat_tokens,
                         max_long_chat_tokens=max_long_chat_tokens,
                         max_docs_tokens=max_docs_tokens,
                         color_print=color_print,
                         max_reason_cn=max_reason_cn
                         )

    def run(self, task: str, filename: str) -> Tuple[str, dict, float]:
        return super().run(task=task, filename=filename)

    def reason(self, task, filename, loop_cn, add_message=False, use_chat_num=2) -> BaseMessage:
        if isinstance(get_excel_meta_description, StructuredTool):
            inspections = get_excel_meta_description.run(tool_input=dict(filename=filename, n=3))
        else:
            inspections = get_excel_meta_description(filename, 3)

        parameters = dict(
            filename=filename,
            inspections=inspections,
        )
        return super().reason(task, loop_cn, parameters, add_message=add_message, use_chat_num=use_chat_num)

    def action(self, ai_reply: BaseMessage, loop_cn: int) -> Tuple[bool, Union[str, None]]:
        codes = extract_python_code(ai_reply.content)
        if len(codes) == 0:
            decision_instruction = "你回复的 Response 中没有找到可执行的 Python代码，请再仔细思考，重新生成。"
        elif len(codes) > 1:
            decision_instruction = "你回复的 Response 中有多个 Python代码块，请汇总成一个。"
        else:
            code = codes[0]
            self.color_print.print_action(code, idx=loop_cn)
            state, result = exec_python_code(code)
            self.color_print.print_result(f'代码执行报错: {state}，代码执行结果：{result}', idx=loop_cn)
            if state is True:
                return state, result

            decision_instruction = f"根据你生成的 Python代码执行报错: {state}，请仔细思考和修正。"

        # 将错误信息追加给大模型，让大模型重新生成代码
        if self.long_chat_vdb_obj:
            memory_to_add = f"Assistant Reply: {ai_reply.content} \nResult: {decision_instruction} "
            self.long_chat_vdb_obj.add_documents([Document(page_content=memory_to_add)])

        self.short_chat_memory_obj.add_message(HumanMessage(content=decision_instruction))
        self.short_chat_memory_obj.add_message(AIMessage(content=ai_reply.content))

        self.color_print.print_result(decision_instruction, idx=loop_cn)
        return False, None

    def as_tool(self):
        return StructuredTool.from_function(
            func=self.run,
            name="AnalyseExcel",
            description="根据需求，生成 Python代码 来分析指定excel中的数据，返回分析结果。"
            # "输入中必须包含文件的完整路径、具体分析方式、分析依据和阈值常量等。"
            # "如果输入信息不完整，你可以拒绝回答",
        )


class ToolObj:
    def __init__(self,
                 llm: BaseChatModel,
                 embeddings: Embeddings,
                 color_print: ColorPrint,
                 excel_analyser_prompt_path: str,
                 chunk_size: int = 300,
                 chunk_overlap: int = 50,
                 ):
        self.llm = llm
        self.embeddings = embeddings
        self.color_print = color_print
        self.excel_analyser_prompt_path = excel_analyser_prompt_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self._init()

    def _init(self):
        self.tool_objs = {
            'ListDirectoryTool': ListDirectoryTool(),
            'FileSearchTool': FileSearchTool(),
            'ReadFileTool': ReadFileTool(),
            'WriteFileTool': WriteFileTool(),
        }

    def get_data_analyse_tools(self):
        self.tool_objs.update(**{
            # 自定义
            'excel_inspection_tool': get_excel_meta_description,
            'AutoExcelAnalyser': AutoExcelAnalyser(
                base_prompt_config_path=self.excel_analyser_prompt_path,
                llm=self.llm,
                color_print=self.color_print
            ),
        })
        tools = list()
        for tool in self.tool_objs.values():
            if not isinstance(tool, BaseTool):
                tool = tool.as_tool()
            tools.append(tool)
        return tools

    def get_doc_tools(self) -> List[BaseTool]:
        # tools = FileManagementToolkit(root_dir=".").get_tools()
        self.tool_objs.update(**{
            'DocumentRetriever': DocumentRetriever(
                llm=self.llm,
                embeddings=self.embeddings,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        })
        tools = list()
        for tool in self.tool_objs.values():
            if not isinstance(tool, BaseTool):
                tool = tool.as_tool()
            tools.append(tool)
        return tools


def test_excel_analyser():
    llm = get_zhipu_chat_llm()
    cp = ColorPrint(print_prefix='Excel Analyse')
    excel_analyser = AutoExcelAnalyser(
        base_prompt_config_path='prompt/tools/excel_analyse.json',
        llm=llm,
        color_print=cp,
    )
    result, tokens, cost = excel_analyser.run(
        task="8月所有供应商的销售额，并汇总成总销售额",
        filename="data/data_analyse/2023年8月-9月销售记录.xlsx"
    )
    print(result, cost)  # 2605636

    # result, cost = excel_analyser.run(
    #     task="9月销售额",
    #     filename="data/2023年8月-9月销售记录.xlsx"
    # )
    # print(result, cost)  # 2851099

    # re = excel_analyser.analyse(
    #     query="8月和9月总销售额",
    #     filename="data/2023年8月-9月销售记录.xlsx"
    # )
    # print(re)  # 5456735
    print(f'total tokens:\t'
          f'prompt_tokens: {excel_analyser.prompt_tokens} \t'
          f'completion_tokens: {excel_analyser.completion_tokens} \t'
          f'total_tokens: {excel_analyser.total_tokens}')


def test_document_retriever():
    llm = get_zhipu_chat_llm()
    embeddings = get_zhipu_embeddings()

    tool = DocumentRetriever(llm=llm, embeddings=embeddings, chunk_size=100, chunk_overlap=10)
    tool_input = dict(
        filename="./data/docs/供应商资格要求.pdf",
        query='月销售额',
    )
    re = tool.run(tool_input)
    print(re)


if __name__ == '__main__':
    test_excel_analyser()
    # test_document_retriever()
