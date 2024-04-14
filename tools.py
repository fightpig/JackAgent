import re
from typing import Tuple, List, Union, Optional

import pandas as pd
from langchain.schema import Document
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_community.tools.file_management import FileSearchTool, ReadFileTool, WriteFileTool, ListDirectoryTool
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.output_parsers.base import T
from langchain_core.tools import StructuredTool
from langchain_core.vectorstores import VectorStoreRetriever

from agent import BaseAgent
from color_print import ColorPrint
from models import get_zhipu_chat_llm
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


def get_excel_short_description(filename: str, n: int = 3) -> str:
    """
    获取Excel的简单描述，返回它所有的sheet名、第一个sheet的列名，以及第一个sheet的前n行
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
        f"3. 第一个Sheet的前{n}行数据：\n{lines}"
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
        inspections = get_excel_short_description(filename, 3)
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
            description="通过Python代码分析一个excel文件的内容。"
                        "输入中必须包含文件的完整路径、具体分析方式、分析依据和阈值常量等。"
                        "如果输入信息不完整，你可以拒绝回答",
        )


def get_tools(llm: BaseChatModel, excel_analyser_prompt_path: str, color_print: ColorPrint):
    excel_inspection_tool = StructuredTool.from_function(
        func=get_excel_short_description,
        name="InspectExcel",
        description="检查指定的Excel文件中的结构和内容，展示它的列名和前n行数据，n默认为3",
    )

    # tools = FileManagementToolkit(root_dir=".").get_tools()
    tools = [
        FileSearchTool(),
        ReadFileTool(),
        WriteFileTool(),
        ListDirectoryTool(),
        excel_inspection_tool,
        # ExcelAnalyser(prompt_path=excel_analyser_prompt_path, llm=llm, color_print=color_print).as_tool()
        AutoExcelAnalyser(
            base_prompt_config_path=excel_analyser_prompt_path, llm=llm, color_print=color_print
        ).as_tool()
    ]
    return tools


def test_excel_analyser():
    llm = get_zhipu_chat_llm()
    cp = ColorPrint(print_prefix='Excel Analyse')
    excel_analyser = AutoExcelAnalyser(
        base_prompt_config_path='prompt/tools/excel_analyse.json', llm=llm, color_print=cp,

    )
    result, cost = excel_analyser.run(
        task="8月销售额",
        filename="data/2023年8月-9月销售记录.xlsx"
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


if __name__ == '__main__':
    test_excel_analyser()
