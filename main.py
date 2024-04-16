from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_community.tools.file_management import ListDirectoryTool
from langchain_core.output_parsers import PydanticOutputParser

from agent import AutoAgent
from color_print import ColorPrint
from models import get_zhipu_chat_llm, get_zhipu_embeddings
from output_parse import Response
from tools import DocumentRetriever, get_excel_meta_description, AutoExcelAnalyser

llm = get_zhipu_chat_llm(max_tokens=2000, temperature=0.1)
embeddings = get_zhipu_embeddings()
main_color_print = ColorPrint(print_prefix='Main')


def coding_task(task: str, resource_dir_path: str):
    excel_color_print = ColorPrint(print_prefix='Excel Analyse')
    excel_analyser_prompt_path = 'prompt/tools/excel_analyse.json'

    tools = [
        ListDirectoryTool(),
        get_excel_meta_description,
        AutoExcelAnalyser(
            base_prompt_config_path=excel_analyser_prompt_path,
            llm=llm,
            color_print=excel_color_print
        ).as_tool(),

    ]

    agent = AutoAgent(
        ai_name='小聪明',
        ai_role='智能助手',
        llm=llm,
        base_prompt_config_path='prompt/code-interpreter/base.json',
        tools=tools,
        output_parser=PydanticOutputParser(pydantic_object=Response),
        short_chat_memory_obj=ChatMessageHistory(),
        color_print=main_color_print,
        max_reason_cn=12,
    )

    agent.run(task=task, resource_dir_path=resource_dir_path)
    print(f'本次消耗token：\n'
          f'\tprompt_tokens: {agent.prompt_tokens}\n'
          f'\tcompletion_tokens: {agent.completion_tokens}\n'
          f'\ttotal_tokens: {agent.total_tokens}\n')


def rag_task(task: str, resource_dir_path: str):
    tools = [
        ListDirectoryTool(),
        DocumentRetriever(
            llm=llm,
            embeddings=embeddings,
            chunk_size=300,
            chunk_overlap=50,
        ),
    ]

    rag_agent = AutoAgent(
        llm=llm,
        base_prompt_config_path='prompt/rag/base.json',
        tools=tools,
        output_parser=PydanticOutputParser(pydantic_object=Response),
        short_chat_memory_obj=ChatMessageHistory(),
        color_print=main_color_print,
        max_reason_cn=5,
    )

    rag_agent.run(task=task, resource_dir_path=resource_dir_path)
    print(f'本次消耗token：\n'
          f'\tprompt_tokens: {rag_agent.prompt_tokens}\n'
          f'\tcompletion_tokens: {rag_agent.completion_tokens}\n'
          f'\ttotal_tokens: {rag_agent.total_tokens}\n')


if __name__ == '__main__':
    coding_task('8月销售额？', resource_dir_path='./data/data_analyse')
    # rag_task('意向供应商应提交哪些材料？', 'data/docs')
