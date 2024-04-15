from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_community.tools.file_management import ListDirectoryTool, ReadFileTool
from langchain_core.output_parsers import PydanticOutputParser

from color_print import ColorPrint
from models import get_zhipu_chat_llm, get_local_chatglm3_6b_embeddings, get_zhipu_embeddings
from output_parse import Response
from toolobj import ToolObj, DocumentRetriever
from agent import AutoAgent


def launch(llm, task, tools, resource_dir_path):
    main_color_print = ColorPrint(print_prefix='Main')

    agent = AutoAgent(
        ai_name='小聪明',
        ai_role='智能助手',
        llm=llm,
        base_prompt_config_path='prompt/base/base.json',
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


def test_1():
    launch('8月销售额？', resource_dir_path='./data/data_analyse')


def test_2():
    llm = get_zhipu_chat_llm(max_tokens=2000, temperature=0.1)
    embeddings = get_zhipu_embeddings()
    tools = [
        ListDirectoryTool(),
        DocumentRetriever(
            llm=llm,
            embeddings=embeddings,
            chunk_size=100,
            chunk_overlap=10,
        ),
    ]
    launch(llm, '月销售额', tools, resource_dir_path='./data/docs')


if __name__ == '__main__':
    # test_1()
    test_2()
