from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_core.output_parsers import PydanticOutputParser

from color_print import ColorPrint
from models import get_zhipu_chat_llm
from output_parse import Response
from tools import get_tools
from agent import AutoAgent


def launch():
    llm = get_zhipu_chat_llm(max_tokens=2000, temperature=0.1)
    main_color_print = ColorPrint(print_prefix='Main')
    excel_analyse_color_print = ColorPrint(print_prefix='Excel analyse')
    tools = get_tools(
        llm=llm,
        excel_analyser_prompt_path='prompt/tools/excel_analyse.json',
        color_print=excel_analyse_color_print
    )

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

    agent.run(task='8月销售额', resource_dir_path='./data')
    print(f'本次消耗token：\n'
          f'\tprompt_tokens: {agent.prompt_tokens}\n'
          f'\tcompletion_tokens: {agent.completion_tokens}\n'
          f'\ttotal_tokens: {agent.total_tokens}\n')


if __name__ == '__main__':
    launch()
