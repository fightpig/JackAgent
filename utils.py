import io
import json
import traceback
from contextlib import redirect_stdout
from typing import Dict, Tuple

from langchain_core.tools import BaseTool
from langchain_core.tools import tool

FINISH_NAME = 'finish'


def format_prompt_command(tool: BaseTool) -> str:
    output = (f"{tool.name}: {tool.description}, "
              f"args json schema: {json.dumps(tool.args)}")
    return output


def format_prompts_with_number(items: list, item_type: str, language='zh') -> str:
    if item_type == "command":
        command_strings = [
            f"{i + 1}. {format_prompt_command(item)}"
            for i, item in enumerate(items)
        ]
        finish_description = (
            "use this to signal that you have finished all your objectives" if language == 'en'
            else "当你完成你所有的目标，请使用这个命令"
        )
        finish_args = (
            '"response": "final response to let people know you have finished your objectives"' if language == 'en'
            else '"response": "最后的 response 是让人们知道你已经完成了你的所有目标"'
        )
        finish_string = (
            f"{len(items) + 1}. {FINISH_NAME}: "
            f"{finish_description}, args: {finish_args}"
        )
        return "\n".join(command_strings + [finish_string])
    else:
        return "\n".join(f"{i + 1}. {item}" for i, item in enumerate(items))


# def exec_python_code(code: str) -> str:
#     # 注意，这是一个不安全的操作，因为用户可以输入任意的Python代码
#     # 实际生产中为了安全起见，可以使用docker隔离执行环境
#     output = io.StringIO()  # 创建一个StringIO对象来捕获输出
#     old_stdout = sys.stdout  # 保存当前的stdout，以便之后可以恢复它
#     sys.stdout = output  # 重定向stdout到StringIO对象
#
#     try:
#         exec(code)
#     # except Exception as e:
#     #     print(e)
#     finally:
#         # 恢复原来的stdout
#         sys.stdout = old_stdout
#     # 获取从StringIO对象中捕获的输出
#     return output.getvalue()


def exec_python_code(code: str, g_context: Dict = globals(), l_context: Dict = locals()) -> Tuple[bool, str]:  # noqa
    io_buffer = io.StringIO()
    state = True
    try:
        with redirect_stdout(io_buffer):
            exec(code, g_context, l_context)
    except Exception as e:  # noqa
        with redirect_stdout(io_buffer):
            print(f'{traceback.format_exception_only(e)}')
            state = False
    return state, io_buffer.getvalue()


class Test:
    @staticmethod
    def test_format_prompts_with_number():
        from langchain_community.agent_toolkits.file_management import FileManagementToolkit
        tools = FileManagementToolkit(root_dir=".").get_tools()
        print(format_prompts_with_number(tools, 'command'))


if __name__ == '__main__':
    # re = exec_python_code('i am')
    # print(1, re)
    Test.test_format_prompts_with_number()
