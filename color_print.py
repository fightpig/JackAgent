import datetime
import enum
import os
from pathlib import Path
from typing import List, Union

from colorama import Style, Fore
from langchain_core.messages import BaseMessage


def color_print(text, color=None, end="\n"):
    print(color + text + Style.RESET_ALL, end=end, flush=True) if color else print(text, end=end, flush=True)


class ColorPrint:
    """带各种颜色的console输出"""

    class PrintLevel(enum.IntEnum):
        DISABLE = -1  # 都不打印
        ALL = 0  # 打印所有
        PROMPT = 1
        REPLY = 2
        TOKEN = 3
        THOUGHT = 4
        ACTION = 5
        FINISH = 6
        RESULT = 7

    class Color(enum.Enum):
        PROMPT = Fore.MAGENTA
        REPLY = Fore.LIGHTYELLOW_EX
        TOKEN = Fore.LIGHTRED_EX
        THOUGHT = Fore.YELLOW
        ACTION = Fore.BLUE
        FINISH = Fore.CYAN
        RESULT = Fore.GREEN

    def __init__(self,
                 print_prefix='',
                 print_level: PrintLevel | str | List[PrintLevel] = PrintLevel.ALL,
                 save_dir_path: Union[str, Path, bool, None] = True):
        """
        :param print_prefix:
        :param print_level:
            1. 若是str，数字0~7中的一个或多个组成，逗号分隔，如: '1,2'
            3. 若是'0'或PrintLevel.ALL，则表示所有都打印
        """
        self.print_prefix = print_prefix
        self.print_levels: list = (
            [print_level.value] if isinstance(print_level, self.PrintLevel)
            else (
                [pl.value for pl in print_level] if isinstance(print_level, list)
                else list(map(int, print_level.split(',')))
            )
        )
        self.save_dir_path = save_dir_path
        self._save_path = None

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = self.save_dir_path / f'{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.log'
        return self._save_path

    @staticmethod
    def print(text, color=None, end="\n"):
        color_print(text, color=color, end=end)

    def print_with_style(self,
                         print_level: PrintLevel,
                         response: BaseMessage | str | dict,
                         title: str,
                         color: Color,
                         style: str = None,
                         idx: int = None):
        if not style:
            title = title if not self.print_prefix else f'{self.print_prefix} - {title}'
            style = f'{">" * 33}{idx}. {title} {"<" * 33}' if idx is not None else f'{">" * 33} {title} {"<" * 33}'
        message = response.content if isinstance(response, BaseMessage) else response
        message = f'{style}\n{message}\n{style}'

        if self.save_dir_path:  # 保存log
            self.save_dir_path = Path(f'./log') if self.save_dir_path is True else Path(self.save_dir_path)
            os.makedirs(self.save_dir_path, exist_ok=True)
            with open(self.save_path, 'a+', encoding='utf') as f:
                f.write(message + '\n')

        if self.PrintLevel.DISABLE in self.print_levels:
            return

        if (
                self.PrintLevel.ALL.value in self.print_levels  # 打印所有
                or print_level.value in self.print_levels  # 打印指定
        ):
            self.print(f'{message}', color.value)

    def print_prompt(self,
                     response: BaseMessage | str,
                     color: Color = None,
                     style: str = None,
                     idx: int = None):
        color = self.Color.PROMPT if not color else color
        self.print_with_style(self.PrintLevel.PROMPT, response, 'prompt', color, style, idx)

    def print_reply(self,
                    response: BaseMessage | str,
                    color: Color = None,
                    style: str = None,
                    idx: int = None):
        color = self.Color.REPLY if not color else color
        self.print_with_style(self.PrintLevel.REPLY, response, 'reply', color, style, idx)

    def print_thought(self,
                      response: BaseMessage | str,
                      color: Color = None,
                      style: str = None,
                      idx: int = None):
        color = self.Color.THOUGHT if not color else color
        self.print_with_style(self.PrintLevel.THOUGHT, response, 'thought', color, style, idx)

    def print_action(self,
                     response: BaseMessage | str,
                     color: Color = None,
                     style: str = None,
                     idx: int = None):
        color = self.Color.ACTION if not color else color
        self.print_with_style(self.PrintLevel.ACTION, response, 'action', color, style, idx)

    def print_finish(self,
                     response: BaseMessage | str,
                     color: Color = None,
                     style: str = None,
                     idx: int = None):
        color = self.Color.FINISH if not color else color
        self.print_with_style(self.PrintLevel.FINISH, response, 'finish', color, style, idx)

    def print_result(self,
                     response: BaseMessage | str,
                     color: Color = None,
                     style: str = None,
                     idx: int = None):
        color = self.Color.RESULT if not color else color
        self.print_with_style(self.PrintLevel.RESULT, response, 'result', color, style, idx)

    def print_token(self,
                    response: BaseMessage | dict,
                    color: Color = None,
                    style: str = None,
                    idx: int = None):
        color = self.Color.TOKEN if not color else color
        message = response if isinstance(response, dict) else response.response_metadata.get("token_usage")
        self.print_with_style(self.PrintLevel.TOKEN, message, 'token消耗', color, style, idx)


def test():
    # 打印所有
    # cp = ColorPrint(print_prefix='Main', print_level=ColorPrint.PrintLevel.ALL)
    # cp.print_prompt('这是prompt！', idx=1)
    # cp.print_reply('这是reply！', idx=1)
    # cp.print_thought('这是thought！', idx=1)
    # cp.print_action('这是action！', idx=1)
    # cp.print_finish('这是finish！', idx=1)
    # cp.print_result('这是result！', idx=1)
    # cp.print_token({'prompt_tokens': 100, 'completion_tokens': 100, 'total_tokens': 200}, idx=1)

    # 打印reply
    # cp = ColorPrint(print_prefix='Main', print_level=ColorPrint.PrintLevel.REPLY)
    # cp.print_prompt('这是prompt！', idx=1)
    # cp.print_reply('这是reply！', idx=1)
    # cp.print_thought('这是thought！', idx=1)
    # cp.print_action('这是action！', idx=1)
    # cp.print_finish('这是finish！', idx=1)
    # cp.print_result('这是result！', idx=1)
    # cp.print_token({'prompt_tokens': 100, 'completion_tokens': 100, 'total_tokens': 200}, idx=1)

    # 打印prompt, action, tokens
    # cp = ColorPrint(print_prefix='Main', print_level='1,3,5')
    # cp.print_prompt('这是prompt！', idx=1)
    # cp.print_reply('这是reply！', idx=1)
    # cp.print_thought('这是thought！', idx=1)
    # cp.print_action('这是action！', idx=1)
    # cp.print_finish('这是finish！', idx=1)
    # cp.print_result('这是result！', idx=1)
    # cp.print_token({'prompt_tokens': 100, 'completion_tokens': 100, 'total_tokens': 200}, idx=1)

    # 打印token, result
    # cp = ColorPrint(print_prefix='Main', print_level=[ColorPrint.PrintLevel.TOKEN, ColorPrint.PrintLevel.RESULT])
    # cp.print_prompt('这是prompt！', idx=1)
    # cp.print_reply('这是reply！', idx=1)
    # cp.print_thought('这是thought！', idx=1)
    # cp.print_action('这是action！', idx=1)
    # cp.print_finish('这是finish！', idx=1)
    # cp.print_result('这是result！', idx=1)
    # cp.print_token({'prompt_tokens': 100, 'completion_tokens': 100, 'total_tokens': 200}, idx=1)

    # 全部不打印
    cp = ColorPrint(print_prefix='Main', print_level=ColorPrint.PrintLevel.DISABLE, save_dir_path=True)
    cp.print_prompt('这是prompt！', idx=1)
    cp.print_reply('这是reply！', idx=1)
    cp.print_thought('这是thought！', idx=1)
    cp.print_action('这是action！', idx=1)
    cp.print_finish('这是finish！', idx=1)
    cp.print_result('这是result！', idx=1)
    cp.print_token({'prompt_tokens': 100, 'completion_tokens': 100, 'total_tokens': 200}, idx=1)


if __name__ == '__main__':
    test()
