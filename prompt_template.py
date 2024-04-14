import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Callable, Tuple, Optional, List, Union

import yaml
from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain_community.agent_toolkits.file_management import FileManagementToolkit
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_core.prompts.loading import load_prompt_from_config


class PromptTemplateConfigParser:
    def __init__(self, base_prompt_template_config_path: str | Path, name='base'):
        self.base_prompt_template_config_path = Path(base_prompt_template_config_path)
        self.config_suffix = self.base_prompt_template_config_path.suffix
        assert self.config_suffix in ['.json', '.yaml']
        self.prompt_template_dir_path = self.base_prompt_template_config_path.parent
        self.name = name

    def get_absolute_path(self, path: str | Path) -> Path:
        return self.prompt_template_dir_path / path if not Path(path).is_absolute() else Path(path)

    def load_config(self, prompt_template_config_path: str | Path) -> Dict[str, Any]:
        """加载模板配置文件，并把txt模板的路径更新为绝对路径"""
        with open(prompt_template_config_path, 'r', encoding='utf-8') as f:
            if self.config_suffix == '.json':
                config = json.load(f)
            else:
                config = yaml.load(f.read(), Loader=yaml.FullLoader)

        template_path = config['template_path']
        config['template_path'] = str(self.get_absolute_path(template_path))
        return config

    @staticmethod
    def load_txt(prompt_template_txt_path: str | Path) -> str:
        with open(prompt_template_txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content

    def parse(self, partial_format: Optional[Dict[str, Tuple[Callable, Dict[str, Any]]]] = None) -> BasePromptTemplate:
        """
        :param partial_format:
            {'tools': (format_func, parameters)}
        :return:
        """
        config = self.load_config(self.base_prompt_template_config_path)
        base_pt: BasePromptTemplate = load_prompt_from_config(config)
        base_pt.name = self.name

        partial_pts = list()
        partial_values = dict()

        for var in base_pt.input_variables:
            json_config = self.get_absolute_path(f'{var}.json')
            yaml_config = self.get_absolute_path(f'{var}.yaml')
            txt_prompt = self.get_absolute_path(f'{var}.txt')

            if (
                    (json_config.exists() and txt_prompt.exists())
                    or (yaml_config.exists() and txt_prompt)
            ):  # prompt模板
                # 解析嵌套的prompt config
                config_path = json_config if json_config.exists() else yaml_config
                partial_pp = self.__class__(config_path, name=var).parse(partial_format)
                partial_pts.append(partial_pp)
                continue

            if txt_prompt.exists():  # prompt txt
                partial_values[var] = self.load_txt(txt_prompt)

            if partial_format and var in partial_format:  # 专门针对某些变量的值进行格式化
                parameters: dict
                format_func: Callable
                format_func, parameters = partial_format[var]
                partial_values[var] = format_func(**parameters) if parameters else format_func()

        # 将PipelinePromptTemplate转为PromptTemplate
        # 因为PipelinePromptTemplate的pipeline_prompts接收的是PromptTemplate
        for idx in range(len(partial_pts)):
            spt = partial_pts[idx]
            spt = self.to_prompt_template(spt)
            partial_pts[idx] = spt

        base_pt = base_pt.partial(**partial_values)
        base_pt = PipelinePromptTemplate(
            final_prompt=base_pt,
            pipeline_prompts=[(spt.name, spt) for spt in partial_pts]
        )

        return base_pt

    @staticmethod
    def to_prompt_template(pt: BasePromptTemplate) -> BasePromptTemplate:
        if isinstance(pt, PipelinePromptTemplate):
            if not pt.pipeline_prompts and not pt.input_variables:
                pt = pt.final_prompt
        return pt


class RecursivePromptTemplate:
    def __init__(
            self,
            base_prompt: BasePromptTemplate,
            partial_prompts: Union[
                List[
                    Tuple[str, Union[BasePromptTemplate, 'RecursivePromptTemplate']]
                ],
                Dict[
                    str, Union[BasePromptTemplate, 'RecursivePromptTemplate']
                ]
            ]
    ):
        """
        base_prompt: The final prompt that is returned.
        partial_prompts: A list of tuples, consisting of a string (`name`) and a Prompt Template.
        """
        self.base_prompt = base_prompt
        self.partial_prompts = partial_prompts
        self.template_format = base_prompt.template_format

    def to_variable(self, var: str) -> str:
        return '{' + var + '}' if self.template_format == 'f-string' else '{{' + var + '}}'

    def build(self) -> BasePromptTemplate:
        partial_prompts: Dict[
            str,
            Union[PromptTemplate, RecursivePromptTemplate]
        ] = (
            {name: prompt for name, prompt in self.partial_prompts} if isinstance(self.partial_prompts, list)
            else self.partial_prompts
        )

        if not partial_prompts:
            return self.base_prompt

        var_pps = dict()
        pp_partial_variables = dict()
        pp_input_variables = list()
        remove_vars = list()
        for var in self.base_prompt.input_variables:
            if var in partial_prompts:
                pp = partial_prompts.get(var)
                if isinstance(pp, self.__class__):
                    pp = pp.build()
                var_pps[var] = pp.template
                pp_partial_variables.update(pp.partial_variables)
                pp_input_variables.extend(pp.input_variables)
                remove_vars.append(var)
            else:
                var_pps[var] = self.to_variable(var)

        partial_variables = self.base_prompt.partial_variables.copy()  # noqa
        self.base_prompt.partial_variables = {key: self.to_variable(key) for key in partial_variables.keys()}
        base_prompt = PromptTemplate.from_template(self.base_prompt.format(**var_pps))
        base_prompt.name = self.base_prompt.name

        base_prompt.partial_variables = partial_variables.copy()
        self.base_prompt.partial_variables = partial_variables

        base_prompt.input_variables = self.base_prompt.input_variables.copy()
        [base_prompt.input_variables.remove(var) for var in remove_vars]
        base_prompt.input_variables.extend(pp_input_variables)

        return base_prompt


class RecursivePromptTemplateConfigParser:
    def __init__(self, base_prompt_template_config_path: str | Path, name='base'):
        self.base_prompt_template_config_path = Path(base_prompt_template_config_path)
        self.config_suffix = self.base_prompt_template_config_path.suffix
        assert self.config_suffix in ['.json', '.yaml']
        self.prompt_template_dir_path = self.base_prompt_template_config_path.parent
        self.name = name

    def get_absolute_path(self, path: str | Path) -> Path:
        return self.prompt_template_dir_path / path if not Path(path).is_absolute() else Path(path)

    def load_config(self, prompt_template_config_path: str | Path) -> Dict[str, Any]:
        """加载模板配置文件，并把txt模板的路径更新为绝对路径"""
        with open(prompt_template_config_path, 'r', encoding='utf-8') as f:
            if self.config_suffix == '.json':
                config = json.load(f)
            else:
                config = yaml.load(f.read(), Loader=yaml.FullLoader)

        template_path = config['template_path']
        config['template_path'] = str(self.get_absolute_path(template_path))
        return config

    @staticmethod
    def load_txt(prompt_template_txt_path: str | Path) -> str:
        with open(prompt_template_txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content

    def parse(self, partial_format: Optional[Dict[str, Tuple[Callable, Dict[str, Any]]]] = None) -> BasePromptTemplate:
        """
        :param partial_format:
            {'tools': (format_func, parameters)}
        :return:
        """
        config = self.load_config(self.base_prompt_template_config_path)
        base_pt: BasePromptTemplate = load_prompt_from_config(config)
        base_pt.name = self.name

        partial_pts = list()
        partial_values = dict()

        for var in base_pt.input_variables:
            json_config = self.get_absolute_path(f'{var}.json')
            yaml_config = self.get_absolute_path(f'{var}.yaml')
            txt_prompt = self.get_absolute_path(f'{var}.txt')

            if (
                    (json_config.exists() and txt_prompt.exists())
                    or (yaml_config.exists() and txt_prompt)
            ):  # 解析嵌套的prompt config
                config_path = json_config if json_config.exists() else yaml_config
                partial_pt = self.__class__(config_path, name=var).parse(partial_format)
                partial_pts.append(partial_pt)
                continue

            if txt_prompt.exists():  # prompt txt
                partial_values[var] = self.load_txt(txt_prompt)

            if partial_format and var in partial_format:  # 专门针对某些变量的值进行格式化
                parameters: dict
                format_func: Callable
                format_func, parameters = partial_format[var]
                partial_values[var] = format_func(**parameters) if parameters else format_func()

        base_pt = base_pt.partial(**partial_values)
        base_pt = RecursivePromptTemplate(
            base_prompt=base_pt,
            partial_prompts=[(spt.name, spt) for spt in partial_pts]
        ).build()

        return base_pt


class Test:
    @staticmethod
    def test_recursive_prompt_template():
        base_template = """{introduction}

{example}

{start}

{tools}

{date}"""
        base_prompt = PromptTemplate.from_template(base_template)

        introduction_template = """You are impersonating {person}."""
        introduction_prompt = PromptTemplate.from_template(introduction_template)

        example_template = """Here's an example of an interaction:
Q: {example_q}
A: {example_a}

{sub_example}"""
        example_prompt = PromptTemplate.from_template(example_template)

        sub_example_template = """Another sub example of an interaction:
Q: {sub_example_q}
A: {sub_example_a}"""
        sub_example_prompt = PromptTemplate.from_template(sub_example_template)

        start_template = """Now, do this for real!
Q: {input}
A:"""
        start_prompt = PromptTemplate.from_template(start_template)

        tools_template = """
{tools}"""
        from utils import format_prompts_with_number
        tools = FileManagementToolkit(root_dir=".").get_tools()
        tools_prompt = PromptTemplate.from_template(tools_template)

        sub_partial_prompts = {
            'sub_example': sub_example_prompt
        }
        example_prompt = RecursivePromptTemplate(example_prompt, sub_partial_prompts)

        partial_prompts = {
            "introduction": introduction_prompt,
            "example": example_prompt,
            "start": start_prompt,
            "tools": tools_prompt,
        }

        base_prompt = RecursivePromptTemplate(base_prompt, partial_prompts).build()
        result = base_prompt.format(
            person="Elon Musk",
            example_q="What's your favorite car?",
            example_a="Tesla",
            input="What's your favorite social media site?",
            date=datetime.now(),
            sub_example_q='sub_example_q',
            sub_example_a='sub_example_a',
            tools=format_prompts_with_number(tools, item_type='command'),
        )

        print(result)

    @staticmethod
    def test_recursive_prompt_template_config_parser():
        from langchain_core.tools import tool
        from utils import format_prompts_with_number

        @tool
        def add(a: int, b: int):
            """返回两个数之和"""
            return a + b

        @tool
        def subtract(a: int, b: int):
            """返回两个数之差"""
            return a - b

        tools = FileManagementToolkit(root_dir=".").get_tools()
        tools.append(add)
        tools.append(subtract)

        partial_format = {
            'tools': (format_prompts_with_number, {'items': tools, 'item_type': 'command', 'language': 'zh'})
        }
        parser = RecursivePromptTemplateConfigParser('prompt/test/base.json')
        parameters = {
            'ai_name': '小聪明',
            'ai_role': '智能助手',
            'task': '你好啊',
            'resource_dir_path': './data',
            'long_term_memory': '待填',
            'chat_history_memory': '待填',
            'example_1_q': 'example_1_q',
            'example_1_a': 'example_1_a',
            'example_2_q': 'example_2_q',
            'example_2_a': 'example_2_a',
        }
        # parser = RecursivePromptTemplateConfigParser('learn/prompt_template/base.json')
        # parameters = dict(
        #     person="Elon Musk",
        #     example_q="What's your favorite car?",
        #     example_a="Tesla",
        #     input="What's your favorite social media site?",
        #     date=datetime.now(),
        #     sub_example_q='sub_example_q',
        #     sub_example_a='sub_example_a',
        #     tools='tools'
        # )
        prompt_template = parser.parse(partial_format)
        # print(prompt_template.input_variables)
        prompt = prompt_template.format(**parameters)
        print(prompt)

    @classmethod
    def test(cls):
        # cls.test_recursive_prompt_template()
        cls.test_recursive_prompt_template_config_parser()


if __name__ == '__main__':
    Test.test()
