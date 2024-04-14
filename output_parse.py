from typing import Tuple, Union

from langchain_core.output_parsers import PydanticOutputParser
from pydantic.v1 import BaseModel, Field


class Thoughts(BaseModel):
    text: str = Field(description='思考')
    reasoning: str = Field(description='推理')
    plan: str = Field(description='- 简短的要点\n- 列表，表达\n- 长期计划')
    criticism: str = Field(description='建设性的自我批评')
    speak: str = Field(description='给用户的思考摘要')


class Command(BaseModel):
    """
    # 自定义输出格式：PydanticOutputParser
    """
    name: str = Field(description='command name', example='list_directory')
    args: dict = Field(description='command arguments', exclude=dict(dir_path='./data'))


class Response(BaseModel):
    thoughts: Thoughts = Field(description='思考与规划')
    command: Command = Field(description='要执行的操作')
    state: Tuple[bool, Union[str, None]] = Field(description='记录解析大模型的返回结果是否异常', default=(True, None))


if __name__ == '__main__':
    parser = PydanticOutputParser(pydantic_object=Response)
    print(parser.get_format_instructions())
