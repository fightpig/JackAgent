import time
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union, Dict

from langchain.schema import (
    BaseChatMessageHistory,
)
from langchain.schema import Document
from langchain.tools.base import BaseTool
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.messages.ai import AIMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever
from pydantic_core import ValidationError

from color_print import ColorPrint
from output_parse import Response, Thoughts, Command
from prompt_template import RecursivePromptTemplateConfigParser
from utils import format_prompts_with_number, FINISH_NAME


class BaseAgent(ABC):
    def __init__(self,
                 ai_name: Optional[str] = None,
                 ai_role: Optional[str] = None,
                 *,
                 llm: BaseChatModel,
                 base_prompt_config_path: str,
                 tools: Optional[List[BaseTool]] = None,
                 output_parser: Optional[JsonOutputParser],
                 short_chat_memory_obj: Optional[BaseChatMessageHistory] = ChatMessageHistory(),
                 long_chat_vdb_obj: Optional[VectorStoreRetriever] = None,
                 docs_vdb_obj: Optional[VectorStoreRetriever] = None,
                 max_send_tokens: int = 4500,
                 max_short_chat_tokens: int = 800,
                 max_long_chat_tokens: int = 400,
                 max_docs_tokens: int = 1200,
                 color_print: ColorPrint,
                 max_reason_cn: int = 10,
                 ):
        self.ai_name = ai_name
        self.ai_role = ai_role
        self.llm = llm
        self.base_prompt_config_path = base_prompt_config_path
        self.tools = tools
        self.output_parser = output_parser
        self.short_chat_memory_obj = short_chat_memory_obj
        self.long_chat_vdb_obj = long_chat_vdb_obj
        self.docs_vdb_obj = docs_vdb_obj
        self.color_print = color_print
        self.max_send_tokens: int = max_send_tokens
        self.max_short_chat_tokens = max_short_chat_tokens
        self.max_long_chat_tokens = max_long_chat_tokens
        self.max_docs_tokens: int = max_docs_tokens
        self.max_reason_cn = max_reason_cn

        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.base_tokens: int = -1

        partial_format = {
            'tools': (format_prompts_with_number, {'items': tools, 'item_type': 'command', 'language': 'zh'})
        }
        prompt = (RecursivePromptTemplateConfigParser(self.base_prompt_config_path)).parse(partial_format)
        self.chain = prompt | llm  # self.chain = LLMChain(llm=llm, prompt=prompt)

    def token_counter(self, txt):
        """计算token数量"""
        return self.chain.last.get_num_tokens(txt)

    def update_tokens(self, ai_reply: BaseMessage | dict):
        ai_reply = ai_reply.response_metadata.get("token_usage") if isinstance(ai_reply, BaseMessage) else ai_reply
        self.prompt_tokens += int(ai_reply.get('prompt_tokens'))
        self.completion_tokens += int(ai_reply.get('completion_tokens'))
        self.total_tokens += int(ai_reply.get('total_tokens'))

    def calc_base_prompt_tokens(self, parameters: dict, ignore_title_keys: Optional[Dict[str, str]]):
        """计算base prompt数理"""
        if self.base_tokens == -1:
            for value in ignore_title_keys.values():
                parameters[value] = value
            prompt = self.chain.first.format(**parameters)
            prompt = prompt.split('\n')
            new_prompt = list()
            length = len(prompt)
            idx = 0
            while idx < length:
                add = True
                for title, value in ignore_title_keys.items():
                    if prompt[idx].startswith(title):
                        add = False
                        break
                if add:
                    new_prompt.append(prompt[idx])
                    idx += 1
                else:
                    idx += 2

            prompt = '\n'.join(new_prompt)
            self.base_tokens = self.token_counter(prompt)

    def update_memory(self,
                      query: str,
                      short_chat_memory_obj: BaseChatMessageHistory,
                      long_chat_vdb_obj: Optional[VectorStoreRetriever] = None,
                      docs_vdb_obj: Optional[VectorStoreRetriever] = None,
                      use_chat_num: int = 12
                      ) -> Tuple[int, str, str, str]:
        """
        默认保存前3轮对话
        ConversationTokenBufferMemory
        """

        def _update_long_memory(vector_store_retriever: VectorStoreRetriever, query_: str, max_tokens: int):
            docs = vector_store_retriever.get_relevant_documents(query_)  # TODO 比较条件
            page_contents = [d.page_content for d in docs]
            while (tokens := sum([self.token_counter(page_content) for page_content in page_contents])) > max_tokens:
                page_contents = page_contents[:-1]  # 去掉最后一条记录
            return page_contents, tokens

        # 1. 文档向量库
        if docs_vdb_obj:
            relevant_docs, docs_tokens = _update_long_memory(docs_vdb_obj, query, self.max_docs_tokens)
            relevant_docs = f'{relevant_docs}'
        else:
            relevant_docs = '暂时没有'
            docs_tokens = self.token_counter(relevant_docs)

        # 2. 长期对话向量库
        if long_chat_vdb_obj:
            long_chat_messages = short_chat_memory_obj.messages[-use_chat_num * 2:-use_chat_num][::-1]
            long_chat_docs, long_chat_tokens = _update_long_memory(
                long_chat_vdb_obj, str(long_chat_messages), self.max_long_chat_tokens
            )
            long_chat_docs = f'{long_chat_docs}'
        else:
            long_chat_docs = '暂时没有'
            long_chat_tokens = self.token_counter(long_chat_docs)

        # 3. 短期对话记录
        short_chat_messages = short_chat_memory_obj.messages[-use_chat_num:][::-1].copy()
        if not short_chat_messages:
            short_chat_memory = '暂时没有'
            short_chat_tokens = self.token_counter(short_chat_memory)
        else:
            keep_short_chat_messages: List[BaseMessage] = list()
            short_chat_tokens = 0
            for message in short_chat_messages:
                short_chat_tokens = self.token_counter(message.content)
                if short_chat_tokens > self.max_short_chat_tokens:
                    break
                keep_short_chat_messages.insert(0, message)
            short_chat_memory = ChatPromptTemplate.from_messages(keep_short_chat_messages).format()

        used_tokens = self.base_tokens + docs_tokens + long_chat_tokens + short_chat_tokens
        return used_tokens, relevant_docs, long_chat_docs, short_chat_memory

    def run(self, **kwargs) -> Tuple[str, dict, float]:
        result, cost = self.loop(**kwargs)
        self.color_print.print_result(f'任务执行结果：{result}，耗时: {cost:.2f}秒')
        tokens = {
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'total_tokens': self.total_tokens
        }
        return result, tokens, cost

    def reason(self,
               task: str,
               loop_cn: int,
               other_parameters: dict,
               add_message=True,
               use_chat_num: int = 12
               ) -> BaseMessage:
        """请求大模型"""
        parameters = dict(
            task=task,
            relevant_docs=None,
            long_chat_docs=None,
            short_chat_memory=None
        )
        parameters.update(other_parameters)
        self.calc_base_prompt_tokens(
            parameters,
            ignore_title_keys={
                '相关文档内容': 'relevant_docs',
                '长时任务记录': 'long_chat_docs',
                '短时任务记录': 'short_chat_memory'
            }
        )

        used_tokens, relevant_docs, long_chat_docs, short_chat_memory = self.update_memory(
            task, self.short_chat_memory_obj, self.long_chat_vdb_obj, self.docs_vdb_obj, use_chat_num
        )
        parameters.update(dict(
            relevant_docs=relevant_docs,
            long_chat_docs=long_chat_docs,
            short_chat_memory=short_chat_memory,
        ))
        self.color_print.print_prompt(
            self.chain.first.format(**parameters) + f'\n\n预计消耗token: {used_tokens}', idx=loop_cn
        )
        ai_reply: BaseMessage = self.chain.invoke(parameters)
        self.update_tokens(ai_reply)
        self.color_print.print_token(ai_reply, idx=loop_cn)
        self.color_print.print_reply(ai_reply, idx=loop_cn)

        if add_message:
            self.short_chat_memory_obj.add_message(
                SystemMessage(content=f"当前日期时间是 {time.strftime('%Y-%m-%d %H:%M:%S')}")
            )
            self.short_chat_memory_obj.add_message(
                HumanMessage(content='确定要使用的下一个命令/操作，并使用上面指定的格式进行回复。')
            )
            self.short_chat_memory_obj.add_message(AIMessage(content=ai_reply.content))
        return ai_reply

    @abstractmethod
    def action(self, ai_reply: BaseMessage, loop_cn: int) -> Union[str, None]:
        pass

    def loop(self, **kwargs) -> Tuple[str, float]:
        loop_cn = 1
        result = None
        s = time.time()
        while loop_cn <= self.max_reason_cn:
            kwargs['loop_cn'] = loop_cn
            ai_reply = self.reason(**kwargs)
            state, result = self.action(ai_reply, loop_cn)
            if state is True:
                break
            loop_cn += 1
        e = time.time()
        cost = e - s
        if loop_cn > self.max_reason_cn and result is None:
            result = f'超过最大{self.max_reason_cn}次限制次数，本次任务失败'
        return result, cost

    def output_parse(self, ai_reply: BaseMessage) -> Response:
        try:
            return self.output_parser.parse(ai_reply.content)
        except Exception as e:
            return Response(
                thoughts=Thoughts(
                    text='',
                    reasoning='',
                    plan='',
                    criticism='',
                    speak='',
                ),
                command=Command(
                    name='',
                    args=dict()
                ),
                state=(False, f'解析{ai_reply.content}，\n报错：{e}')
            )


class AutoAgent(BaseAgent):
    def __init__(self,
                 ai_name: Optional[str] = None,
                 ai_role: Optional[str] = None,
                 *,
                 llm: BaseChatModel,
                 base_prompt_config_path: str,
                 tools: Optional[List[BaseTool]] = None,
                 output_parser: JsonOutputParser,
                 short_chat_memory_obj: Optional[BaseChatMessageHistory] = ChatMessageHistory(),
                 long_chat_vdb_obj: Optional[VectorStoreRetriever] = None,
                 docs_vdb_obj: Optional[VectorStoreRetriever] = None,
                 max_send_tokens: int = 4500,
                 max_short_chat_tokens: int = 800,
                 max_long_chat_tokens: int = 400,
                 max_docs_tokens: int = 1200,
                 color_print: ColorPrint,
                 max_reason_cn: int = 10,
                 ):
        super().__init__(ai_name,
                         ai_role,
                         llm=llm,
                         base_prompt_config_path=base_prompt_config_path,
                         tools=tools,
                         output_parser=output_parser,
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

    def reason(self, task, resource_dir_path, loop_cn, add_message=True, use_chat_num=12) -> BaseMessage:
        """请求大模型"""
        parameters = dict(
            ai_name=self.ai_name,
            ai_role=self.ai_role,
            resource_dir_path=resource_dir_path,
        )
        return super().reason(task, loop_cn, parameters, add_message, use_chat_num)

    def action(self, ai_reply: BaseMessage, loop_cn: int) -> Tuple[bool, Union[str, None]]:
        response = self.output_parse(ai_reply)
        if response.state[0] is False:
            result = response.state[1]
        else:
            thoughts = response.thoughts
            action = response.command
            # self.color_print.print_thought(thoughts, idx=loop_cn)
            # self.color_print.print_action(action, idx=loop_cn)

            tools = {t.name: t for t in self.tools}
            if action.name == FINISH_NAME:
                response = action.args['response']
                self.color_print.print_finish(response, idx=loop_cn)
                return True, response

            if action.name in tools:
                tool = tools[action.name]
                try:
                    observation = tool.run(action.args)
                    if action.name == 'AnalyseExcel':
                        observation, tokens, cost = observation
                        self.update_tokens(tokens)
                except ValidationError as e:
                    observation = f'参数Validation Error: {str(e)}, args: {action.args}'
                except Exception as e:
                    observation = f'Error: {str(e)}, {type(e).__name__}, args: {action.args}'
                result = f'Command: {tool.name}, Returned: {observation}'
            elif action.name == 'ERROR':
                result = f'Error: {action.args}'
            else:
                result = f"未知Command: {action.name}, 请参阅COMMANDS列表以获取可用的”命令“，并且只以指定的JSON格式响应"

        if self.long_chat_vdb_obj:
            memory_to_add = f"Assistant Reply: {ai_reply.content} \nResult: {result} "
            self.long_chat_vdb_obj.add_documents([Document(page_content=memory_to_add)])
        self.short_chat_memory_obj.add_message(SystemMessage(content=result))
        self.color_print.print_result(result, idx=loop_cn)
        return False, None
