import os

from dotenv import find_dotenv, load_dotenv
from langchain_community.chat_models.fake import FakeListChatModel
from langchain_zhipu import ZhipuAIEmbeddings, ChatZhipuAI
from transformers import AutoTokenizer

load_dotenv(find_dotenv())
FINISH_NAME = 'finish'
ZHIPU_API_KEY = os.environ.get('ZHIPU_API_KEY')


def get_fake_llm():
    return FakeListChatModel(name='FakeLLM', responses=[
        'Hello World!',
        'Hi',
        'How are you?',
        'How are you doing?',
        'How is it going?',
    ], i=3)


def get_zhipu_embeddings(**kwargs) -> ZhipuAIEmbeddings:
    return ZhipuAIEmbeddings(api_key=ZHIPU_API_KEY, **kwargs)


def get_local_chatglm3_6b_embeddings(model_path=r'D:\workspaces\python\chatglm3-6b-tokenizer',
                                     **kwargs) -> AutoTokenizer:
    """
    从https://modelscope.cn/models/ZhipuAI/chatglm3-6b/files中下载以下3个文件，放到指定文件夹中，再加载
    1. tokenization_chatglm.py
    2. tokenizer.model
    3. tokenizer_config.json
    """
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, **kwargs)


def get_zhipu_chat_llm(model_name="glm-4",
                       max_tokens=1000,
                       tokenizer=None,
                       **kwargs) -> ChatZhipuAI:
    llm = ChatZhipuAI(
        api_key=ZHIPU_API_KEY,
        model=model_name,
        max_tokens=max_tokens,
        **kwargs
    )
    tokenizer = tokenizer if tokenizer else get_local_chatglm3_6b_embeddings()
    llm.__class__.get_num_tokens = lambda self, x: len(tokenizer.encode(x))
    return llm


class Test:
    @staticmethod
    def test_get_fake_llm():
        llm = get_fake_llm()
        print(llm.invoke('hello'))

    @classmethod
    def test(cls):
        cls.test_get_fake_llm()


if __name__ == '__main__':
    Test.test()
