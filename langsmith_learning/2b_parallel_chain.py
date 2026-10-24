import os

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

os.environ['LANGCHAIN_PROJECT'] = 'Parallel Chain Test'

prompt1 = PromptTemplate(
    template='Mention a few good things about the following topic in 80 to 100 words.\nTopic: {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Mention the problems and negative outcomes because of the occurence of the following topic in 80 to 100 words.\nTopic::\n{topic}',
    input_variables=['topic']
)

llm = ChatOpenAI(
    model = 'local qwen3.5:4b',
    base_url='http://127.0.0.1:9931/v1',
    api_key='Not needed'                    # type: ignore
)

parser = StrOutputParser()

chain = RunnableParallel({
    'positives': prompt1 | llm | parser,
    'negatives': prompt2 | llm | parser
})

config = {
    'runnable_name': 'parallel_chain',
    'tags': ['parallel', 'test', '123checkcheck'],
    'metadata': {'model': 'local qwen3.5:4b', 'all_local?': 'yes'}
}

user_input = input('Enter a topic to test out: ')
result = chain.invoke({'topic': user_input}, config=config)

print(result)
