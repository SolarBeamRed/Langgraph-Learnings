from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt1 = PromptTemplate(
    template='Mention a few good things about the following topic in 80 to 100 words.\nTopic: {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Understand the topic being talked about in the given text, and argue against the positive things bought up in the text in about 80 to 100 words.\nText:\n{text}',
    input_variables=['text']
)

llm = ChatOpenAI(
    model = 'local qwen3.5:4b',
    base_url='http://127.0.0.1:9931/v1',
    api_key='Not needed'                    # type: ignore
)

parser = StrOutputParser()

chain = prompt1 | llm | parser | prompt2 | llm | parser

user_input = input('Enter a topic to test out: ')
result = chain.invoke({'topic': user_input})

print(result)
