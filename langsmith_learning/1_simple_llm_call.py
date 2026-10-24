from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Simple one-line prompt
prompt = PromptTemplate.from_template("{question}")

llm = ChatOpenAI(
    model = 'local qwen3.5:4b',
    base_url='http://127.0.0.1:9931/v1',
    api_key='Not needed'                    # type: ignore
)

parser = StrOutputParser()

chain = prompt | llm | parser


user_input = input('Enter a question: ')
result = chain.invoke({"question": user_input})
print(result)
