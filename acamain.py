import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


st.set_page_config(page_title="Your Friendly Math Tutor")
st.title("Your Friendly Math Tutor")
openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

prompt = """
Act as a tutor that helps students solve math and arithmetic reasoning questions.
Students will ask you questions. Think step-by-step to reach the answer. Write down each reasoning step.
You will be asked to show the answer or give clues that help students reach the answer on their own.

Here are a few example questions with expected answer and clues:

Question: John has 2 houses. Each house has 3 bedrooms and there are 2 windows in each bedroom.
Each house has 1 kitchen with 2 windows. Also, each house has 5 windows that are not in the bedrooms or kitchens.
How many windows are there in John's houses?
Answer: Each house has 3 bedrooms with 2 windows each, so that's 3 x 2 = 6 windows per house. \
Each house also has 1 kitchen with 2 windows, so that's 2 x 1 = 2 windows per house. \
Each house has 5 windows that are not in the bedrooms or kitchens, so that's 5 x 1 = 5 windows per house. \
In total, each house has 6 + 2 + 5 = 13 windows. \
Since John has 2 houses, he has a total of 2 x 13 = 26 windows. The answer is 26.
Clues: 1. Find the number of bedroom windows, kitchen windows, and other windows separately \
2. Add them together to find the total number of windows at each house \
3. Find the total number of windows for all the houses.

Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer: There are originally 15 trees. After the workers plant some trees, \
there are 21 trees. So the workers planted 21 - 15 = 6 trees. The answer is 6.",
Clues: 1. Start with the total number of trees after planting and subtract the original \
number of trees to find how many were planted. \
2. Use subtraction to find the difference between the two numbers.

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total? 
Answer: Originally, Leah had 32 chocolates. Her sister had 42. \
So in total they had 32 + 42 = 74. After eating 35, they \
had 74 - 35 = 39. The answer is 39.
Clues: 1. Start with the total number of chocolates they had. \
    2. Subtract the number of chocolates they ate.

Question: {question}
"""


def generate_response(question):
    chat = ChatOpenAI(temperature=0.0, openai_api_key=openai_api_key)
    prompt_template = ChatPromptTemplate.from_template(template=prompt)
    messages = prompt_template.format_messages(
        question=question
    )
    response = chat(messages)
    return response.content


with st.form('myform'):
    question = st.text_input('Enter question:', '')
    clues = st.form_submit_button('Give me clues')
    answer = st.form_submit_button('Show me the answer')
    if not openai_api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
    if clues and openai_api_key.startswith('sk-'):
        st.info(generate_response(question).split("Clues")[1][2:])
    if answer and openai_api_key.startswith('sk-'):
        st.info(generate_response(question).split("Clues")[0])
