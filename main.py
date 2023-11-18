import fcheck as factChecking
import streamlit as st


#sentenceFromUser = 'What is Python programming language?'
st.title("Fact Checking for llama")




promt = st.text_input('Введите сюда свой запрос в llama: ')


if st.button('start'):
    outputData = factChecking.run(promt)
    st.write(outputData)
