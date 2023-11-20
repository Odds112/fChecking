import fcheck as factChecking
import streamlit as st
import os
import json


def load_serp(directory):
    serp_data = []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            with open(f, 'r') as js:
                data = json.load(js)
                serp_data.append(data)
    return serp_data


clean_sentences_bears = ['Polar bears are amazing animals that live in the Arctic.',
                         "Polar bears have thick fur and a layer of fat to keep them warm in the cold climate.",
                         'Polar bears hunt seals for food, and can swim very well.',
                         'Polar bears are big, with adult males weighing up to 1,500 pounds!',
                         'Polar bears have a long neck and powerful legs that help them move across the snow and ice.']

bears_data = load_serp('data/polar_bears')


bears = [{'querry': sentence, 'google': serp}
         for sentence, serp in zip(clean_sentences_bears, bears_data)]

# sentenceFromUser = 'What is Python programming language?'
st.title("Fact Checking for llama")

factChecking.run_no_google(bears)

# На данном этапе пока не нужно давать пользователю отправлять произвольные запросы к LLM, сделай selectbox


# promt = st.text_input('Введите сюда свой запрос в llama: ')


# if st.button('start'):
# outputData = factChecking.run(promt)
# st.write(outputData)
