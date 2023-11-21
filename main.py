import llm_data
import no_google
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

# # Если версия Python > 3.9
# # def start_option(selected_option):
# #     match selected_option:
# #         case 'Polar bears':
# #             bears_data = load_serp('data/polar_bears')
# #             bears = [{'querry': sentence, 'google': serp}
# #                     for sentence, serp in zip(llm_data.polar_bears, bears_data)]
# #             return bears
# #         case 'operation overlord':
# #             datajs = load_serp('data/overlords')
# #             ready_data = [{'querry': sentence, 'google': serp}
# #                     for sentence, serp in zip(llm_data.overlord, datajs)]
# #             return ready_data
# #         case 'football player Arshavin':
# #             datajs = load_serp('data/footballs')
# #             ready_data = [{'querry': sentence, 'google': serp}
# #                           for sentence, serp in zip(llm_data.arshavin, datajs)]
# #             return ready_data
# #         case 'New-Jersey':
# #             datajs = load_serp('data/jersey')
# #             ready_data = [{'querry': sentence, 'google': serp}
# #                           for sentence, serp in zip(llm_data.new_jersey, datajs)]
# #             return ready_data
# #         case 'OS Ubuntu':
# #             datajs = load_serp('data/ubuntu')
# #             ready_data = [{'querry': sentence, 'google': serp}
# #                           for sentence, serp in zip(llm_data.ubuntu, datajs)]
# #             return ready_data
#
def start_option(selected_option):
    if selected_option == 'Polar bears':
        bears_data = load_serp('data/polar_bears')
        bears = [{'querry': sentence, 'google': serp}
            for sentence, serp in zip(llm_data.polar_bears, bears_data)]
        return bears
    if selected_option == 'operation overlord':
            datajs = load_serp('data/overlords')
            ready_data = [{'querry': sentence, 'google': serp}
                    for sentence, serp in zip(llm_data.overlord, datajs)]
            return ready_data
    if selected_option == 'football player Arshavin':
            datajs = load_serp('data/footballs')
            ready_data = [{'querry': sentence, 'google': serp}
                          for sentence, serp in zip(llm_data.arshavin, datajs)]
            return ready_data
    if selected_option == 'New-Jersey':
            datajs = load_serp('data/jersey')
            ready_data = [{'querry': sentence, 'google': serp}
                          for sentence, serp in zip(llm_data.new_jersey, datajs)]
            return ready_data
    if selected_option == 'OS Ubuntu':
            datajs = load_serp('data/ubuntu')
            ready_data = [{'querry': sentence, 'google': serp}
                          for sentence, serp in zip(llm_data.ubuntu, datajs)]
            return ready_data


st.title("Fact Checking for llama")



va = ['Polar bears','operation overlord','football player Arshavin', 'New-Jersey', 'OS Ubuntu']

option = st.selectbox(
    'Выберите то, о чем вы хотите узнать: ', va)

if st.button('Start'):
    ready_sen = no_google.run_no_google(start_option(option))
    for x in ready_sen:
        st.write(x)