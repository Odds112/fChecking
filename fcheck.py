import requests
import json
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer, util
# from serpapi import GoogleSearch
# import spacy
from llamaapi import LlamaAPI
# import neuralcoref
# from spacy import displacy
# from allennlp.predictors.predictor import Predictor
# import allennlp_models.tagging


# SRL_MODEL_PATH = "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz"


# nltk.download('punkt')


# intros = set(["Accordingly",
#               "Additionally",
#               "After all",
#               "Alas",
#               "As a result",
#               "Be that as it may",
#               "Behold",
#               "Besides",
#               "Certainly",
#               "Consequently",
#               "Correspondingly",
#               "Despite this",
#               "Equally",
#               "Finally",
#               "First of all",
#               "For example",
#               "Furthermore",
#               "Hence",
#               "Hitherto",
#               "However",
#               "In addition",
#               "In conclusion",
#               "In fact",
#               "In short"
#               "In spite of",
#               "Indeed",
#               "Instead",
#               "Likewise",
#               "Meanwhile",
#               "Moreover",
#               "Namely",
#               "Nevertheless"])
# punct = set(['.', '!', '?', '/n'])


# Зачем ты скопировал код функций, которые нигде не вызываются?


# def get_framegroups(sentence):
#     predictor = Predictor.from_path(SRL_MODEL_PATH)
#     groupes = predictor.predict(sentence)
#     words = groupes['words']
#     res = []
#     for i, group in enumerate(groupes['verbs']):
#         frames = {      }
#         for word, tag in zip(words, group['tags']):
#           if tag != 'O':
#             tag_name = tag[2::]
#             if tag_name  in frames:
#               frames[tag_name].append(word)
#             else:
#               frames[tag_name] = [word]
#         temp_res = []
#         for tag_name in frames.keys():
#           temp_res.append({
#               'group_id': i,
#               'tag': tag_name,
#               'text': ' '.join(frames[tag_name])
#           })
#         res.append(temp_res)
#     return res

# def framegroup_is_broken(framegroup):
#   return len(framegroup) < 2


# def get_embedded_framegroups(frame, framegroups):
#   embedded_framegroups = []
#   for framegroup in framegroups:
#     framegroup_text = [frame['text'] for frame in framegroup]

#     if all([word in frame['text'] for word in framegroup_text]):
#       if not framegroup_is_broken(framegroup):
#         embedded_framegroups.append(framegroup)
#   return embedded_framegroups

# def framegroup_is_embedded(framegroup,  sentences):
#   for sentence in sentences:
#     if all([word['text'] in sentence for word in framegroup]):
#       return True
#   return False

# stop_tags = set([
#     'R-ARG',
#     'ARGM-PRP',
#     'ARGM-ADV',
#     'ARGM-PRD',
#   #  'ARGM-MNR',
# ])

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# def update_current_sentence(current_sentence, new_text):
#    current_sentence['text'].append(new_text)
#    current_sentence['status'] = 0

# def add_current_sentence_to_result(global_sentences, current_sentence, need_to_check=False, frame=None):
#     func = lambda f: any([ tag in f['tag']   for tag in stop_tags])
#     if (need_to_check and func(frame)) or need_to_check == False:
#             new_sentence = ' '.join(current_sentence['text'])
#             global_sentences.append(new_sentence)
#             current_sentence['status'] = 1


# def get_segmented_sentence(framegroups):
#   global_sentences = []
#   for framegroup in framegroups:
#     current_sentence = {'text':[], 'status':0}
#     if framegroup_is_broken(framegroup) or framegroup_is_embedded(framegroup, global_sentences):
#       continue
#     for frame in framegroup:
#       add_current_sentence_to_result(global_sentences, current_sentence, True, frame=frame)

#       embedded_framegroups = get_embedded_framegroups(frame, framegroups)
#       if len(embedded_framegroups) == 0:
#           update_current_sentence(current_sentence, frame['text'])
#       else:
#           for embedded_framegroup in embedded_framegroups:

#               for emb_frame in embedded_framegroup:
#                 if emb_frame['text'] not in ' '.join(current_sentence['text']):
#                   add_current_sentence_to_result(global_sentences, current_sentence, True, frame=emb_frame)

#                   update_current_sentence(current_sentence, emb_frame['text'])

#               add_current_sentence_to_result(global_sentences, current_sentence, False)

#     if current_sentence['status'] == 0:
#       add_current_sentence_to_result(global_sentences, current_sentence, False)
#   return global_sentences


# def semantic_analys(clean_sentences):
#     args_dict = { }
#     analized_sentences = []
#     for sentence in clean_sentences:
#         framegroups = get_framegroups(sentence)
#         sentences = get_segmented_sentence(framegroups)

#         for framegroup in framegroups:
#           for frame in framegroup:
#             if 'ARG' in frame['tag']:
#               arg = frame['text']
#               if arg in args_dict:
#                   args_dict[arg] += 1
#               else:
#                   args_dict[arg] = 1
#         sem_analys = {
#             'topic': None,
#             'sentence': sentence,
#             'segmented_sentences': sentences
#         }
#         analized_sentences.append(sem_analys)

#     topic = max(args_dict, key=args_dict.get)
#     for analized_sentence in analized_sentences:
#      analized_sentence['topic'] = topic

#     return analized_sentences


# def tokenize_sentences(text):
#     # text = coref_resolution(text)

#     # Remove 'However', 'Besides', 'Hence' and other intro-expressions.
#     for intro in intros:
#         if intro in text:
#             text = text.replace(intro, '')

#     sentences = []
#     if text[-1] in punct:  # Remove incomplited sentences
#         sentences = [t.strip() for t in sent_tokenize(text)]
#     else:
#         sentences = [t.strip() for t in sent_tokenize(text)[:-1]]
#     print(sentences)
#     # res = semantic_analys(sentences)

#     return sentences


# #


def get_relevant_refs(refs, source):
    '''
      Filter Google's response, remains only relevant sentences.
      If there is no relevant sentence,
    '''

    all_sentences = []
    for j in refs:
        sentences = sent_tokenize(j['snippet'])
        all_sentences += sentences

    scores = []
    for ref in all_sentences:
        score = similarity(source, ref)
        scores.append({
            'ref': ref,
            'score': score
        })

    candidates = sorted(list(filter(
        lambda r: r['score'] > 0.725, scores)),  key=lambda x: x['score'], reverse=True)
    if len(candidates) > 1:
        for i, c in enumerate(candidates):
            current_refs = candidates[i+1:]
            for ref in current_refs:
                score = similarity(c['ref'], ref['ref'])
                if score > 0.45:
                    candidates.remove(ref)
    elif len(candidates) == 0:
        return None
    return candidates


# Не используй в разработке функцию run, это абсолютно бессмысленно! Чтобы она работала нужен вызов run_google_search, который у тебя закомментирован (и правильно, что закомментирован!)

# def run(prompt):
#     reliable_sentences = []

#     prompt += "\nDon't use complicated sentences or jargon."

#     #llm_output = get_llm_output(prompt, max_tokens=200)

#     llm_output_tokenized = tokenize_sentences(llm_output.strip())
#     google_sentense = ['For starters, Python is straightforward to learn, easy to code, and has a large library. On the other hand, Java is more compatible and excels at creating games and apps for mobile devices. They are both powerful, widely used programming languages that can evolve to accommodate cutting-edge technologies.',
#                        'Python is widely considered among the easiest programming languages for beginners to learn. If you are interested in learning a programming language, Python is a good place to start.',
#                        'Data analysts also assist web developers by identifying trends in data, working with tools such as Tableau, SAS, Microsoft Excel, Google Analytics, and Apache Spark to do this.',
#                        'This programming language is known for its simplicity and readability, with beginners finding it easy to grasp. Python is versatile and can be used for a range of applications from web development, data analysis, artificial intelligence, automation and more.']
#     lol = -1

#     for sentence in llm_output_tokenized:
#         # google_outputs = run_google_search(sentence)
#         lol += 1
#         google_outputs = google_sentense[lol]
#         relevant_google_outputs = get_relevant_refs(google_outputs, sentence)
#         if relevant_google_outputs == None:
#             print(
#                 f"{ sentence } --> Suspicious sentence. It is either meaningless or completely false")
#             continue

#         prompt = f'''Source: "{sentence}" '''
#         for i, fact in enumerate(relevant_google_outputs):
#             prompt += f'Fact_{i+1}: "{fact}\n"'

#         facts_name = "Fact_1"
#         if len(relevant_google_outputs) > 1:
#             for i, fact in enumerate(relevant_google_outputs[1:]):
#                 facts_name += f'and Fact_{i}'
#         prompt += f'New_Source = Source modified to reflect {facts_name}. If the Source approximately reflects the {facts_name}, New_Source matches exactly Source. Do not type explanation, comments and score. Start type with: "New_Source: ...'

#         llm_output = get_llm_output(prompt, 250)
#         llm_output = llm_output.strip('\n. "').split('New_Source:')[-1].strip()

#         if '"' in llm_output:
#             llm_output = llm_output.split('"')[1]

#         score = similarity(sentence, llm_output)
#         if score < 0.60:
#             print(f"{ sentence } --> Something went wrong!")
#         elif score < 0.90:
#             print(f"{ sentence} --> {llm_output}, score={score}")
#             reliable_sentences.append(llm_output)
#         else:
#             print(f"{ sentence } --> Absolutely correct!")
#             reliable_sentences.append(sentence)

#     if len(reliable_sentences) == 0:
#         print('The generated text was completely incorrect and cannot be corrected!')
#         reliable_sentences = ",".join(
#             ['"' + sentence + '"' for sentence in reliable_sentences])
#         prompt = f'Write a text using following sentences: {reliable_sentences}\nDo not use any additional information.  Do not type explanation and comments. Start with "Text: ..."'
#         llm_output = get_llm_output(prompt, 300)
#         print('\nFixed text: ', llm_output.split('Text:')[-1].strip())


def similarity(sentence_1, sentence_2):
    embedding_source = model.encode(sentence_1, convert_to_tensor=True)
    embedding_ref = model.encode(sentence_2, convert_to_tensor=True)
    score = float(util.pytorch_cos_sim(embedding_source, embedding_ref)[0][0])
    return score


# Используй пока эту функцию: на данном этапе пока не нужно давать пользователю отправлять произвольные запросы к LLM,
# пускай он будет ВЫБИРАТЬ один из пяти запросов, по которому у нас уже есть SERP;
# напомню, что результаты serp  (google_outputs) находились в секции "Что говорит гугл".
# Только пожалуйста, предворительно сохрани их в формате JSON.

# В итоге у тебя будет так: Пользователь выбирает запрос (например, "Tell me about polar bears"), затем в окне ниже ему как будто бы печатается ответ LLM,
# эти ответы, с уже разделенные на предложения и с замененными местоимениями, есть в блокноте в секции Experiments.
# После этого читай соответсвующие JSON-ы и отправляй это всё в функцию  run_no_google. Сделай так, чтобы run_no_google возвращал результаты! Зачем их печатать в консоль??


def run_no_google(google_outputs):
    reliable_sentences = []
    for sentence in google_outputs:
        relevant_google_outputs = get_relevant_refs(
            sentence['google'], sentence['querry'])
        if relevant_google_outputs == None:
            print(
                f"{ sentence['querry'] } --> Suspicious sentence. It is either meaningless or completely false")
            continue

        prompt = f'''Source: "{sentence['querry']}" '''
        for i, fact in enumerate(relevant_google_outputs):
            prompt += f'Fact_{i+1}: "{fact}\n"'

        facts_name = "Fact_1"
        if len(relevant_google_outputs) > 1:
            for i, fact in enumerate(relevant_google_outputs[1:]):
                facts_name += f'and Fact_{i}'
        prompt += f'New_Source = Source modified to reflect {facts_name}. If the Source approximately reflects the {facts_name}, New_Source matches exactly Source. Do not type explanation, comments and score. Start type with: "New_Source: ...'

        llm_output = get_llm_output(prompt, 250)
        llm_output = llm_output.strip('\n. "').split('New_Source:')[-1].strip()

        if '"' in llm_output:
            llm_output = llm_output.split('"')[1]

        score = similarity(sentence['querry'], llm_output)
        if score < 0.60:
            print(f"{ sentence['querry'] } --> Something went wrong!")
        elif score < 0.90:
            print(f"{ sentence['querry'] } --> {llm_output}, score={score}")
            reliable_sentences.append(llm_output)
        else:
            print(f"{ sentence['querry'] } --> Absolutely correct!")
            reliable_sentences.append(sentence['querry'])

    if len(reliable_sentences) == 0:
        print('The generated text was completely incorrect and cannot be corrected!')
    reliable_sentences = ",".join(
        ['"' + sentence + '"' for sentence in reliable_sentences])
    prompt = f'Write a text using following sentences: {reliable_sentences}\nDo not use any additional information.  Do not type explanation and comments. Start with "Text: ..."'
    llm_output = get_llm_output(prompt, 300)
    print('\nFixed text: ', llm_output.split('Text:')[-1].strip())


def get_llm_output(prompt, max_tokens=100):

    llama = LlamaAPI(
        "LL-qujsIzM1LLbEaaAgkbfAUTaZc0mIKUQtvrzYRf7uZGSaoJ9UFiVH2BJDW79GKEDp")

    api_request_json = {
        "temperature": 0.1,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }
    response = llama.run(api_request_json)
    return json.dumps(response.json()['choices'][0]['message']['content'], indent=2)
