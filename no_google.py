import nltk
import json
from llamaapi import LlamaAPI
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer, util



model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def get_relevant_refs(refs, source):


    all_sentences = []
    for j in refs:
        sentences = sent_tokenize(j['snippet'])
        all_sentences += sentences

        embedding_source = model.encode(source, convert_to_tensor=True)
        scores = []
        for ref in all_sentences:
            score = similarity(source, ref)
            scores.append({
                'ref': ref,
                'score':score
            })

    candidates = sorted(list(filter(lambda r: r['score'] > 0.725, scores)),  key=lambda x:x['score'], reverse=True)
    if len(candidates) > 1:
        for i, c in enumerate(candidates):
            current_refs = candidates[i+1:]
            embedding_source = model.encode(c['ref'], convert_to_tensor=True)

            for ref in current_refs:
              score = similarity(c['ref'], ref['ref'])
              if score > 0.45:
                candidates.remove(ref)
    elif len(candidates) == 0:
            return None
    return candidates

def similarity(sentence_1, sentence_2):
    embedding_source = model.encode(sentence_1, convert_to_tensor=True)
    embedding_ref = model.encode(sentence_2, convert_to_tensor=True)
    score = float(util.pytorch_cos_sim(embedding_source, embedding_ref)[0][0])
    return score

# def run_no_google(google_outputs):
#     reliable_sentences = []
#     for sentence in google_outputs:
#         relevant_google_outputs = get_relevant_refs(sentence['google'], sentence['querry'])
#         if relevant_google_outputs == None:
#            new_sen1 = (sentence['querry'] + "  --> " +"Suspicious sentence. It is either meaningless or completely false")
#            reliable_sentences.append(new_sen1)
#            continue
#
#         # prompt = f'''Source: "{sentence['querry']}" '''
#         # for i, fact in enumerate(relevant_google_outputs):
#         #   prompt += f'Fact_{i+1}: "{fact}\n"'
#         #
#         # facts_name = "Fact_1"
#         # if len(relevant_google_outputs) > 1:
#         #     for i, fact in enumerate(relevant_google_outputs[1:]):
#         #         facts_name += f'and Fact_{i}'
#         # prompt += f'New_Source = Source modified to reflect {facts_name}. If the Source approximately reflects the {facts_name}, New_Source matches exactly Source. Do not type explanation, comments and score. Start type with: "New_Source: ...'
#         #
#         # llm_output = get_llm_output(prompt, 250)
#         # llm_output = llm_output.strip('\n. "').split('New_Source:')[-1].strip()
#
#
#
#         score = similarity(sentence['querry'], relevant_google_outputs)
#         if score < 0.60:
#           new_sen = sentence['querry'] + "   --> " +"Something went wrong!"
#           reliable_sentences.append(new_sen)
#         elif score < 0.90:
#           new_sen = sentence['querry'] + "   -->  " + str(score)
#           reliable_sentences.append(new_sen)
#         else:
#            new_sen = sentence['querry'] + "  --> " + "Absolutely correct!"
#            reliable_sentences.append(new_sen)
#     return(reliable_sentences)

def run_no_google(google_outputs):
    reliable_sentences = []
    ready_sen = []
    for sentence in google_outputs:
        relevant_google_outputs = get_relevant_refs(sentence['google'], sentence['querry'])
        if relevant_google_outputs == None:
            new_sen1 = (sentence['querry'] + "  --> " + "Suspicious sentence. It is either meaningless or completely false")
            ready_sen.append(new_sen1)
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
                new_sen = sentence['querry'] + "   --> " + "Something went wrong!"
                ready_sen.append(new_sen)
            elif score < 0.90:
                new_sen = sentence['querry'] + "   -->  " + str(score)
                ready_sen.append(new_sen)
                reliable_sentences.append(llm_output)
            else:
                new_sen = sentence['querry'] + "  --> " + "Absolutely correct!"
                ready_sen.append(new_sen)
                reliable_sentences.append(sentence['querry'])



    if len(reliable_sentences) == 0:
        print('The generated text was completely incorrect and cannot be corrected!')
    rel = reliable_sentences
    rel.insert(0,'Reliable sentences: ')
    reliable_sentences = ",".join(['"' + sentence + '"' for sentence in reliable_sentences])
    prompt = f'Write a text using following sentences: {reliable_sentences}\nDo not use any additional information.  Do not type explanation and comments. Start with "Text: ..."'
    llm_output = get_llm_output(prompt, 300)
    text2 = '\nFixed text: ', llm_output.split('Text:')[-1].strip()
    answer = [ready_sen, text2, rel]
    return (answer)

def get_llm_output(prompt, max_tokens=100):

  llama = LlamaAPI("LL-qujsIzM1LLbEaaAgkbfAUTaZc0mIKUQtvrzYRf7uZGSaoJ9UFiVH2BJDW79GKEDp")

  api_request_json = {
      "temperature": 0.1,
      "messages": [
          {"role": "user", "content": prompt},
      ],
      "stream": False,
  }
  response = llama.run(api_request_json)
  return json.dumps(response.json()['choices'][0]['message']['content'], indent=2)
