import os
import openai
import json
import numpy as np
from numpy.linalg import norm
import re
from time import time,sleep
from uuid import uuid4
import datetime
from lib import Muse


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)


def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return json.load(infile)


def save_json(filepath, payload):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        json.dump(payload, outfile, ensure_ascii=False, sort_keys=True, indent=2)


def timestamp_to_datetime(unix_time):
    return datetime.datetime.fromtimestamp(unix_time).strftime("%A, %B %d, %Y at %I:%M%p %Z")


def gpt3_embedding(content, engine='text-embedding-ada-002'):
    content = content.encode(encoding='ASCII',errors='ignore').decode()  # fix any UNICODE errors
    response = openai.Embedding.create(input=content,engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector


def chatgpt_completion(messages, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(model=model, messages=messages)
    text = response['choices'][0]['message']['content']
    filename = 'chat_%s_muse.txt' % time()
    if not os.path.exists('chat_logs'):
        os.makedirs('chat_logs')
    save_file('chat_logs/%s' % filename, text)
    return text


def gpt3_completion(prompt, engine='text-davinci-003', temp=0.0, top_p=1.0, tokens=400, freq_pen=0.0, pres_pen=0.0, stop=['USER:', 'RAVEN:']):
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()
    while True:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                temperature=temp,
                max_tokens=tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen,
                stop=stop)
            text = response['choices'][0]['text'].strip()
            #text = re.sub('[\r\n]+', '\n', text)
            #text = re.sub('[\t ]+', ' ', text)
            filename = '%s_gpt3.txt' % time()
            if not os.path.exists('gpt3_logs'):
                os.makedirs('gpt3_logs')
            save_file('gpt3_logs/%s' % filename, prompt + '\n\n==========\n\n' + text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)


def flatten_convo(conversation):
    convo = ''
    for i in conversation:
        convo += '%s: %s\n' % (i['role'].upper(), i['content'])
    return convo.strip()


if __name__ == '__main__':
    convo_length = 30
    api_key = open_file('key_openai.txt')
    org_key = open_file('key_org.txt')
    muse = Muse(api_key, org_key)
    
    counter = 0
    while True:
        # get user input, save to file
        a = input('\n\nUSER: ')
        conversation.append({'role': 'user', 'content': a})
        filename = 'chat_%s_user.txt' % time()
        if not os.path.exists('chat_logs'):
            os.makedirs('chat_logs')
        save_file('chat_logs/%s' % filename, a)
        flat = flatten_convo(conversation)
        #print(flat)
        # infer user intent, disposition, valence, needs
        prompt = open_file('prompt_anticipate.txt').replace('<<INPUT>>', flat)
        anticipation = gpt3_completion(prompt)
        print('\n\nANTICIPATION: %s' % anticipation)
        # summarize the conversation to the most salient points
        prompt = open_file('prompt_salience.txt').replace('<<INPUT>>', flat)
        salience = gpt3_completion(prompt)
        print('\n\nSALIENCE: %s' % salience)
        # update SYSTEM based upon user needs and salience
        conversation[0]['content'] = default_system + ''' Here's a brief summary of the conversation: %s - And here's what I expect the user's needs are: %s''' % (salience, anticipation)
        # generate a response
        response = chatgpt_completion(conversation)
        conversation.append({'role': 'assistant', 'content': response})
        print('\n\nMUSE: %s' % response)
        # increment counter and consolidate memories
        counter += 2
        if counter >= 10:
            # reset conversation
            conversation = list()
            conversation.append({'role': 'system', 'content': default_system})