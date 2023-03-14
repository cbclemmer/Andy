import os
import openai
import json
from time import time,sleep
import datetime
from lib import Muse
import signal

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
    api_key = open_file('key_openai.txt').split('\n')[0]
    org_key = open_file('key_org.txt').split('\n')[0]
    muse = Muse(api_key, org_key)

    # save whatever is in the chat log when key interrupt
    def keyboardInterruptHandler(_, __):
        if len(muse._messages) > 2:
            print("Conversation saved")
        muse.save_messages()
        exit(0)
    signal.signal(signal.SIGINT, keyboardInterruptHandler)
    
    while True:
        user_input = input('\n\nUSER: ')
        if user_input == 'RESET':
            muse.reset()
            continue
        if user_input == 'LOAD':
            [_, _, msg] = muse.load()
            print('Context:\n ' + msg)
            continue
        print('Sending message...')
        msg_res = muse.send_chat(user_input)
        print('\n\nMUSE:\n%s' % msg_res + '\n\n')
