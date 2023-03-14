import requests
import tiktoken
import os
import json
from time import time
from openai.embeddings_utils import cosine_similarity

class GptCompletion:
    def __init__(self, api_key, org, model) -> None:
        self.api_key = api_key
        self.org = org
        self.model = model
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self._encoding = tiktoken.encoding_for_model(self.model)

    def complete(self, prompt, config={}):
        defaultConfig = {
            "model": self.model,
            "prompt": prompt
        }

        defaultConfig.update(config)
        res = requests.post('https://api.openai.com/v1/completions', 
            headers={
                'Authorization': 'Bearer ' + self.api_key,
                'OpenAI-Organization': self.org
            },
            json = defaultConfig
        ).json()
        if res.get('error') is not None:
            raise SyntaxError("error getting chat message: " + res.get('error').get('message'))
        self.prompt_tokens += res.get('usage').get('prompt_tokens')
        self.completion_tokens += res.get('usage').get('completion_tokens')
        self.total_tokens += res.get('usage').get('total_tokens')
        msg = res.get('choices')[0].get('text')
        return msg

class Chat:
    def __init__(
        self,
        api_key,
        org,
        model,
        config={ }
    ):
        self.api_key = api_key
        self.org = org
        self.model = model
        self.config = config
        self._encoding = tiktoken.encoding_for_model(self.model)

        # total tokens of the conversation updated every chat message
        self._total_tokens = 0
        self._messages = []

    def add_message(self, msg, role):
        self._messages.append({ 'role': role, 'content': msg})
    
    def send(self, msg, role="user"):
        if self.api_key is None or \
            self.org is None or \
            self.model is None:
            raise AssertionError("API key, org, or model is not defined")

        self.add_message(msg, role)
        return self.run()

    def run(self):
        cfg = {
            "model": self.model,
            "messages": self._messages
        }
        cfg.update(self.config)
        res = requests.post('https://api.openai.com/v1/chat/completions', 
            headers={
                'Authorization': 'Bearer ' + self.api_key,
                'OpenAI-Organization': self.org
            },
            json = cfg
        ).json()
        if res.get('error') is not None:
            raise SyntaxError("error getting chat message: " + res.get('error').get('message'))
        self._total_tokens = res.get('usage').get('total_tokens')
        choice = res.get('choices')[0].get('message')
        self._messages.append(choice)
        return choice.get('content')

class EmbeddingFactory:
    def __init__(self, api_key, org_key):
        self.api_key = api_key
        self.org_key = org_key

    def get_embedding(self, data):
        d = {
            "model": 'text-embedding-ada-002',
            "input": json.dumps(data)
        }
        res = requests.post('https://api.openai.com/v1/embeddings', 
            headers={
                'Authorization': 'Bearer ' + self.api_key,
                'OpenAI-Organization': self.org_key
            },
            json=d
        ).json()
        if res.get('error') is not None:
            raise SyntaxError("Error getting embedding: " + res.get('error').get('message'))
        return res.get('data')[0].get('embedding')


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)

def stringify_conversation(conversation):
    convo = ''
    for i in conversation:
        convo += '%s: %s\n' % (i['role'].upper(), i['content'])
    return convo.strip()

completion_config = {
    'temperature': 0,
    'max_tokens': 400,
    'stop': ['USER:', 'RAVEN:']
}

class Anticipation(GptCompletion):
    def __init__(self, api_key, org_key):
        super().__init__(api_key, org_key, 'text-davinci-003')
        self._anticipation_prompt = open_file('prompts/prompt_anticipate.txt')
        self._prompt_tokens = len(self._encoding.encode(self._anticipation_prompt))

    def anticipate(self, conversation):
        conv_s = stringify_conversation(conversation)
        prompt = self._anticipation_prompt.replace('<<INPUT>>', conv_s)
        return self.complete(prompt, completion_config)

class Salience(GptCompletion):
    def __init__(self, api_key, org_key):
        super().__init__(api_key, org_key, 'text-davinci-003')
        self._salience_prompt = open_file('prompts/prompt_salience.txt')
        self._prompt_tokens = len(self._encoding.encode(self._salience_prompt))

    def get_salient_points(self, conversation, anticipation, memories):
        conv_s = stringify_conversation(conversation)
        prompt = self._salience_prompt\
            .replace('<<INPUT>>', conv_s)\
            .replace('<<ANTICIPATION>>', anticipation)\
            .replace('<<MEMORIES>>', memories)

        print('SAL')
        print(prompt)
        print('\nSAL')
        return self.complete(prompt, completion_config)

class Memory(EmbeddingFactory):
    def __init__(self, api_key, org_key):
        super().__init__(api_key, org_key)

    def get_memories(self, query, top_n=3):
        q_embed = self.get_embedding(query)
        most_similar = [ ]
        for e_file in os.listdir('embeddings'):
            embed_obj = json.loads(open_file('embeddings/' + e_file))
            embed_dat = embed_obj['embedding']
            similarity = cosine_similarity(q_embed, embed_dat)
            most_similar.append((similarity, embed_obj))
            def sort_objs(o):
                return o[0]
            most_similar.sort(key=sort_objs)
            most_similar = most_similar[:top_n]
        return most_similar

class Muse(Chat):
    def __init__(self, api_key, org_key, max_tokens = 4096, max_chat_length = 512):
        super().__init__(api_key, org_key, 'gpt-3.5-turbo', {
            "max_tokens": max_chat_length
        })
        self._max_tokens = max_tokens
        self._max_chat_length = max_chat_length
        self._anticipation = Anticipation(api_key, org_key)
        self._salience = Salience(api_key, org_key)
        self.embedFactory = EmbeddingFactory(api_key, org_key)
        self.memories = Memory(api_key, org_key)
        self.embeddings = []
        self._default_system_msg = open_file('prompts/prompt_system_default.txt')

        self.add_message(self._default_system_msg, 'system')

    def send_chat(self, msg, sys_msg = None):
        self.add_message(msg, 'user')
        system_prompt = self._default_system_msg if sys_msg == None else sys_msg
        anticipation = ''
        salient_points = ''

        # if there has already been at least one message sent
        if len(self._messages) > 1:
            # infer user intent, disposition, valence, needs
            anticipation = self._anticipation.anticipate(self._messages)

            memories = self.memories.get_memories(self._messages[0]['content'] + '\nUSER:' + msg)
            mem_list = ''
            for mem in memories:
                mem_list += '\n- ' + mem[1]['salient_points']
            
            # summarize the conversation to the most salient points
            salient_points = self._salience.get_salient_points(self._messages, anticipation, mem_list)
            
            # update SYSTEM based upon user needs and salience
            system_prompt += '\nHere\'s a brief summary of the conversation:\n %s \n- And here\'s what I expect the user\'s needs are:\n%s' % (salient_points, anticipation)
        
        self._messages[0]['content'] = system_prompt

        # not a perfect calculation but close enough
        conversation_tokens = len(self._encoding.encode(stringify_conversation(self._messages + [{ 'role': 'user', 'content': msg }])))
        
        # reset conversation and save conversation
        if conversation_tokens + self._max_chat_length > self._max_tokens:
            self.save_messages()
            self._messages.clear()
            self.add_message(system_prompt, 'system')

        # generate a response
        msg_res = self.run()

        embed_object = {
            'time': str(time()),
            'anticipation': anticipation,
            'salient_points': salient_points,
            'message': msg,
            'response': msg_res
        }
        embedding = self.embedFactory.get_embedding(json.dumps(embed_object))
        embed_object['embedding'] = embedding
        self.embeddings.append(embed_object)

        return [anticipation, salient_points, msg_res]
    
    def reset(self):
        self.save_messages()
        self._messages[0] = self._default_system_msg

    def save_messages(self):
        t = time()
        if len(self._messages) < 2:
            return # bail if theres nothing to save
        filename = 'chat_%s_user.txt' % t
        if not os.path.exists('chat_logs'):
            os.makedirs('chat_logs')
        conv = stringify_conversation(self._messages)
        save_file('chat_logs/%s' % filename, conv)

        if not os.path.exists('embeddings'):
            os.makedirs('embeddings')
        emb_str = ''
        for emb in self.embeddings:
            emb_str += json.dumps(emb) + ',\n'
        emb_str = emb_str[:len(emb_str)-2]
        filename = 'embedding_%s.json' % t
        save_file('embeddings/%s' % filename, emb_str)

    def load(self):
        if not os.path.exists('chat_logs'):
            print('Error: there are no chat logs to load')
        
        logs = []
        for l in os.listdir('chat_logs'):
            edit_time = os.path.getmtime('chat_logs/' + l)
            logs.append((l, edit_time))
        
        def sort_logs(l):
            return l[1]
        logs.sort(key=sort_logs, reverse=True)
        log = logs[0][0]
        log = open_file('chat_logs/' + log).split('USER:')[0]
        self.reset()
        self._messages[0] = { 'role': 'system', 'content': log }
        print('Last conversation loaded...')
        return self.send_chat('Summarize the current conversation, be short and concise', log)
        
