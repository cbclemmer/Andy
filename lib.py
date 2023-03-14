import requests
import tiktoken
import os
import json
import uuid
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

def get_closest_embeddings(folder, q_embed, top_n):
    for e_file in os.listdir(folder):
        e_file_data = json.loads(open_file(folder + '/' + e_file))
        for line in e_file_data.split('\n'):
            embed_obj = json.loads(line)
            embed_dat = embed_obj['embedding']
            similarity = cosine_similarity(q_embed, embed_dat)
            embed_obj['file_name'] = e_file
            most_similar.append((similarity, embed_obj))
            def sort_objs(o):
                return o[0]
            most_similar.sort(key=sort_objs)
            most_similar = most_similar[:top_n]
    return most_similar

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

    def get_salient_points(self, conversation):
        conv_s = stringify_conversation(conversation)
        prompt = self._salience_prompt.replace('<<INPUT>>', conv_s)
        return self.complete(prompt, completion_config)

class Memory(EmbeddingFactory):
    def __init__(self, api_key, org_key):
        super().__init__(api_key, org_key)

    def get_memories(self, query, top_n=3):
        q_embed = self.get_embedding(query)
        most_similar = get_closest_embeddings('embeddings', q_embed, top_n)
        memories = ''
        for o in most_similar:
            memories += '\n- ' + o['salient_points']
        return memories

class Summary(GptCompletion):
    def __init__(self, api_key, org_key):
        super().__init__(api_key, org_key, 'text-davinci-003')
        self._summary_prompt = open_file('prompts/prompt_executive_summary.txt')
        self.prompt_tokens = self._encoding.encode(self._summary_prompt)

    def summarize(self, conversation):
        conv_s = stringify_conversation(conversation)
        prompt = self._summary_prompt.replace('<<INPUT>>', conv_s)
        return self.complete(prompt, completion_config)


class Concept(GptCompletion):
    def __init__(self, api_key, org_key):
        super().__init__(api_key, org_key, 'text-davinci-003')
        self.embeddingFactory = EmbeddingFactory(api_key, org_key)
        self._retrieve_prompt = open_file('prompts/prompt_concept_retieve.txt')
        self._update_prompt = open_file('prompts/prompt_concept_update.txt')

    def retrieve_concepts(self, salient_points):
        prompt = self._retrieve_prompt.replace('<<INPUT>>', salient_points)
        concept_keys = self.complete(prompt, completion_config).split('\n')
        concepts = ''
        for c in concept_keys:
            q_embed = self.embeddingFactory.get_embedding(c)
            most_similar = get_closest_embeddings('concepts', q_embed, 1)
            if len(most_similar) == 0:
                continue
            concepts += '\n- ' + concepts['data']
        return concepts
    
    def update_concepts(self, salient_points, retrieved_concepts, bot_message, user_message):
        prompt = self._update_prompt\
            .replace('<<SALIENT_POINTS>>', salient_points)\
            .replace('<<CONCEPTS>>', retrieved_concepts)\
            .replace('<<CHAT_MESSAGE>>', bot_message)\
            .replace('<<USER_MESSAGE>>', user_message)
        concepts_to_update = self.complete(prompt, completion_config).split('\n')
        add_concepts = []
        update_concepts = []
        for c in concepts_to_update:
            c_embed_o = self.embed_concept(c)
            if 'add' in c.lower():
                add_concepts.append(c_embed_o)
            if 'update' in c.lower():
                c_to_update = get_closest_embeddings('concepts', c_embed_o['embedding'], 1)
                c_embed_o['file_name'] = c_to_update['file_name']
                update_concepts.append(c_embed_o)
        return {
            'add': add_concepts,
            'update': update_concepts
        }

    def embed_concept(self, concept):
        c_embed = self.embeddingFactory.get_embedding(concept)
        return {
            'data': concept,
            'embedding': c_embed
        }   
    
class Muse(Chat):
    def __init__(self, api_key, org_key, max_tokens = 4096, max_chat_length = 512):
        super().__init__(api_key, org_key, 'gpt-3.5-turbo', {
            "max_tokens": max_chat_length
        })
        self._max_tokens = max_tokens
        self._max_chat_length = max_chat_length
        self._anticipation = Anticipation(api_key, org_key)
        self._salience = Salience(api_key, org_key)
        self._summaryFactory = Summary(api_key, org_key)
        self.embedFactory = EmbeddingFactory(api_key, org_key)
        self.memories = Memory(api_key, org_key)
        self.concepts = Concept(api_key, org_key)
        self.memory_data = []
        self.concept_data = {
            'add': [],
            'update': []
        }
        self._default_system_msg = open_file('prompts/prompt_system_default.txt')
        self._system_context_msg = open_file('prompts/prompt_system_context.txt')

        self.add_message(self._default_system_msg, 'system')

    def send_chat(self, msg, sys_msg = None):
        self.add_message(msg, 'user')
        system_prompt = self._default_system_msg if sys_msg == None else sys_msg
        anticipation = ''
        salient_points = ''
        concepts = ''

        # if there has already been at least one message sent
        if len(self._messages) > 1:
            # infer user intent, disposition, valence, needs
            anticipation = self._anticipation.anticipate(self._messages)

            memories = self.memories.get_memories(self._messages[0]['content'] + '\nUSER:' + msg)
            
            # summarize the conversation to the most salient points
            salient_points = self._salience.get_salient_points(self._messages)

            concepts = self.concepts.retrieve_concepts(salient_points)
            
            # update SYSTEM based upon user needs and salience
            system_prompt += self._system_context_msg\
                .replace('<<CONVERSATION>>', salient_points)\
                .replace('<<ANTICIPATION>>', anticipation)\
                .replace('<<MEMORIES>>', memories)\
                .replace('<<CONCEPTS>>', concepts)
        
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

        memory_embed = {
            'time': str(time()),
            'anticipation': anticipation,
            'salient_points': salient_points,
            'message': msg,
            'response': msg_res
        }
        memory_embed['embedding'] = self.embedFactory.get_embedding(json.dumps(memory_embed))
        self.memory_data.append(memory_embed)

        concepts_to_add_or_update = self.concepts.update_concepts(salient_points, concepts, msg_res, msg)
        self.concept_data['add'] += concepts_to_add_or_update['add']
        self.concept_data['update'] += concepts_to_add_or_update['update']

        return [anticipation, salient_points, msg_res]
    
    def reset(self):
        self.save_messages()
        self._messages[0] = self._default_system_msg

    def _save_chat_log(self, t):
        if len(self._messages) < 2:
            return # bail if theres nothing to save
        print('Summarizing conversation...')
        summary = self._summaryFactory.summarize(self._messages)
        self._messages[0] = self._default_system_msg + \
            '\nI am continuing from a previous conversation, here is a summary of that conversation:\n' \
            + summary
        filename = 'chat_%s_user.txt' % t
        if not os.path.exists('chat_logs'):
            os.makedirs('chat_logs')
        conv = stringify_conversation(self._messages)
        save_file('chat_logs/%s' % filename, conv)

    def _save_embedding(self, t):
        if not os.path.exists('embeddings'):
            os.makedirs('embeddings')
        emb_str = ''
        for emb in self.memory_data:
            emb_str += json.dumps(emb) + '\n'
        emb_str = emb_str[:len(emb_str)-1]
        filename = 'embedding_%s.jsonl' % t
        save_file('embeddings/%s' % filename, emb_str)

    def _save_concepts(self):
        for c in self.concept_data['add']:
            f_name = uuid.uuid1() + '.json'
            if not os.path.exists('concepts'):
                os.makedirs('concepts')
            save_file('concepts/' + f_name, json.dumps(c))
        for c in self.concept_data['update']:
            os.remove('concepts/' + c['file_name'])
            save_file('concepts/' + c['file_name'], json.dumps(c))

    def save_messages(self):
        t = time()
        self._save_chat_log(t)
        self._save_embedding(t)
        self._save_concepts()

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
        return self.run()
        # return self.send_chat('Summarize the current conversation, be short and concise', log)
        
