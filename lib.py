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
            if 'overloaded' in res.get('error').get('message'):
                print('ERROR: Server overloaded. Retrying request...')
                self.complete(self, prompt, config)
            raise SyntaxError("error getting chat message: " + res.get('error').get('message'))
        # self.prompt_tokens += res.get('usage').get('prompt_tokens')
        # self.completion_tokens += res.get('usage').get('completion_tokens')
        # self.total_tokens += res.get('usage').get('total_tokens')
        print('TOKENS')
        print(res.get('usage').get('prompt_tokens'))
        print(res.get('usage').get('completion_tokens'))
        print(res.get('usage').get('total_tokens'))
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
            if 'overloaded' in res.get('error').get('message'):
                print('ERROR: Server overloaded. Retrying request...')
                self.run(self)
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
            if 'overloaded' in res.get('error').get('message'):
                print('ERROR: Server overloaded. Retrying request...')
                self.get_embedding(self, data)
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

def clean_embedding_folder(folder, max_sim):
    files = os.listdir(folder)
    rerun = False
    for f in files:
        full_path = folder + '/' + f
        lines = open_file(full_path).split('\n')
        idx = 0
        lines_to_remove = [ ]
        for line in lines:
            if len(line) == 0:
                continue
            obj = json.loads(line)
            emb = obj['embedding']
            closest = get_closest_embeddings(folder, emb, 2)
            if len(closest) > 1 and closest[1][0] > max_sim:
                lines_to_remove.append(idx)
            idx += 1
        if len(lines_to_remove) > 0:
            rerun=True
            new_file = ''
            idx = 0
            for line in lines:
                if idx in lines_to_remove:
                    continue
                new_file += line + '\n'
                idx += 1
            os.remove(full_path)
            if len(new_file) > 0:
                save_file(full_path, new_file)
    if rerun:
        clean_embedding_folder(folder, max_sim)

def get_closest_embeddings(folder, q_embed, top_n, exclude=[], exclude_key=None):
    if not os.path.exists(folder):
        return []
    most_similar = []
    for e_file in os.listdir(folder):
        e_file_data = open_file(folder + '/' + e_file)
        if e_file_data == '':
            continue
        for line in e_file_data.split('\n'):
            if len(line) == 0:
                continue
            embed_obj = json.loads(line)
            if exclude_key is not None:
                if embed_obj[exclude_key] in exclude:
                    continue
            embed_dat = embed_obj['embedding']
            similarity = cosine_similarity(q_embed, embed_dat)
            embed_obj['file_name'] = e_file
            most_similar.append((similarity, embed_obj))
            def sort_objs(o):
                return o[0]
            most_similar.sort(key=sort_objs, reverse=True)
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
            memories += '\n- ' + o[1]['salient_points']
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
        self._retrieve_prompt = open_file('prompts/prompt_concept_retrieve.txt')
        self._update_prompt = open_file('prompts/prompt_concept_update.txt')

    def retrieve_concepts(self, salient_points):
        prompt = self._retrieve_prompt.replace('<<INPUT>>', salient_points)
        concept_keys = self.complete(prompt, completion_config).split('\n')
        concepts = ''
        concept_list = []
        for c in concept_keys:
            print('Attempting to retrieve concept: ' + c)
            q_embed = self.embeddingFactory.get_embedding(c)
            most_similar = get_closest_embeddings('concepts', q_embed, 1, concept_list, 'data')
            if len(most_similar) == 0:
                continue
            concept_list.append(most_similar[0][1]['data'])
            concepts += '\n- ' + most_similar[0][1]['data']
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
                c_embed_o['data'] = c_embed_o['data'].replace('Add', '')
                add_concepts.append(c_embed_o)
            if 'update' in c.lower():
                c_embed_o['data'] = c_embed_o['data'].replace('Update', '')
                c_to_update = get_closest_embeddings('concepts', c_embed_o['embedding'], 1)
                if len(c_to_update) == 0:
                    continue
                c_embed_o['file_name'] = c_to_update[0][1]['file_name']
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

class Write(GptCompletion):
    def __init__(self, api_key, org_key):
        super().__init__(api_key, org_key, 'text-davinci-003')
        self.embedFactory = EmbeddingFactory(api_key, org_key)
    
    def write_document(self, last_memory):
        prompt = open_file('prompts/prompt_write_query.txt')\
            .replace('<<MEMORY>>', last_memory)
        
        queries = self.complete(prompt, completion_config)
        queries = queries.split('\n')
        def get_memories(q_list):
            memories = [ ]
            for q in q_list:
                emb = self.embedFactory.get_embedding(q)
                best_match = get_closest_embeddings('embeddings', emb, 1)
                new_mem = best_match[0][1]['salient_points']
                if new_mem in memories:
                    continue
                memories.append(new_mem)
            
            s_memories = ''
            for m in memories:
                s_memories += '- ' + m + '\n'
            return s_memories

        s_memories = get_memories(queries)
        concept_files = os.listdir('concepts')
        concepts = [ ]
        for f in concept_files:
            f_data = open_file('concepts/' + f)
            if len(f_data) == 0:
                continue
            for line in f_data.split('\n'):
                if len(line) == 0:
                    continue
                obj = json.loads(line)
                concepts.append(obj['data'])

        s_concepts = ''
        for c in concepts:
            s_concepts += '- ' + c + '\n'

        prompt = open_file('prompts/prompt_write.txt')\
            .replace('<<MEMORIES>>', s_memories)\
            .replace('<<CONCEPTS>>', s_concepts)

        document_beginning = self.complete(prompt, completion_config)
        print(document_beginning)

        prompt = open_file('prompts/prompt_write_memory.txt')\
            .replace('<<MEMORIES>>', s_memories)\
            .replace('<<DOCUMENT>>', document_beginning)
        
        new_memory_queries = self.complete(prompt, completion_config)
        new_memories = get_memories(new_memory_queries.split('\n'))
        prompt = open_file('prompts/prompt_write.txt')\
            .replace('<<MEMORIES>>', new_memories)\
            .replace('<<CONCEPTS>>', s_concepts)\
            + '\n' + document_beginning
        
        document_chunk = self.complete(prompt, completion_config)
        return document_chunk



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
        self.write = Write(api_key, org_key)
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
            print('Anticipating User needs...')
            anticipation = self._anticipation.anticipate(self._messages)

            print('Retrieving relevant memories...')
            memories = self.memories.get_memories(self._messages[0]['content'] + '\nUSER:' + msg)
            
            # summarize the conversation to the most salient points
            print('Summarizing salient points of conversation...')
            salient_points = self._salience.get_salient_points(self._messages)

            print('Retrieving relevant concepts...')
            concepts = self.concepts.retrieve_concepts(salient_points)
            
            # update SYSTEM based upon user needs and salience
            system_prompt += self._system_context_msg\
                .replace('<<CONVERSATION>>', salient_points)\
                .replace('<<ANTICIPATION>>', anticipation)\
                .replace('<<MEMORIES>>', memories)\
                .replace('<<CONCEPTS>>', concepts)
        
        self._messages[0]['content'] = system_prompt
        print('SYSTEM PROMPT:\n' + system_prompt + '\n\n')

        # not a perfect calculation but close enough
        conversation_tokens = len(self._encoding.encode(stringify_conversation(self._messages + [{ 'role': 'user', 'content': msg }])))
        
        # reset conversation and save conversation
        if conversation_tokens + self._max_chat_length > self._max_tokens:
            print('Reached maximum tokens, summarizing and resetting conversation...')
            self.save_messages()
            self._messages.clear()
            self._total_tokens = 0
            self.add_message(system_prompt, 'system')

        # generate a response
        print('Getting bot response...')
        msg_res = self.run()

        memory_embed = {
            'time': str(time()),
            'anticipation': anticipation,
            'salient_points': salient_points,
            'message': msg,
            'response': msg_res
        }
        print('Getting embedding for message...')
        memory_embed['embedding'] = self.embedFactory.get_embedding(json.dumps(memory_embed))
        self.memory_data.append(memory_embed)

        print('Updating concepts based on bot response...')
        concepts_to_add_or_update = self.concepts.update_concepts(salient_points, concepts, msg_res, msg)
        print('\n\nNew or updated concepts:\n')
        for c in concepts_to_add_or_update['add']:
            print('Added concept: ' + c['data'])
        for c in concepts_to_add_or_update['update']:
            print('Updated concept: ' + c['data'])
        self.concept_data['add'] += concepts_to_add_or_update['add']
        self.concept_data['update'] += concepts_to_add_or_update['update']

        print('Current tokens: ' + str(self._total_tokens))
        return msg_res
    
    def clean_memories(self):
        clean_embedding_folder('embeddings', 0.99)
        clean_embedding_folder('concepts', 0.9)
    
    def write_document(self):
        return self.write.write_document(self._messages[0]['content'])

    def reset(self):
        self.save_messages()
        self._messages[0] = self._default_system_msg

    def _save_chat_log(self, t):
        if len(self._messages) < 2:
            return # bail if theres nothing to save
        print('Summarizing conversation...')
        summary = self._summaryFactory.summarize(self._messages)
        self._messages[0] = {
            'role': 'system',
            'content': self._default_system_msg + \
                '\nI am continuing from a previous conversation, here is a summary of that conversation:\n' \
                + summary
        }
        
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
            if emb == '':
                continue
            emb_str += json.dumps(emb) + '\n'
        emb_str = emb_str[:len(emb_str)-1]
        filename = 'embedding_%s.jsonl' % t
        if len(emb_str) == 0:
            return
        save_file('embeddings/%s' % filename, emb_str)

    def _save_concepts(self):
        for c in self.concept_data['add']:
            f_name = 'concept_' + str(uuid.uuid1()) + '.json'
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
        self.clean_memories()

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
        self._messages[0] = { 'role': 'system', 'content': log }
        print('Last conversation loaded...')
        return self.send('What were we doing?')
        # return self.send_chat('Summarize the current conversation, be short and concise', log)
        
