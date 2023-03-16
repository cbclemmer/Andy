import os
from util import open_file, stringify_conversation, clean_embedding_folder, save_file
from brain.anticipation import Anticipation
from brain.salience import Salience
from brain.memory import Memory
from brain.concept import Concept
from summary import Summary
from gpt import EmbeddingFactory, Chat
from write import Write
import uuid
import json
import time

class Andy(Chat):
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
        
