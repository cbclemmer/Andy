import requests
import tiktoken
import os
from time import time

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
        return (res, msg)

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
    
    def send(self, msg, role="user"):
        if self.api_key is None or \
            self.org is None or \
            self.model is None:
            raise AssertionError("API key, org, or model is not defined")

        self._messages.append({ "role": role, "content": msg })
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
    'tokens': 400,
    'stop': ['USER:', 'RAVEN:']
}

class Anticipation(GptCompletion):
    def __init__(self, api_key, org_key):
        super(GptCompletion, self).__init__(api_key, org_key, 'text-davinci-003')
        self._anticipation_prompt = open_file('prompts/prompt_anticipate.txt')
        self._prompt_tokens = len(self._encoding.encode(self._anticipation_prompt))

    def anticipate(self, conversation):
        conv_s = stringify_conversation(conversation)
        prompt = self._anticipation_prompt.replace('<<INPUT>>', conv_s)
        return self.complete(prompt, completion_config)

class Salience(GptCompletion):
    def __init__(self, api_key, org_key):
        super(GptCompletion, self).__init__(api_key, org_key, 'text-davinci-003')
        self._salience_prompt = open_file('prompts/prompt_salience.txt')
        self._prompt_tokens = len(self._encoding.encode(self._salience_prompt))

    def get_salient_points(self, conversation):
        conv_s = stringify_conversation(conversation)
        prompt = self._salience_prompt.replace('<<INPUT>>', conv_s)
        return self.complete(prompt, completion_config)

class Muse(Chat):
    def __init__(self, api_key, org_key, max_tokens = 4096, max_chat_length = 512):
        super(Chat, self).__init__(api_key, org_key, 'gpt-3.5-turbo', {
            "max_tokens": max_chat_length
        })
        self._max_tokens = max_tokens
        self._max_chat_length = max_chat_length
        self._anticipation = Anticipation(api_key, org_key, 'text-davinci-003')
        self._salience = Salience(api_key, org_key, 'text-davinci-003')
        self._default_system_msg = open_file('prompts/prompt_system_default.txt')

        self._messages.append({ 'role': 'system', 'content': self._default_system_msg})

    def send_chat(self, msg):
        system_prompt = self._default_system_msg
        anticipation = ''
        salient_points = ''

        # if there has already been at least one message sent
        if len(self._messages) > 1:
            # infer user intent, disposition, valence, needs
            anticipation = self._anticipation.anticipate(self._messages)
            
            # summarize the conversation to the most salient points
            salient_points = self._salience.get_salient_points(self._messages)
            
            # update SYSTEM based upon user needs and salience
            system_prompt += '\nHere\'s a brief summary of the conversation:\n %s \n- And here\'s what I expect the user\'s needs are:\n%s' % (salient_points, anticipation)
        
        self._messages[0]['content'] = system_prompt

        # not a perfect calculation but close enough
        conversation_tokens = len(self._encoding.encode(stringify_conversation(self._messages + [msg])))
        
        # reset conversation and save conversation
        if conversation_tokens + self._max_chat_length > self._max_tokens:
            self.save_messages()
            self._messages.clear()
            self._messages.append(system_prompt)

        # generate a response
        msg_res = self.send(msg)
        return [anticipation, salient_points, msg_res]
    
    def reset(self):
        self.save_messages()
        self._messages[0] = self._default_system_msg

    def save_messages(self):
        if len(self._messages) < 2:
            return # bail if theres nothing to save
        filename = 'chat_%s_user.txt' % time()
        if not os.path.exists('chat_logs'):
            os.makedirs('chat_logs')
        conv = stringify_conversation(self._messages)
        save_file('chat_logs/%s' % filename, conv)

    def load(self):
        if not os.path.exists('chat_logs'):
            print('Error: there are no chat logs to load')
        
        logs = []
        for l in os.listdir('chat_logs'):
            edit_time = os.path.getmtime('chat_logs' + l)
            logs.append(l, edit_time)
        
        def sort_logs(l):
            return l[1]
        logs.sort(key=sort_logs, reverse=True)
        log = logs[0].split('user:')[0]
        self.reset()
        self._messages[0] = log
        
