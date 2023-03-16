from gpt import GptCompletion, EmbeddingFactory, completion_config
from util import open_file, get_closest_embeddings

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
