import json
import os
from openai.embeddings_utils import cosine_similarity

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