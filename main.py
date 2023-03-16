from src.andy import Andy
from src.util import open_file
import signal

convo_length = 30
api_key = open_file('key_openai.txt').split('\n')[0]
org_key = open_file('key_org.txt').split('\n')[0]
muse = Andy(api_key, org_key)

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
        msg = muse.load()
        print('Context:\n ' + msg)
        continue
    if user_input == 'CLEAN':
        msg = muse.clean_memories()
        continue
    if user_input == 'WRITE':
        res = muse.write_document()
        print('--------------------')
        print(res)
        continue
    print('Sending message...')
    msg_res = muse.send_chat(user_input)
    print('\n\nMUSE:\n%s' % msg_res + '\n\n')
