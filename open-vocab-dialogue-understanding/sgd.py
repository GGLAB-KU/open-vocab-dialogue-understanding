import os
import json
import copy

def parse_dataset(data_dir):
    print('parsing training data')
    train_data = read_fold_data(data_dir, 'train')
    train_dialogues = parse_dialogues(train_data)
    
    print('training data size: ', len(train_dialogues))

    print('parsing development data')
    dev_data = read_fold_data(data_dir, 'dev')
    dev_dialogues = parse_dialogues(dev_data)
    print('development data size: ', len(dev_dialogues))
    

    print('parsing testing data')
    test_data = read_fold_data(data_dir, 'test')
    test_dialogues = parse_dialogues(test_data)
    print('testing data size: ', len(test_dialogues))

    return train_dialogues, dev_dialogues, test_dialogues


def read_fold_data(data_dir, fold):
    data = []
    fold_dir = os.path.join(data_dir, fold)
    for filename in os.listdir(fold_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(fold_dir, filename)
            with open(file_path, 'r') as file:
                file_data = json.load(file)
                data.extend(file_data)
    return data

def parse_dialogues(dialogues_data):
    dialogues = []
    for dialogue_data in dialogues_data:
        if not 'dialogue_id' in dialogue_data:
            print('WARNING: dialogue_id not found')
            continue
        dialogue_id = dialogue_data['dialogue_id']
        turns = []
        turn_index = 0
        for turn_data in dialogue_data['turns']:
            frames = []
            speaker = turn_data['speaker']
            if 'turn_id' in turn_data:
                turn_id = turn_data['turn_id']
            else:
                turn_id = str(turn_index)
            text = turn_data['utterance']

            for frame_data in turn_data['frames']:
                slots = []
                requested_slots = []
                state_slots = []
                intent = None

                actions = frame_data['actions']
                domain = frame_data['service']
                for frame_slot in frame_data['slots']:
                    # added to handle SGD data
                    if not 'value' in frame_slot:
                        continue
                    slot = dict()
                    slot['slot'] = frame_slot['slot']
                    slot['value'] = frame_slot['value']
                    slots.append(slot)

                if 'state' in frame_data.keys():
                    state_data = frame_data['state']
                    intent = state_data['active_intent']

                    for slot_name in state_data['slot_values'].keys():
                        slot_values = state_data['slot_values'][slot_name]
                        slot = dict()
                        slot['slot'] = slot_name
                        slot['value'] = slot_values
                        state_slots.append(slot)
                    requested_slots = state_data['requested_slots']
                
                #if speaker == 'USER' and len(requested_slots) == 0 and len(state_slots) == 0:
                #    continue
                
                frame = dict()
                frame['actions'] = actions
                frame['domain'] = domain
                frame['intent'] = intent
                frame['slots'] = slots
                frame['requested_slots'] = requested_slots
                frame['state_slots'] = state_slots
                frames.append(frame)

            turn = dict()
            turn['frames'] = frames
            turn['id'] = turn_id
            turn['speaker'] = speaker
            turn['text'] = text

            turns.append(turn)
            turn_index += 1
        dialogue = dict()
        dialogue['id'] = dialogue_id
        dialogue['turns'] = turns
        dialogues.append(dialogue)

    return format_data(dialogues)

def format_data(dialogues):

    processed_dialogues = dict()

    for dialogue in dialogues:
        dialogue_history = []
        dialogue_id =  dialogue['id']
        processed_dialogue = dict()
        for turn in dialogue['turns']:
            processed_turn = dict()
            domains = []
            frames = dict()
            turn_id =turn['id']
            
            text = turn['text']
            history= copy.deepcopy(dialogue_history)
            speaker = turn['speaker']

            for frame in turn['frames']:
                domain_slots = dict()
                domain = frame['domain']
                if speaker == 'USER':
                    for slot in frame['state_slots']:
                        domain_slots[slot['slot']] = slot['value']
                
                elif speaker == 'SYSTEM':
                    for slot in frame['slots']:
                        domain_slots[slot['slot']] = slot['value']
                
                #if domain_slots:
                if True:
                     domains.append(domain)
                     frame_dict = dict()
                     frame_dict['requested_slots'] = frame['requested_slots']
                     frame_dict['slots'] = domain_slots
                     frame_dict['intent'] = frame['intent']
                     frame_dict['actions'] = frame['actions']
                     frames[domain] = frame_dict
            
            
            processed_turn['text'] = text
            processed_turn['history'] = history
            processed_turn['speaker'] = speaker
            processed_turn['domains'] = frames
            

            processed_dialogue[turn_id] = processed_turn
            dialogue_history.append(speaker+': '+text)

        processed_dialogues[dialogue_id] = processed_dialogue

    return processed_dialogues