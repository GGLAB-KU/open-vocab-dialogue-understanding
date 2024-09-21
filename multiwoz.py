import os
import json
import copy

def parse_dataset(data_dir):
    print('reading all dialoge')
    data = read_data(data_dir)
    dialogues = parse_dialogues(data)

    dev_dialogues_file_path = os.path.join(data_dir, 'valListFile.txt')
    with open(dev_dialogues_file_path) as file:
        dev_dialogue_ids = [line.strip() for line in file.readlines()]
    
    test_dialogues_file_path = os.path.join(data_dir, 'testListFile.txt')
    with open(test_dialogues_file_path) as file:
        test_dialogue_ids = [line.strip() for line in file.readlines()]

    train_dialogues, dev_dialogues, test_dialogues = split_dialogues(dialogues, dev_dialogue_ids, test_dialogue_ids)
    print('training data size: ', len(train_dialogues))
    print('development data size: ', len(dev_dialogues))
    print('testing data size: ', len(test_dialogues))

    '''
    with open('debug.json', 'w') as file:
        json.dump(train_dialogues['SNG0258.json'], file, indent=4)
    '''
    return train_dialogues, dev_dialogues, test_dialogues


def read_data(data_dir):
    data = []
    data_file = os.path.join(data_dir,'data.json')
    with open(data_file, 'r') as file:
        data = json.load(file)
    return data

def parse_dialogues(dialogues_data):
    dialogues = {}
    for dialogue_id, dialogue_data in dialogues_data.items():
        dialogue_history = []
        processed_turns = {}
        turns = dialogue_data['log']
        for turn_index, turn in enumerate(turns):
            domains = {}
            speaker = 'USER' if turn_index % 2 == 0 else 'SYSTEM'
            text = turn['text']
            metadata = turn['metadata']
            for domain, domain_data in metadata.items():
                slots = {}
                # Process "book" slots, including "booked"
                for slot_name, slot_value in domain_data['book'].items():
                    if slot_name == 'booked':
                        for booked_item in slot_value:
                            for key, value in booked_item.items():
                                if key in slots:
                                    if not isinstance(slots[key], list):
                                        slots[key] = [slots[key]]
                                    slots[key].append(value)
                                else:
                                    slots[key] = value
                    elif slot_value not in ['not mentioned', ""]:
                        slots[slot_name] = slot_value
                # Process "semi" slots
                for slot_name, slot_value in domain_data['semi'].items():
                    if slot_value not in ['not mentioned', ""]:
                        slots[slot_name] = slot_value
                # Assuming a process for "requested_slots" or the third type of slot goes here

                # Only add domain data if slots or requested slots are present
                if slots:
                    domains[domain] = {'slots': slots, 'requested_slots': []}  # Adjust as needed for requested_slots

            turn_data = {
                 'domains': domains,
                 'id': turn_index,
                 'speaker': speaker,
                 'text': text,
                 'history': copy.deepcopy(dialogue_history)
             }
            dialogue_history.append(f"{speaker}: {text}")
            # to append slots to the user turn as well
            #### WARNING: commented for debugging
            #if speaker == 'SYSTEM':
                #processed_turns[str(turn_index-1)]['domains']=domains
            processed_turns[str(turn_index)]=turn_data

        dialogues[dialogue_id] = processed_turns

    return dialogues


def split_dialogues(dialogues, dev_dialogue_ids, test_dialogue_ids):
    train_dialogues = {}
    dev_dialogues = {}
    test_dialogues = {}
    for dialogue_id in dialogues:
        if dialogue_id in dev_dialogue_ids:
            dev_dialogues[dialogue_id] = dialogues[dialogue_id]
        elif dialogue_id in test_dialogue_ids:
            test_dialogues[dialogue_id] = dialogues[dialogue_id]
        else:
            train_dialogues[dialogue_id] = dialogues[dialogue_id]
    return train_dialogues, dev_dialogues, test_dialogues