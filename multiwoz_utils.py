import os
import time
import csv
import threading
from string_utils import compare_strings

empty_slot_values = ["", 'unknown', 'not specified', 'not available', 'to be determined']

class MultiWOZ:
    domains = []
    slots = {}
    slot_value_pool = {}

    def __init__(self, domains, slots, slot_value_pool_file_path):
        self.domains = domains
        self.slots = slots
        self.slot_value_pool_file_path = slot_value_pool_file_path
        self.slot_value_pool= self.read_slot_value_pool(slot_value_pool_file_path)
       
       # context manager
        self.write_interval = 1200
        self.stop_event = threading.Event()
        self.writer_thread = threading.Thread(target=self.periodic_write)
        self.writer_thread.start()
    
    def periodic_write(self):
        while not self.stop_event.is_set():
            self.dump_slot_value_pool(self.slot_value_pool_file_path)
            time.sleep(self.write_interval)

    def stop(self):
        self.stop_event.set()
        self.writer_thread.join()
        self.dump_slot_value_pool(self.slot_value_pool_file_path)

    def __enter__(self):
        # Any setup code if needed
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Ensure cleanup is done
        self.stop()
    
    def read_slot_value_pool(self, slot_value_pool_file_path):
        slot_value_pool = {}
        try:
            with open(slot_value_pool_file_path,'r') as slot_value_pool_file:
                reader = csv.DictReader(slot_value_pool_file, delimiter='|')
                for row in reader:
                    key = (row['predicted_value'], row['dataset_value'])
                    slot_value_pool[key] = row['equal'].lower() == 'true'
            print(f'Finished reading the {slot_value_pool_file_path}')
        except:
            print(f'{slot_value_pool_file_path} couldn\'t be read')
        
        print(f'Number of entries in {slot_value_pool_file_path} is {len(self.slot_value_pool)}')
        return slot_value_pool
    
    def dump_slot_value_pool(self, slot_value_pool_file_path):
        print(f'Dumping {slot_value_pool_file_path}'    )
        with open(slot_value_pool_file_path, 'w') as slot_value_pool_file:
            writer = csv.writer(slot_value_pool_file, delimiter='|')
            writer.writerow(['predicted_value', 'dataset_value', 'equal'])
            for key, value in self.slot_value_pool.items():
                writer.writerow([key[0], key[1], value])
    
    def __del__(self):
        print('Exit')
        self.dump_slot_value_pool(slot_value_pool_file_path=self.slot_value_pool_file_path)
        
    
    def get_slot_name(self, domain, slot):
        slot_name = domain+'-'+self.slots[domain][slot]['multiwoz_2.2']
        return slot_name
    
    def is_trackable(self, domain, slot):
        if slot not in self.slots[domain]:
            slot = self.get_slot_key(slot_domain= domain, slot_name=slot)
        trackable_slot = self.slots[domain][slot]['trackable']
        return trackable_slot
    
    def is_valid_slot(self, domain, slot):
        if slot in self.slots[domain]:
            valid_slot = True
        else:
            valid_slot = False
        return valid_slot

    def compare_intents(self, intent1, intent2, strict=True):
        result = False
        
        normalized_intent1 = intent1.replace('-', ' ').replace('_',' ')
        normalized_intent2 = intent2.replace('-', ' ').replace('_', ' ')
        if compare_strings(normalized_intent1, normalized_intent2, strict=strict, use_similarity=False):
            result = True
        else:
            result = False
        return result
    
    def normalize_slot(self, slot, domain_names=None):
        slot_clean = slot.lower()
        if domain_names is None:
            slot_clean = slot_clean.replace('value_', '')
        else:
            for domain in domain_names:
                if domain.lower()+'-' in slot_clean:
                    slot_clean = slot_clean.replace(domain+'-', "")

        return slot_clean

    def is_empty_slot(self, slot_name, slot_value):
        empty_slot = False

        if isinstance(slot_value, list):
            if len(slot_value) == 0:
                empty_slot = True
            else:
                empty_slot = False
        
        elif isinstance(slot_value, dict):
            if len(slot_value) == 0:
                empty_slot = True
            else:
                empty_slot = False
        else:
            slot_str_value = str(slot_value).lower()
            if slot_value is None or slot_str_value.upper() == 'N/A' or slot_str_value in empty_slot_values :
                empty_slot = True
            else:
                empty_slot = False

        return empty_slot

    def filter_empty_slot_values(self, slots):
        non_empty_slots = {}
        for slot in slots:
            if not self.is_empty_slot(slot, slots[slot]):
                non_empty_slots[slot] = slots[slot]

        return non_empty_slots

    def get_slot_key(self, slot_name, slot_domain):
        slot_key = None

        domain_slots = self.slots[slot_domain]
        if slot_name in domain_slots:
            slot_key = slot_name
        else:
            slot_clean = self.normalize_slot(slot_name, self.domains)
            if slot_clean in domain_slots:
                slot_key = slot_clean
            else:
                for slot in domain_slots.keys():
                    if slot_name in domain_slots[slot]['aliases'] or slot_clean in domain_slots[slot]['aliases']:
                        slot_key = slot
                        break
        return slot_key

