import sgd
import multiwoz
import abcd
import json


class Dataset:

    def __init__(self, type, metadata_file_path, dir):
        self.type = type
        self.dir = dir
        self.train_data = None
        self.dev_data = None
        self.test_data = None
        self.domains = {}
        self.intents = {}
        self.slots = {}
        self.services = {}
        self.parse_metadata(metadata_file_path)

    def parse_metadata(self, metadata_file_path):

        with open(metadata_file_path, 'r') as metadata_file:
            metadata = json.load(metadata_file)
        domains = list(metadata['domains'].keys())
        for domain in domains:
            self.domains[domain.lower()] = metadata['domains'][domain]['description']

        # parse the sercices
        for domain in metadata['domains']:
            print('parsing domain', domain)
            services = metadata['domains'][domain]['services']
            self.services[domain.lower()] = [item.lower() for item in services]
        # parse the intents
        for domain in metadata['domains']:
            intents = metadata['domains'][domain]['intents']
            self.intents[domain.lower()] = [item.lower() for item in intents]

        for domain in domains:
            domain_slots = metadata['domains'][domain]['slots']
            # convert keys and values to lowercase
            self.slots[domain.lower()] = {
                k.lower(): v for k, v in domain_slots.items()}

    def parse(self):
        if self.type == 'SGD':
            self.train_data, self.dev_data, self.test_data = sgd.parse_dataset(
                self.dir)
        elif self.type == 'MWZ':
            self.train_data, self.dev_data, self.test_data = multiwoz.parse_dataset(
                self.dir)
        elif self.type == 'ABCD':
            self.train_data, self.dev_data, self.test_data = abcd.parse_dataset(
                self.dir)

    def get_split_data(self, split):
        if split == 'test':
            raw_data = self.test_data
        elif split == 'dev':
            raw_data = self.dev_data
        elif split == 'train':
            raw_data = self.train_data

        if self.type == 'SGD':
            data = self.service_to_domain(raw_data)
        elif self.type == 'MWZ':
            data = raw_data
        elif self.type == 'ABCD':
            data = abcd.get_prediction_data(raw_data)

        return data

    def get_service_domain(self, service):
        service_domain = None
        service = service.lower()
        for domain in self.domains:
            if service in self.services[domain]:
               service_domain = domain
               break
        return service_domain
    
    def service_to_domain(self, raw_data):
        for dialogue in raw_data:
            for turn in raw_data[dialogue]:
                turn_services = list(raw_data[dialogue][turn]['domains'].keys())
                for service in turn_services:
                    service_domain = self.get_service_domain(service)
                    raw_data[dialogue][turn]['domains'][service_domain] = raw_data[dialogue][turn]['domains'][service]
                    del raw_data[dialogue][turn]['domains'][service]
        return raw_data

def main():
    dataset = Dataset('MWZ', '/home/asafa/workspace/zero-shot-qa-driven-dst/data/MultiWOZ_2.5/domains.json',
                      '/home/asafa/workspace/datasets/MultiWOZ2.4/data/MULTIWOZ2.4')
    dataset.parse()
    predicted_data = dataset.get_prediction_data('test')
    # print(predicted_data)
