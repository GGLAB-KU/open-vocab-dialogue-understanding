from multiwoz_utils import MultiWOZ
import os
import json


class Database:
    debug = False
    tables = {}

    def __init__(self, dir, table_names, domains, slots):
        self.dir = dir
        self.utils = MultiWOZ(domains, slots)
        self.parse_tables(dir, table_names)

    def parse_tables(self, dir, table_names):
        for table_name in table_names:
            file_path = os.path.join(dir, table_name+'_db'+'.json')
            if not os.path.isfile(file_path):
                print('table ', table_name,
                      ' has no data. File couldn\'t be found or parsed ', file_path)
                continue
            with open(file_path, 'r') as file:
                table_data = []
                table_keys = []
                column_keys = {}

                data = json.load(file)

                # loop over tables
                for record in data:
                    parsed_record = {}
                    for column in record.keys():
                        # get columns keys
                        if column in column_keys.keys():
                            column_key = column_keys[column]
                        else:
                            slots_key = self.utils.get_slot_key(column, table_name)
                            if slots_key is None:
                                column_key = column
                            else:
                                column_key = slots_key
                            # cache the column key
                            column_keys[column] = column_key

                        if self.is_key(table_name, column) or self.is_key(table_name, column_key):
                            table_keys.append(column_key)
                        # construct the record
                        parsed_record[column_key] = record[column]

                    table_data.append(parsed_record)

                self.tables[table_name] = {}
                self.tables[table_name]['keys'] = table_keys
                self.tables[table_name]['columns'] = column_keys
                self.tables[table_name]['data'] = table_data
                print('Table', table_name, 'has: ',
                      len(table_data), ' records')

    def query_db(self, table, query_values, use_similarity=False):
        if self.debug:
            print('Table: ', table)
            print('query_values= ', query_values)

        results_set = []
        parsed_query_values = {}
        for column in query_values:
            slot_keys = self.utils.get_slot_key(column, table)
            if slot_keys is None:
                continue
            column_key = slot_keys[0]
            parsed_query_values[column_key] = query_values[column]

        records = self.tables[table]['data']
        if self.debug:
            print('parsed_query_values= ', parsed_query_values)

        for record in records:
            matched_column_count = 0
            column_matched = False
            for column_key in parsed_query_values:
                if not column_key in record:
                    continue
                if self.utils.compare_slot_values(column_key, parsed_query_values[column_key], record[column_key], strict=True, use_similarity=use_similarity):
                    column_matched = True
                    matched_column_count += 1
                else:
                    column_matched = False
                    break
            if column_matched:
                results_set.append(record)
        if self.debug:
            print('Total of ', len(results_set), ' records retrieved')
            
        return results_set

    def is_key(self, table_name, column_name):
        if column_name.lower() == 'name':
            return True
        elif table_name.lower() in column_name.lower() and 'id' in column_name.lower():
            return True
        else:
            return False

    def compare_records(self, table, base_record, record, use_similarity=False):
        columns = self.tables[table]['columns']
        records_match = False
        matched_column_count = 0
        result = True

        for column in columns:
            if column in base_record:
                if self.utils.compare_slot_values(base_record[column], record[column], strict=True, use_similarity=use_similarity):
                    records_match = True
                    matched_column_count += 1
                else:
                    records_match = False
                    break
        if records_match and matched_column_count > 0:
            result = True
        else:
            result = False

        return result
