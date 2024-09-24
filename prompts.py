def get_domain_prediction_prompt(model, prediction_level):
    match (model,prediction_level):
        case (_,'DIALOGUE'):
            template = DOMAIN_PREDICTION_TEMPLATE
        case (_,'TURN'):
            template = TURN_DOMAIN_PREDICTION_TEMPLATE
    return template

def get_turn_entity_extraction_prompt(model, prediction_level):
    match (model,prediction_level):
        case (_,'TURN'):
            template = TURN_ENTITY_EXTRACTION_TEMPLATE
    return template

def get_slot_value_selection_prompt(model, prediction_level):
    match (model,prediction_level):
        case (_,'TURN'):
            template = TURN_SLOT_VALUE_SELECTION_TEMPLATE
    return template

def get_dst_task_prompt(model, prediction_level):
    match (model,prediction_level):
        case (_,'DIALOGUE'):
            template = DIALOGUE_DST_PREDICTION_TEMPLATE
        case ("CHATGPT",'TURN'):
            template = TURN_DST_PREDICTION_CHATGPT_TEMPLATE
        case ("LLAMA",'TURN'):
            template = TURN_DST_PREDICTION_LLAMA_TEMPLATE
        case ("GEMINI",'TURN'):
            template = TURN_DST_PREDICTION_GEMINI_TEMPLATE
        case ("VICUNA",'TURN'):
            template = TURN_DST_PREDICTION_GEMINI_TEMPLATE
        case("MISTRALAI", 'TURN'):
            template = TURN_DST_PREDICTION_MISTRALAI_TEMPLATE
    return template

DOMAIN_PREDICTION_TEMPLATE = '''Dialogue Classification Task:
Consider the dialogue below that contains {dialogue_length} turns between two parties: USER and a SYSTEM:
Dialogue below:
{dialogue_string}Objective:
Which of the following domains (one or more domains) the user is asking service for? {domains_names}

Guidelines:
- Classify each turn based on the intents and the context in which the domain-specific terms are used.
- For turns that involve multiple domains, classify the turn under all relevant domains.
- For General turns that don't include services, return None
Format the output in json with zero-based turn index as key and domains as value array with no more details. Make sure to classify every turn.'''

TURN_DOMAIN_PREDICTION_TEMPLATE = '''Dialogue Classification Task:
Consider the dialogue below that contains {dialogue_length} turns between two parties: USER and a SYSTEM:
Dialogue below:
{dialogue_string}Objective:
Which of the following domains (one or more domains) the user is asking service for in the last turn? {domains_names}

Guidelines:
- Classify the last turn (turn index = {turn_index}) based on the intents and the context in which the domain-specific terms are used.
- If the last turn involves multiple domains, classify it under all relevant domains.
- If the last turn doesn't include a service inquiry, return None.
Return the domains in json array with no more explanation or details. The response should be only json array.'''

TURN_DOMAIN_PREDICTION_TEMPLATE_CHAT = '''Consider the following domains or services:
{domains_names}
Now consider the successive turns that I will provide you between two speakers: a USER and a SYSTEM. Which of the following domains (one or more domains) the user is asking service for? Follow the following 3 instructions:

Guidelines:
- Classify the user's turn based on the intents, context and previouse turns
- If the user's turn involves multiple domains, classify it under all relevant domains.
- If the user's turn doesn't include a service inquiry, return None.
Format the output in json array with 'domains' as key and no more details.'''

TURN_ENTITY_EXTRACTION_TEMPLATE = '''I will provide you the definition of the entities you need to extract, the sentence from where your extract the entities and the output format:
Entity definition:
-DAY: Any format of explicit dates.
-TIME: explicit time values like 8:00, 17:00. Please normalize the time to 24-format.
-NUMBER: Any format of number.
-PRICE: price
-LOCATION:  geographic location, address, city, town or area
-NAME: Name of hotel, train station, restaurant or attraction
-TYPE: food category (value could be Italian, ...), taxi car type (value could be tesla, white VW, ...), attraction type or accommodation type (value could be hotel or guesthouse)
-RANGE: price range
-CODE:  reference number, postcode or id.
-BOOLEAN: true or false for exists or doesn't exist
-DONTCARE: no preference or doesn't matter expressions
Output Format: json with the following keys:
'NAME': [list of entities present], 
'DAY': [list of entities present], 
'TIME': [list of entities present], 
'LOCATION': [list of entities present], 
'TIME': [list of entities present], 
'NUMBER': [list of entities present],
'PRICE': [list of entities present],
'TYPE': [list of entities present],
'RANGE': [list of entities present],
'DONTCARE': [list of entities present],
'CODE': [list of entities present],
'BOOLEAN': [list of entities present]

If no entities are presented in any categories keep it [].
Sentence:
{turn}
Output:Let's analyze it step-by-step and extract the values carefully. If you are not sure about any value, don't return it. Focus on the value, not the abstract entity.'''

TURN_SLOT_VALUE_SELECTION_TEMPLATE = '''Consider the dialogue below between USER and SYSTEM:
{dialogue_string}
Can you select the value of the {slot_name} ({slot_description}) in the last turn (turn index = {turn_index}) from the list below? {slot_values_str}

Guidelines:
- Please return the answer in JSON with the {slot_key} as key.
- Don't assume value and just return values from the last turn (turn index = {turn_index}).'''


DIALOGUE_DST_PREDICTION_TEMPLATE = '''Consider the list of concepts, called "slots", provided below with their definitions:{slots_description}
\nNow consider the dialogue below that contains {dialogue_length} turns between two speakers: a USER and a SYSTEM. Please meticulously extract and catalog the slot values from each turn of the dialogue based on the provided slot definitions and follow the following 6 instructions
1. Carefully identify the slot values explicitly mentioned by the speaker in that turn.
2. Ensure you incorporate any acknowledged or accepted slot values from the directly previous turn within the current speaker's turn.
3. For any direct inquiry by the speaker about a specific slot, mark its value as "?".
4. Carefully identify the slots being asked about by the speaker and mark their values as "?".
5. If the speaker explicitly mentions they have no preference or it doesn't matter for a specific slot, mark its value as "*".
6. If a slot isn't mentioned in a turn, do not include it.
\nDialogue:
{dialogue_string}
Ensure thoroughness and accuracy in the identification process. Return the output as json object with zero-based turn index as key and the following as value with no more details:{slots_names}'''

TURN_DST_PREDICTION_CHATGPT_TEMPLATE = '''Consider the list of concepts, called "slots", provided below with their definitions:{slots_description}
\nNow consider the successive turns that I will provide you between two speakers: a USER and a SYSTEM about {domain}. Please meticulously extract and catalog the slot values from each pairs of turns based on the provided slot definitions and follow the following 6 instructions
1. Carefully identify the slot values explicitly mentioned by the speaker in that turn.
2. Ensure you incorporate any acknowledged or accepted slot values from the directly previous turn within the current speaker's turn.
3. For any direct inquiry by the speaker about a specific slot, mark its value as "?".
4. Carefully identify the slots being asked about by the speaker and mark their values as "?".
5. If the speaker explicitly mentions they have no preference or it doesn't matter for a specific slot, mark its value as "*".
6. If a slot isn't mentioned in a turn, do not include it.

Ensure thoroughness and accuracy in the identification process. Return the output as json object with the following as key and their values and no more details:{slots_names}'''

TURN_DST_PREDICTION_CHATGPT_35_TURBO_TEMPLATE = '''
As a dialogue state tracker, your task is to track the slot values that are important to the user during a series of dialogue turns between a USER and a SYSTEM. We are interested in capturing the user's preferences and inquiries about {domain} regarding specific slots.
Slots to Track:
- {slots_description}

Instructions:
1. Track slot values mentioned by the user during each dialogue turn.
2. If the system mentions relevant slot values that are important to the user's context or preferences, track those as well.
3. If the user explicitly states they have no preference or don't care about a specific slot, set its value to *
4. Provide the slot values in a JSON format.
5. Make sure to check all the slots, and don't miss any.
Output Format: json object
{{
  "slot_name": "slot_value"
}}
With the following key: {slots_names}
'''
TURN_DST_PREDICTION_CHATGPT_TEMPLATE = '''
As a dialogue state tracker, your task is to track the slot values that are important to the user during a series of dialogue turns between a USER and a SYSTEM about {domain}.
Slots to Track:
- {slots_description}

Instructions:
1. Track slot values mentioned by the user during each dialogue turn.
2. If the user refers to a slot mentioned by the system and accepts it, retrun that slot with its actual value.
3. If the system mentions relevant slot values that are important to the user's context or preferences, track those as well.
4. If the user states they have no preference or don't care about a specific slot, set its value to *
5. Otherwise, don't return the slot.
6. Return the slot values exactly as they appear in the dialogue without any normalization or conversion, including time and date slots.
7. Provide the slot values in a JSON format.
8. Ensure all specified slots are checked, and none are missed.
Output Format: json object
{{
  "slot_name": "slot_value"
}}
With the following key: {slots_names}
'''
TURN_DST_PREDICTION_LLAMA_TEMPLATE = '''
As a dialogue state tracker, your task is to track the following {domain} slots during the dialogue turns that I will provide afterwards:
slots:
  {slots_description}

Instructions:
- If the slot belongs to another domain, don\'t return it
- The slot value can be one of the following:
1- Slot actual value: if the user mentioned the slot value, return the value
2- Slot actual value: if the system mentioned the slot value and the user didn\'t reject it.
3- *: if the user explicitly states he has no preference
4- ?: if the user explicitly states he is inquiring about a specific slot
5- Otherwise, don\'t return it.
-OutputFormat: json object
{{
  "slot_name": slot_value
}}
With the following key: {slots_names}
'''
TURN_DST_PREDICTION_LLAMA_TEMPLATE = '''
As a dialogue state tracker, your task is to track the following {domain} slots during the dialogue turns that I will provide afterwards:
slots:
  {slots_description}

Instructions:
1- Slot actual value: if the user mentioned the slot value
2- Slot actual value: if the system mentioned the slot value and the user didn\'t reject it.
3- *: if the user states he has no preference
4- Sometimes you need to get the slot value from another domain if the user refers to it.
4- Otherwise, don\'t return it.
-OutputFormat: json object
{{
  "slot_name": slot_value
}}
With the following key: {slots_names}

'''
TURN_DST_PREDICTION_MISTRALAI_TEMPLATE = '''
As a dialogue state tracker, your task is to track the following {domain} slots during the dialogue turns that I will provide afterwards:
slots:
  {slots_description}

Instructions:
- If the slot belongs to another domain, don\'t return it
- The slot value can be one of the following:
1- Slot actual value: if the user mentioned the slot value, return the value
2- Slot actual value: if the system mentioned the slot value and the user didn\'t reject it.
3- Otherwise, don\'t return it.
-OutputFormat: json object
{{
  "slot_name": slot_value
}}
With the following key: {slots_names}
The response should be only json object. Don't return nested json object.
Answer:
'''

TURN_DST_PREDICTION_GEMINI_TEMPLATE = '''
As a dialogue state tracker, your task is to track the following {domain} slots during the dialogue turns that I will provide afterwards:
slots:
  {slots_description}

Instructions:
- If the slot belongs to another domain, don\'t return it
- The slot value can be one of the following:
1- Slot actual value: if the user mentioned the slot value, return the value
2- Slot actual value: if the system mentioned the slot value and the user didn\'t reject it.
3- *: if the user explicitly states he has no preference
4- ?: if the user explicitly states he is inquiring about a specific slot
5- Otherwise, don\'t return it.
-OutputFormat:
{{
  "slot_name": slot_value
}}
'''
TURN_DST_PREDICTION_VICUNA_TEMPLATE = '''
As a dialogue state tracker, your task is to track the following {domain} slots during the dialogue turns that I will provide afterwards:
slots:
  {slots_description}

Instructions:
- If the slot belongs to another domain, don\'t return it
- The slot value can be one of the following:
1- Slot actual value: if the user mentioned the slot value, return the value
2- Slot actual value: if the system mentioned the slot value and the user didn\'t reject it.
3- *: if the user explicitly states he has no preference
4- ?: if the user explicitly states he is inquiring about a specific slot
5- Otherwise, don\'t return it.
-OutputFormat: json object
{{
  "slot_name": slot_value
}}
With the following key: {slots_names}
'''
############################