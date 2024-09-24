import os
import json
import os
import google.generativeai as googleai
import requests
from openai import OpenAI
import cohere
import time
from string_utils import extract_json
import requests
import re
import replicate
import prompts
from octoai.text_gen import ChatMessage
from octoai.client import OctoAI
import logging
import logging.config


class ChatBot:
    dst_parameters = {  # dialogue state tracking
        'CHATGPT': {'temperature': 0.5, 'top_p': 0.9, 'EOS': '\n', 'max_request_repeat_count': 5},
        'PALM': {'temperature': 0.7, 'top_p': 1, 'EOS': '\n', 'max_request_repeat_count': 5},
        'GEMINI': {'temperature': 0.9, 'top_p': 1, 'EOS': '\n', 'max_request_repeat_count': 5},
        'LLAMA': {'temperature': 0.7, 'top_p': 0.9, 'EOS': '\n', 'max_request_repeat_count': 5},
        'MISTRAL': {'temperature': 0.25, 'top_p': 1, 'EOS': '\n', 'max_request_repeat_count': 5},
        'QWEN': {'temperature': 0.25, 'top_p': 1, 'EOS': '<|im_end|>', 'max_request_repeat_count': 5},
        'STARLING': {'temperature': 0.7, 'top_p': 1, 'EOS': '\n', 'max_request_repeat_count': 5},
        'MISTRALAI': {'temperature': 0.7, 'top_p': 1, 'EOS': '\n', 'max_request_repeat_count': 5},
        'COHERE': {'temperature': 0.7, 'top_p': 1, 'EOS': '\n', 'max_request_repeat_count': 5},
        'VICUNA': {'temperature': 0.7, 'top_p': 1, 'EOS': '\n', 'max_request_repeat_count': 5},
        'WIZARDLM': {'temperature': 0.7, 'top_p': 0.9, 'EOS': '\n', 'max_request_repeat_count': 5}
    }
    dc_parameters = {  # domain classification
        'CHATGPT': {'temperature': 0.3, 'top_p': 0.9, 'EOS': '\n', 'max_request_repeat_count': 5},
        'PALM': {'temperature': 0.8, 'top_p': 1, 'EOS': '\n', 'max_request_repeat_count': 5},
        'GEMINI': {'temperature': 0.8, 'top_p': 1, 'EOS': '\n', 'max_request_repeat_count': 5},
        'LLAMA': {'temperature': 0.25, 'top_p': 0.9, 'EOS': '\n', 'max_request_repeat_count': 5},
        'QWEN': {'temperature': 0.25, 'top_p': 1, 'EOS': '<|im_end|>', 'max_request_repeat_count': 5},
        'VICUNA': {'temperature': 0.1, 'top_p': 1, 'EOS': '\n', 'max_request_repeat_count': 5},
        'WIZARDLM': {'temperature': 0.25, 'top_p': 1, 'EOS': '\n', 'max_request_repeat_count': 5},
        'MISTRALAI': {'temperature': 0.25, 'top_p': 0.9, 'EOS': '\n', 'max_request_repeat_count': 5},
    }

    open_source_models = ['STARLING']
    octoai_models = ['LLAMA', 'QWEN', 'MISTRALAI']
    togetherai_models = ['KOALA', 'VICUNA']
    deepinfra_models = ['WIZARDLM']
    open_source_model_host = '127.0.0.1'
    open_source_model_port = '9999'
    response_format = 'json'  # text or json
    model = 'WIZARDLM'  # BARD, PALM, LLAMA, WIZARDLM, COHERE, QWEN or CHATGPT
    # CHATGPT: 'gpt-3.5-turbo', 'gpt-4', gpt-4-1106-preview
    # QWEN: 'qwen1.5-32b-chat'
    # PALM: 'chat', 'text'
    # COHERE: command
    sub_model = 'command'

    def __init__(self, model, sub_model, response_format, prompt_templates_file, log_dir='./debug', debug=True):
        os.environ['GOOGLE_API_KEY'] = ""
        os.environ['OPENAI_API_KEY'] = ""
        os.environ['COHERE_API_KEY'] = ""
        os.environ['REPLICATE_API_TOKEN'] = ""
        os.environ['OCATOAI_API_TOKEN'] = ""
        os.environ['TOGETHER_API_KEY'] = ""
        os.environ['DEEPINFRA_API_KEY'] = ""

        self.model = model
        self.sub_model = sub_model
        self.response_format=response_format

        # set the logger
        if debug:
            self.set_default_logger()
            print('Use the default logger')
        else:
            with open('log_config.json', 'rt') as log_config_file:
                config = json.load(log_config_file)
            config['handlers']['fileHandler']['filename'] = model.replace(
                '/', '-')+'-'+sub_model.replace('/', '-')+'_'+config['handlers']['fileHandler']['filename']
            config['handlers']['fileHandler']['filename'] = os.path.join(
                log_dir, config['handlers']['fileHandler']['filename'])
            print(config['handlers']['fileHandler']['filename'])
            logging.config.dictConfig(config)

            # Get the logger
            self.logger = logging.getLogger('chatbot')

        self.logger.info('Logger setup complete, starting Chatbot')
        # set the prompt template
        with open(prompt_templates_file, 'r', encoding="utf-8") as file:
            prompt_templates = json.load(file)
        self.prompt_template = prompt_templates[model]
        self.logger.info(f'Prompt template = {self.prompt_template}')

        if model in self.open_source_models:
            self.load_open_source_model()

        elif model in self.octoai_models:
            self.logger.info('Initiate OCTOAI')
            self.octoai = OctoAI(api_key=os.getenv("OCATOAI_API_TOKEN"))

        elif model in self.togetherai_models:
            self.logger.info('Initiate TogetherAI')
            self.togtherai = OpenAI(api_key=os.environ.get(
                "TOGETHER_API_KEY"), base_url="https://api.together.xyz/v1/openai",)
        
        elif model in self.deepinfra_models:
            self.logger.info('Initiate DeepInfra')
            self.deepinfra = OpenAI(api_key=os.environ.get(
                "DEEPINFRA_API_KEY"), base_url="https://api.deepinfra.com/v1/openai",)

        elif model == 'PALM':
            self.logger.info('Initiate PALM')

        elif model == 'GEMINI':
            self.logger.info('Initiate GEMINI')
            googleai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

        elif model == 'CHATGPT':
            self.logger.info('Initiate ChatGPT')
            self.chatGPT = OpenAI()
            self.chatGPT.api_key = os.getenv("OPENAI_API_KEY")

        elif (model == 'LLAMA'):
            self.logger.info('Initiate Llama')

        elif (model == "COHERE"):
            self.cohere = cohere.Client(os.getenv("COHERE_API_KEY"))

        else:
            raise NotImplementedError

    def set_default_logger(self):

        logger = logging.getLogger('chatbot')
        logger.setLevel(logging.DEBUG)
        file_handler_normalization = logging.FileHandler(
            filename='debug/chatbot_debug.log', mode='w')
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s -%(message)s')
        file_handler_normalization.setFormatter(formatter)
        logger.addHandler(file_handler_normalization)
        self.logger = logger

    def load_open_source_model(self):
        api_url = 'http://'+self.open_source_model_host+':' + \
            self.open_source_model_port+'/'+'load'
        request = {'model': self.sub_model}
        response = requests.post(api_url, json=request)
        answer = response.json()['result']

    def query_open_source_model(self, prompt, temperature, top_p):
        api_url = 'http://'+self.open_source_model_host+':' + \
            self.open_source_model_port+'/'+'predict'
        request = {'model': self.sub_model, "prompt": prompt,
                   "temperature": temperature, "top_p": top_p, "max_gen_len": 2048}
        response = requests.post(api_url, json=request)
        answer = response.json()['prediction'].replace(prompt, '')
        return answer

    def query_palm(self, prompt, temperature):
        googleai.configure()
        safety_settings = [{"category": "HARM_CATEGORY_DEROGATORY", "threshold": "BLOCK_NONE"}, {"category": "HARM_CATEGORY_TOXICITY", "threshold": "BLOCK_NONE"}, {"category": "HARM_CATEGORY_VIOLENCE", "threshold": "BLOCK_NONE"}, {
            "category": "HARM_CATEGORY_SEXUAL", "threshold": "BLOCK_NONE"}, {"category": "HARM_CATEGORY_MEDICAL", "threshold": "BLOCK_NONE"}, {"category": "HARM_CATEGORY_DANGEROUS", "threshold": "BLOCK_NONE"}]
        if self.sub_model == 'text':
            response = googleai.generate_text(
                prompt=prompt, temperature=temperature, safety_settings=safety_settings)
            answer = response.result

        elif self.sub_model == 'chat':
            response = googleai.chat(
                prompt=prompt, temperature=temperature, top_k=40, top_p=0.95, candidate_count=1, safety_settings=safety_settings)
            answer = response.last

        return answer

    def query_gemini(self, prompt, temperature):
        generation_config = {
            "temperature": temperature,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 2048,
        }
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
        ]

        model = googleai.GenerativeModel(model_name=self.sub_model,
                                         generation_config=generation_config,
                                         safety_settings=safety_settings)
        response = model.generate_content([prompt])
        self.logger.info(response.text)
        return response.text

    def query_chatgpt(self, prompt, temperature=0.7, top_p=0.9):
        answer = ''

        if 'gpt-' in self.sub_model:
            if self.response_format == 'json':
                completion = self.chatGPT.chat.completions.create(
                    model=self.sub_model,
                    messages=[{"role": "system", "content": "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible."}, {
                        "role": "user", "content": "{0}".format(prompt)}],
                    temperature=temperature,
                    response_format={"type": "json_object"},
                    max_tokens=2048
                )
                answer = completion.choices[0].message.content
            else:
                completion = self.chatGPT.chat.completions.create(
                    model=self.sub_model,
                    messages=[{"role": "system", "content": "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible."}, {
                        "role": "user", "content": "{0}".format(prompt)}],
                    temperature=temperature,
                    max_tokens=2048
                )
                answer = completion.choices[0].message.content

        elif 'davinci' in self.sub_model:
            raise NotImplementedError
        else:
            raise NotImplementedError

        return answer

    def query_octoai(self, prompt, temperature, top_p):
        completion = self.octoai.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Keep your responses limited to one short paragraph if possible."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=self.sub_model,
            max_tokens=128,
            presence_penalty=0,
            temperature=temperature,
            top_p=top_p,
        )
        return completion.choices[0].message.content

    def query_togtherai(self, prompt, temperature, top_p):
        completion = self.togetherai.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Be precise and prvoide accurate answers."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=self.sub_model,
            max_tokens=1024,
            temperature=temperature,
            top_p=top_p,
        )
        return completion.choices[0].message.content

    def query_deepinfra(self, prompt, temperature=0.7, top_p=0.9):
        self.logger.info(f'query_deepinfra prompt: {prompt}')
        if self.response_format == 'json':
            completion = self.deepinfra.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.sub_model,
                max_tokens=2028,
                temperature=temperature,
                top_p=top_p,
                response_format={"type": "json_object"}
            )
        else:
            completion = self.deepinfra.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.sub_model,
                max_tokens=2028,
                temperature=temperature,
                top_p=top_p,
            )
        self.logger.info(f'promp: {prompt}')
        self.logger.info(f'Anser: {completion.choices[0].message.content}')
        return completion.choices[0].message.content

    def query_llama_replicate(self, prompt, temperature, top_p):
        output = replicate.run(
            "meta/llama-2-70b-chat",
            input={
                "debug": False,
                "top_p": top_p,
                "prompt": prompt,
                "temperature": temperature,
                "system_prompt": "Return the result in json with no space in the key and no more details",
                "max_new_tokens": 500,
                "min_new_tokens": -1
            }
        )
        self.logger.info('llama response = ', output)
        return ''.join(output)

    def query_cohere(self, prompt, temperature):
        response = self.cohere.chat(
            chat_history=[],
            message=prompt,
            connectors=[]
        )
        answer = response.text
        '''response = self.cohere.generate(
            model=self.sub_model,
            prompt=prompt,
            max_tokens=300,
            temperature=temperature)
        
        answer = response.generations[0].text'''
        return answer

    def query_llm(self, prompt, temperature=0.7, top_p=0.9, max_try_count=10):
        answer = ''
        answer_dic = {}

        answer_received = False
        try_counter = 0

        self.logger.info(
            '------------------------------------------------------')
        self.logger.info('PROCESS STARTED: quer_llm()')

        # create prompt from template
        if isinstance(prompt, tuple):
            # prompt with instructions
            prompt = self.prompt_template.format(
                instructions=prompt[0], prompt_input=prompt[1])
        else:
            prompt = self.prompt_template.format(prompt_input=prompt)

        self.logger.info('question= %s', prompt)

        # Define a dictionary to mimic switch-case
        query_methods = {
            'CHATGPT': lambda: self.query_chatgpt(prompt=prompt, temperature=temperature, top_p=top_p),
            'BARD': lambda: self.query_bard(prompt),
            'PALM': lambda: self.query_palm(prompt, temperature=temperature),
            'GEMINI': lambda: self.query_gemini(prompt, temperature=temperature),
            'WIZARDLM': lambda: self.query_deepinfra(prompt, temperature=temperature),
            'LLAMA': lambda: self.query_octoai(prompt, temperature=temperature, top_p=top_p),
            'COHERE': lambda: self.query_cohere(prompt, temperature=temperature),
            'MISTRAL': lambda: self.query_octoai(prompt, temperature=temperature, top_p=top_p),
            'STARLING': lambda: self.query_open_source_model(prompt, temperature=temperature, top_p=top_p),
            'MISTRALAI': lambda: self.query_octoai(prompt, temperature=temperature, top_p=top_p),
            'QWEN': lambda: self.query_octoai(prompt, temperature=temperature, top_p=top_p),
            'VICUNA': lambda: self.query_togtherai(prompt, temperature=temperature, top_p=top_p)
        }

        while not answer_received and try_counter < max_try_count:
            try_counter += 1
            self.logger.info(f'Trial#{try_counter} of {max_try_count}')
            try:
                query_func = query_methods.get(self.model)
                if query_func is None:
                    raise NotImplementedError

                answer = query_func()
                self.logger.info(f'answer: {answer}')
                answer_dic = json.loads(answer)

                if len(answer_dic) > 0:
                    answer_received = True

            except Exception as exception:
                self.logger.exception(exception)
                self.logger.info(f'Rescived answer: {answer}')
                self.logger.info(
                    'Couldn\' parse the answer as josn. Will try to clean the answer first')
                answer = extract_json(answer)
                self.logger.info(f'clean answer= {answer}')
                if len(answer) > 0:
                    answer_dic = json.loads(answer)
                    if len(answer_dic) > 0:
                        answer_received = True
                    else:
                        self.logger.info('Answer dictionary is empty')
                        time.sleep(0.20)
                else:
                    self.logger.info(
                        'Answer is empty after cleaning. Here is the original exception:')
                    self.logger.info(exception)
                    time.sleep(0.20)
        self.logger.info(f'answer_received: {answer_received}')
        if not answer_received or answer_dic is None:
            answer_dic = {}
            self.logger.info(
                'Finished all the possible trials but couldn\'t get answer')

        self.logger.info('Answer = %s', str(answer_dic))
        self.logger.info('PROCESS COMPLETED: quer_llm()')
        return answer_dic

    def run_openai_chat(self, system_prompt, turns, speakers, temperature, top_p=1.0):
        # Start with the system providing the task description
        messages = [{"role": "system", "content": system_prompt}]

        # Process each turn, alternating between user and system
        active_turns = []
        active_speakers = []
        for turn_index in range(len(turns)):
            # Determine if it's a user or system turn
            speaker = speakers[turn_index]
            turn = turns[turn_index]

            active_turns.append(turn)
            active_speakers.append(speaker)

            # Only send a request to the API for user inputs; system inputs are scripted/planned
            if speaker.lower() == "user":
                dialogue_str = self.dialogue_to_string(
                    speakers=active_speakers, turns=active_turns, eos=self.dst_parameters[self.model]['EOS'], append_turn_index=False)
                messages.append({"role": 'user', "content": dialogue_str})

                if self.response_format == 'json':
                    response = self.chatGPT.chat.completions.create(
                        model=self.sub_model,
                        messages=messages.copy(),
                        temperature=temperature,
                        response_format={"type": "json_object"},
                        max_tokens=2048
                    )
                else:
                    response = self.chatGPT.chat.completions.create(
                        model=self.sub_model,
                        messages=messages.copy(),
                        temperature=temperature,
                        max_tokens=2048
                    )

                system_response = response.choices[0].message.content

                # Append the system's response to the conversation history
                messages.append(
                    {"role": "assistant", "content": system_response})

                active_turns = []
                active_speakers = []

        chat_log = messages
        return chat_log

    def run_octoai_chat(self, system_prompt, turns, speakers, temperature, top_p=0.9):
        # Start with the system providing the task description
        messages = [{"role": "system", "content": system_prompt}]

        # Process each turn, alternating between user and system
        active_turns = []
        active_speakers = []
        for turn_index in range(len(turns)):
            # Determine if it's a user or system turn
            speaker = speakers[turn_index]
            turn = turns[turn_index]

            active_turns.append(turn)
            active_speakers.append(speaker)

            # Only send a request to the API for user inputs; system inputs are scripted/planned
            if speaker.lower() == "user":
                dialogue_str = self.dialogue_to_string(
                    speakers=active_speakers, turns=active_turns, eos=self.dst_parameters[self.model]['EOS'], append_turn_index=False)
                messages.append({"role": 'user', "content": dialogue_str})
                if self.response_format == 'json':
                    response = self.octoai.chat.completions.create(
                        model=self.sub_model,
                        messages=messages.copy(),
                        temperature=temperature,
                        top_p=top_p,
                        presence_penalty=0,
                        response_format={"type": "json_object"},
                        max_tokens=2048
                    )
                else:
                    response = self.octoai.chat.completions.create(
                        model=self.sub_model,
                        messages=messages.copy(),
                        temperature=temperature,
                        top_p=top_p,
                        presence_penalty=0,
                        max_tokens=2048
                    )

                system_response = response.choices[0].message.content
                try:
                    json.loads(system_response)
                except:
                    self.logger.info(
                        f'system_response before cleaning: {system_response}')
                    system_response = extract_json(system_response)
                    self.logger.info(
                        f'system_response after cleaning: {system_response}')

                # Append the system's response to the conversation history
                messages.append(
                    {"role": "assistant", "content": system_response})

                active_turns = []
                active_speakers = []

        chat_log = messages
        return chat_log

    def run_togetherai_chat(self, system_prompt, turns, speakers, temperature, top_p=0.9):
        # Start with the system providing the task description
        messages = [{"role": "system", "content": system_prompt}]

        # Process each turn, alternating between user and system
        active_turns = []
        active_speakers = []
        for turn_index in range(len(turns)):
            # Determine if it's a user or system turn
            speaker = speakers[turn_index]
            turn = turns[turn_index]

            active_turns.append(turn)
            active_speakers.append(speaker)

            # Only send a request to the API for user inputs; system inputs are scripted/planned
            if speaker.lower() == "user":
                dialogue_str = self.dialogue_to_string(
                    speakers=active_speakers, turns=active_turns, eos=self.dst_parameters[self.model]['EOS'], append_turn_index=False)
                messages.append({"role": 'user', "content": dialogue_str})
                if self.response_format == 'json':
                    response = self.togetherai.chat.completions.create(
                        model=self.sub_model,
                        messages=messages.copy(),
                        temperature=temperature,
                        top_p=top_p,
                        presence_penalty=0,
                        response_format={"type": "json_object"},
                        max_tokens=2048
                    )
                else:
                    response = self.togetherai.chat.completions.create(
                        model=self.sub_model,
                        messages=messages.copy(),
                        temperature=temperature,
                        top_p=top_p,
                        presence_penalty=0,
                        max_tokens=2048
                    )

                system_response = response.choices[0].message.content
                try:
                    json.loads(system_response)
                except:
                    self.logger.info(
                        f'system_response before cleaning: {system_response}')
                    system_response = extract_json(system_response)
                    self.logger.info(
                        f'system_response after cleaning: {system_response}')

                # Append the system's response to the conversation history
                messages.append(
                    {"role": "assistant", "content": system_response})

                active_turns = []
                active_speakers = []

        chat_log = messages
        return chat_log
    
    def run_deepinfra_chat(self, system_prompt, turns, speakers, temperature, top_p=0.9):
        # Start with the system providing the task description
        messages = [{"role": "system", "content": system_prompt}]

        # Process each turn, alternating between user and system
        active_turns = []
        active_speakers = []
        for turn_index in range(len(turns)):
            # Determine if it's a user or system turn
            speaker = speakers[turn_index]
            turn = turns[turn_index]

            active_turns.append(turn)
            active_speakers.append(speaker)

            # Only send a request to the API for user inputs; system inputs are scripted/planned
            if speaker.lower() == "user":
                dialogue_str = self.dialogue_to_string(
                    speakers=active_speakers, turns=active_turns, eos=self.dst_parameters[self.model]['EOS'], append_turn_index=False)
                messages.append({"role": 'user', "content": dialogue_str})
                if self.response_format == 'json':
                    response = self.deepinfra.chat.completions.create(
                        model=self.sub_model,
                        messages=messages.copy(),
                        temperature=temperature,
                        top_p=top_p,
                        presence_penalty=0,
                        response_format={"type": "json_object"},
                        max_tokens=2048
                    )
                else:
                    response = self.deepinfra.chat.completions.create(
                        model=self.sub_model,
                        messages=messages.copy(),
                        temperature=temperature,
                        top_p=top_p,
                        presence_penalty=0,
                        max_tokens=2048
                    )

                system_response = response.choices[0].message.content
                try:
                    json.loads(system_response)
                except:
                    self.logger.info(
                        f'system_response before cleaning: {system_response}')
                    system_response = extract_json(system_response)
                    self.logger.info(
                        f'system_response after cleaning: {system_response}')

                # Append the system's response to the conversation history
                messages.append(
                    {"role": "assistant", "content": system_response})

                active_turns = []
                active_speakers = []

        chat_log = messages
        return chat_log
    
    def run_gemini_chat(self, system_prompt, turns, speakers, temperature, top_p=0.9):
        generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": 0,  # review
            "max_output_tokens": 2048,
        }
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_ONLY_HIGH"
            },
        ]

        model = googleai.GenerativeModel(model_name=self.sub_model,
                                         generation_config=generation_config,
                                         safety_settings=safety_settings)

        # Start with the system providing the task description
        messages = [{"role": "user", "parts": [system_prompt]}]
        convo = model.start_chat(history=[])
        convo.send_message(system_prompt)

        # Process each turn, alternating between user and system
        active_turns = []
        active_speakers = []
        for turn_index in range(len(turns)):
            # Determine if it's a user or system turn
            speaker = speakers[turn_index]
            turn = turns[turn_index]

            active_turns.append(turn)
            active_speakers.append(speaker)

            # Only send a request to the API for user inputs; system inputs are scripted/planned
            if speaker.lower() == "user":
                dialogue_str = self.dialogue_to_string(
                    speakers=active_speakers, turns=active_turns, eos=self.dst_parameters[self.model]['EOS'], append_turn_index=False)
                messages.append({"role": 'user', "content": dialogue_str})

                convo.send_message(dialogue_str)
                system_response = convo.last.text
                self.logger.info(f'system_response: {system_response}')
                try:
                    json.loads(system_response)
                except:
                    self.logger.info(
                        f'system_response before cleaning: {system_response}')
                    system_response = extract_json(system_response)
                    self.logger.info(
                        f'system_response after cleaning: {system_response}')

                # Append the system's response to the conversation history
                messages.append(
                    {"role": "assistant", "content": system_response})

                active_turns = []
                active_speakers = []

        chat_log = messages
        return chat_log

    def run_llm_chat(self, system_prompt, turns, speakers, temperature=0.7, top_p=0.9):
        chat_log = []

        self.logger.info(
            '------------------------------------------------------')
        self.logger.info('PROCESS STARTED: run_llm_chat()')
        self.logger.info(f'question= {system_prompt}')
        self.logger.info(system_prompt)
        self.logger.info(str(turns))
        self.logger.info(str(speakers))

        # Define a dictionary to mimic switch-case
        query_methods = {
            'CHATGPT': lambda: self.run_openai_chat(system_prompt=system_prompt, turns=turns, speakers=speakers, temperature=temperature),
            'QWEN': lambda: self.run_octoai_chat(system_prompt=system_prompt, turns=turns, speakers=speakers, temperature=temperature),
            'LLAMA': lambda: self.run_octoai_chat(system_prompt=system_prompt, turns=turns, speakers=speakers, temperature=temperature),
            'MISTRALAI': lambda: self.run_octoai_chat(system_prompt=system_prompt, turns=turns, speakers=speakers, temperature=temperature),
            'KOALA': lambda: self.run_togetherai_chat(system_prompt=system_prompt, turns=turns, speakers=speakers, temperature=temperature),
            'VICUNA': lambda: self.run_togetherai_chat(system_prompt=system_prompt, turns=turns, speakers=speakers, temperature=temperature),
            'GEMINI': lambda: self.run_gemini_chat(system_prompt=system_prompt, turns=turns, speakers=speakers, temperature=temperature),
            'WIZARDLM': lambda: self.run_deepinfra_chat(system_prompt=system_prompt, turns=turns, speakers=speakers, temperature=temperature)
        }

        try:
            query_func = query_methods.get(self.model)
            if query_func is None:
                raise NotImplementedError

            chat_log = query_func()
        except Exception as exception:
            self.logger.info('chat_log is none')
            self.logger.info(str(exception))
            chat_log = []

        self.logger.info('Chatlog= %s', str(chat_log))
        self.logger.info('PROCESS COMPELTED: run_llm_chat()')

        return chat_log

    def predict_dialogue_domains(self, turns, speakers, available_domains, prediction_level, use_domain_description=False):

        domains_names = self.domains_to_string(available_domains, append_description=use_domain_description)
        predicted_domains = {}

        task_prompt = prompts.get_domain_prediction_prompt(
            model=self.model, prediction_level=prediction_level)

        # Predict the dialogue domains in one shot
        if prediction_level == 'DIALOGUE':

            dialogue_str = self.dialogue_to_string(
                speakers=speakers, turns=turns, eos=self.dc_parameters[self.model]['EOS'], append_turn_index=True)

            prompt = task_prompt.format(dialogue_length=len(
                turns), dialogue_string=dialogue_str, domains_names=domains_names)

            predicted_domains = self.query_llm(prompt=prompt, temperature=self.dc_parameters[self.model]['temperature'], top_p=self.dc_parameters[
                                               self.model]['top_p'], max_try_count=self.dc_parameters[self.model]['max_request_repeat_count'])

        # Predict the dialogue domains turn by turn
        elif prediction_level == 'TURN':
            active_conversion = []
            active_conversion_speakers = []
            for turn_index in range(0, len(turns), 2):
                user_turn = turns[turn_index]
                system_turn = turns[turn_index+1]
                active_conversion.append(user_turn)
                active_conversion_speakers.append('USER')

                dialogue_str = self.dialogue_to_string(
                    speakers=active_conversion_speakers, turns=active_conversion, eos=self.dc_parameters[self.model]['EOS'], append_turn_index=True)
                prompt = task_prompt.format(dialogue_length=len(
                    active_conversion), dialogue_string=dialogue_str, domains_names=domains_names, turn_index=len(active_conversion)-1)
                predicted_domain_batch = self.query_llm(prompt=prompt, temperature=self.dc_parameters[self.model]['temperature'], top_p=self.dc_parameters[
                                                        self.model]['top_p'], max_try_count=self.dc_parameters[self.model]['max_request_repeat_count'])

                active_conversion.append(system_turn)
                active_conversion_speakers.append('SYSTEM')
                # if statement to handle somecases returned from Llama
                if 'domains' in predicted_domain_batch:
                    turn_predicted_domains = predicted_domain_batch['domains']
                elif isinstance(predicted_domain_batch, list) and 'domains' in predicted_domain_batch[0]:
                   turn_predicted_domains =  predicted_domain_batch[0]['domains']
                elif isinstance(predicted_domain_batch, list):
                    turn_predicted_domains = predicted_domain_batch
                else:
                    self.logger.info(
                        f'ERROR: couldn\'t extract domains from json {predicted_domain_batch}')
                    turn_predicted_domains = []
                if turn_predicted_domains == None or (isinstance(turn_predicted_domains,str) and turn_predicted_domains.lower() == 'none'):
                    turn_predicted_domains = {}
                predicted_domains[str(turn_index)] = turn_predicted_domains
                predicted_domains[str(turn_index+1)] = turn_predicted_domains

        elif prediction_level == 'TURN-CHAT':
            prompt = task_prompt.format(domains_names=domains_names)
            chat_log = self.run_llm_chat(prompt, turns, speakers, temperature=self.dc_parameters[self.model][
                'temperature'], top_p=self.dc_parameters[self.model]['top_p'])
            for message_index in range(len(chat_log)):
                if chat_log[message_index]['role'].lower() == 'assistant':
                    turn_predicted_values = chat_log[message_index]['content']
                    try:
                        turn_predicted_domains = json.loads(
                            turn_predicted_values)
                    except:
                        self.logger.info(
                            f'turn_pred: {turn_predicted_values}')
                        turn_predicted_values = extract_json(
                            turn_predicted_values)
                        self.logger.info(
                            f'turn_predicted_values after cleaning: {turn_predicted_values}')
                        turn_predicted_domains = json.loads(
                            turn_predicted_values)

                    # if statement to handle somecases returned from Llama
                    if 'domains' in turn_predicted_domains:
                        turn_predicted_domains = turn_predicted_domains['domains']

                    elif isinstance(predicted_domains, list) and 'domains' in predicted_domains[0]:
                        turn_predicted_domains = turn_predicted_domains[0]['domains']
                    elif isinstance(predicted_domains, list):
                        turn_predicted_domains = turn_predicted_domains
                    else:
                        self.logger.info(
                            f'ERROR: couldn\'t extract domains from json {turn_predicted_domains}')
                    turn_index = len(predicted_domains)
                    predicted_domains[str(turn_index)] = turn_predicted_domains
                    predicted_domains[str(turn_index+1)] = turn_predicted_domains

        if isinstance(predicted_domains,str) and predicted_domains.lower() == 'none':
            predicted_domains = {}
        return predicted_domains

    def extract_turn_entities(self, turn, prediction_level='TURN'):

        task_prompt = prompts.get_turn_entity_extraction_prompt(
            model=self.model, prediction_level=prediction_level)
        prompt = task_prompt.format(turn=turn)
        extracted_slot_values = self.query_llm(prompt=prompt, temperature=self.dst_parameters[self.model]['temperature'], top_p=self.dst_parameters[
                                               self.model]['top_p'], max_try_count=self.dst_parameters[self.model]['max_request_repeat_count'])

        return extracted_slot_values

    def predict_dialogue_slots(self, turns, speakers, domain, domain_slots, use_slots_possible_values, prediction_level):
        task_prompt = prompts.get_dst_task_prompt(
            self.model, prediction_level)
        # construct the slots string
        turn_slots = domain_slots
        slots_names, slots_description_str = self.slots_to_string(
            slots=turn_slots, add_possible_values=use_slots_possible_values)

        if prediction_level == 'DIALOGUE':
            # construct the dialogue string
            dialogue_str = self.dialogue_to_string(
                speakers=speakers, turns=turns, eos=self.dst_parameters[self.model]['EOS'], append_turn_index=True)

            prompt = task_prompt.format(slots_description=slots_description_str, dialogue_length=len(
                turns), dialogue_string=dialogue_str, slots_names=slots_names)

            answer_dict = self.query_llm(
                prompt=prompt, temperature=self.dst_parameters[self.model]['temperature'], top_p=self.dst_parameters[self.model]['top_p'], max_try_count=self.dst_parameters[self.model]['max_request_repeat_count'])

            predicted_slot_values = answer_dict

        elif prediction_level == 'TURN':
            prompt = task_prompt.format(
                slots_description=slots_description_str, slots_names=slots_names,  domain=domain)
            chat_log = self.run_llm_chat(prompt, turns, speakers, temperature=self.dst_parameters[self.model][
                'temperature'], top_p=self.dst_parameters[self.model]['top_p'])
            predicted_slot_values = {}
            for message_index in range(len(chat_log)):
                if chat_log[message_index]['role'].lower() == 'assistant':
                    turn_predicted_values = chat_log[message_index]['content']
                    try:
                        turn_predicted_slots = json.loads(
                            turn_predicted_values)
                    except:
                        self.logger.info(
                            f'turn_predicted_values before cleaning: {turn_predicted_values}')
                        turn_predicted_values = extract_json(
                            turn_predicted_values)
                        self.logger.info(
                            f'turn_predicted_values after cleaning: {turn_predicted_values}')
                        turn_predicted_slots = json.loads(
                            turn_predicted_values)

                    if turn_predicted_slots == 'none':
                        turn_predicted_slots = {}
                    # handle cases where the response contains the slots as value for 'answer:' key (Mistral)
                    elif 'answer' in turn_predicted_slots:
                        turn_predicted_slots = extract_json(turn_predicted_slots['answer'])
                        turn_predicted_slots = json.loads(turn_predicted_slots)
                    predicted_slot_values[str(
                        len(predicted_slot_values)*2)] = turn_predicted_slots

        return predicted_slot_values

    def choose_slot_value(self, slot, turns, speakers, domain, domain_slots, possible_values, prediction_level='TURN'):
        # get the task prompt
        task_prompt = prompts.get_slot_value_selection_prompt(
            model=self.model, prediction_level=prediction_level)

        if domain in slot:
            slot_name = slot.replace('-', ' ')
            slot_key = slot
        else:
            slot_name = domain+' '+slot.replace('-', ' ')
            slot_key = domain+'-'+slot

        dialogue_str = self.dialogue_to_string(
            speakers=speakers, turns=turns, eos=self.dst_parameters[self.model]['EOS'])
        possible_values_str = self.slot_values_to_string(possible_values)

        # create the slot description #
        slot_description_str = domain_slots[slot]['description']
        # format the question
        prompt = task_prompt.format(dialogue_string=dialogue_str, slot_name=slot_name, slot_key=slot_key,
                                    slot_description=slot_description_str, turn_index=len(turns)-1, slot_values_str=possible_values_str)
        # query the LLM
        answer_dic = self.query_llm(
            prompt=prompt, temperature=self.dst_parameters[self.model]['temperature'], top_p=self.dst_parameters[self.model]['top_p'], max_try_count=self.dst_parameters[self.model]['max_request_repeat_count'])
        # Check answer
        if slot_key in answer_dic:
            predicted_answer = answer_dic[slot_key]
        else:
            predicted_answer = None

        return predicted_answer

    def dialogue_to_string(self, speakers, turns, eos, append_turn_index=True):
        dialogue_str = ''
        turn_index = 0
        for turn, speaker in zip(turns, speakers):
            if append_turn_index:
                turn_str = str(turn_index)+'-'
            else:
                turn_str = ''
            turn_str += speaker+': '+turn
            dialogue_str = dialogue_str+turn_str+eos
            turn_index += 1

        return dialogue_str

    def domains_to_string(self, domains, append_description=False):
        domains_str = ''
        for domain in domains:
            if append_description:
                domains_str += '\n-'+domain+': '+domains[domain]
            else:
                domains_str += '\n-'+domain
        return domains_str
    
    def slots_to_string(self, slots, add_possible_values=True):
        slots_names = ''
        slots_description_str = ''

        for slot in slots:
            slots_names += '\n-'+slot
            slots_description_str += '\n-'+slot + \
                ': '+slots[slot]['description']
            if add_possible_values and len(slots[slot]['possible_values']) > 0:
                slots_description_str += '. The value should be one of the following possible values: ' + \
                    str(slots[slot]['possible_values'])
        return slots_names, slots_description_str

    def slot_values_to_string(self, possible_values):
        slot_values_str = ''

        possible_values.add(
            'None: the value is not mentioned or references in the turn')
        for value in possible_values:
            slot_values_str += '\n-'+str(value)

        return slot_values_str
