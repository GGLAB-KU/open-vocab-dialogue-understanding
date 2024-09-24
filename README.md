# A Zero-Shot Open-Vocabulary Pipeline for Dialogue Understanding

This repository contains the prompt templates and chatbot-related files for the work: **A Zero-Shot Open-Vocabulary Pipeline for Dialogue Understanding**.

## Datasets

- **MultiWOZ dataset** can be downloaded from: [MultiWOZ GitHub](https://github.com/budzianowski/multiwoz)
- **MultiWOZ 2.4 dataset** can be downloaded from: [MultiWOZ 2.4 GitHub](https://github.com/smartyfh/MultiWOZ2.4)
- **SGD dataset** can be downloaded from: [SGD GitHub](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue)

## Reproducing the Experiments

To reproduce the experiments, follow these steps:

1. Download the datasets from the links above and extract them into a `dataset` directory.
2. Due to the size and cost of performing the experiments for all the test splits, we provide a Jupyter Notebook (`dst.ipynb`) that can be configured to run on specific parts of the dataset.
   
### Domain Experiment

The first step is to run the **Domain Experiment** by configuring the following parameters in the notebook:

```python
predict_domains = True
predict_user_state_slots = False
prediction_level = 'TURN'  # 'TURN' or 'DIALOGUE': predict domain turns one by one or as part of the full dialogue
use_domain_description = True  # Append domain description to the list of domains
overwrite_predictions = True  # Overwrite existing predictions if running the experiment again
dataset_split = "test"  # Dataset split: 'train', 'dev', or 'test'
model = 'CHATGPT'  # LLM family
sub_model = 'gpt-4o'  # LLM model name
response_format = 'json'  # Request format: 'json' or 'text'
dataset_name = 'SGD'  # Dataset name: 'SGD' or 'MultiWOZ'
dataset_type = 'SGD'  # Dataset format: 'SGD' or 'MWZ'
```
After running the experiment, you will get a prediction file in data/{dataset_split} with the following format:
{model_name}-{sub_model}_{prompt_type}_{prediction_level}_dialogues.json

Additionally, specify the dialogue range with the following parameters:
```python
test_dialogue_lower_index = 0  # Lower index for test dialogues
test_dialogue_upper_index = 1001  # Upper index for test dialogues
```
### DST Expirement

Next, run the **DST Experiment** by configuring these parameters in the notebook:
```python
update_prediction_files = False
predict_domains = False
predict_user_state_slots = True
prediction_level = 'TURN'  # 'TURN' or 'DIALOGUE'
prompt_type = 'TASK'  # 'QA' or 'TASK'
predict_dontcare_slots = True  # Predict "dontcare" slots
use_ontology = False  # Use possible slot values (ontology)
dataset_driver = 'USER'  # 'USER' or 'SYSTEM': For SGD, slots are attached to the user turn; for MWZ, they're attached to the system turn
```
This experiment will add slots to the predicted domain file.

The predicted domains and slots are attached to the dialogus by ID. For evalaution, you can use eval scripts from https://github.com/WoodScene/LDST/blob/main/eval.py.

You can track the geenrated prompts and retreived responses from the LLMs in the `debug` log dir.
