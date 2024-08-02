# Question-Answering-Model-using-Bert
# BERT Question Answering Project

This repository contains a BERT-based question answering system, utilizing a pre-trained BERT model to answer questions based on a given passage.

## Setup

### Google Colab Instructions

1. **Clone Repository & Install Dependencies:**
    ```python
    !git clone https://github.com/your-username/your-repo.git
    %cd your-repo
    !pip install -r requirements.txt
    ```

2. **Mount Google Drive:**
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

3. **Open & Run Notebook:**
    - Upload and open `your_notebook.ipynb` in Colab.
    - Run all cells.

## Usage Example

```python
from transformers import BertForQuestionAnswering, BertTokenizer
import torch
import numpy as np

# Load model and tokenizer
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

def bert_question_answer(question, passage):
    input_ids = tokenizer.encode(question, passage, max_length=500, truncation=True)
    segment_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
    start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]))
    start, end = torch.argmax(start_scores), torch.argmax(end_scores)
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[start:end+1]))
    return answer

question = "What is the name of the YouTube channel?"
passage = "Watch complete playlist of Natural Language Processing. Don't forget to like, share and subscribe to my channel IG Tech Team."
print(bert_question_answer(question, passage))

"**Data**"
Place any required data files in the data/ directory and access them as needed.

**License**

### `requirements.txt`

```txt
transformers==4.12.5
torch==1.10.0
numpy==1.21.2

**.gitignore**
*.pyc
__pycache__/
*.ipynb_checkpoints
venv/

**LICENSE**
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
...

