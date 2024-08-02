# BERT Question Answering

This repository provides a BERT-based question answering system. It utilizes a pre-trained BERT model to answer questions based on a provided passage.

## Getting Started in Google Colab

1. **Clone the Repository and Install Dependencies:**
    ```python
    !git clone https://github.com/your-username/your-repo.git
    %cd your-repo
    !pip install -r requirements.txt
    ```

2. **Open and Run the Notebook:**
    - Upload `your_notebook.ipynb` to Google Colab.
    - Mount Google Drive for accessing any required files:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

## Usage

1. **Load the Model:**
    ```python
    from transformers import BertForQuestionAnswering, BertTokenizer
    import torch

    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    ```

2. **Ask Questions:**
    ```python
    def answer_question(question, passage):
        inputs = tokenizer.encode_plus(question, passage, return_tensors='pt')
        answer_start_scores, answer_end_scores = model(**inputs)
        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs.input_ids[0][answer_start:answer_end]))
        return answer

    question = "What is the capital of France?"
    passage = "Paris is the capital of France."
    print(answer_question(question, passage))
    ```

## Data

Upload any required data files to the `data/` directory in your Google Drive.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


    
