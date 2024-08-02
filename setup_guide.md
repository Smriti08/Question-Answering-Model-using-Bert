# Setup Guide for BERT Question Answering Project

## Prerequisites

- **Python**: Version 3.7 or later.
- **Git**: Install from [git-scm.com](https://git-scm.com/downloads).
- **IDE**: Optional, e.g., Visual Studio Code or PyCharm.

## Setup Steps

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/yourusername/your-repo.git
    cd your-repo
    ```

2. **Create and Activate a Virtual Environment**:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```

3. **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Notebook**:

    - **Google Colab**: Upload and run `your_notebook.ipynb` in Colab.
    - **Local**: Ensure Jupyter Notebook is installed:

        ```bash
        pip install notebook
        jupyter notebook your_notebook.ipynb
        ```

5. **Verify Installation**:

    Run a test cell in the notebook:

    ```python
    import torch
    from transformers import BertTokenizer, BertForQuestionAnswering

    print(torch.cuda.is_available())

    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    print("Setup is complete and BERT model is loaded successfully!")
    ```

6. **Optional: Set Up Environment Variables**

    Create a `.env` file in the project root and add variables as `KEY=value`.

7. **Explore and Modify**

    Open `your_notebook.ipynb` and make modifications as needed. Refer to the `README.md` for project details.

## Troubleshooting

- **Installation Issues**: Check `requirements.txt` and library compatibility.
- **Runtime Errors**: Ensure correct Python and library versions, and environment activation.

For help, check [GitHub Issues](https://github.com/yourusername/your-repo/issues) or open a new issue.

---

Happy coding!
