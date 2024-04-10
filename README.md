# langchain-cohere-document-qa

Small application to ask questions to a PDF document using Cohere LLM and langchain. This application does not intend to be perfect, it's only a starting point to create more complex applications that use langchain and Cohere (or any other LLM solution).

It runs in a Flask server that allows the users to ask questions in the text area and obtain responses from the PDF. As mentioned before, no optimization to obtain the best responeses are made.

It's possible to run using

```python
python main.py
```

and access the browser in http://127.0.0.1:5000 (if you do not change anything in the code).

To run the code, you must create a ```.env``` with the variable COHERE_API_KEY. Here you must put your own COHERE api key. You should also create a data folder and put a PDF file inside it, named document.pdf.
