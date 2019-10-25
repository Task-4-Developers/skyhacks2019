# Development and usage

To run virtual environment and install dependencies run:

```bash
virtualenv .env && source .env/bin/activate && pip install -r requirements.txt
```

To save the requirements after development run:

```bash
pip freeze > requirements.txt
```

To exit the virtual environment run:

```bash
deactivate
```
