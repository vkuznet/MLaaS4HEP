"""
Basic example of preprocessing function
"""

def preprocessing(rec):
    "Simple preprocessing function"
    # for example our JSON record has the following structure:
    # {'data': {... payload record ..}} and we want to extract the payload from it
    if isinstance(rec, dict) and 'data' in rec:
        return rec['data']
    return rec
