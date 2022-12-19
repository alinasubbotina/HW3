import pickle


def save_model(dir: str, model) -> None:
    pickle.dump(model, open(dir, 'wb'))


def load_model(dir: str):
    return pickle.load(open(dir, 'rb'))
