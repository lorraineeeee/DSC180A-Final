import pickle
def make_dataset(route):
    with open(route, 'rb') as f:
        data = pickle.load(f)
        return data
