import torch

def load_weights(NN, filename):
    try:
        d_file = open(filename, 'r')
        NN.load_state_dict(d_file)
        return 0
    except FileNotFoundError:
        print(f"Error: {__file__}: load_weights: File '{filename}' doesn't exist.")
    except PermissionError:
        print(f"Error: {__file__}: load_weights: Permission denied: '{filename}'.")
    except Exception as e:
        print(f"Error: {__file__}: load_weights: Error: {e}")
    return 1

def save_nn(neuralN, filename):
    try:
        torch.save(neuralN.state_dict(), filename)
    except Exception as e:
        raise e
