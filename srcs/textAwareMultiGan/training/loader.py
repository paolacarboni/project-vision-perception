import ast
from ..training.utils import trainParameters

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

def parse_line(line, par: trainParameters):
    if len(line) > 1:
        parameter = line[1]
        if line[0] in ["d_lr", "g_lr"]:
            try:
                parameter = float(line[1])
            except ValueError:
                print(f'Error: {__file__}: parse_line: {line[1]} isn\'t a float.')
                return 1
        if line[0] in ["d_betas", "g_betas"]:
            try:
                parameter = ast.literal_eval(line[1])
            except ValueError:
                print(f'Error: {__file__}: parse_line: {line[1]} isn\'t a tuple.')
                return 1
        if line[0] == "d_lr":
            par.d_lr = parameter
        elif line[0] == "g_lr":
            par.g_lr = parameter
        elif line[0] == "d_betas":
            par.d_betas = parameter
        elif line[0] == "g_betas":
            par.g_betas = parameter
        elif line[0] == "d_filename":
            par.d_filename = parameter
        elif line[0] == "g_filename":
            par.g_filename = parameter
    return 0

def load_parameters(pars: [trainParameters], par_file):
    
    par: trainParameters

    try:
        file = open(par_file, 'r')
        for line in file:
            words = line.split()
            if len(words) > 0:
                if words[0] == "32":
                    par = pars[0]
                elif words[0] == "64":
                    par = pars[1]
                elif words[0] == "128":
                    par = pars[2]
                elif words[0] == "256":
                    par = pars[3]
                else:
                    if parse_line(words, par):
                        return 1
        return 0
    except FileNotFoundError:
        print(f"Error: {__file__}: load_parameters: File '{par_file}' doesn't exist.")
    except PermissionError:
        print(f"Error: {__file__}: load_parameters: Permission denied: '{par_file}'.")
    except Exception as e:
        print(f"Error: {__file__}: load_parameters: Error: {e}")
    return 1