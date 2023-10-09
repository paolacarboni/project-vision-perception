import sys
import os
import torch
from interface import window_ex
from textAwareMultiGan.training.train2 import train_gan
from textAwareMultiGan.training.datasets import create_datasets
from textDetection.training.train import train_text

def is_int(string: str, args):
    try:
        num = int(string)
        if (num > 0):
            return True
        return False
    except ValueError:
        return False

def is_in(string: str, args):
    if string.casefold() in args:
        return True
    return False

def is_file(file: str, args):
    if os.path.exists(file):
        return True
    return False

def is_folder(folder: str, args):
    print("La folder è: ", folder)
    if os.path.exists(folder):
        if os.path.isdir(folder):
            return True
    return False

def loop_input(header: str, checkInput, error, args):
    flag = 1
    output = ""
    while flag:
        try:
            output = input(header)
            if output.casefold() == "exit":
                exit(0)
            if checkInput(output, args):
                flag = 0
            elif output == "":
                pass
            else:
                sys.stderr.write(error + " Print exit to exit\n")
        except KeyboardInterrupt:
            print("\n")
        except EOFError:
            print("\n")
    return output

def main():
    args = sys.argv[1:]
    if len(args) < 1:
        return 1
    if args[0] == '1':
        window_ex()
    elif args[0] == '2':
        net = loop_input("Insert network to train (text or gan): ", is_in, "Error: The network must be \"text\" or \"gan\".", ["text", "gan"])
        if net == "gan":
            res = int(loop_input("Insert GAN resolution (32, 64, 128, 256): ", is_in, "Error: bad resolution.", ["32", "64", "128", "256"]))
        batch_size = int(loop_input("Batch size: ", is_int, "Error: batch size must be a real number.", 0))
        epoch = int(loop_input("Epoches: ", is_int, "Error: batch size must be a real number.", 0))
        parameter_file = loop_input("Choose the parameter file: ", is_file, "Error: file not found", 0)
        dataset_folder = loop_input("Choose the dataset: ", is_folder, "Error: dataset not found", 0)
        result_folder = loop_input("Choose where save the results: ", is_folder, "Error: folder not found", 0)
        if (net == "gan"):
            train_gan(dataset_folder, result_folder, batch_size, res, epoch)
        elif (net == "text"):
            train_text(batch_size, epoch, dataset_folder, result_folder)
    elif args[0] == '3':
        old_data = loop_input("Insert dataset path: ", is_folder, "Error: folder not found", 0)
        new_data = loop_input("Insert where save dataset: ", is_folder, "Error: folder not found", 0)
        create_datasets(old_data, new_data)

    return 0


if __name__ == "__main__":
    main()