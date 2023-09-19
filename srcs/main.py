import sys
import os
from interface import window_ex
from textAwareMultiGan.training.train import train_gan
from textAwareMultiGan.training.datasets import create_datasets

def is_int(string: str, args):
    try:
        num = int(string)
        if (num > 0):
            return True
        return False
    except ValueError:
        return False

def is_in(string: str, args):
    if string in args:
        return True
    return False

def is_file(file: str, args):
    if os.path.exists(file):
        return True
    return False

def is_folder(folder: str, args):
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
            output = output.casefold()
            if output == "exit":
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
            train_gan(res, dataset_folder, result_folder, parameter_file, epoch=epoch, batch_size=batch_size)
    elif args[0] == '3':
        flag = loop_input("Please note that all images in resrcs will be deleted. Continue? (yes), (no): ", is_in, "Exit to exit", ["yes", "y", "no", "n"])
        if flag == "yes" or "y":
            create_datasets()

    return 0


if __name__ == "__main__":
    main()