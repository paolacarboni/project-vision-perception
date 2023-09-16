import sys
from interface import window_ex

def is_int(string: str, *args):
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

def loop_input(header: str, checkInput, error, args):
    flag = 1
    output = ""
    while flag:
        try:
            output = input(header)
            output = output.casefold()
            if output == "exit" or checkInput(output, args):
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
        if net == "exit":
            exit(0)
        if net == "gan":
            res = loop_input("Insert GAN resolution (32, 64, 128, 256 or all): ", is_in, "Error: bad resolution.", ["32", "64", "128", "256", "all"])
            if res == "exit":
                exit(0)
        batch_size = loop_input("Batch size: ", is_int, "Error: batch size must be a real number.", 0)
        if batch_size == "exit":
            exit(0)
    return 0


if __name__ == "__main__":
    main()