from termcolor import colored
import torch
import datetime


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        return "mps"
    else:
        return "cpu"


def terminal_msg(msg="text msg here", mode="E"):
    if mode == "C":
        print(colored("[Completed]", "green"), msg)
    elif mode == "E":
        print(colored("[Executing]", "blue"), msg)
    elif mode == "F":
        print(colored("[Failed]", "red"), msg)
    else:
        print("Something wrong with terminal_msg().")
        exit()


def epic_start(project_name, env="requirements.txt"):
    # http://patorjk.com/software/taag/#p=display&f=Slant&t=Hanshi%20Sun
    print(
        """
        __  __                 __    _    _____            
       / / / /___ _____  _____/ /_  (_)  / ___/__  ______  
      / /_/ / __ `/ __ \/ ___/ __ \/ /   \__ \/ / / / __ \ 
     / __  / /_/ / / / (__  ) / / / /   ___/ / /_/ / / / / 
    /_/ /_/\__,_/_/ /_/____/_/ /_/_/   /____/\__,_/_/ /_/  
    """
    )
    print(colored("[Project]", "magenta"), project_name)
    device = get_device()
    print(colored("[Device]", "cyan"), device)
    print(colored("[Environment]", "yellow"), "Check the packages below:")
    with open(env) as f:
        content_list = f.readlines()
    converted_list = []
    for element in content_list:
        converted_list.append(element.strip())
    print(converted_list)

def pretty_print(*values):
    col_width = 13

    def format_val(v):
        if not isinstance(v, str):
            v = np.array2string(v, precision=5, floatmode='fixed')
        return v.ljust(col_width)

    str_values = [format_val(v) for v in values]
    print("   ".join(str_values))