import os
import json
# import win32com.client
import numpy as np
from scipy import signal


def read_log(log_name):
    results = {}
    log_name = os.path.join('logs', log_name+'.txt')

    with open(log_name, 'r') as f:
        for line in f:
            j = line.index(")")
            run = line[27:j]
            arr = run.split(', ')
            # run = tuple([a[1:-1] for a in arr])
            run = tuple([(a[1:-1]  if not a.isnumeric() else int(a)) for a in arr])
            res = line[j+4:-1]
            res = res.replace("\'", "\"")
            res = res.replace(")", "]")
            res = res.replace("(", "[")
            try:
                res = json.loads(res)
                results[run] = res
            except json.JSONDecodeError:
                print(f"could not decode line {line}")
                continue

    return results

# def get_target_path(shortcut_path):
#     shell = win32com.client.Dispatch("WScript.Shell")
#     shortcut = shell.CreateShortcut(shortcut_path)
#     return shortcut.Targetpath


def multiconvolve(inputs):
    result = inputs[0]
    for array in inputs[1:]:
        result = np.convolve(result, array, mode='full')
    return result

def deconvolve(output, input):
    input_fixed = np.trim_zeros(input, 'f')
    return signal.deconvolve(output, input_fixed)[0][len(input) - len(input_fixed):]

def compute_grad_for_conv_base_and_base_val(conv_base, grad_size, base_val):
    res = []
    for i in range(grad_size):
        conv_result = np.trim_zeros(np.pad(conv_base, (i, 0)), 'b')
        res.append(np.pad(conv_result, (0, max(len(conv_result), len(base_val)) - len(conv_result))) - np.zeros(len(base_val)))
    
    return np.array(res)


        