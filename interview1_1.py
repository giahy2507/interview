
import string

def check_completestring(test_str = "qwertyuioplkjhgfdsazxcvbnm"):
    alphabet = string.ascii_lowercase
    dict_alpha = {}
    for c in alphabet:
        dict_alpha[c] = 0

    for c in test_str:
        if c in dict_alpha:
            dict_alpha[c] = 1

    for key, value in dict_alpha.items():
        if value == 0:
            return "NO"
    return "YES"

if __name__ == "__main__":
    alphabet = string.ascii_lowercase
    dict_alpha = {}
    for c in alphabet:
        dict_alpha[c] = 0

    print(check_completestring("ejuxggfsts"))



