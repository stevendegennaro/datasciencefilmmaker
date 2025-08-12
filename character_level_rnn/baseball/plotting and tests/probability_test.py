from collections import Counter
import numpy as np
from name_network_scratch import import_names
import tqdm

test_list = ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'd', 'd', 'd', 'd', 'e', 'e', 'e', 'e', 'e', 'f', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'l', 'l', 'l', 'm', 'm', 'm', 'm', 'm', 'm', 'm','m', 'm', 'm', 'm', 'm', 'm', 'm', 'm', 'm', 'm', 'm', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'o', 'o','o', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 't', 't', 'u', 'u', 'u']

def predict_accuracy(letters_list):
    count = Counter(letters_list)
    probs = np.array(list(count.values()))

    sum_of_squares = np.sum(probs**2)
    square_of_sum = np.sum(probs)**2
    return sum_of_squares / square_of_sum

# prob = predict_accuracy(test_list)

def monte_carlo_accuracy(letters_list, n = 100000):
    count = 0
    for _ in range(n):
        l1 = np.random.choice(letters_list)
        l2 = np.random.choice(letters_list)
        if l1 == l2:
            count += 1

    return count/n

# mc = monte_carlo_accuracy(test_list)

def mc_inputs(set_dict, n = 100):

    # Import the names and stick them in a dictionary
    firstnames, lastnames, suffixes = import_names()
    names = firstnames
    
    inputs = []
    targets = []
    for name in names:
        for i in range(1,len(name)):
            inputs.append(name[:i])
            targets.append(name[i])

    correct = 0
    with tqdm.trange(n) as t:
        for i in t:
            # Choose an input (and corresponding target) at random
            index = np.random.randint(len(inputs))

            # Look up list of letters for that input in set_dict
            # Choose one of those letters at random
            guess = np.random.choice(set_dict[inputs[index]])

            # See if that letter matches the target
            if guess == targets[index]:
                # If so, add 1 to the correct guesses
                correct += 1

            if i > 0:
                t.set_description(f"{correct/i}")

        return correct / n

