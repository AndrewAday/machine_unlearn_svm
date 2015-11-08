
def sec_to_english(sec):
    sec_trunc_1 = int(sec // 60) * 60
    s = sec - sec_trunc_1
    sec -= s
    sec /= 60
    sec_trunc_2 = int(sec // 60) * 60
    m = int(sec - sec_trunc_2)
    h = sec_trunc_2 / 60
    return str(h) + " hours, " + str(m) + " minutes, and " + str(s) + " seconds."

def compose(train_y, train_x, pol_y, pol_x):
    data_y = train_y + pol_y
    data_x = train_x + pol_x
    return (data_y, data_x)

def compose_set(working_set):
    return compose(working_set[0], working_set[1], working_set[2], working_set[3])
    
def delist(l):
    """ [[a], [b], [c]] --> [a,b,c] """
    return [e[0] for e in l]

def update(working_set, i):
    if i >= len(working_set[0]): # remove from pol, not train
        i -= len(working_set[0])
        working_set[2][i] = None
        working_set[3][i] = None
    else:
        working_set[0][i] = None
        working_set[1][i] = None

def refresh(working_set, i):
    pass


def update_word_frequencies(current, new):
    new_word_vector = _vectorize(new)
    for word in new_word_vector:
        if word in current:
            current[word] += 1
        else:
            current[word] = 1
    return current

def revert_word_frequencies(current, forget):
    forget_word_vector = _vectorize(forget)
    for word in forget_word_vector:
        current[word] -= 1
    return current


def get_word_frequencies(msg):
    word_freq = {}
    word_vector = _vectorize(msg)
    for word in word_vector:
        word_freq[word] = 1
    return word_freq

def _vectorize(msg):
    return [t[1] for t in msg.clues]
