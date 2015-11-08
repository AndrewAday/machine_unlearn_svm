
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

def delist(l):
    """ [[a], [b], [c]] --> [a,b,c] """
    return [e[0] for e in l]
