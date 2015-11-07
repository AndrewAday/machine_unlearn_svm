
def sec_to_english(sec):
    sec_trunc_1 = int(sec // 60) * 60
    s = sec - sec_trunc_1
    sec -= s
    sec /= 60
    sec_trunc_2 = int(sec // 60) * 60
    m = int(sec - sec_trunc_2)
    h = sec_trunc_2 / 60
    return str(h) + " hours, " + str(m) + " minutes, and " + str(s) + " seconds."