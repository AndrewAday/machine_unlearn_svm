
def seconds_to_english(seconds):
    seconds_trunc_1 = int(seconds // 60) * 60
    s = seconds - seconds_trunc_1
    seconds -= s
    seconds /= 60
    seconds_trunc_2 = int(seconds // 60) * 60
    m = int(seconds - seconds_trunc_2)
    h = seconds_trunc_2 / 60
    return str(h) + " hours, " + str(m) + " minutes, and " + str(s) + " seconds."