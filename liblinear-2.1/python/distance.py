import sys

def distance(email, frequency, distance_opt):
    if distance_opt == "frequency5":
        # (1/(N2+1)+1/(N3+1)+1/(N8+1)) / (|M|*|M|), 
        # where M is the number of common features between the cluster and the sample.  
        # That is, M equals 3 in the example (N2, N3, and N8). 
        distance = 0.0
        common_features = 1
        email_indices = _vectorize(email)
        if len(email_indices) == 0:
            return sys.maxint
        for k in email_indices:
            if frequency[k] != 0:
                common_features += 1
            distance += 1.0/(frequency[k] + 1.0)
        return distance/(common_features ** 2)

def _vectorize(email):
    return [k for k in email if email[k] == 1]