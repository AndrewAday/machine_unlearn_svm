""" Helper functions for the unlearning process """
import copy
from cluster import Cluster
import helpers as h

def cluster_au(au, gold=True):
    """Clusters the training space of an ActiveUnlearner and returns the list of clusters."""
    
    print "\n----------------------Beginning the Clustering Process-----------------------\n"
    cluster_list = [] # list of tuples (net_rate_change, cluster)
    train_y = copy.deepcopy(au.train_y)
    train_x = copy.deepcopy(au.train_x)
    pol_y = copy.deepcopy(au.pol_y)
    pol_x = copy.deepcopy(au.pol_x)

    original_training_size = len(h.strip(pol_y)) + len(h.strip(train_y))

    print "\nResetting mislabeled...\n"
    mislabeled = au.get_mislabeled(update=True) # gets an array of all false positives, false negatives
    au.mislabeled_chosen = [] # reset set of clustered mislabeled emails in this instance of au

    print "\n Clustering...\n"
    pre_cluster_rate = au.current_detection_rate
    training_size = len(h.strip(pol_y)) + len(h.strip(train_y))
    while training_size > 0: # loop until all emails in phantom training space have been assigned
        print "\n-----------------------------------------------------\n"
        print "\n" + str(training_size) + " emails out of " + str(original_training_size) + \
              " still unclustered.\n"

        training = [train_y, train_x, pol_y, pol_x]

        # Choose an arbitrary email from the mislabeled emails and returns the training email closest to it.
        # Final call and source of current_seed is mislabeled_initial() function
        # current_seed = cluster_methods(au, "mislabeled", training, mislabeled) 
        current_seed = None 
        label = None
        while current_seed is None:
            label, init_pos, current_seed = au.select_initial(mislabeled, "weighted", training) 

        if str(current_seed) == 'NO_CENTROIDS':
            cluster_result = cluster_remaining(au, training, impact=True) # TODO implement cluster_result
        else:
            cluster_result = determine_cluster(current_seed, au, label, init_pos, working_set=training, gold=gold) # if true, relearn clusters after returning them
        if cluster_result is None:
            print "!!!How did this happen?????"
            sys.exit(cluster_result)

        net_rate_change, cluster = cluster_result
        # After getting the cluster and net_rate_change, you relearn the cluster in original dataset if impact=True

        post_cluster_rate = au.current_detection_rate

        # make sure the cluster was properly relearned
        assert(post_cluster_rate == pre_cluster_rate), str(pre_cluster_rate) + " " + str(post_cluster_rate)
        print "cluster relearned successfully: au detection rate back to ", post_cluster_rate

        cluster_list.append([net_rate_change, cluster])

        print "\nRemoving cluster from shuffled training set...\n"

        h.unlearn(training, cluster.cluster_set)
        training_size = len(h.strip(pol_y)) + len(h.strip(train_y))

    cluster_list.sort() # sorts by net_rate_change
    print "\nClustering process done and sorted.\n"
    return cluster_list 

def determine_cluster(center, au, label, init_pos, working_set=None, gold=False, impact=True):
    """Given a chosen starting center and a given increment of cluster size, it continues to grow and cluster more
    until the detection rate hits a maximum peak (i.e. optimal cluster); if first try is a decrease, reject this
    center and return False.
    """

    print "\nDetermining appropriate cluster around msg at position", init_pos, "...\n"
    old_detection_rate = au.current_detection_rate
    first_state_rate = au.current_detection_rate
    counter = 0

    cluster = Cluster((center,init_pos), au.increment, au, label, distance_opt=au.distance_opt, working_set=working_set)
    # Test detection rate after unlearning cluster
    au.unlearn(cluster)
    au.init_ground()
    new_detection_rate = au.current_detection_rate

    if new_detection_rate <= old_detection_rate:    # Detection rate worsens - Reject
        print "\nCenter is inviable. " + str(new_detection_rate) + " < " + str(old_detection_rate) + "\n" 
        print "relearning cluster... "
        
        au.learn(cluster)

        second_state_rate = new_detection_rate
        net_rate_change = second_state_rate - first_state_rate
        print "cluster rejected with a net rate change of ", second_state_rate, " - ", first_state_rate, " = ", net_rate_change
        au.current_detection_rate = first_state_rate

        return net_rate_change, cluster

    elif cluster.size < au.increment:
        if impact:
            au.learn(cluster)
            second_state_rate = new_detection_rate
            net_rate_change = second_state_rate - first_state_rate
            au.current_detection_rate = first_state_rate
            print "no more emails to cluster, returning cluster of size ", cluster.size
            return net_rate_change, cluster

    else:   # Detection rate improves - Grow cluster
        if gold:
            cluster = au.cluster_by_gold(cluster, old_detection_rate, new_detection_rate, counter)
        if impact: #include net_rate_change in return
            au.learn(cluster) # relearn cluster in real training space so deltas of future cluster are not influenced
            second_state_rate = au.current_detection_rate
            net_rate_change = second_state_rate - first_state_rate
            print "cluster found with a net rate change of ", second_state_rate, " - ", first_state_rate, " = ", net_rate_change
            au.current_detection_rate = first_state_rate
            return net_rate_change, cluster

