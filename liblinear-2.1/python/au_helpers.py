""" Helper functions for the unlearning process """
import copy


def cluster_au(au, gold=False):
    """Clusters the training space of an ActiveUnlearner and returns the list of clusters."""
    
    print "\n----------------------Beginning the Clustering Process-----------------------\n"
    cluster_list = [] # list of tuples (net_rate_change, cluster)
    train_y = copy.deepcopy(au.train_y)
    train_x = copy.deepcopy(au.train_x)
    pol_y = copy.deepcopy(au.pol_y)
    pol_x = copy.deepcopy(au.pol_x)

    print "\nResetting mislabeled...\n"
    mislabeled = au.get_mislabeled(update=True) # gets an array of all false positives, false negatives
    # ^ also runs init_ground, which will update accuracy of classifier on test emails
    
    au.mislabeled_chosen = set() # reset set of clustered mislabeled emails in this instance of au

    print "\n Clustering...\n"
    original_training_size = len(training)
    pre_cluster_rate = au.current_detection_rate
    while len(training) > 0: # loop until all emails in phantom training space have been assigned
        print "\n-----------------------------------------------------\n"
        print "\n" + str(len(training)) + " emails out of " + str(original_training_size) + \
              " still unclustered.\n"

        # Choose an arbitrary email from the mislabeled emails and returns the training email closest to it.
        # Final call and source of current_seed is mislabeled_initial() function
        # current_seed = cluster_methods(au, "mislabeled", training, mislabeled) 
        current_seed = cluster_methods(au, "weighted", training, mislabeled) # weighted function selects most confident falsely labeled email
        while current_seed is None:
            current_seed = cluster_methods(au, "weighted", training, mislabeled) # weighted function selects most confident falsely labeled email
        if str(current_seed) == NO_CENTROIDS:
            current_seed = choice(training) # choose random remail from remaining emails as seed
            cluster_result = cluster_remaining(current_seed, au, training, impact=True)
        else:
            cluster_result = determine_cluster(current_seed, au, working_set=training, gold=gold, impact=True) # if true, relearn clusters after returning them
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
        original_len = len(training)
        for email in cluster.cluster_set: # remove emails from phantom training set so they are not assigned to other clusters
            training.remove(email)
        #print "\nTraining space is now at ", original_len, " --> ", len(training), " emails"

    cluster_list.sort() # sorts by net_rate_change
    print "\nClustering process done and sorted.\n"
    return cluster_list 