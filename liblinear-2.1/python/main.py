""" Test Driver for the Unlearning Process """
import os
import sys
import time
import process_data as pd
import helpers
import svm_driver as svm

# Set options liblinear
params = '-c .001'

# Choose directory we want to process
directory = "Mislabeled-Both-1.1-processed"
output = directory + "-unlearn-stats"

# rename helpers variables
seconds_to_english = helpers.sec_to_english


def unlearn_stats(au, outfile, data_set, train, test, polluted, total_polluted, total_unpolluted,
                  train_time, clusters=False, vanilla=None, noisy_clusters=False):
        """Runs an unlearn algorithm on an ActiveUnlearner and prints out the resultant stats."""
        outfile.write("---------------------------\n")
        outfile.write("Data Set: " + data_set + "\n")
        outfile.write("Vanilla Training: " + str(train[0]) + " ham and " + str(train[1]) + " spam.\n")
        outfile.write("Testing: " + str(test[0]) + " ham and " + str(test[1]) + " spam.\n")
        outfile.write("Pollution Training: " + str(polluted[0]) + " ham and " + str(polluted[1]) +
                      " spam.\n")
        if vanilla is not None:
            outfile.write("Vanilla Detection Rate: " + str(vanilla[0]) + ".\n")
        outfile.write("---------------------------\n")
        outfile.write("\n\n")
        outfile.write("CLUSTER AND RATE COUNTS:\n")
        outfile.write("---------------------------\n")

        original_detection_rate = au.driver.tester.correct_classification_rate()

        outfile.write("0: " + str(original_detection_rate) + "\n")

        time_start = time.time()
        
        # get the unlearned cluster list
        # Testing shrinking the rejected clusters
        # cluster_list = au.impact_active_unlearn(outfile, test=True, pollution_set3=pollution_set3, gold=True, shrink_rejects=True) 
        cluster_list = au.impact_active_unlearn(outfile, test=True, pollution_set3=pollution_set3, gold=True, shrink_rejects=False) 
        
        time_end = time.time()
        unlearn_time = seconds_to_english(time_end - time_start)
        
        total_polluted_unlearned = 0
        total_unlearned = 0
        total_unpolluted_unlearned = 0
        total_noisy_unlearned = 0
        final_detection_rate = au.current_detection_rate
        noise = []

        print "\nTallying up final counts...\n"
        for cluster in cluster_list:
            cluster = cluster[1]
            total_unlearned += cluster.size # total no. emails unlearned
            total_polluted_unlearned += cluster.target_set3()
            total_unpolluted_unlearned += (cluster.size - cluster.target_set3())

        outfile.write("\nSTATS\n")
        outfile.write("---------------------------\n")
        outfile.write("Initial Detection Rate: " + str(original_detection_rate) + "\n")
        outfile.write("Final Detection Rate: " + str(final_detection_rate) + "\n")
        outfile.write("Total Unlearned:\n")
        outfile.write(str(total_unlearned) + "\n")
        outfile.write("Polluted Percentage of Unlearned:\n")
        outfile.write(str(total_polluted_unlearned) + "/" + str(total_unlearned) + " = " + str(float(total_polluted_unlearned) / float(total_unlearned)) + "\n")
        outfile.write("Unpolluted Percentage of Unlearned:\n")
        outfile.write(str(total_unpolluted_unlearned) + "/" + str(total_unlearned) + " = " + str(float(total_unpolluted_unlearned) / float(total_unlearned)) + "\n")
        outfile.write("Percentage of Polluted Unlearned:\n")
        outfile.write(str(total_polluted_unlearned) + "/" + str(total_polluted) + " = " +  str(float(total_polluted_unlearned) / float(total_polluted)) + "\n")
        outfile.write("Percentage of Unpolluted Unlearned:\n")
        outfile.write(str(total_unpolluted_unlearned) + "/" + str(total_unpolluted) + " = " + str(float(total_unpolluted_unlearned) / float(total_unpolluted)) + "\n")
        if noisy_clusters:
            if vanilla is not None:
                # get list of clusters with 0 polluted emails, but unlearning still improves classification accuracy
                noise = noisy_data_check(find_pure_clusters(cluster_list, ps_3=pollution_set3), vanilla[1]) #vanilla[1] is the v_au instance
                for cluster in noise:
                    total_noisy_unlearned += cluster.size
                outfile.write("Percentage of Noisy Data in Unpolluted Unlearned:\n")
                outfile.write(str(total_noisy_unlearned) + "/" + str(total_unpolluted_unlearned) + " = " +  str(float(total_noisy_unlearned) / float(total_unpolluted_unlearned)) + "\n")
        outfile.write("Time for training:\n")
        outfile.write(train_time + "\n")
        outfile.write("Time for unlearning:\n")
        outfile.write(unlearn_time)
        outfile.write("\n") #always end files w/ newline

        if clusters:
            return cluster_list


def find_pure_clusters(cluster_list, ps_3):
    pure_clusters = []
    for cluster in cluster_list:
        cluster = cluster[1]
        if ps_3:
            pure_clusters.append(cluster.target_set3_get_unpolluted())
            # if cluster.target_set3() == 0:
            #     pure_clusters.append(cluster)

        else:
            if cluster.target_set4() == 0:
                pure_clusters.append(cluster)

    return pure_clusters


def noisy_data_check(pure_clusters, v_au):
    """
    Returns a list of all clusters which had no polluted emails, 
    but unlearning them improves classification accuracy 
    """
    noisy_clusters = []
    original_detection_rate = v_au.current_detection_rate
    counter = 1
    for cluster in pure_clusters:
        print "testing for noise in cluster ", counter, "/", len(pure_clusters)
        v_au.unlearn(cluster)
        v_au.init_ground(True)
        new_detection_rate = v_au.driver.tester.correct_classification_rate()
        if new_detection_rate > original_detection_rate:
            noisy_clusters.append(cluster)

        v_au.learn(cluster)
        counter += 1

    return noisy_clusters


def main():
    print "Processing ", directory

    # Collect the processed data
    emails = pd.get_emails(directory, vanilla=False)
    van_emails = pd.get_emails(directory, vanilla=True)

    # assign variables to train and test data
    pol_y, pol_x = emails[0]
    train_y, train_x = emails[1]
    test_y, test_x = emails[2]

    data_y = train_y + pol_y
    data_x = train_x + pol_x

    van_pol_y, van_pol_x = emails[0]
    van_train_y, van_train_x = van_emails[1]
    van_test_y, van_test_x = van_emails[2]

    # group polluted/unpolluted data to train
    van_data_y = van_train_y + van_pol_y 
    van_data_x = van_train_x + van_pol_x
    
    print "Calculating initial vanilla detection rate:"
    van_m = svm.train(van_data_y, van_data_x, params)
    van_acc = svm.predict(van_test_y, van_test_x, van_m)
    print "Initial vanilla accuracy: ", van_acc

    print "Calculating initial pollued detection rate:"
    m = svm.train(data_y, data_x, params)
    acc = svm.predict(test_y, test_x, m)
    print "Initial polluted accuracy: ", acc

    # Calculate the number of emails for polluted, train, test, and total data sets
    size = emails[3]
    ham_polluted = size['ham_polluted']
    spam_polluted = size['spam_polluted']
    train_ham = size['train_ham']
    train_spam = size['train_spam']
    test_ham = size['test_ham']
    test_spam = size['test_spam']
    total_polluted = size['total_polluted']
    total_unpolluted = size['total_unpolluted']

    train

    print size
    return
    try:
        time_1 = time.time() # begin timer
        # Instantiate ActiveUnlearner object
        au = ActiveUnlearnDriver.ActiveUnlearner([msgs.HamStream(ham_train, [ham_train]),
                                                  msgs.HamStream(ham_p, [ham_p])],        # Training Ham 
                                                 [msgs.SpamStream(spam_train, [spam_train]),
                                                  msgs.SpamStream(spam_p, [spam_p])],     # Training Spam
                                                 msgs.HamStream(ham_test, [ham_test]),          # Testing Ham
                                                 msgs.SpamStream(spam_test, [spam_test]),       # Testing Spam
                                                 distance_opt="frequency5", all_opt=True,      
                                                 update_opt="hybrid", greedy_opt=True,          
                                                 include_unsures=False, multi_process=True) # Don't unclude unsure emails        

        # vanilla active unlearner
        # v_au = ActiveUnlearnDriver.ActiveUnlearner([msgs.HamStream(ham_train, [ham_train]), []],
        #                                            [msgs.SpamStream(spam_train, [spam_train]), []],
        #                                            msgs.HamStream(ham_test, [ham_test]),
        #                                            msgs.SpamStream(spam_test, [spam_test]))

        # vanilla_detection_rate = v_au.current_detection_rate

        time_2 = time.time()
        train_time = seconds_to_english(time_2 - time_1)
        print "Train time:", train_time, "\n"

        

        with open(dest + data_set + " (unlearn_stats).txt", 'w+') as outfile:
            try:
                # unlearn_stats(au, outfile, data_set, [train_ham, train_spam], [test_ham, test_spam],
                #               [ham_polluted, spam_polluted], total_polluted, total_unpolluted,
                #               train_time, vanilla=[vanilla_detection_rate, v_au], noisy_clusters=True)
                unlearn_stats(au, outfile, data_set, [train_ham, train_spam], [test_ham, test_spam],
                              [ham_polluted, spam_polluted], total_polluted, total_unpolluted,
                              train_time, vanilla=None, noisy_clusters=True)

            except KeyboardInterrupt:
                outfile.flush()
                sys.exit()

        # In the hopes of keeping RAM down between iterations
        del au
        del v_au

    except KeyboardInterrupt:
        sys.exit()

if __name__ == "__main__":
    main()
