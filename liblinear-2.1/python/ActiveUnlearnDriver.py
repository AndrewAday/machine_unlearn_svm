import sys
import svm_driver as svm
import helpers as h
import au_helpers as au_h
from distance import distance
from math import sqrt

phi = (1 + sqrt(5)) / 2
grow_tol = 50 # window tolerance for gold_section_search to maximize positive delta
shrink_tol = 10 # window tolerance for golden_section_search to minize negative delta
dec = 20

class ActiveUnlearner:
    """
    Core component of the unlearning algorithm. Container class for most relevant methods, driver/classifier,
    and data.
    """
    def __init__(self, train_y, train_x, pol_y, pol_x, test_y, test_x, params='-c .001 -q', threshold=100, increment=100,
                 distance_opt="frequency5", update_opt="hybrid", greedy_opt=False):
        # Training Data
        self.train_y = train_y
        self.train_x = train_x
        self.pol_y = pol_y
        self.pol_x = pol_x

        # Testing Data
        self.test_y = test_y
        self.test_x = test_x

        # Unlearn Options
        self.params = params
        self.distance_opt = distance_opt
        self.greedy = greedy_opt
        self.increment = increment
        self.threshold = threshold

        # Performance Variables
        self.p_label = None
        self.p_val = None
        
        # Train algorithm normally
        self.current_detection_rate = 0
        self.init_ground()
        self.mislabeled_chosen = set()
        self.training_chosen = set()

        print "Initial detection rate: ", self.current_detection_rate

    def init_ground(self):
        """ Predicts on testing data and updates self.current_detection_rate """
        data_y, data_x = h.compose(self.train_y, self.train_x, self.pol_y, self.pol_x)
        data_y = h.strip(data_y) # strip None
        data_x = h.strip(data_x)
        m = svm.train(data_y, data_x, self.params)
        p_label, p_acc, p_val = svm.predict(self.test_y, self.test_x, m)
        self.p_label = p_label
        self.p_val = h.delist(p_val)
        self.current_detection_rate = p_acc[0]

    def unlearn(self, cluster):
        """Unlearns a cluster from the ActiveUnlearner."""
        print "unlearning cluster of size ", len(cluster.cluster_set), " from au"
        if len(cluster.ham) + len(cluster.spam) != cluster.size:
            print "\nUpdating cluster ham and spam sets...\n"
            cluster.divide()

        h.unlearn([self.train_y, self.train_x, self.pol_y, self.pol_x], cluster.cluster_set)

    def learn(self, cluster):
        """Learns a cluster from the ActiveUnlearner."""
        if len(cluster.ham) + len(cluster.spam) != cluster.size:
            print "\nUpdating cluster ham and spam sets...\n"
            cluster.divide()

        h.relearn([self.train_y, self.train_x, self.pol_y, self.pol_x], cluster.working_set, cluster.cluster_set)
        


    # --------------------------------------------------------------------------------------------------------------

    def detect_rate(self, cluster):
        """Returns the detection rate if a given cluster is unlearned.
        Relearns the cluster afterwards.
        """
        self.unlearn(cluster)
        self.init_ground()
        detection_rate = self.driver.tester.correct_classification_rate()
        self.learn(cluster)
        return detection_rate

    def start_detect_rate(self, cluster):
        """Determines the detection rate after unlearning an initial cluster."""
        self.unlearn(cluster)
        self.init_ground()
        detection_rate = self.driver.tester.correct_classification_rate()
        return detection_rate

    def continue_detect_rate(self, cluster, n):
        """Determines the detection rate after growing a cluster to be unlearned."""
        old_cluster = copy.deepcopy(cluster.cluster_set)
        cluster.cluster_more(n)
        new_cluster = cluster.cluster_set

        new_unlearns = new_cluster - old_cluster
        assert(len(new_unlearns) == len(new_cluster) - len(old_cluster))
        assert(len(new_unlearns) == n), len(new_unlearns)

        unlearn_hams = []
        unlearn_spams = []

        for unlearn in new_unlearns:
            if unlearn.train == 1 or unlearn.train == 3:
                unlearn_hams.append(unlearn)

            elif unlearn.train == 0 or unlearn.train == 2:
                unlearn_spams.append(unlearn)

            self.driver.tester.train_examples[unlearn.train].remove(unlearn)

        self.driver.untrain(unlearn_hams, unlearn_spams)
        self.init_ground()
        detection_rate = self.driver.tester.correct_classification_rate()
        return detection_rate

    # --------------------------------------------------------------------------------------------------------------

    def divide_new_elements(self, messages, unlearn):
        """Divides a given set of emails to be unlearned into ham and spam lists and unlearns both."""
        hams = []
        spams = []
        for message in messages:
            if message.train == 1 or message.train == 3:
                hams.append(message)

            elif message.train == 0 or message.train == 2:
                spams.append(message)

            else:
                raise AssertionError("Message lacks train attribute.")

            if unlearn:
                self.driver.tester.train_examples[message.train].remove(message)

            else:
                self.driver.tester.train_examples[message.train].append(message)

        if unlearn:
            self.driver.untrain(hams, spams)

        else:
            self.driver.train(hams, spams)

    def cluster_by_gold(self, cluster, old_detection_rate, new_detection_rate, counter):
        """Finds an appropriate cluster around a msg by using the golden section search method."""
        sizes = [0]
        detection_rates = [old_detection_rate]

        new_unlearns = ['a', 'b', 'c']

        if len(new_unlearns) > 0:
            if new_detection_rate > old_detection_rate:
                return self.try_gold(cluster, sizes, detection_rates, old_detection_rate, new_detection_rate, counter)

            else:
                new_learns = cluster.cluster_less(self.increment)
                self.divide_new_elements(new_learns, False)
                return cluster

        else:
            return cluster

    def try_gold(self, cluster, sizes, detection_rates, old_detection_rate, new_detection_rate, counter,shrink_rejects=False):
        
        """
        Performs golden section search on the size of a cluster; grows/shrinks exponentially at a rate of phi to ensure that
        window ratios will be same at all levels (except edge cases), and uses this to determine the initial window.
        """
        if shrink_rejects:
            shrink_cluster = cluster.size - int(cluster.size/phi)
            while counter == 0 or new_detection_rate > old_detection_rate:
                counter += 1
                sizes.append(cluster.size)
                detection_rates.append(new_detection_rate)
                old_detection_rate = new_detection_rate
                print "\n Exploring a shrunk cluster of size", cluster.size - shrink_cluster, "...\n"
                
                new_learns = cluster.cluster_less(shrink_cluster)
                shrink_cluster = cluster.size - int(cluster.size/phi)

                self.divide_new_elements(new_learns, False)
                self.init_ground()
                new_detection_rate = self.driver.tester.correct_classification_rate()

        else:
            extra_cluster = int(phi * cluster.size)
            while new_detection_rate > old_detection_rate:
                counter += 1

                sizes.append(cluster.size)
                detection_rates.append(new_detection_rate)
                old_detection_rate = new_detection_rate
                print "\nExploring cluster of size", cluster.size + int(round(extra_cluster)), "...\n"

                new_unlearns = cluster.cluster_more(int(round(extra_cluster))) # new_unlearns is array of newly added emails
                extra_cluster *= phi

                self.divide_new_elements(new_unlearns, True) # unlearns the newly added elements
                self.init_ground() # rerun test to find new classification accuracy
                new_detection_rate = self.driver.tester.correct_classification_rate()

        sizes.append(cluster.size) # array of all cluster sizes
        detection_rates.append(new_detection_rate) # array of all classification rates

        cluster, detection_rate, iterations = self.golden_section_search(cluster, sizes, detection_rates)
        print "\nAppropriate cluster found, with size " + str(cluster.size) + " after " + \
              str(counter + iterations) + " tries.\n"

        self.current_detection_rate = detection_rate
        return cluster

    def golden_section_search(self, cluster, sizes, detection_rates):
        """Performs golden section search on a cluster given a provided initial window."""
        print "\nPerforming golden section search...\n"

        # left, middle_1, right = sizes[len(sizes) - 3], sizes[len(sizes) - 2], sizes[len(sizes) - 1]
        left, middle_1, right = sizes[-3],sizes[-2],sizes[-1]
        pointer = middle_1
        iterations = 0
        new_relearns = cluster.cluster_less(right - middle_1)
        self.divide_new_elements(new_relearns, False)

        assert(len(sizes) == len(detection_rates)), len(sizes) - len(detection_rates)
        f = dict(zip(sizes, detection_rates))

        middle_2 = right - (middle_1 - left)

        while abs(right - left) > grow_tol:
            print "\nWindow is between " + str(left) + " and " + str(right) + ".\n"
            try:
                assert(middle_1 < middle_2)

            except AssertionError:
                middle_1, middle_2, pointer = self.switch_middles(middle_1, middle_2, cluster)

            print "Middles are " + str(middle_1) + " and " + str(middle_2) + ".\n"

            try:
                rate_1 = f[middle_1]

            except KeyError:
                rate_1, middle_1, pointer = self.evaluate_left_middle(pointer, middle_1, cluster, f)
                iterations += 1

            try:
                rate_2 = f[middle_2]

            except KeyError:
                rate_2, middle_2, pointer = self.evaluate_right_middle(pointer, middle_2, cluster, f)
                iterations += 1

            if rate_1 > rate_2:
                right = middle_2
                middle_2 = middle_1
                middle_1 = right - int((right - left) / phi)

            else:
                left = middle_1
                middle_1 = middle_2
                middle_2 = left + int((right - left) / phi)

        size = int(float(left + right) / 2)
        assert (left <= size <= right), str(left) + ", " + str(right)
        if pointer < size:
            new_unlearns = cluster.cluster_more(size - pointer)
            assert(cluster.size == size), str(size) + " " + str(cluster.size)
            self.divide_new_elements(new_unlearns, True)

        elif pointer > size:
            new_relearns = cluster.cluster_less(pointer - size)
            assert(cluster.size == size), str(size) + " " + str(cluster.size)
            self.divide_new_elements(new_relearns, False)

        else:
            raise AssertionError("Pointer is at the midpoint of the window.")

        self.init_ground()
        detection_rate = self.driver.tester.correct_classification_rate()
        iterations += 1

        return cluster, detection_rate, iterations

    def switch_middles(self, middle_1, middle_2, cluster):
        """
        Switches the middles during golden section search. This is necessary when the exponential probing reaches the
        end of the training space and causes problems of truncation.
        """
        print "\nSwitching out of order middles...\n"
        middle_1, middle_2 = middle_2, middle_1
        pointer = middle_1
        if cluster.size > pointer:
            new_relearns = cluster.cluster_less(cluster.size - pointer)
            self.divide_new_elements(new_relearns, False)

        elif cluster.size < pointer:
            new_unlearns = cluster.cluster_more(pointer - cluster.size)
            self.divide_new_elements(new_unlearns, True)

        return middle_1, middle_2, pointer

    def evaluate_left_middle(self, pointer, middle_1, cluster, f):
        """Evaluates the detection rate at the left middle during golden section search."""
        if pointer > middle_1:
            new_relearns = cluster.cluster_less(pointer - middle_1)
            pointer = middle_1
            print "Pointer is at " + str(pointer) + ".\n"
            assert(cluster.size == pointer), cluster.size
            self.divide_new_elements(new_relearns, False)
            self.init_ground()
            rate_1 = self.driver.tester.correct_classification_rate()
            f[middle_1] = rate_1

        elif pointer < middle_1:
            raise AssertionError("Pointer is on the left of middle_1.")

        else:
            assert(cluster.size == pointer), cluster.size
            self.init_ground()
            rate_1 = self.driver.tester.correct_classification_rate()
            if middle_1 in f:
                raise AssertionError("Key should not have been in f.")

            else:
                f[middle_1] = rate_1

        return rate_1, middle_1, pointer

    def evaluate_right_middle(self, pointer, middle_2, cluster, f):
        """Evaluates the detection rate at the right middle during the golden section search."""
        if pointer < middle_2:
            new_unlearns = cluster.cluster_more(middle_2 - pointer)
            pointer = middle_2
            print "Pointer is at " + str(pointer) + ".\n"
            assert(cluster.size == pointer), cluster.size
            self.divide_new_elements(new_unlearns, True)
            self.init_ground()
            rate_2 = self.driver.tester.correct_classification_rate()
            f[middle_2] = rate_2

        elif pointer > middle_2:
            raise AssertionError("Pointer is on the right of middle_2.")

        else:
            raise AssertionError("Pointer is at the same location as middle_2.")

        return rate_2, middle_2, pointer

    # -----------------------------------------------------------------------------------

    def impact_active_unlearn(self, outfile, pollution_set3=True, gold=False):
        """
        Attempts to improve the machine by first clustering the training space and then unlearning clusters based off
        of perceived impact to the machine.

        pos_cluster_opt values: 0 = treat negative clusters like any other cluster, 1 = only form positive clusters,
        2 = shrink negative clusters until positive, 3 = ignore negative clusters after clustering (only applicable in
        greedy checking)
        """
        unlearned_cluster_list = []
        try:
            cluster_count = 0
            attempt_count = 0
            detection_rate = self.current_detection_rate

            cluster_count, attempt_count = self.lazy_unlearn(detection_rate, unlearned_cluster_list,
                                                             cluster_count, attempt_count,
                                                             outfile, pollution_set3, gold)

            print "\nThreshold achieved or all clusters consumed after", cluster_count, "clusters unlearned and", \
                attempt_count, "clustering attempts.\n"

            print "\nFinal detection rate: " + str(self.current_detection_rate) + ".\n"
            
            return unlearned_cluster_list

        except KeyboardInterrupt:
            return unlearned_cluster_list

    def lazy_unlearn(self, detection_rate, unlearned_cluster_list, cluster_count, attempt_count, outfile,
                     pollution_set3, gold):
        """
        After clustering, unlearns all clusters with positive impact in the cluster list, in reverse order. This is
        due to the fact that going in the regular order usually first unlearns a large cluster that is actually not
        polluted. TODO: is there anyway to determine if this set is actually polluted?

        This is because in the polluted state of the machine, this first big cluster is perceived as a high
        impact cluster, but after unlearning several (large) polluted clusters first (with slightly smaller impact but
        still significant), this preserves the large (and unpolluted) cluster.
        """

        # returns list of tuples contained (net_rate_change, cluster)
        cluster_list = au_h.cluster_au(self, gold=gold) 
        
        rejection_rate = .1 # Reject all clusters <= this threshold delta value
        attempt_count += 1

        print "Lazy Unlearn Attempt " + str(attempt_count) + " cluster length: ", len(cluster_list)
        print "----------The Cluster List------------"
        print cluster_list
        print "----------/The Cluster List------------"

        # ANDREW CHANGED: while detection_rate <= self.threshold and cluster_list[len(cluster_list) - 1][0] > 0:
        while detection_rate <= self.threshold and cluster_list[-1][0] > rejection_rate:
            list_length = len(cluster_list)
            j = 0
            while cluster_list[j][0] <= rejection_rate:
                j += 1 # move j pointer until lands on smallest positive delta cluster

            if not self.greedy: # unlearn the smallest positive delta clusters first
                indices = range(j, list_length)
            else:
                indices = list(reversed(range(j, list_length)))

            for i in indices:
                cluster = cluster_list[i]
                print "\n-----------------------------------------------------\n"
                print "\nChecking cluster " + str(j + 1) + " of " + str(list_length) + "...\n"
                print "\nOriginal increase in detection rate is ", cluster[0]
                j += 1
                old_detection_rate = detection_rate

                self.unlearn(cluster[1]) # unlearn the cluster
                self.init_ground(update=True) # find new accuracy, update the cached training space
                detection_rate = self.current_detection_rate
                if detection_rate > old_detection_rate: # if improved, record stats
                    cluster_count += 1 # number of unlearned clusters
                    unlearned_cluster_list.append(cluster)
                    self.current_detection_rate = detection_rate
                    cluster_print_stats(outfile, pollution_set3, detection_rate, cluster, cluster_count, attempt_count)
                    print "\nCurrent detection rate achieved is " + str(detection_rate) + ".\n"

                else:
                    self.learn(cluster[1]) # else relearn cluster and move to the next one
                    detection_rate = old_detection_rate

            if detection_rate > self.threshold:
                break

            else: # do the whole process again, this time with the training space - unlearned clusters
                del cluster_list
                cluster_list = cluster_au(self, gold)
                attempt_count += 1
                gc.collect()

        return cluster_count, attempt_count

    # -----------------------------------------------------------------------------------

    def get_mislabeled(self, update=False):
        """
        Returns the set of mislabeled emails (from the ground truth) based off of the
        current classifier state. By default assumes the current state's numbers and
        tester false positives/negatives have already been generated; if not, it'll run the
        predict method from the tester.
        """
        if update:
            self.init_ground()
        # find indices where pred != actual
        assert len(self.p_label) == len(self.test_y), \
            "self.p_val length: %r != self.test_y length: %r" % (len(self.p_val), len(self.test_y))
        mis_indices = []
        for x in range(len(self.p_val)):
            if self.p_label[x] != self.test_y[x]:
                mis_indices.append(x)

        mislabeled = [(self.p_val[i], self.test_x[i]) for i in mis_indices] # (p_val, data vector)
        mislabeled.sort(key=lambda x: abs(x[0]), reverse=True) # sort in descending order
        print "Generated mislabeled list of length: ", len(mislabeled)
        print [e[0] for e in mislabeled[:5]]
        
        return mislabeled

    def select_initial(self, mislabeled=None, option="mislabeled", working_set=None):
        """Returns an email to be used as the next seed for a cluster."""

        if option == "weighted":
            return self.weighted_initial(working_set,mislabeled)

    def weighted_initial(self, working_set, mislabeled):
        print "Total Cluster Centroids Chosen: ", len(self.mislabeled_chosen)

        print len(mislabeled), " mislabeled emails remaining as possible cluster centroids" 
        if len(mislabeled) == 0: #No more centers to select
            return (None, None, 'NO_CENTROIDS')
        else:
            prob, mislabeled_point = mislabeled.pop(0) # Choose most potent mislabeled email 
            self.mislabeled_chosen.append(mislabeled_point)

            print "Chose the mislabeled point with z = ", prob

            data_y, data_x = h.compose_set(working_set)

            init_email = None
            init_pos = None
            label = None
            if "frequency" in self.distance_opt:
                min_distance = sys.maxint
                for i,email in enumerate(data_x):
                    if email is not None:
                        current_distance = distance(email, mislabeled_point, self.distance_opt)
                        if current_distance < min_distance:
                            init_email = email
                            init_pos = i
                            min_distance = current_distance

            if init_email is None:
                print "Training emails remaining: ", len(data_x)
            else:
                label = data_y[init_pos]
                print "-> selected cluster centroid with label: ", label, " and distance: ", min_distance, " from mislabeled point"
            return (label, init_pos, init_email)
