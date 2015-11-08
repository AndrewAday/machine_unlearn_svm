from distance import distance
class Cluster:
    def __init__(self, msg, size, active_unlearner, label, distance_opt, 
                working_set=None, separate=True):
        
        self.clustroid = msg[1] # index of msg
        self.label = label
        self.common_features = []
        self.separate = separate
        self.size = size # arbitrarily set to 100
        self.active_unlearner = active_unlearner # point to calling au instance
        self.opt = distance_opt

        self.working_set = working_set
        self.train_y = self.working_set[0]
        self.train_x = self.working_set[1]
        self.pol_y = self.working_set[2]
        self.pol_x = self.working_set[3]
        self.data_y, self.data_x = h.compose_set(self.working_set)

        self.ham = set()
        self.spam = set()

        if 'frequency' in self.opt:
            self.cluster_word_frequency = msg[0] # actual vector representation of msg
            self.added = [] # keeps track of order emails are added

        self.dist_list = self.distance_array(self.separate) # returns list containing dist from all emails in phantom space to center clustroid
        self.cluster_set = self.make_cluster() # adds closest emails to cluster
        self.divide() # adds cluster emails to ham and spam

    def __repr__(self):
        return repr((self.label,self.clustroid))

    def distance_array(self, separate):
        """Returns a list containing the distances from each email to the center."""

        if separate: # if true, all emails must be same type (spam or ham) as centroid
            dist_list = []
            for x in range(len(self.data_y)):
                if self.data_y[x] == self.label: # same type 
                    dist_list.append((distance(self.data_x[x], self.cluster_word_frequency, self.opt), x))

        dist_list.sort() # sorts tuples by first element default, the distance

        print "\n ----------------Generated Distance Array----------------\n"
        print [email[1] for email in dist_list[:5]]

        return dist_list

    def update_dist_list(self, separate=True): 
        """Updates self.dist_list for the frequency method"""
        indices = [train[1] for train in self.dist_list] # get array of indices
        self.dist_list = [(distance(self.data_x[i], self.cluster_word_frequency, self.opt), i) for i in indices]
        self.dist_list.sort()

    def make_cluster(self):
        """Constructs the initial cluster of emails."""
        # self.dist_list = [t for t in self.dist_list if t is not None]
        if self.size > len(self.dist_list):
            print "\nTruncating cluster size...\n"
            self.size = len(self.dist_list)

        if 'frequency' in self.opt:
            emails = [self.clustroid] # list of added emails

            current_size = 1

            while current_size < self.size:
                d,i = self.dist_list.pop(0) # get nearest email
                emails.append(i) # add to list
                self.added.append(i) # track order in which emails are added
                self.cluster_word_frequency = helpers.update_word_frequencies(self.cluster_word_frequency, self.data_x[i]) # update word frequencies
                self.update_dist_list() # new cluster_word_frequency, so need to resort closest emails
                current_size += 1
            print "-> cluster initialized with size", len(emails)
        return set(emails)

    def divide(self):
        """Divides messages in the cluster between spam and ham."""
        for msg in self.cluster_set: # indices of added msgs
            if self.data_y[msg] == -1:
                self.ham.add(msg)
            else:
                self.spam.add(msg)

    def target_spam(self):
        """Returns a count of the number of spam emails in the cluster."""
        return len(self.spam)

    def target_set3(self):
        """Returns a count of the number of Set3 emails in the cluster."""
        counter = 0
        for i in self.cluster_set:
            if msg >= len(self.train_y):
                counter += 1
        return counter

    def target_set3_get_unpolluted(self):
        cluster_set_new = []
        spam_new = set()
        ham_new = set()
        for msg in self.cluster_set:
            if msg >= len(self.train_y): #msg is polluted, remove from cluster
                self.size -= 1
            else:
                cluster_set_new.append(msg)
                if self.train_y[msg] == -1:
                    ham_new.add(msg)
                else:
                    spam_new.add(msg)
        self.cluster_set = cluster_set_new
        self.spam = spam_new
        self.ham = ham_new
        return self # return the cluster

    def cluster_more(self, n):
        """Expands the cluster to include n more emails and returns these additional emails.
           If n more is not available, cluster size is simply truncated to include all remaining
           emails."""
        if 'frequency' in self.opt:
            if n >= len(self.dist_list):
                n = len(self.dist_list)
            print "Adding ", n, " more emails to cluster of size ", self.size, " via ", self.opt,  " method"
            self.size += n

            new_elements = []
            added = 0
            while added < n:
                nearest = self.dist_list[0][1] # get nearest email
                new_elements.append(nearest) # add to new list
                self.added.append(nearest)
                self.cluster_set.add(nearest) # add to original cluster set
                self.cluster_word_frequency = helpers.update_word_frequencies(self.cluster_word_frequency, nearest) # update word frequencies
                # self.dist_list = self.distance_array(self.separate) # update distance list w/ new frequency list
                del self.dist_list[0]
                self.update_dist_list()
                added += 1
            assert(len(new_elements) == n), str(len(new_elements)) + " " + str(n)
            assert(len(self.cluster_set) == self.size), str(len(self.cluster_set)) + " " + str(self.size)
            for msg in new_elements:
                if msg.train == 1 or msg.train == 3:
                    self.ham.add(msg)
                elif msg.train == 0 or msg.train == 2:
                    self.spam.add(msg)
            return new_elements 

        old_cluster_set = self.cluster_set
        if self.size + n <= len(self.dist_list):
            self.size += n

        else:
            print "\nTruncating cluster size...\n"
            if len(self.dist_list) > 0:
                self.size = len(self.dist_list)

        if self.sort_first:
            new_cluster_set = set(item[1] for item in self.dist_list[:self.size])
        else:
            k_smallest = quickselect.k_smallest
            new_cluster_set = set(item[1] for item in k_smallest(self.dist_list, self.size))

        new_elements = list(item for item in new_cluster_set if item not in old_cluster_set)
        self.cluster_set = new_cluster_set

        assert(len(self.cluster_set) == self.size), len(self.cluster_set)

        for msg in new_elements:
            if msg.train == 1 or msg.train == 3:
                self.ham.add(msg)
            elif msg.train == 0 or msg.train == 2:
                self.spam.add(msg)

        return new_elements

    def learn(self, n): # relearn only set.size elements. unlearning is too convoluted
        print "-> relearning a cluster of size ", self.size, " via intersection method"
        old_cluster_set = self.cluster_set
        self.ham = set()
        self.spam = set()
        self.cluster_set = set()
        self.dist_list = self.distance_array(self.separate)
        self.cluster_set = self.make_cluster()
        self.divide()
        new_cluster_set = self.cluster_set
        new_elements = list(item for item in old_cluster_set if item not in new_cluster_set)
        assert(len(self.cluster_set) == self.size), str(len(self.cluster_set)) + " " + str(self.size)
        assert(len(new_elements) == n), len(new_elements)        
        return new_elements

    def cluster_less(self, n):
        """Contracts the cluster to include n less emails and returns the now newly excluded emails."""

        old_cluster_set = self.cluster_set
        self.size -= n
        assert(self.size > 0), "Cluster size would become negative!"
        if self.sort_first:
            if self.opt == "intersection":
                new_elements = self.learn(n)
                return new_elements
            elif "frequency" in self.opt:
                unlearned = 0
                new_elements = []
                while unlearned < n:
                    email = self.added.pop() # remove most recently added email
                    new_elements.append(email) # add to new emails list
                    self.cluster_set.remove(email)
                    # self.working_set.append(email)
                    self.cluster_word_frequency = helpers.revert_word_frequencies(self.cluster_word_frequency, email) # update word frequencies
                    self.dist_list.append((0, email))
                    unlearned += 1
                #self.dist_list = self.distance_array(self.separate) 
                self.update_dist_list()
                assert(len(new_elements) == n), str(len(new_elements)) + " " + str(n)
                assert(len(self.cluster_set) == self.size), str(len(self.cluster_set)) + " " + str(self.size)

                for msg in new_elements:
                    if msg.train == 1 or msg.train == 3:
                        self.ham.remove(msg)
                    elif msg.train == 0 or msg.train == 2:
                        self.spam.remove(msg)

                return new_elements
            else:
                new_cluster_set = set(item[1] for item in self.dist_list[:self.size])
        else:
            k_smallest = quickselect.k_smallest
            new_cluster_set = set(item[1] for item in k_smallest(self.dist_list, self.size))

        new_elements = list(item for item in old_cluster_set if item not in new_cluster_set)
        self.cluster_set = new_cluster_set

        assert(len(self.cluster_set) == self.size), str(len(self.cluster_set)) + " " + str(self.size)
        assert(len(new_elements) == n), len(new_elements)        

        for msg in new_elements:
            if msg.train == 1 or msg.train == 3:
                self.ham.remove(msg)
            elif msg.train == 0 or msg.train == 2:
                self.spam.remove(msg)

        return new_elements