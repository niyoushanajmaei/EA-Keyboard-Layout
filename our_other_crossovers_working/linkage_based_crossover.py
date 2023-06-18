import numpy as np
from permutationsga.problem import Solution
from permutationsga.qap import QAP
import logging


def indices_of_highest_frequency(bigram):
    return np.unravel_index(np.argmax(bigram, axis=None), bigram.shape)

def get_better_region(distance_matrix):
    row_sum = np.sum(distance_matrix, axis = 1)
    col_sum = np.sum(distance_matrix, axis = 0)
    distances = np.add(row_sum, col_sum)
    better_region = np.argsort(distances)[:8]
    return better_region
    
def get_highest_frequency(frequency_matrix):
    row_sum = np.sum(frequency_matrix, axis = 1)
    col_sum = np.sum(frequency_matrix, axis = 0)
    frequencies = np.add(row_sum, col_sum)
    high_frequency = np.argsort(frequencies)[-8:]
    return high_frequency


class LinkageBasedCrossover():

    def __init__(self):
        logging.basicConfig(filename="cross.log", level=logging.DEBUG, filemode='w')
        self.logger = logging.getLogger(__name__)
        self.logger.debug("Initialized LinkageBasedCrossover")
        self.logger.disabled = False

    def linkage_based_crossover(self, matrices, parent1: Solution, parent2: Solution):
        """
        Performs a crossover between two parents
        Maintain the linkages of highly correlated characters from parents
        Start copying from the bigrams from highest frequency to lowest frequency to decrease the chance of breaking the linkage
        """
        self.better_region = get_better_region(matrices[1])
        self.worse_region = [i for i in range(26) if i not in self.better_region]
        self.highest_frequency = get_highest_frequency(matrices[0])
        bifreq = np.copy(matrices[0])

        self.logger.debug("Entering linkage_based_crossover")

        assert parent1.e is not None, "Ensure solution s0 is initialized before use."
        assert parent2.e is not None, "Ensure solution s1 is initialized before use."

        children = [np.array([-1 for i in range(26)]), np.array([-1 for i in range(26)])]
        parents = [parent1.e, parent2.e]

        copied_characters = []

        while True:
            # terminate if all characters are copied\
            self.logger.debug(f"Currently copied characters: {len(set(copied_characters))}")
            if len(set(copied_characters)) == 26:
                self.logger.debug("All characters are copied")
                break

            # find the highest frequency bigram that is not yet copied
            # mark this bigram as copied by setting its frequency to 0
            while True:
                char_one, char_two = indices_of_highest_frequency(bifreq)
                self.logger.debug(f"Bigram {char_one} {char_two} has highest frequency of {bifreq[char_one, char_two]}")
                if bifreq[char_one, char_two] == 0:
                    break
                if char_one not in copied_characters or char_two not in copied_characters:
                    break
                else:
                    bifreq[char_one, char_two] = 0
                    bifreq[char_two, char_one] = 0
            
            if bifreq[char_one, char_two] == 0:
                self.logger.debug("Max frequency was 0, breaking")
                break
            else:
                bifreq[char_one, char_two] = 0
                bifreq[char_two, char_one] = 0
                
            if char_one == char_two:
                self.logger.debug(f"Both characters are the same, skipping. frequency was {bifreq[char_one, char_two]}")
                bifreq[char_one, char_two] = 0
                continue

            self.logger.debug(f"Copying bigram {char_one} {char_two}")

            # copy the bigram to the children
            # if both characters are not copied yet, copy the whole bigram from one parent to one child
            if char_one not in copied_characters and char_two not in copied_characters:
                self.logger.debug(f"Both characters are not copied yet")
                # randomly choose a parent
                parent = np.random.choice([0, 1])
                skipped_one = self.copy_chars_from_parent_to_child([char_one, char_two], parents[parent], children[0])
                skipped_two = self.copy_chars_from_parent_to_child([char_one, char_two], parents[1-parent], children[1])
                copied_characters.extend([char_one, char_two])
                self.logger.debug(f"copied {char_one} {char_two} to both children, skipped {skipped_one} for one, and {skipped_two} for two")

            # if one of the characters is already copied, copy the other character from the same parent to the same child
            else:
                chars = [char_one, char_two]
                for i, char in enumerate(chars):
                    if char in children[0] or char in children[1]:
                        self.logger.debug(f"Character {char} is already copied but {chars[1-i]} is not")
                        self.copy_when_one_char_is_already_copied(chars[i], chars[1-i], parents, children)
                        copied_characters.append(chars[1-i])
                        self.logger.debug(f"Copied {chars[1-i]} to both children")
                        break
        
        
        # account for skipped characters
        self.logger.debug("Accounting for skipped characters")
        for char in [i for i in range(26) if i not in children[0]]:
            self.logger.debug(f"Placing {char} randomly in child one")
            self.place_randomly_in_child(char, children[0])  # only if not
        for char in [i for i in range(26) if i not in children[1]]:
            self.logger.debug(f"Placing {char} randomly in child two")
            self.place_randomly_in_child(char, children[1])
        
        # check if both children are valid
        self.logger.debug("Checking if both children are valid")
        if not self.is_valid(children[0]):
            self.logger.debug("Child 1 is not valid")
            self.logger.debug(children[0])
        if not self.is_valid(children[1]):
            self.logger.debug("Child 2 is not valid")
            self.logger.debug(children[1])
        
        self.logger.debug("Exiting linkage_based_crossover")
        return [Solution(children[0]), Solution(children[1])]


    def place_randomly_in_child(self, char, child):
        self.logger.debug(f"Placing {char} randomly in child")
        if char in child:
            return
        if -1 not in child:
            self.logger.debug("Child is full")
            return
        placed = False
        for i in range(10):
            if char in self.highest_frequency:
                self.logger.debug(f"Placing {char} in highest frequency region")
                rand = np.random.randint(0, len(self.better_region), size=1)[0]
                loc = self.better_region[rand]
            else:
                self.logger.debug(f"Placing {char} in lowest frequency region")
                rand = np.random.randint(0, len(self.worse_region), size=1)[0]
                loc = self.worse_region[rand]
            if child[loc] == -1:
                child[loc] = char
                placed = True
                break
        if not placed:
            self.logger.debug(f"Could not place {char} in child, placing randomly")
            indices = np.where(child == -1)[0]
            index = np.random.choice(indices)
            child[index] = char
        self.logger.debug(f"Placed {char} in child")
        return


    def is_valid(self, child):
        """
        Returns True if all characters are unique and there is no -1 in the child
        """
        if -1 in child:
            return False
        if len(set(child)) != 26:
            return False
        return True
    

    def copy_when_one_char_is_already_copied(self, char_one, char_two, parents, children):
        """
        char_one is in copied_characters
        char_two is not in copied_characters
        """
        # if char_one was copied from a parent, copy char_two from the same parent
        # copy from a random parent if char_one was placed randomly
        index_of_child_one = np.where(children[0] == char_one)[0][0]
        parent_of_child_one = self.which_parent_was_char_taken_from(char_one, parents, index_of_child_one)
        if parent_of_child_one is not None:
            self.copy_chars_from_parent_to_child([char_two], parents[parent_of_child_one], children[0])
        else:
            parent_one = np.random.choice([0, 1])
            self.copy_chars_from_parent_to_child([char_two], parents[parent_one], children[0])
        
        index_of_child_two = np.where(children[1] == char_one)[0][0]
        parent_of_child_two = self.which_parent_was_char_taken_from(char_one, parents, index_of_child_two)
        if parent_of_child_two is not None:
            self.copy_chars_from_parent_to_child([char_two], parents[parent_of_child_two], children[1])
        else:
            parent_two = np.random.choice([0, 1])
            self.copy_chars_from_parent_to_child([char_two], parents[parent_two], children[1])
            


    def copy_chars_from_parent_to_child(self, chars, parent, child):
        """
        If the index is already filled, place randomly
        """
        for char in chars:
            i = np.where(parent == char)
            if child[i] != -1: 
                self.place_randomly_in_child(char, child)
            else:
                self.logger.debug(f"Copying {char} in child")
                child[i] = char
        return 
    
    
    def which_parent_was_char_taken_from(self, char, parents, index):
        self.logger.debug(f"Finding which parent {char} was taken from with index {index}")
        for i, parent in enumerate(parents):
            j = np.where(parent == char)
            if j == index:
                return i
        return None

