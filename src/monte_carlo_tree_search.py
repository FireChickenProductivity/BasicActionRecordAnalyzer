#This defines an implementation of monte Carlo tree search usable for finding better combinations of voice command recommendations when picking the best n

from recommendation_generation import PotentialCommandInformation
import random

class ScoredNode:
    def __init__(self, index: int, depth: int=0, parent=None):
        self.index = index
        self.score = 0
        self.children = {}
        self.times_explored = 0
        self.depth = depth
        self.parent = parent

    def get_depth(self):
        return self.depth
    
    def get_index(self):
        return self.index

    def get_children(self):
        return self.children.values()

    def get_parent(self):
        return self.parent

    def get_score(self):
        return self.score

    def get_times_explored(self) -> int:
        return self.times_explored

    def handle_score(self, score):
        self.score = max(score, self.score)
        self.times_explored += 1

    def add_child(self, node):
        self.children[node.get_index()] = node

    def has_child(self, index):
        return index in self.children

    def get_child(self, index):
        if not self.has_child(index):
            self.children[index] = ScoredNode(index, self.depth + 1, self)
        return self.children[index]

class MonteCarloExplorationData:
    def __init__(self):
        """Contains data on the exploration done so far searching for a good set of recommendations"""
        self.roots = {}

    def back_propagate_score(self, path: list[int], score: int):
        root = self.roots[path[0]]
        root.handle_score(score)
        for index in range(1, len(path)):
            root = root.get_child(path[index])
            root.handle_score(score)

    def get_progress_from_choice(self, choice: int, progress):
        if not progress:
            if choice not in self.roots:
                self.roots[choice] = ScoredNode(choice)
            return self.roots[choice]
        return progress.get_child(choice)

    def compute_score_and_times_explored_for_values(self, progress: ScoredNode):
        children = progress.get_children()
        result = {}
        for child in children:
            result[child.get_index()] = (child.get_score(), child.get_times_explored())
        return result

    def compute_best_child(self, progress: ScoredNode):
        if progress:
            children = progress.get_children()
        else:
            children = self.roots.values()
        if not children:
            return None
        best_score = 0
        best_index = -1
        for child in children:
            score = child.get_score()
            if score > best_score:
                best_index = child.get_index()
                best_score = score
        return best_index

    def handle_expansion(self, path):
        progress = None
        for choice in path:
            progress = self.get_progress_from_choice(choice, progress)

class MonteCarloTreeSearcher:
    def __init__(
        self,
        scoring_function,
        recommendation_limit: int,
        recommendations: list[PotentialCommandInformation]
    ):
        self.scoring_function = scoring_function
        self.best_recommendation: list[PotentialCommandInformation]
        self.best_score: int = 0
        self.recommendation_limit = recommendation_limit
        self.exploration_data = MonteCarloExplorationData()
        self.recommendations = recommendations

    def get_best_score(self):
        return self.best_score

    def get_best_recommendation(self):
        return self.best_recommendation

    def simulate_play_out(
            self, 
            starting_path: list[PotentialCommandInformation],
        ):
        path = starting_path[:]
        last_potential_index = len(self.recommendations) - self.recommendation_limit - len(starting_path)
        next_possible_index = len(starting_path)
        for _ in range(self.recommendation_limit - len(starting_path)):
            choice = random.randint(next_possible_index, last_potential_index)
            next_possible_index = choice + 1
            last_potential_index += 1
            path.append(choice)
        potential_recommendations = [self.recommendations[i] for i in path]
        score = self.scoring_function(potential_recommendations)
        if score > self.best_score:
            self.best_score = score
            print("New best score from random exploration", self.best_score)
            self.best_recommendation = potential_recommendations
        self.exploration_data.back_propagate_score(starting_path, score)
    
    def select_next_starting_path(self):
        pass

    def expand(self, path):
        self.exploration_data.handle_expansion(path)

    def explore_solution(self):
        #Need to pick a good node to explore
        #Need to do a play out
        #Back propagate
        starting_path = self.select_next_starting_path()
        self.expand(starting_path)
        self.simulate_play_out(starting_path)


    def explore_solutions(self, num_trials: int):
        for _ in range(num_trials):
            self.explore_solution()
    
def perform_monte_carlo_tree_search(recommendations, recommendation_limit, scoring_function, number_of_trials):
    searcher = MonteCarloTreeSearcher(scoring_function, recommendation_limit, recommendations)
    searcher.explore_solutions(number_of_trials)
    return searcher.get_best_recommendation(), searcher.get_best_score()