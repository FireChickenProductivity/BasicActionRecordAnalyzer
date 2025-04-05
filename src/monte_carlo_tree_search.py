#This defines an implementation of monte Carlo tree search usable for finding better combinations of voice command recommendations when picking the best n

from recommendation_generation import PotentialCommandInformation
import random
import math
from calculation_utilities import compute_max

NUM_ALTERNATIVES_TO_EXPLORE = 5

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
        self.roots: dict[int, ScoredNode] = {}
        self.total_explored = 0

    def back_propagate_score(self, path: list[int], score: int):
        root = self.roots[path[0]]
        root.handle_score(score)
        for index in range(1, len(path)):
            root = root.get_child(path[index])
            root.handle_score(score)
        self.total_explored += 1

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

    def compute_times_explored(self, progress: ScoredNode) -> int:
        return progress.get_times_explored() if progress else self.total_explored

    def compute_best_child(self, progress: ScoredNode):
        """Computes the best child and corresponding value using UCT"""
        if progress:
            children = progress.get_children()
            times_parent_explored = progress.get_times_explored()
        else:
            children = self.roots.values()
            times_parent_explored = self.total_explored
        if not children:
            return None, 0
        best_value = 0
        best_index = -1

        best_score = max(children, key=lambda x: x.get_score()).get_score()
        for child in children:
            value = child.get_score()/best_score + math.sqrt(math.log(times_parent_explored)/child.get_times_explored())
            if value > best_value:
                best_index = child.get_index()
                best_value = value
        return best_index, value

    def return_next_index_after_exploration(self, progress: ScoredNode) -> int:
        if not progress:
            return max(self.roots, default=0) + 1
        return max(progress.get_children(), key=lambda x: x.get_index(), default=progress.get_index())
                    
    def handle_expansion(self, path):
        progress = None
        for choice in path:
            progress = self.get_progress_from_choice(choice, progress)

    def compute_depth(self, progress: ScoredNode):
        return progress.get_depth() if progress else 0

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
        self.recommendations = sorted(
            recommendations, 
            key=lambda r: r.get_number_of_words_saved(),
            reverse=True,
        )

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
    
    def compute_alternative_score(self, progress, index: int) -> float:
        return math.sqrt(math.log(self.exploration_data.compute_times_explored(progress)))
        

    def compute_best_alternative(self, progress) -> tuple[int, float]:
        next_index = self.exploration_data.return_next_index_after_exploration(
            progress
        )
        if next_index == len(self.recommendations):
            return -1, -1
        #If exactly the limit left, just return the next one
        num_remaining = len(self.recommendations) - self.exploration_data.compute_depth(progress)
        if num_remaining == self.recommendation_limit:
            return next_index, self.compute_alternative_score(progress, next_index)
        ending_index = min(
            next_index + NUM_ALTERNATIVES_TO_EXPLORE, 
            len(self.recommendations)
            )
        return compute_max(
                    range(next_index, ending_index), 
                    lambda i: self.compute_alternative_score(progress, i)
                    )

    def select_next_starting_path(self):
        #Recursively pick best node until reaching leaf
        if not self.exploration_data.compute_times_explored(None):
            return [0]
        path = []
        progress = None
        best_child, value = self.exploration_data.compute_best_child(progress)
        alternative, alternative_value = self.compute_best_alternative(progress)
        if alternative_value > value:
            path.append(alternative)
            return path
        while best_child:
            path.append(best_child)
            progress = self.exploration_data.get_progress_from_choice(best_child, progress)
            best_child, value = self.exploration_data.compute_best_child(progress)
            alternative, alternative_value = self.compute_best_alternative(progress)
            if alternative_value > value:
                path.append(alternative)
                return path
        if len(path) < self.recommendation_limit:
            alternative, _ = self.compute_best_alternative(progress)
            path.append(alternative)
        return path

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