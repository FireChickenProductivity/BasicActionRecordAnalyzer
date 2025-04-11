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
        return self.score/self.times_explored

    def get_times_explored(self) -> int:
        return self.times_explored

    def handle_score(self, score):
        self.score += score
    
    def handle_exploration(self):
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

    def compute_best_child(self, progress: ScoredNode, c=1):
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
            value = child.get_score()/best_score + c*math.sqrt(math.log(times_parent_explored)/child.get_times_explored())
            if value > best_value:
                best_index = child.get_index()
                best_value = value
        return best_index, value

    def compute_next_index_after_exploration(self, progress: ScoredNode) -> int:
        if not progress:
            return max(self.roots, default=0) + 1
        return max(progress.get_children(), key=lambda x: x.get_index(), default=progress).get_index() + 1
                    
    def handle_expansion(self, path):
        progress = None
        for choice in path:
            progress = self.get_progress_from_choice(choice, progress)

    def compute_depth(self, progress: ScoredNode):
        return progress.get_depth() if progress else 0

    def handle_exploration(self, path):
        self.total_explored += 1
        progress = None
        for choice in path:
            progress = self.get_progress_from_choice(choice, progress)
            progress.handle_exploration()

    def create_initial_for_path(self, path: list[int]):
        progress = None
        for choice in path:
            progress = self.get_progress_from_choice(choice, progress)
            progress.handle_exploration()
        return progress

class MonteCarloTreeSearcher:
    def __init__(
        self,
        scoring_function,
        recommendation_limit: int,
        recommendations: list[PotentialCommandInformation],
        start,
        maximum_depth: int,
        greedy_function,
        greedy_depth: int=3,
        rollouts_per_exploration: int=10,
        rollouts_per_child_expansion: int=1,
    ):
        """recommendations should be sorted in ascending order of value"""
        self.scoring_function = scoring_function
        self.best_recommendation: list[PotentialCommandInformation]
        self.best_score: int = 0
        self.best_recommendation_indexes: list[int]
        self.recommendation_limit = recommendation_limit
        self.exploration_data = MonteCarloExplorationData()
        self.recommendations = recommendations
        self.start = start
        self.initial_progress = self.exploration_data.create_initial_for_path(self.start)
        self.maximum_depth = min(len(start) + maximum_depth, recommendation_limit)
        self.rollouts_per_exploration = rollouts_per_exploration
        self.rollouts_per_child_expansion = rollouts_per_child_expansion
        self.greedy_function = greedy_function
        self.greedy_depth = greedy_depth

    def get_best_score(self):
        return self.best_score

    def get_best_recommendation(self):
        return self.best_recommendation

    def get_best_recommendation_indexes(self):
        return self.best_recommendation_indexes

    def compute_best_average_score_move(self):
        return self.exploration_data.compute_best_child(self.initial_progress, 0)[0]

    def simulate_play_out(
            self, 
            starting_path: list[int],
            use_greedy: bool=False
        ):
        path = starting_path[:]
        num_remaining = self.recommendation_limit - len(starting_path)
        next_possible_index = path[-1]
        last_potential_index = len(self.recommendations) - num_remaining
        if use_greedy:
            _, _, path = self.greedy_function(min(self.greedy_depth, num_remaining), self.recommendations, self.scoring_function, start=path, index_range=(next_possible_index, last_potential_index + 1))
            path = sorted(path)
            num_remaining = self.recommendation_limit - len(path)
        last_potential_index = len(self.recommendations) - num_remaining
        next_possible_index = path[-1]
        for _ in range(num_remaining):
            choice = random.randint(next_possible_index, last_potential_index)
            next_possible_index = choice + 1
            last_potential_index += 1
            path.append(choice)
        potential_recommendations = [self.recommendations[i] for i in path]
        score = self.scoring_function(potential_recommendations)
        if score > self.best_score:
            self.best_score = score
            print("New best score from random exploration", self.best_score, "with depth", len(starting_path), "and starting path score", self.scoring_function([self.recommendations[i] for i in starting_path]), starting_path)
            self.best_recommendation = potential_recommendations
            self.best_recommendation_indexes = path
        self.exploration_data.back_propagate_score(starting_path, score)
    
    def select_next_starting_path(self):
        #Recursively pick best node until reaching leaf
        path = self.start[:]
        progress = self.initial_progress
        best_child, _ = self.exploration_data.compute_best_child(progress)
        while best_child is not None and len(path) < self.maximum_depth - 1:
            path.append(best_child)
            progress = self.exploration_data.get_progress_from_choice(best_child, progress)
            best_child, _ = self.exploration_data.compute_best_child(progress)
        if len(path) < self.maximum_depth - 1 and best_child is None:
            self.explore_every_child(path)
            best_child, _ = self.exploration_data.compute_best_child(progress)
            path.append(best_child)
        return path

    def explore_every_child(self, starting_path: list[int]):
        if len(starting_path) < self.maximum_depth:
            start = starting_path[-1] + 1 if starting_path else 0
            ending = len(self.recommendations) - self.recommendation_limit + (len(starting_path))
            for i in range(start, ending):
                starting_path.append(i)
                self.expand(starting_path)
                for _ in range(self.rollouts_per_child_expansion): self.simulate_play_out(starting_path, use_greedy=True)
                self.exploration_data.handle_exploration(starting_path)
                starting_path.pop()

    def expand(self, path):
        self.exploration_data.handle_expansion(path)

    def explore_solution(self):
        #Need to pick a good node to explore
        #Need to do a play out
        #Back propagate
        starting_path = self.select_next_starting_path()
        assert len(starting_path) <= self.recommendation_limit, (starting_path, self.recommendation_limit)
        self.expand(starting_path)
        for _ in range(self.rollouts_per_exploration): self.simulate_play_out(starting_path, use_greedy=True)
        self.exploration_data.handle_exploration(starting_path)

    def explore_solutions(self, num_trials: int):
        for _ in range(num_trials):
            self.explore_solution()

    def seed(self, seed):
        indexes = [self.recommendations.index(i) for i in seed]
        self.expand(indexes)
        self.simulate_play_out(indexes)
        self.exploration_data.handle_exploration(indexes)
        self.best_score = 0
    
def perform_monte_carlo_tree_search(recommendations, recommendation_limit, scoring_function, number_of_trials, seed=None, greedy_function=None):
    recommendations = sorted(
            recommendations, 
            key=lambda r: r.get_number_of_words_saved(),
            reverse=True,
        )
    indexes = []
    best: list[PotentialCommandInformation]
    best_score = 0
    for i in range(recommendation_limit - 1):
        print(f"Running round {i + 1} of tree search")
        searcher = MonteCarloTreeSearcher(scoring_function, recommendation_limit, recommendations, indexes, maximum_depth=recommendation_limit, rollouts_per_exploration=10, greedy_depth=1, greedy_function=greedy_function)
        if seed: searcher.seed(seed)
        searcher.explore_solutions(number_of_trials)
        new_index = searcher.get_best_recommendation_indexes()[i]
        if new_index != i:
            recommendations[i], recommendations[new_index] = recommendations[new_index], recommendations[i],
        indexes.append(i)
        if searcher.get_best_score() > best_score:
            best_score = searcher.get_best_score()
            best = searcher.get_best_recommendation()
        if greedy_function:
            greedy_result, greedy_score, _ = greedy_function(recommendation_limit, recommendations, start=indexes)
            if greedy_score > best_score:
                best_score = greedy_score
                best = greedy_result
                print(f"Got better result with greedy {best_score}")

    return best, best_score