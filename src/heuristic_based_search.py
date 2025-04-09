from queue import PriorityQueue, Empty

class Heuristic:
    def __init__(self, can_be_applied, update_state):
        self.can_be_applied = can_be_applied
        self.update_state = update_state

class HeuristicSearchState:
    def __init__(self, chosen, recommendations, score, scoring_function):
        self.chosen = chosen
        self.recommendations = recommendations
        self.score = score
        self.scoring_function = scoring_function
    
    def set_chosen(self, chosen):
        self.chosen = chosen
        self.score = self.scoring_function(self.recommendations, chosen)

    def get_chosen(self):
        return self.chosen

    def set_recommendations(self, recommendations):
        self.recommendations = recommendations

    def get_recommendations(self):
        return self.recommendations

    def __lt__(self, other) -> bool:
        return self.score < other.score

    def copy(self):
        return HeuristicSearchState(self.chosen, self.recommendations, self.score, self.scoring_function)

class HeuristicSearcher:
    def __init__(self, safe_heuristics: list[Heuristic], unsafe_heuristics: list[Heuristic], recommendations, score_function, best_greedy_score: float):
        self.safe_heuristics = safe_heuristics
        self.unsafe_heuristics = unsafe_heuristics
        self.recommendations = recommendations
        self.score_function = score_function
        self.best_score = best_greedy_score
        self.chosen = []
        self.queue = PriorityQueue()
    
    def enqueue(self, chosen, recommendations):
        self.queue.put_nowait(HeuristicSearchState(chosen, recommendations, self.score_function(chosen)))

    def apply_safe_heuristics(self, state: HeuristicSearchState):
        for heuristic in self.safe_heuristics:
            if heuristic.can_be_applied(state):
                heuristic.update_state(state)

    def handle_new_state(self, state: HeuristicSearchState):
        if state.get_score() > self.best_score:
            self.best_score = state.get_score()
            self.chosen = state.get_chosen()

    def search(self, max_operations: int):
        try:
            for _ in range(max_operations):
                current_best: HeuristicSearchState = self.queue.get_nowait()
                self.apply_safe_heuristics(current_best)
                self.handle_new_state(current_best)
        except Empty:
            pass


