from recommendation_generation import PotentialCommandInformation, compute_string_representation_of_actions, PotentialAbstractCommandInformation 
from input_parsing import NO_NUMBER_OF_RECOMMENDATIONS_LIMIT

def compute_words_saved_per_use(command: PotentialCommandInformation):
    return command.get_number_of_words_saved()/command.get_number_of_times_used()

def compute_number_of_elements_in_range(start: int, final: int) -> int:
    return final - start + 1

def compute_action_subsequences(actions):
    for i in range(len(actions)):
        for j in range(i, len(actions)):
            if compute_number_of_elements_in_range(i, j) < len(actions):
                sub_actions = actions[i:j + 1]
                subsequence = compute_string_representation_of_actions(sub_actions)
                yield subsequence

class AbstractRecommendationInformation:
    """Holds information associated with an abstract command recommendation useful for tracking overlap with other commands"""
    def __init__(self, abstract_command_information: PotentialAbstractCommandInformation):
        self.abstract_command_information = abstract_command_information
        self.concrete_instantiations: set[str] = set()
        self.concrete_sequences: set[str] = set()

    def add_instantiation(self, instantiation: str, actions):
        self.concrete_instantiations.add(instantiation)
        for subsequence in compute_action_subsequences(actions):
            self.concrete_sequences.add(subsequence)

    def get_number_of_non_concrete_occurrences(self) -> int:
        return self.abstract_command_information.get_number_of_instantiations() - len(self.concrete_instantiations)

    def get_command(self):
        return self.abstract_command_information

def _compute_recommendations_sequences(recommendations: list[PotentialCommandInformation]):
    action_sequences: dict[str, PotentialCommandInformation] = {}
    abstract_information: dict[str, AbstractRecommendationInformation] = {}
    concrete_sequences: list[str] = []
    for command in recommendations:
        representation = compute_string_representation_of_actions(command.get_actions())
        action_sequences[representation] = command
        if command.is_abstract():
            abstract_information[representation] = AbstractRecommendationInformation(command)
        else:
            concrete_sequences.append(representation)
    return action_sequences, abstract_information, concrete_sequences

def _accumulate_instantiation_information_for_abstract_sequences(
    action_sequences: dict[str, PotentialCommandInformation],
    abstract_information: dict[str, AbstractRecommendationInformation],
    concrete_sequences: list[str]
    ):
    for concrete_sequence in concrete_sequences:
        for abstract_sequence in abstract_information:
            corresponding_abstract_information = abstract_information[abstract_sequence]
            if concrete_sequence in corresponding_abstract_information.get_command().get_instantiation_set():
                corresponding_abstract_information.add_instantiation(concrete_sequence, action_sequences[concrete_sequence].get_actions())

def _compute_concrete_recommendations_score_ignoring_overlap(
    action_sequences: dict[str, PotentialCommandInformation],
    concrete_sequences: list[str]
    ) -> int:
    score: int = 0
    for c in concrete_sequences: score += action_sequences[c].get_number_of_words_saved()
    return score

def _compute_abstract_recommendations_score_ignoring_overlap(
    abstract_information: dict[str, AbstractRecommendationInformation]
) -> int:
    score = 0
    for sequence in abstract_information:
        relevant_information = abstract_information[sequence]
        score += relevant_information.get_number_of_non_concrete_occurrences()*compute_words_saved_per_use(relevant_information.get_command())
    return score

def _compute_score_overlap(
    action_sequences: dict[str, PotentialCommandInformation]
) -> int:
    overlap = 0
    for sequence in action_sequences:
        command: PotentialCommandInformation = action_sequences[sequence]
        actions = command.get_actions()
        for subsequence in compute_action_subsequences(actions):
            if subsequence in action_sequences:
                smaller_command = action_sequences[subsequence]
                #For every instance of the bigger command, the smaller command was present so subtract the number of words that we thought the smaller command had saved during those instances of the bigger command
                overlap += compute_words_saved_per_use(smaller_command)*command.get_number_of_times_used()
    return overlap

def _compute_score_ignoring_overlap(
    action_sequences: dict[str, PotentialCommandInformation],
    abstract_information: dict[str, AbstractRecommendationInformation],
    concrete_sequences: list[str]
) -> int:
    concrete_score: int = _compute_concrete_recommendations_score_ignoring_overlap(
        action_sequences,
        concrete_sequences
    )
    abstract_score: int = _compute_abstract_recommendations_score_ignoring_overlap(abstract_information)
    score: int = concrete_score + abstract_score
    return score

def compute_recommendations_score(recommendations: list[PotentialCommandInformation]):
    sequences = _compute_recommendations_sequences(recommendations)
    action_sequences = sequences[0]
    _accumulate_instantiation_information_for_abstract_sequences(*sequences)
    score: int = _compute_score_ignoring_overlap(*sequences)
    overlap: int = _compute_score_overlap(action_sequences)
    result: int = score - overlap
    assert result >= 0
    return result

#TODO: Potentially Deal with recommendations for this function with a linked list class. Using a list may be faster because of cache optimization
#Try to optimize to not need repeatedly recomputing the action representations
def compute_best_recommendations(recommendation_limit, recommendations):
    if recommendation_limit == NO_NUMBER_OF_RECOMMENDATIONS_LIMIT:
        return recommendations
    best_recommendations = []
    for i in range(recommendation_limit):
        best_score = 0
        best_recommendation_index = None
        for index, recommendation in enumerate(recommendations):
            best_recommendations.append(recommendation)
            score = compute_recommendations_score(best_recommendations)
            if score > best_score:
                best_score = score
                best_recommendation_index = index
            best_recommendations.pop()
        if best_score == 0:
            break
        else:
            best_recommendations.append(recommendations[best_recommendation_index])
            recommendations.pop(best_recommendation_index)
    return best_recommendations