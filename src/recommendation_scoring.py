from recommendation_generation import PotentialCommandInformation, compute_string_representation_of_actions, PotentialAbstractCommandInformation 
from input_parsing import NO_NUMBER_OF_RECOMMENDATIONS_LIMIT
from collections import Counter
from action_records import BasicAction
from action_utilities import create_insert_action, is_insert, get_insert_text, is_insert_only_actions, get_insert_text_from_insert_only_actions
from monte_carlo_tree_search import perform_monte_carlo_tree_search
import time
import multiprocessing
import math

def compute_words_saved_per_use(command: PotentialCommandInformation):
    return command.get_number_of_words_saved()/command.get_number_of_times_used()

def compute_number_of_elements_in_range(start: int, final: int) -> int:
    return final - start + 1

def compute_string_subsequences(text: str):
    for i in range(len(text)):
        for j in range(i, len(text)):
            if compute_number_of_elements_in_range(i, j) < len(text):
                yield text[i:j + 1]

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

def _compute_number_of_commands_including_action(recommendations: list[PotentialCommandInformation]) -> dict[str, int]:
    result = Counter()
    for recommendation in recommendations:
        unique_actions = set(
            [compute_string_representation_of_actions([action])
             for action in recommendation.get_actions()
             ]
        )
        for unique_action in unique_actions:
            result[unique_action] += 1
    return result

def _compute_single_inserts_from_commands(recommendations: list[PotentialCommandInformation]):
    single_inserts: set[str] = set()
    for recommendation in recommendations:
        actions = recommendation.get_actions()
        if is_insert_only_actions(actions):
            single_inserts.add(get_insert_text_from_insert_only_actions(actions))
    return single_inserts

def _compute_max_nonidentical_prefix_or_suffix_similarity(text: str, others: set[str]):
    best = 0
    for other in others:
        if other != text and len(other) >= best:
            smallest_size = min(len(other), len(text))
            for i in range(1, smallest_size + 1):
                text_sub_string = text[-i:]
                if other[-i:] == text_sub_string:
                    best = max(len(text_sub_string), best)
                else:
                    break
            for i in range(1, smallest_size + 1):
                text_sub_string = text[:i]
                if other[:i] == text_sub_string:
                    best = max(len(text_sub_string), best)
                else:
                    break
    return best

def _score_recommendations_weighting_by_inverse_action_frequency(
    recommendations: list[PotentialCommandInformation],
    num_commands_including_action: dict[str, int],
    single_inserts: set[str]
) -> float:
    score = 0.0
    for recommendation in recommendations:
        actions = recommendation.get_actions()
        if is_insert_only_actions(actions) and len(single_inserts) > 1:
            inserted_text = get_insert_text_from_insert_only_actions(actions)
            similarity = _compute_max_nonidentical_prefix_or_suffix_similarity(inserted_text, single_inserts)
            if similarity == 0:
                weight = 1
            else:
                weight = (similarity/len(inserted_text))**2
        else:
            weight = 0
            for action in actions:
                representation = compute_string_representation_of_actions([action])
                weight += 1/(num_commands_including_action[representation])
            weight /= len(actions)
        score += weight*recommendation.get_number_of_words_saved()
    return score

def compute_heuristic_recommendation_score(recommendations: list[PotentialCommandInformation]) -> float:
    num_commands_including_action = _compute_number_of_commands_including_action(recommendations)
    single_inserts = _compute_single_inserts_from_commands(recommendations)
    return _score_recommendations_weighting_by_inverse_action_frequency(
        recommendations,
        num_commands_including_action,
        single_inserts
    )

def _append_insert_subsequences(collection: list, action: BasicAction):
    inserted_text = get_insert_text(action)
    for s in compute_string_subsequences(inserted_text):
        action = create_insert_action(s)
        rep = compute_string_representation_of_actions([action])
        collection.append(rep)

def _append_insert_subsequences_with_multiple_actions(
    collection: list[str],
    sub_actions: list[BasicAction]
):
    """This assumes that there is more than one action"""
    beginning_inserts: list[str]
    ending_inserts: list[str]
    if is_insert(sub_actions[0]):
        beginning_inserts = []
        inserted_text = get_insert_text(sub_actions[0])
        if len(inserted_text) > 1:
            for i in range(1, len(inserted_text)):
                beginning_inserts.append(inserted_text[i:])
    if is_insert(sub_actions[-1]):
        ending_inserts = []
        inserted_text = get_insert_text(sub_actions[-1])
        if len(inserted_text) > 1:
            for i in range(1, len(inserted_text)):
                ending_inserts.append(inserted_text[0:i])
    if is_insert(sub_actions[0]) and not is_insert(sub_actions[-1]):
        other_representation = compute_string_representation_of_actions(sub_actions[1:])
        for s in beginning_inserts:
            s_rep = compute_string_representation_of_actions(
                [create_insert_action(s)]
            )
            collection.append(s_rep + other_representation)
    elif is_insert(sub_actions[-1]) and not is_insert(sub_actions[0]):
        other_representation = compute_string_representation_of_actions(sub_actions[:-1])
        for s in ending_inserts:
            s_rep = compute_string_representation_of_actions(
                [create_insert_action(s)]
            )
            collection.append(other_representation + s_rep)
    elif is_insert(sub_actions[0]) and is_insert(sub_actions[-1]):
        other_representation = compute_string_representation_of_actions(sub_actions[1:-1])
        beginning_inserts.append(get_insert_text(sub_actions[0]))
        ending_inserts.append(get_insert_text(sub_actions[-1]))
        for i, b in enumerate(beginning_inserts):
            b_rep = compute_string_representation_of_actions(
                [create_insert_action(b)]
            )
            for j, e in enumerate(ending_inserts):
                if i != len(beginning_inserts) - 1 or j != len(ending_inserts) - 1:
                    e_rep = compute_string_representation_of_actions(
                        [create_insert_action(e)]
                    )
                    collection.append(b_rep + other_representation + e_rep)

def compute_action_subsequences_including_leading_and_trailing_inserts(
    actions: list[BasicAction]
):
    for i in range(len(actions)):
        for j in range(i, len(actions)):
            subsequences = []
            sub_actions = actions[i:j + 1]
            if compute_number_of_elements_in_range(i, j) < len(actions):
                subsequences.append(compute_string_representation_of_actions(sub_actions))
            if len(sub_actions) == 1 and is_insert(sub_actions[0]):
                _append_insert_subsequences(subsequences, sub_actions[0])
            elif len(sub_actions) > 1:
                _append_insert_subsequences_with_multiple_actions(subsequences, sub_actions)
            
            for subsequence in subsequences:
                yield subsequence

def filter_out_recommendations_redundant_smaller_commands(
    recommendations: list[PotentialCommandInformation]
) -> list[PotentialCommandInformation]:
    #For every command that is a shorter version of another command but is not used any more times: remove it
    action_sequences: dict[str, PotentialCommandInformation] = {}
    for command in recommendations:
        representation = compute_string_representation_of_actions(command.get_actions())
        action_sequences[representation] = command
    to_remove = set()
    for sequence in action_sequences:
        command = action_sequences[sequence]
        for sub_sequence in compute_action_subsequences_including_leading_and_trailing_inserts(command.get_actions()):
            if sub_sequence in action_sequences and \
                action_sequences[sub_sequence].get_number_of_times_used() == command.get_number_of_times_used():
                to_remove.add(sub_sequence)
    for sequence in to_remove:
        action_sequences.pop(sequence)
    result = [action_sequences[s] for s in action_sequences]
    return result

def compute_overlapping_and_nonoverlapping(non_overlapping_limit: int, total: list[PotentialCommandInformation]):
    non_overlapping = []
    num_commands_including_action = _compute_number_of_commands_including_action(total)
    overlapping = []
    for r in total:
        overlaps = False
        for a in r.get_actions():
            if num_commands_including_action[compute_string_representation_of_actions([a])] > 1:
                overlaps = True
                break
        if overlaps:
            overlapping.append(r)
        else:
            non_overlapping.append(r)
    return overlapping, sorted(non_overlapping, key=lambda r: r.get_number_of_words_saved(), reverse=True)[:non_overlapping_limit]

def filter_out_inferior_within_nonoverlapping_regions(recommendation_limit: int, recommendations: list[PotentialCommandInformation]):
    done = False
    #These overlap nothing
    recommendations, non_overlapping = compute_overlapping_and_nonoverlapping(recommendation_limit, recommendations)
    while not done:
        #Separate into groups that do not overlap with each other
        #Keep the best from each group
        previous_number = len(recommendations) + len(non_overlapping)
        groups: list[tuple[set, list]] = []
        for recommendation in recommendations:
            belongs_to_a_group = False
            action_representations = [
                compute_string_representation_of_actions([a])
                for a in recommendation.get_actions()
            ]
            for group in groups:
                group_set, group_list = group
                belongs_to_group = True
                for action in action_representations:
                    if action in group_set:
                        belongs_to_group = False
                        break
                if belongs_to_group:
                    group_list.append(recommendation)
                    for a in action_representations:
                        group_set.add(a)
                    belongs_to_a_group = True
                    break
            if not belongs_to_a_group:
                groups.append((set(action_representations), [recommendation]))
        group_lists = [group[1] for group in groups]
        recommendations = []
        for group in group_lists:
            if len(group) > recommendation_limit:
                new_group = sorted(group, key=lambda r: r.get_number_of_words_saved(), reverse=True)[:recommendation_limit]
                recommendations.extend(new_group)
            else:
                recommendations.extend(group)
        total_count = len(recommendations) + len(non_overlapping)
        done = total_count == previous_number or total_count == recommendation_limit
    recommendations.extend(non_overlapping)
    return recommendations


def filter_out_recommendations_using_safe_heuristics(recommendation_limit: int, recommendations: list[PotentialCommandInformation]):
    recommendations = filter_out_recommendations_redundant_smaller_commands(recommendations)
    if len(recommendations) > recommendation_limit:
        recommendations = filter_out_inferior_within_nonoverlapping_regions(recommendation_limit, recommendations)
    return recommendations

worker_recommendations = None
def initialize_worker(recommendations):
    global worker_recommendations
    worker_recommendations = recommendations
def _parallelly_compute_recommendation_based_on_greedy_local_max(best_recommendations, index_range, consumed, scoring_function, cpu_count: int, pool):
    work_per_worker = math.ceil((index_range[1] - index_range[0])/cpu_count)
    start: int = index_range[0]
    results = []
    for _ in range(cpu_count):
        ending = min(index_range[1], start + work_per_worker)
        worker_range = (start, ending)
        result = pool.apply_async(_sequentially_compute_best_recommendation_based_on_greedy_local_max, (None, best_recommendations, worker_range, consumed, scoring_function))
        results.append(result)
        start += work_per_worker

    best_score = -1
    best_index = None
    for result in results:
        index, score = result.get()
        if score > best_score:
            best_score = score
            best_index = index
    return best_index, best_score

def _sequentially_compute_best_recommendation_based_on_greedy_local_max(recommendations, best_recommendations, index_range, consumed, scoring_function):
    best_score = 0
    best_recommendation_index = -1
    for index in range(index_range[0], index_range[1]):
        if index not in consumed:
            recommendation = recommendations[index] if recommendations else worker_recommendations[index]
            best_recommendations.append(recommendation)
            score = scoring_function(best_recommendations + [recommendation])
            if score > best_score:
                best_score = score
                best_recommendation_index = index
            best_recommendations.pop()
    return best_recommendation_index, best_score

def _compute_best_recommendations_based_on_greedy_local_max_helper(recommendation_limit, recommendations, best_recommendations, index_range, consumed, scoring_function, *, parallelize: bool):
    num_remaining = recommendation_limit - len(best_recommendations)
    best_score = 0
    cpu_count = multiprocessing.cpu_count()
    should_parallelize = parallelize and cpu_count > 1 and (recommendation_limit - len(best_recommendations))*(index_range[1] - index_range[0])/cpu_count > len(recommendations)
    if should_parallelize:
        with multiprocessing.Pool(cpu_count, initializer=initialize_worker, initargs=(recommendations,)) as p:
            for _ in range(num_remaining):
                best_recommendation_index, best_score = _parallelly_compute_recommendation_based_on_greedy_local_max(best_recommendations, index_range, consumed, scoring_function, cpu_count, p)
                if best_score == 0:
                    break
                else:
                    best_recommendations.append(recommendations[best_recommendation_index])
                    consumed.add(best_recommendation_index)
    else:
        for _ in range(num_remaining):
            best_recommendation_index, best_score = _sequentially_compute_best_recommendation_based_on_greedy_local_max(recommendations, best_recommendations, index_range, consumed, scoring_function)
            if best_score == 0:
                break
            else:
                best_recommendations.append(recommendations[best_recommendation_index])
                consumed.add(best_recommendation_index)
    return best_recommendations, best_score, [i for i in consumed]

#TODO: Potentially Deal with recommendations for this function with a linked list class. Using a list may be faster because of cache optimization
#Try to optimize to not need repeatedly recomputing the action representations
def compute_best_recommendations_based_on_greedy_local_max(recommendation_limit, recommendations, scoring_function=compute_heuristic_recommendation_score, start=None, index_range=None, parallelize=False):
    if start is not None:
        if isinstance(start[0], int):
            best_recommendations = [recommendations[i] for i in start]
            consumed = set(start)
        else:
            raise ValueError("Must provide elements as indexes to use optional start argument with compute_best_recommendations_based_on_greedy_local_max")
    else:
        best_recommendations = []
        consumed = set()
    if not index_range:
        index_range = (0, len(recommendations))
    return _compute_best_recommendations_based_on_greedy_local_max_helper(recommendation_limit, recommendations, best_recommendations, index_range, consumed, scoring_function, parallelize=parallelize)

def compute_best_recommendations(recommendation_limit, recommendations, scoring_function=compute_heuristic_recommendation_score, is_verbose=False):
    if is_verbose: print(f"Narrowing it down from {len(recommendations)}.")
    if recommendation_limit == NO_NUMBER_OF_RECOMMENDATIONS_LIMIT:
        return recommendations
    if is_verbose: 
        print("Using safe heuristic preprocessing")
        current_time = time.time()
    recommendations = filter_out_recommendations_using_safe_heuristics(recommendation_limit, recommendations)
    if is_verbose:
        print(f"Safe heuristics took {time.time() - current_time} seconds")
        print(f"Narrowed it down to {len(recommendations)}.")
        print("Finding the best combination of recommendations")
        current_time = time.time()
    best_recommendations, greedy_score, _ = compute_best_recommendations_based_on_greedy_local_max(
        recommendation_limit,
        recommendations,
        scoring_function,
        parallelize=True,
    )
    if is_verbose:
        print(f'greedy took {time.time() - current_time} seconds')
        print('greedy_score', greedy_score)
        current_time = time.time()
    monte_carlo_recommendation, monte_carlo_score = perform_monte_carlo_tree_search(
        recommendations,
        recommendation_limit,
        scoring_function,
        round(len(recommendations)/recommendation_limit),
        greedy_function=compute_best_recommendations_based_on_greedy_local_max,
    )
    if is_verbose:
        print(f"Search took {time.time() - current_time} seconds")
    if monte_carlo_score > greedy_score:
        best_recommendations = monte_carlo_recommendation
    print('monte_carlo_score', monte_carlo_score)
    return best_recommendations