from monte_carlo_tree_search import *
from basic_action_record_analysis import *
from recommendation_scoring import *
import json
import os
import time

if __name__ == '__main__':
    program_directory = compute_main_program_directory()
    data_directory = compute_data_directory(program_directory)
    record_names = ("/Users/sam/projects/ArtificialTalonCommandHistoryGenerator/recommendation", "/Users/sam/projects/ArtificialTalonCommandHistoryGenerator/tasks")
    chain_sizes = (5, 20)
    c_values = (1/1000000, 1/100000, 1/10000, 1.0, math.sqrt(2), 2)
    cores_to_use = (1, 10)
    numbers_of_recommendations = (10, 30)
    c_scores = {}
    results = {}
    number_of_trials = 5
    total = len(record_names)*len(chain_sizes)*len(numbers_of_recommendations)*len(cores_to_use)*len(c_values)*number_of_trials
    print(f"Running {total} iterations.")
    iteration = 1
    start = time.time()
    for record_name in record_names:
        record = obtain_file_record(data_directory, record_name)
        for chain_size in chain_sizes:
            unfiltered_recommendations = compute_recommendations_from_record(record, chain_size, verbose = False)
            for number_of_recommendations in numbers_of_recommendations:
                recommendations = filter_out_recommendations_using_safe_heuristics(number_of_recommendations, unfiltered_recommendations)
                for number_of_cores in cores_to_use:
                    for c in c_values:
                        for trial in range(number_of_trials):
                            _, score = perform_monte_carlo_tree_search(recommendations, number_of_recommendations, compute_heuristic_recommendation_score, round(len(recommendations)/number_of_recommendations),greedy_function=compute_best_recommendations_based_on_greedy_local_max, cores_override=number_of_cores, c=c)
                            result_representation = f"{record_name} cs{chain_size} nr {number_of_recommendations} cores: {number_of_cores} c{c}"
                            if result_representation in c_scores:
                                c_scores[result_representation].append(score)
                            else:
                                c_scores[result_representation] = [score]
                            results[result_representation + f" trial{trial+1}"] = (record_name, chain_size, number_of_recommendations, number_of_cores, c)
                            print('c_scores', c_scores)
                            print(f"Progress: {iteration}/{total}")
                            estimated_time_remaining = ((time.time() - start)/(iteration))*(total - iteration)
                            estimated_remaining_hours = estimated_time_remaining/(60**2)
                            print(f"Estimated remaining time {estimated_remaining_hours} hours")
                            iteration += 1
    for c in c_scores:
        c_scores[c] = sum(c_scores[c])/len(c_scores[c])
    print('c_scores', c_scores)
    with open(os.path.join(data_directory, "traininglog"), "a") as f:
        f.write(json.dumps((c_scores, results)) + "\n")
    
                        



            


