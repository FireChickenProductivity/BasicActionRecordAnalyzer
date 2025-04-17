from monte_carlo_tree_search import *
from basic_action_record_analysis import *
from recommendation_scoring import *
import json
import os

if __name__ == '__main__':
    program_directory = compute_main_program_directory()
    data_directory = compute_data_directory(program_directory)
    record_names = ("/Users/sam/projects/ArtificialTalonCommandHistoryGenerator/out", "Users/sam/projects/ArtificialTalonCommandHistoryGenerator/recommendation")
    chain_sizes = (5, 10, 20)
    c_values = (1/10000, 1/1000, 1/100, 0.5, 0.1, 1, math.sqrt(2), 2, 3)
    cores_to_use = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    numbers_of_recommendations = (10, 20, 30)
    c_scores = {}
    results = {}
    for record_name in record_names:
        record = obtain_file_record(data_directory, record_name)
        for chain_size in chain_sizes:
            recommendations = compute_recommendations_from_record(record, chain_size, verbose = False)
            for number_of_recommendations in numbers_of_recommendations:
                recommendations = filter_out_recommendations_using_safe_heuristics(number_of_recommendations, recommendations)
                for number_of_cores in cores_to_use:
                    for c in c_values:
                        score = perform_monte_carlo_tree_search(recommendations, number_of_recommendations, compute_heuristic_recommendation_score, round(len(recommendations)/number_of_recommendations),greedy_function=compute_best_recommendations_based_on_greedy_local_max,)
                        if c in c_scores:
                            c_scores[c].append(score)
                        else:
                            c_scores[c] = [score]
                        results[f"{record_name} cs{chain_size} nr {number_of_recommendations} cores: {number_of_cores} c{c}"] = (record_name, chain_size, number_of_recommendations, number_of_cores, c)
                        print('c_scores', c_scores)
    for c in c_scores:
        c_scores[c] = sum(c_scores[c])/len(c_scores[c])
    print('number_of_recommendations', number_of_recommendations)  
    with open(os.path.join(data_directory, "traininglog")) as f:
        f.write(json.dumps((c_scores, results)))
    
                        



            


