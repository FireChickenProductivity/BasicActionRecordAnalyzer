import datetime
from pathlib import PurePath
import os
import time
import multiprocessing

from action_records import BasicAction, read_file_record
from input_parsing import InputParameters, get_input_parameters_from_user, NO_NUMBER_OF_RECOMMENDATIONS_LIMIT
from recommendation_generation import *
from recommendation_scoring import compute_best_recommendations

RECOMMENDATION_OUTPUT_DIRECTORY = 'Recommendations'
DATA_DIRECTORY = 'Data'
EXPECTED_GRANDPARENT = 'talon'
INPUT_FILENAME = 'record.txt'
OUTPUT_FILENAME_PREFIX = 'recommendations '
OUTPUT_FILE_EXTENSION = '.txt'
COMMANDS_TO_IGNORE_FILENAME = 'commands_to_ignore.txt'

class ProgramDirectoryInvalidException(Exception):
    pass

def compute_main_program_directory():
    program_path = PurePath(__file__)
    grandparent = program_path.parent.parent
    return grandparent

def compute_recommendation_output_directory(main_program_directory):
    return os.path.join(main_program_directory, RECOMMENDATION_OUTPUT_DIRECTORY)

def compute_data_directory(main_program_directory):
    return os.path.join(main_program_directory, DATA_DIRECTORY)

def create_file_if_nonexistent(path):
    if not os.path.exists(path):
        with open(path, 'w') as file:
            pass

def create_file_at_directory_if_nonexistent(directory, file):
    path = os.path.join(directory, file)
    create_file_if_nonexistent(path)

def read_commands_to_ignore(directory):
    create_file_at_directory_if_nonexistent(directory, COMMANDS_TO_IGNORE_FILENAME)
    path = os.path.join(directory, COMMANDS_TO_IGNORE_FILENAME)
    commands = ActionSequenceSet()
    current_command_actions = []
    with open(path, 'r') as file:
        line = file.readline()
        while line:
            line_without_trailing_newline = line.strip()
            if line_without_trailing_newline:
                current_command_actions.append(BasicAction.from_json(line_without_trailing_newline))
            else:
                commands.insert(current_command_actions)
                current_command_actions = []
            line = file.readline()
        if current_command_actions:
            commands.insert(current_command_actions)
    return commands

def compute_record_without_stuff_to_ignore(directory, record):
    commands_to_ignore = read_commands_to_ignore(directory)
    filtered_record = [command for command in record if not command.is_command_record() or not commands_to_ignore.contains_command_actions(command)]
    return filtered_record

def obtain_file_record(data_directory, input_path):
    record = read_file_record(input_path)
    filtered_record = compute_record_without_stuff_to_ignore(data_directory, record)
    return filtered_record

def write_command_to_file(file, command: PotentialCommandInformation):
    file.write(f'#Number of times used: {command.get_number_of_times_used()}\n')
    file.write(f'#Number of words saved: {command.get_number_of_words_saved()}\n')
    if command.is_abstract(): file.write(f'#Number of instantiations of abstract command: {command.get_number_of_instantiations()}\n')
    for action in command.get_actions(): file.write('\t' + action.compute_talon_script() + '\n')
    file.write('\n\n')

def generate_output_filename(output_directory):
    timestamp = datetime.datetime.now()
    formatted_timestamp = str(timestamp).replace('.', ',').replace(':', '-')
    output_path = os.path.join(output_directory, OUTPUT_FILENAME_PREFIX + formatted_timestamp + OUTPUT_FILE_EXTENSION)
    return output_path

def output_recommendations(recommended_commands, output_directory):
    output_path = generate_output_filename(output_directory)
    with open(output_path, 'w') as file:
        for command in recommended_commands: write_command_to_file(file, command)

def create_command_information_set_from_record(record, max_command_chain_considered, *, verbose = False):
    command_set: CommandInformationSet = CommandInformationSet()    
    num_cpus = multiprocessing.cpu_count()
    if num_cpus > 1:
        maximum_parallelism = min(num_cpus, max_command_chain_considered)
        with multiprocessing.Pool(
                maximum_parallelism,
                initializer=initialize_worker_with_record,
                initargs=(record,)
            ) as pool:
            for chain in range(len(record)): command_set.process_chain_usage(record, chain, max_command_chain_considered, verbose = verbose, pool = pool)
    else:
        for chain in range(len(record)): command_set.process_chain_usage(record, chain, max_command_chain_considered, verbose = verbose)
    return command_set

def compute_recommendations_from_record(record, max_command_chain_considered = 100, *, verbose = False, filter = basic_command_filter):
    command_set = create_command_information_set_from_record(record, max_command_chain_considered, verbose = verbose)
    recommended_commands = command_set.get_commands_meeting_condition(filter)
    sorted_recommended_commands = sorted(recommended_commands, key = lambda command: command.get_number_of_times_used(), reverse = True)
    return sorted_recommended_commands

        
def generate_recommendations(recommendation_directory, data_directory, parameters: InputParameters):
    record = obtain_file_record(data_directory, parameters.input_path)
    print('finished reading record')
    recommendations_start_time = time.time()
    recommendations = compute_recommendations_from_record(record, parameters.max_chain_length, verbose = True)
    print(f"created {len(recommendations)} recommendations in {(time.time() - recommendations_start_time)} seconds")
    if parameters.max_number_of_recommendations != NO_NUMBER_OF_RECOMMENDATIONS_LIMIT:
        print('identifying the best', parameters.max_number_of_recommendations, 'recommendations')
        recommendations = compute_best_recommendations(
            parameters.max_number_of_recommendations,
            recommendations,
            is_verbose=True
        )
    print('outputting recommendations')
    output_recommendations(recommendations, recommendation_directory)
    print('completed')
#/Users/sam/projects/ArtificialTalonCommandHistoryGenerator/out
def guarantee_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    program_directory = compute_main_program_directory()
    recommendation_output_directory = compute_recommendation_output_directory(program_directory)
    guarantee_directory_exists(recommendation_output_directory)
    data_directory = compute_data_directory(program_directory)
    guarantee_directory_exists(data_directory)
    parameters = get_input_parameters_from_user()
    generate_recommendations(recommendation_output_directory, data_directory, parameters)

if __name__ == '__main__':
    main()
