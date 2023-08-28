import math
import datetime
from pathlib import PurePath
from typing import List
import os

from action_records import BasicAction, read_file_record, TalonCapture, CommandChain
from text_separation import TextSeparationAnalyzer

RECOMMENDATION_OUTPUT_DIRECTORY = 'Recommendations'
DATA_DIRECTORY = 'Data'
EXPECTED_GRANDPARENT = 'talon'
INPUT_FILENAME = 'record.txt'
OUTPUT_FILENAME_PREFIX = 'recommendations '
OUTPUT_FILE_EXTENSION = '.txt'
COMMANDS_TO_IGNORE_FILENAME = 'commands_to_ignore.txt'

class PotentialCommandInformation:
    def __init__(self, actions):
        self.actions = actions
        self.number_of_times_used: int = 0
        self.total_number_of_words_dictated: int = 0
        self.number_of_actions: int = len(self.actions)
        self.count_repetitions_appropriately_for_number_of_actions()
        self.chain = None
        
    def count_repetitions_appropriately_for_number_of_actions(self):
        for action in self.actions: self.count_repetition_appropriately_for_a_number_of_actions(action)
    
    def count_repetition_appropriately_for_a_number_of_actions(self, action):
        argument = action.get_arguments()[0]
        if action.get_name() == 'repeat' and type(argument) == int:
            self.number_of_actions += argument - 1
    
    def get_number_of_actions(self):
        return len(self.actions)
    
    def get_average_words_dictated(self):
        return self.total_number_of_words_dictated/self.number_of_times_used
    
    def get_number_of_times_used(self):
        return self.number_of_times_used

    def get_actions(self):
        return self.actions
    
    def is_abstract(self):
        return False

    def process_usage(self, command_chain):
        if self.should_process_usage(command_chain.get_chain_number()):
            self.process_relevant_usage(command_chain)
    
    def should_process_usage(self, chain):
        return self.chain is None or chain > self.chain

    def process_relevant_usage(self, command_chain):
        self.number_of_times_used += 1
        self.chain = command_chain.get_chain_ending_index()
        words = command_chain.get_name().split(' ')
        self.total_number_of_words_dictated += len(words)

    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        return f'actions: {CommandInformationSet.compute_representation(self)}, number of times used: {self.number_of_times_used}, total number of words dictated: {self.total_number_of_words_dictated}'

class ActionSequenceSet:
    def __init__(self):
        self.set = set()
    
    def insert(self, actions):
        representation = compute_string_representation_of_actions(actions)
        self.set.add(representation)
    
    def contains(self, actions):
        return compute_string_representation_of_actions(actions) in self.set
    
    def contains_command_actions(self, command):
        return self.contains(command.get_actions())
    
    def get_size(self):
        return len(self.set)


class PotentialAbstractCommandInformation(PotentialCommandInformation):
    def __init__(self, actions):
        self.instantiation_set = ActionSequenceSet()
        super().__init__(actions)
    
    def process_usage(self, command_chain, instantiation):
        if self.should_process_usage(command_chain.get_chain_number()):
            self.instantiation_set.insert(instantiation.get_actions())
            self.process_relevant_usage(command_chain)
    
    def get_number_of_instantiations(self):
        return self.instantiation_set.get_size()
    
    def is_abstract(self):
        return True

def compute_repeat_simplified_command_chain(command_chain):
    new_actions = []
    last_non_repeat_action = None
    repeat_count: int = 0
    for action in command_chain.get_actions():
        if action == last_non_repeat_action:
            repeat_count += 1
        else:
            if repeat_count > 0:
                new_actions.append(BasicAction('repeat', [repeat_count]))
                repeat_count = 0
            new_actions.append(action)
            last_non_repeat_action = action
    if repeat_count > 0:
        new_actions.append(BasicAction('repeat', [repeat_count]))
    new_command = CommandChain(command_chain.get_name(), new_actions, command_chain.get_chain_number(), command_chain.get_size())
    return new_command

def compute_insert_simplified_command_chain(command_chain):
    new_actions = []
    current_insert_text = ''
    for action in command_chain.get_actions():
        if action.get_name() == 'insert':
            current_insert_text += action.get_arguments()[0]
        else:
            if current_insert_text:
                new_actions.append(BasicAction('insert', [current_insert_text]))
                current_insert_text = ''
            new_actions.append(action)
    if current_insert_text: new_actions.append(BasicAction('insert', [current_insert_text]))
    new_command = CommandChain(command_chain.get_name(), new_actions, command_chain.get_chain_number(), command_chain.get_size())
    return new_command

def compute_string_representation_of_actions(actions):
    representation = ''
    for action in actions:
        representation += action.to_json()
    return representation

def should_make_abstract_repeat_representation(command):
    actions = command.get_actions()
    if len(actions) <= 2:
        return False
    return any(action.get_name() == 'repeat' for action in actions)

def compute_command_chain_copy_with_new_name_and_actions(command_chain, new_name, new_actions):
    return CommandChain(new_name, new_actions, command_chain.get_chain_number(), command_chain.get_size())

def make_abstract_repeat_representation_for(command_chain):
    actions = command_chain.get_actions()
    instances = 0
    new_actions = []
    new_name = command_chain.get_name()
    for action in actions:
        if action.get_name() == 'repeat':
            instances += 1
            argument = TalonCapture('number_small', instances, ' - 1')
            repeat_action = BasicAction('repeat', [argument])
            new_actions.append(repeat_action)
            new_name += ' ' + argument.compute_command_component()
        else:
            new_actions.append(action)
    new_command = compute_command_chain_copy_with_new_name_and_actions(command_chain, new_name, new_actions)
    return new_command

def is_prose_inside_inserted_text_with_consistent_separator(prose: str, text: str) -> bool:
    text_separation_analyzer = TextSeparationAnalyzer(text)
    text_separation_analyzer.search_for_prose_in_separated_part(prose)
    text_separation_analyzer.is_prose_separator_consistent()
    return text_separation_analyzer.has_found_prose()

class InvalidCaseException(Exception): pass

def compute_case_string(text: str) -> str:
    if text.islower(): return 'lower'
    elif text.isupper(): return 'upper'
    elif text[0].isupper() and text[1:].islower(): return 'capitalized'
    else: raise InvalidCaseException()

def has_valid_case(analyzer: TextSeparationAnalyzer) -> bool:
    try:
        for word in analyzer.compute_prose_portion_of_text(): compute_case_string(word)
        return True
    except:
        return False

def compute_simplified_case_strings_list(case_strings: List) -> List:
    simplified_case_strings = []
    new_case_found = False
    final_case = case_strings[-1]
    simplified_case_strings.append(final_case)
    for index in range(len(case_strings) - 2, -1, -1):
        case = case_strings[index]
        if case != final_case or new_case_found:
            simplified_case_strings.append(case)
            new_case_found = True
    simplified_case_strings.reverse()
    return simplified_case_strings

def compute_case_string_for_prose(analyzer: TextSeparationAnalyzer):
    prose = analyzer.compute_prose_portion_of_text()
    case_strings = [compute_case_string(prose_word) for prose_word in prose]
    simplified_case_strings = compute_simplified_case_strings_list(case_strings)
    case_string = ' '.join(simplified_case_strings)
    return case_string

class ProseMatch:
    def __init__(self, analyzer: TextSeparationAnalyzer, name: str):
        self.analyzer = analyzer
        self.name = name

def make_abstract_representation_for_prose_command(command_chain, match: ProseMatch, insert_to_modify_index: int):
    analyzer = match.analyzer
    actions = command_chain.get_actions()
    new_actions = actions[:insert_to_modify_index]
    text_before = analyzer.compute_text_before_prose()
    if text_before: new_actions.append(BasicAction('insert', [text_before]))
    prose_argument = TalonCapture('user.text', 1)
    new_actions.append(BasicAction('user.fire_chicken_auto_generated_command_action_insert_formatted_text', [prose_argument, compute_case_string_for_prose(analyzer), analyzer.get_first_prose_separator()]))
    text_after = analyzer.compute_text_after_prose()
    if text_after: new_actions.append(BasicAction('insert', [text_after]))
    if insert_to_modify_index + 1 < len(actions): new_actions.extend(actions[insert_to_modify_index + 1:])
    new_command = compute_command_chain_copy_with_new_name_and_actions(command_chain, match.name, new_actions)
    return new_command

class InsertAction:
    def __init__(self, text: str, index: int):
        self.text = text
        self.index = index

def obtain_inserts_from_command_chain(command_chain):
    return [InsertAction(action.get_arguments()[0], index) for index, action in enumerate(command_chain.get_actions()) if action.get_name() == 'insert']

def generate_prose_command_command_name(words, starting_index: int, prose_size: int) -> str:
    command_name_parts = words[:starting_index]
    command_name_parts.append('<user.text>')
    command_name_parts.extend(words[starting_index + prose_size:])
    command_name = ' '.join(command_name_parts)
    return command_name

def generate_prose_from_words(words, starting_index: int, prose_size: int) -> str:
    prose = ' '.join(words[starting_index:starting_index + prose_size])
    return prose

def compute_text_analyzer_for_prose_and_insert(prose: str, insert: InsertAction):
    analyzer = TextSeparationAnalyzer(insert.text)
    analyzer.search_for_prose_in_separated_part(prose)
    return analyzer

class ValidProseNotFoundException(Exception):
    pass

def find_prose_match_for_command_given_insert_at_interval(words, insert, starting_index, prose_size):
    prose = generate_prose_from_words(words, starting_index, prose_size)
    analyzer = compute_text_analyzer_for_prose_and_insert(prose, insert)
    if analyzer.is_prose_separator_consistent() and analyzer.has_found_prose() and has_valid_case(analyzer):
        command_name = generate_prose_command_command_name(words, starting_index, prose_size)
        return ProseMatch(analyzer, command_name)
    raise ValidProseNotFoundException()

def find_prose_matches_for_command_given_insert_at_starting_index(words, insert, starting_index, max_prose_size_to_consider):
    matches = []
    maximum_size = min(max_prose_size_to_consider, len(words) - starting_index + 1)
    for prose_size in range(1, maximum_size):
        try: matches.append(find_prose_match_for_command_given_insert_at_interval(words, insert, starting_index, prose_size))
        except ValidProseNotFoundException: break
    return matches

def find_prose_matches_for_command_given_insert(command_chain, insert, max_prose_size_to_consider):
    dictation: str = command_chain.get_name()
    words = dictation.split(' ')
    matches = []
    for starting_index in range(len(words)): matches.extend(find_prose_matches_for_command_given_insert_at_starting_index(words, insert, starting_index, max_prose_size_to_consider))
    return matches

def is_acceptable_abstract_representation(representation):
    return len(representation.get_actions()) > 1

def make_abstract_prose_representations_for_command_given_insert(command_chain, insert, max_prose_size_to_consider):
    abstract_representations = []
    prose_matches = find_prose_matches_for_command_given_insert(command_chain, insert, max_prose_size_to_consider)
    for match in prose_matches:
        abstract_representation = make_abstract_representation_for_prose_command(command_chain, match, insert.index)
        if is_acceptable_abstract_representation(abstract_representation): abstract_representations.append(abstract_representation)
    return abstract_representations

def make_abstract_prose_representations_for_command_given_inserts(command_chain, inserts, max_prose_size_to_consider):
    abstract_representations = []
    for insert in inserts: 
        representations_given_insert = make_abstract_prose_representations_for_command_given_insert(command_chain, insert, max_prose_size_to_consider)
        abstract_representations.extend(representations_given_insert)
    return abstract_representations

def make_abstract_prose_representations_for_command(command_chain, max_prose_size_to_consider = 10):
    inserts = obtain_inserts_from_command_chain(command_chain)
    if len(inserts) == 0: return []
    else: return make_abstract_prose_representations_for_command_given_inserts(command_chain, inserts, max_prose_size_to_consider)

def basic_command_filter(command: PotentialCommandInformation):
    return command.get_average_words_dictated() >= 2 and command.get_number_of_times_used() > 1 and \
            (not command.is_abstract() or command.get_number_of_instantiations() > 2 and command.get_average_words_dictated() > 2) and \
            (command.get_number_of_actions()/command.get_average_words_dictated() < 2 or \
            command.get_number_of_actions()*math.sqrt(command.get_number_of_times_used()) > command.get_average_words_dictated())

class CommandInformationSet:
    def __init__(self):
        self.commands = {}

    def insert_command(self, command, representation):
        self.commands[representation] = command
    
    def process_abstract_command_usage(self, command_chain, abstract_command_chain):
        representation = CommandInformationSet.compute_representation(abstract_command_chain)
        if representation not in self.commands:
            self.insert_command(PotentialAbstractCommandInformation(abstract_command_chain.get_actions()), representation)
        self.commands[representation].process_usage(abstract_command_chain, command_chain)

    def create_abstract_commands(self, command_chain):
        commands = []
        if should_make_abstract_repeat_representation(command_chain):
            abstract_repeat_representation = make_abstract_repeat_representation_for(command_chain)
            commands.append(abstract_repeat_representation)
        abstract_prose_commands = make_abstract_prose_representations_for_command(command_chain)
        commands.extend(abstract_prose_commands)
        return commands
    
    def handle_needed_abstract_commands(self, command_chain):
        abstract_commands = self.create_abstract_commands(command_chain)
        for abstract_command in abstract_commands: self.process_abstract_command_usage(command_chain, abstract_command)

    def process_command_usage(self, command_chain):
        representation = CommandInformationSet.compute_representation(command_chain)
        if representation not in self.commands:
            self.insert_command(PotentialCommandInformation(command_chain.get_actions()), representation)
        self.commands[representation].process_usage(command_chain)
        self.handle_needed_abstract_commands(command_chain)
    
    def process_partial_chain_usage(self, record, command_chain):
        command_chain.append_command(record[command_chain.get_next_chain_index()])
        simplified_command_chain = compute_insert_simplified_command_chain(command_chain)
        simplified_command_chain = compute_repeat_simplified_command_chain(simplified_command_chain)
        self.process_command_usage(simplified_command_chain)

    def process_chain_usage(self, record, chain, max_command_chain_considered, verbose = False):
        command_chain: CommandChain = CommandChain(None, [], chain)
        chain_target = min(len(record), chain + max_command_chain_considered)
        for chain_ending_index in range(chain, chain_target): self.process_partial_chain_usage(record, command_chain)
        if verbose: print('chain', chain + 1, 'out of', len(record) - 1, 'target: ', chain_target - 1)

    @staticmethod
    def compute_representation(command):
        actions = command.get_actions()
        representation = compute_string_representation_of_actions(actions)
        return representation
    
    def get_commands_meeting_condition(self, condition):
        commands_to_output = [command for command in self.commands.values() if condition(command)]
        return commands_to_output
    
    def contains_command_with_representation(self, representation: str):
        return representation in self.commands
    
    def contains_command(self, command):
        representation = CommandInformationSet.compute_representation(command)
        return self.contains_command_with_representation(representation)

    def get_size(self):
        return len(self.commands)

    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        representation: str = ''
        for command in self.commands.values():
            representation += str(command) + '\n'
        return representation

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
    filtered_record = [command for command in record if not commands_to_ignore.contains_command_actions(command)]
    return filtered_record

def obtain_file_record(data_directory, input_path):
    record = read_file_record(input_path)
    filtered_record = compute_record_without_stuff_to_ignore(data_directory, record)
    return filtered_record

def write_command_to_file(file, command):
    file.write(f'#Number of times used: {command.get_number_of_times_used()}\n')
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
    for chain in range(len(record)): command_set.process_chain_usage(record, chain, max_command_chain_considered, verbose = verbose)
    return command_set

def compute_recommendations_from_record(record, max_command_chain_considered = 100, *, verbose = False):
    command_set = create_command_information_set_from_record(record, max_command_chain_considered, verbose = verbose)
    recommended_commands = command_set.get_commands_meeting_condition(basic_command_filter)
    sorted_recommended_commands = sorted(recommended_commands, key = lambda command: command.get_number_of_times_used(), reverse = True)
    return sorted_recommended_commands

def generate_recommendations(recommendation_directory, data_directory, input_path):
    record = obtain_file_record(data_directory, input_path)
    print('finished reading record')
    recommendations = compute_recommendations_from_record(record, 20, verbose = True)
    print('outputting recommendations')
    output_recommendations(recommendations, recommendation_directory)
    print('completed')

def guarantee_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_file_input_path_from_user() -> str:
    path = ''
    needs_valid_path = True
    while needs_valid_path:
        path = input('Input the path to the command record:')
        if os.path.exists(path) and os.path.splitext(path)[1] == '.txt':
            needs_valid_path = False
        else:
            print('Please input a valid path!')
    return path

def main():
    program_directory = compute_main_program_directory()
    recommendation_output_directory = compute_recommendation_output_directory(program_directory)
    guarantee_directory_exists(recommendation_output_directory)
    data_directory = compute_data_directory(program_directory)
    guarantee_directory_exists(data_directory)
    input_path = get_file_input_path_from_user()
    generate_recommendations(recommendation_output_directory, data_directory, input_path)

if __name__ == '__main__':
    main()
