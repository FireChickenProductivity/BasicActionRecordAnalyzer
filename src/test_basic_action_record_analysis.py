import unittest
from action_records import *
from basic_action_record_analysis import *
from text_separation import *

class TestPotentialCommandInformation(unittest.TestCase):
    def test_potential_command_information_for_press_a_has_one_action(self):
        potential_command_information = generate_potential_command_information_on_press_a()
        self.assertEqual(potential_command_information.get_number_of_actions(), 1)
    
    def test_potential_command_information_for_press_a_with_single_word_has_average_number_of_words_dictated_one(self):
        self._assert_potential_command_information_for_press_a_with_words_dictated_has_specified_number_of_average_words_dictated('air', 1)
    
    def test_potential_command_information_for_press_a_with_two_words_dictated_has_average_number_of_words_dictated_two(self):
        self._assert_potential_command_information_for_press_a_with_words_dictated_has_specified_number_of_average_words_dictated('press air', 2)
    
    def test_potential_command_information_for_press_a_with_sequence_dictated_has_correct_average_number_of_words_dictated(self):
        self._assert_potential_command_information_for_press_a_with_words_dictated_has_specified_number_of_average_words_dictated(['air', 'chicken', 'this is a test'], 2)

    def test_potential_command_information_for_press_a_reports_single_usage_with_single_usage(self):
        potential_command_information = generate_potential_command_information_on_press_a()
        potential_command_information.process_usage(generate_named_press_a_command_chain('this is a test'))
        self.assertEqual(potential_command_information.get_number_of_times_used(), 1)
    
    def test_potential_command_information_for_press_a_reports_two_usages_with_two_usages(self):
        potential_command_information = generate_potential_command_information_on_press_a()
        potential_command_information.process_usage(generate_named_press_a_command_chain('this is a test', 0))
        potential_command_information.process_usage(generate_named_press_a_command_chain('chicken', 1))
        self.assertEqual(potential_command_information.get_number_of_times_used(), 2)

    def test_potential_command_information_with_two_actions_has_correct_number_of_actions(self):
        self._assert_potential_command_information_with_key_actions_has_correct_number_of_actions(['a', 'ctrl-c'])

    def _assert_potential_command_information_for_press_a_with_words_dictated_has_specified_number_of_average_words_dictated(self, words, number):
        potential_command_information = generate_potential_command_information_on_press_a()
        if isinstance(words, str):
            potential_command_information.process_usage(generate_named_press_a_command_chain(words))
        else:
            for index, utterance in enumerate(words):
                potential_command_information.process_usage(generate_named_press_a_command_chain(utterance, index))
        self.assertEqual(potential_command_information.get_average_words_dictated(), number)
    
    def _assert_potential_command_information_with_key_actions_has_correct_number_of_actions(self, keystrokes):
        potential_command_information = PotentialCommandInformation(generate_multiple_key_pressing_actions(keystrokes))
        self.assertEqual(potential_command_information.get_number_of_actions(), len(keystrokes))

class TestCommandSet(unittest.TestCase):
    def test_command_set_with_single_command_used_once_gives_correct_information(self):
        command_set = CommandInformationSet()
        command = generate_copy_all_command_chain(0, 0)
        command_set.process_command_usage(command)
        potential_command_information = command_set.get_commands_meeting_condition(return_true)
        self.assertEqual(len(potential_command_information), 1)
        command_information: PotentialCommandInformation = potential_command_information[0]
        self.assertEqual(command_information.get_average_words_dictated(), 2)
        self.assertEqual(command_information.get_number_of_actions(), 2)
        self.assertEqual(command_information.get_number_of_times_used(), 1)
    
    def test_command_set_handles_multiple_commands_with_multiple_uses(self):
        command_set = CommandInformationSet()
        press_a = generate_press_a_command()
        copy_all = generate_copy_all_command()

        command_set.process_command_usage(generate_press_a_command_chain(0, 1))
        command_set.process_command_usage(generate_copy_all_command_chain(1, 1))
        command_set.process_command_usage(generate_copy_all_command_chain(2, 1))
        command_set.process_command_usage(generate_press_a_command_chain(3, 1))
        command_set.process_command_usage(generate_press_a_command_chain(4, 1))

        expected_press_a_information = generate_potential_command_information_with_uses(generate_press_a_action_list(), [press_a.get_name()]*3)
        expected_copy_all_information = generate_potential_command_information_with_uses(generate_copy_all_action_list(), [copy_all.get_name()]*2)
        press_a_information = get_command_set_information_matching_actions(command_set, press_a.get_actions())
        copy_all_information = get_command_set_information_matching_actions(command_set, copy_all.get_actions())
        
        self.assertTrue(potential_command_informations_match(press_a_information, expected_press_a_information))
        self.assertTrue(potential_command_informations_match(copy_all_information, expected_copy_all_information))

class TestCommandSimplification(unittest.TestCase):
    def test_repeat_simplify_command_does_nothing_to_press_a(self):
        command = generate_press_a_command_chain()
        expected_command = generate_press_a_command_chain()
        simplified_command = compute_repeat_simplified_command_chain(command)
        self.assertEqual(simplified_command.get_actions(), expected_command.get_actions())
    
    def test_repeat_simplify_command_handles_multiple_repetitions(self):
        command = CommandChain('test', generate_multiple_key_pressing_actions(['b', 'b', 'c', 'd', 'd', 'd', 'a', 'l', 'l']))
        expected_actions = [generate_key_press_action('b'), BasicAction('repeat', [1]), generate_key_press_action('c'), generate_key_press_action('d'), BasicAction('repeat', [2]), 
        generate_key_press_action('a'), generate_key_press_action('l'), BasicAction('repeat', [1])]
        expected_command = CommandChain('test', expected_actions)

        simplified_command = compute_repeat_simplified_command_chain(command)

        self.assertEqual(simplified_command.get_actions(), expected_command.get_actions())
    
    def test_insert_simplification_does_nothing_to_press_a(self):
        command = generate_press_a_command_chain()
        expected_command = generate_press_a_command_chain()
        self.assert_command_insert_simplifies_correctly(command, expected_command)
    
    def test_insert_simplification_handles_multiple_inserts(self):
        command = CommandChain('name', [generate_insert_action('this'), generate_insert_action('is'), generate_key_press_action('a'), generate_insert_action('a'), generate_insert_action('test')], 0, 1)
        expected_command = CommandChain('name', [generate_insert_action('thisis'), generate_key_press_action('a'), generate_insert_action('atest')], 0, 1)
        self.assert_command_insert_simplifies_correctly(command, expected_command)
    
    def test_insert_simplification_handles_single_insert(self):
        command = CommandChain('name', [generate_insert_action('this'), generate_key_press_action('a')], 0, 1)
        expected_command = CommandChain('name', [generate_insert_action('this'), generate_key_press_action('a')], 0, 1)
        self.assert_command_insert_simplifies_correctly(command, expected_command)

    def assert_command_insert_simplifies_correctly(self, original, expected):
        simplified_command = compute_insert_simplified_command_chain(original)
        self.assert_command_chains_match(simplified_command, expected)
    
    def assert_command_chains_match(self, actual, expected):
        assert_command_chains_match(self, actual, expected)
    
class TestGeneratingCommandSetFromRecord(unittest.TestCase):
    def test_can_handle_simple_record(self):
        record = generate_simple_command_record()
        command_set = create_command_information_set_from_record(record, 100)

        expected_command_information = [generate_rain_potential_command_information(), generate_copy_all_potential_command_information(), generate_air_potential_command_information(), 
                                        generate_rain_copy_all_potential_command_information(), generate_rain_copy_all_air_potential_command_information(), 
                                        generate_copy_all_air_potential_command_information()]

        self._assert_command_set_matches_expected_potential_command_information(command_set, expected_command_information)
    
    def test_can_handle_record_start(self):
        record = generate_simple_command_record()
        record.insert(-1, RecordingStart())
        command_set = create_command_information_set_from_record(record, 100)
        
        expected_command_information = [generate_rain_potential_command_information(), generate_copy_all_potential_command_information(),
                                        generate_air_potential_command_information(), generate_rain_copy_all_potential_command_information()]

        self._assert_command_set_matches_expected_potential_command_information(command_set, expected_command_information)

    def test_can_handle_long_pause_before_command(self):
        record = generate_command_record_with_many_seconds_before_middle_command()
        command_set = create_command_information_set_from_record(record, 100)

        expected_command_information = [generate_rain_potential_command_information(), generate_copy_all_potential_command_information(), generate_air_potential_command_information(), 
                                        generate_copy_all_air_potential_command_information()]

        self._assert_command_set_matches_expected_potential_command_information(command_set, expected_command_information)    
    
    def _assert_command_set_matches_expected_potential_command_information(self, command_set, expected_command_information):
        self.assertTrue(command_set_matches_expected_potential_command_information(command_set, expected_command_information))

class TestFindingProseInText(unittest.TestCase):
    def test_can_handle_identical_text(self):
        text = 'a'
        self.assertTrue(is_prose_inside_inserted_text_with_consistent_separator(text, text))

    def test_false_given_with_empty_string_target(self):
        self.assertFalse(is_prose_inside_inserted_text_with_consistent_separator('testing', ''))
    
    def test_can_handle_sub_string_match(self):
        target = 'this is a test'
        prose = 'is'
        self.assertTrue(is_prose_inside_inserted_text_with_consistent_separator(prose, target))
    
    def test_can_handle_multiple_words(self):
        target = 'this is a test'
        prose = 'this is'
        self.assertTrue(is_prose_inside_inserted_text_with_consistent_separator(prose, target))
    
    def test_can_handle_multiple_words_with_different_separators(self):
        target = 'this-is_____a test'
        prose = 'this is a'
        self.assertTrue(is_prose_inside_inserted_text_with_consistent_separator(prose, target))
    
    def test_can_handle_multiple_cases(self):
        target = 'ChickenEATSgrainstonight'
        prose = 'chicken eats grains'
        self.assertTrue(is_prose_inside_inserted_text_with_consistent_separator(prose, target))
    
    def test_fails_with_first_word_off(self):
        target = 'this is a test'
        prose = 'ths is'
        self.assertFalse(is_prose_inside_inserted_text_with_consistent_separator(prose, target))
    
    def test_fails_with_last_word_off(self):
        target = 'this is a test'
        prose = 'this s'
        self.assertFalse(is_prose_inside_inserted_text_with_consistent_separator(prose, target))

    def test_consistent_separator_without_separators(self):
        target = 'thisisatest'
        self.assertTrue(self.is_consistent_separator(target))
    
    def test_handles_single_character_separator(self):
        target = 'this_is_a_test'
        self.assertTrue(self.is_consistent_separator(target))
    
    def test_inconsistent_separator_with_multiple_separators(self):
        target = 'this_is__a_test'
        self.assertFalse(self.is_consistent_separator(target))
    
    def test_consistent_separator_with_multiple_characters(self):
        target = 'this!!!!!is!!!!!a!!!!!test'
        self.assertTrue(self.is_consistent_separator(target))
    
    def test_no_prefix_without_prefix(self):
        target = 'this_is_a_test'
        expected_prefix = ''
        self.assert_prefix_matches(target, expected_prefix)
    
    def test_handles_simple_prefix(self):
        target = '!this_is_a_test'
        expected_prefix = '!'
        self.assert_prefix_matches(target, expected_prefix)
    
    def test_can_find_one_word_prose_at_beginning(self):
        self.assert_indices_match('test', 'testing this here', 0, 0, 4)
    
    def test_can_find_multiple_words_at_beginning(self):
        self.assert_indices_match('this is a test', 'this_is_a_test_', 0, 0, 4)
    
    def test_can_find_one_word_in_middle(self):
        self.assert_indices_match('test', 'this_is_a_realtest_right_here', 3, 4, 8)
    
    def test_can_find_multiple_words_in_middle(self):
        self.assert_indices_match('this is a testr', 'yes_forrealthis_is_a_testrighthere_', 1, 7, 5)
    
    def test_can_find_one_word_at_ending(self):
        self.assert_indices_match('testing', 'this_is_actuallytestingstuff', 2, 8, 15)
    
    def test_can_find_multiple_words_at_ending(self):
        self.assert_indices_match('this is a test', 'once_again_this_is_a_test', 2, 0, 4)
    
    def test_can_find_multiple_words_at_middle_of_ending(self):
        self.assert_indices_match('this is a test', 'once_againthis_is_a_testing', 1, 5, 4)
    
    def test_can_find_empty_text_before_prose_with_one_word(self):
        original_text: str = 'test'
        prose: str = 'test'
        expected: str = ''
        self.assert_text_before_prose_matches(original_text, prose, expected)
    
    def test_can_find_text_before_prose_with_one_word(self):
        original_text: str = 'test'
        prose: str = 'st'
        expected: str = 'te'
        self.assert_text_before_prose_matches(original_text, prose, expected)
    
    def test_can_find_text_before_prose_with_multiple_words(self):
        original_text: str = '_This is_a!test today'
        prose: str = 'a test'
        expected = '_This is_'
        self.assert_text_before_prose_matches(original_text, prose, expected)
    
    def test_can_find_empty_text_after_prose_with_one_word(self):
        original_text: str = 'test'
        prose: str = 'test'
        expected: str = ''
        self.assert_text_after_prose_matches(original_text, prose, expected)
    
    def test_can_find_text_after_prose_with_one_word(self):
        original_text: str = 'test'
        prose: str = 'te'
        expected: str = 'st'
        self.assert_text_after_prose_matches(original_text, prose, expected)
    
    def test_can_find_text_after_prose_with_multiple_words(self):
        original_text: str = '_This is_a!test today'
        prose: str = 'is a'
        expected = '!test today'
        self.assert_text_after_prose_matches(original_text, prose, expected)
    
    def is_consistent_separator(self, target_text: str):
        analyzer = TextSeparationAnalyzer(target_text)
        return analyzer.is_separator_consistent()
    
    def assert_prefix_matches(self, target_text: str, prefix: str):
        analyzer = TextSeparation(target_text, is_character_alpha)
        self.assertEqual(analyzer.get_prefix(), prefix)
    
    def assert_indices_match(self, prose, text, prose_index, beginning_index, ending_index):
        analyzer = TextSeparationAnalyzer(text)
        analyzer.search_for_prose_in_separated_part(prose)
        self.assertEqual(analyzer.get_prose_index(), prose_index)
        self.assertEqual(analyzer.get_prose_beginning_index(), beginning_index)
        self.assertEqual(analyzer.get_prose_ending_index(), ending_index)
    
    def assert_text_before_prose_matches(self, original_text: str, prose: str, expected: str):
        analyzer = TextSeparationAnalyzer(original_text)
        analyzer.search_for_prose_in_separated_part(prose)
        actual: str = analyzer.compute_text_before_prose()
        self.assertEqual(actual, expected)
    
    def assert_text_after_prose_matches(self, original_text: str, prose: str, expected: str):
        analyzer = TextSeparationAnalyzer(original_text)
        analyzer.search_for_prose_in_separated_part(prose)
        actual: str = analyzer.compute_text_after_prose()
        self.assertEqual(actual, expected)

class TestConsistentProseSeparatorDetection(unittest.TestCase):
    def test_handles_single_word_prose(self):
        self.assert_text_with_prose_gives_the_result('this_is_a_test', 'is', True)
    
    def test_handles_snake_case(self):
        self.assert_text_with_prose_gives_the_result('chicken!!this_is_a_testchicken', 'this is a test', True)
    
    def test_handle_spaces(self):
        self.assert_text_with_prose_gives_the_result('for real this is a test', 'this is a test', True)
    
    def test_handles_two_words(self):
        self.assert_text_with_prose_gives_the_result('this_is!_@_____a_test', 'is a', True)
    
    def test_handles_final_word(self):
        self.assert_text_with_prose_gives_the_result('this_is_a_test!', 'is a test', True)
    
    def test_false_with_two_different_separators(self):
        self.assert_text_with_prose_gives_the_result('this_is a test', 'this is a', False)

    def assert_text_with_prose_gives_the_result(self, text: str, prose: str, expected: bool):
        analyzer = TextSeparationAnalyzer(text)
        analyzer.search_for_prose_in_separated_part(prose)
        result: bool = analyzer.is_prose_separator_consistent()
        self.assertEqual(result, expected)

class TestDetectingProseCases(unittest.TestCase):
    def test_handles_single_lowercase_word(self):
        self.assert_prose_cases_match_expected('word', 'word', 'lower')
    
    def test_handles_single_upper_case_word(self):
        self.assert_prose_cases_match_expected('WORD', 'word', 'upper')
    
    def test_handles_single_capitalized_word(self):
        self.assert_prose_cases_match_expected('Word', 'word', 'capitalized')

    def test_handles_single_uppercase_character(self):
        self.assert_prose_cases_match_expected('A', 'a', 'upper')
    
    def test_handles_camel_case_correctly(self):
        self.assert_prose_cases_match_expected('thisIsATest', 'this is a test', 'lower capitalized upper capitalized')
    
    def test_handles_snake_case_correctly(self):
        self.assert_prose_cases_match_expected('this_is_a_test', 'this is a test', 'lower')
    
    def test_handles_substring_prose(self):
        self.assert_prose_cases_match_expected('yesthisIsaTESThere', 'this is a test', 'lower capitalized lower upper')
    
    def test_handles_separated_substring(self):
        self.assert_prose_cases_match_expected('stuff!THIS_IS_A_TEST!stuff', 'this is a test', 'upper')

    def assert_prose_cases_match_expected(self, text: str, prose: str, expected: str):
        analyzer = TextSeparationAnalyzer(text)
        analyzer.search_for_prose_in_separated_part(prose)
        case_string = compute_case_string_for_prose(analyzer)
        self.assertEqual(case_string, expected)
    
class TestProseCasesSimplification(unittest.TestCase):
    def test_handles_single_case(self):
        self.assert_simplification_matches_expected(['lower'], ['lower'])
    
    def test_handles_two_different_cases(self):
        self.assert_simplification_matches_expected(['lower', 'upper'], ['lower', 'upper'])
    
    def test_handles_two_identical_cases(self):
        self.assert_simplification_matches_expected(['lower', 'lower'], ['lower'])
    
    def test_handles_identical_cases_followed_by_different(self):
        self.assert_simplification_matches_expected(['upper', 'capitalized', 'lower', 'upper', 'lower', 'lower', 'lower'], ['upper', 'capitalized', 'lower', 'upper', 'lower'])
        
    def assert_simplification_matches_expected(self, original, expected):
        actual = compute_simplified_case_strings_list(original)
        self.assertEqual(actual, expected)
    
class TestGettingFirstSeparator(unittest.TestCase):
    def test_handles_single_word_no_separator(self):
        self.assert_separator_matches_expected('this', 'this', '')

    def test_handles_single_word_in_text_with_no_internal_separator(self):
        self.assert_separator_matches_expected('stuff this test', 'this', '')
    
    def test_handles_two_words_with_separator(self):
        self.assert_separator_matches_expected('two  words', 'two words', '  ')
    
    def test_handles_two_words_in_text_with_separator(self):
        self.assert_separator_matches_expected('This contains two_words in the middle', 'two words', '_')
    
    def test_handles_three_words_in_text(self):
        self.assert_separator_matches_expected('this_is_a_bigger_test_case', 'bigger test', '_')
    
    def test_handles_two_words_at_beginning(self):
        self.assert_separator_matches_expected('two_words_at_the_beginning', 'two words', '_')
    
    def test_handles_two_words_at_ending(self):
        self.assert_separator_matches_expected('at_ending_there_are_two_words', 'two words', '_')
        
    def assert_separator_matches_expected(self, text: str, prose: str, expected: str):
        analyzer = TextSeparationAnalyzer(text)
        analyzer.search_for_prose_in_separated_part(prose)
        actual = analyzer.get_first_prose_separator()
        self.assertEqual(actual, expected)
    
class TestMakeAbstractRepresentationForProseCommand(unittest.TestCase):
    def test_handles_simple_insert_only(self):
        actions = [generate_insert_action('simple')]
        input_command_chain = generate_command_chain_with_actions(actions)
        prose = 'simple'
        insert_to_modify_index = 0
        expected_actions = [generate_abstract_prose_action('lower', '')]
        expected_command_chain = generate_command_chain_with_actions(expected_actions)
        self.assert_actual_matches_expected_given_arguments(input_command_chain, prose, insert_to_modify_index, expected_command_chain)
    
    def test_handles_insert_only_with_text_to_the_left(self):
        actions = [generate_insert_action('prefixed_simple')]
        input_command_chain = generate_command_chain_with_actions(actions)
        prose = 'simple'
        insert_to_modify_index = 0
        expected_actions = [generate_insert_action('prefixed_'), generate_abstract_prose_action('lower', '')]
        expected_command_chain = generate_command_chain_with_actions(expected_actions)
        self.assert_actual_matches_expected_given_arguments(input_command_chain, prose, insert_to_modify_index, expected_command_chain)
    
    def test_handles_insert_only_with_text_to_the_right(self):
        actions = [generate_insert_action('simple_text_postfix_text')]
        input_command_chain = generate_command_chain_with_actions(actions)
        prose = 'simple text'
        insert_to_modify_index = 0
        expected_actions = [generate_abstract_prose_action('lower', '_'), generate_insert_action('_postfix_text')]
        expected_command_chain = generate_command_chain_with_actions(expected_actions)
        self.assert_actual_matches_expected_given_arguments(input_command_chain, prose, insert_to_modify_index, expected_command_chain)
    
    def test_handles_insert_only_with_text_on_both_sides(self):
        actions = [generate_insert_action('prefixsimple_text_postfix_text')]
        input_command_chain = generate_command_chain_with_actions(actions)
        prose = 'simple text'
        insert_to_modify_index = 0
        expected_actions = [generate_insert_action('prefix'), generate_abstract_prose_action('lower', '_'), generate_insert_action('_postfix_text')]
        expected_command_chain = generate_command_chain_with_actions(expected_actions)
        self.assert_actual_matches_expected_given_arguments(input_command_chain, prose, insert_to_modify_index, expected_command_chain)
    
    def test_handles_prior_actions(self):
        actions = [generate_press_a_action(), generate_insert_action('test'), generate_insert_action('simple')]
        input_command_chain = generate_command_chain_with_actions(actions)
        prose = 'simple'
        insert_to_modify_index = 2
        expected_actions = [generate_press_a_action(), generate_insert_action('test'), generate_abstract_prose_action('lower', '')]
        expected_command_chain = generate_command_chain_with_actions(expected_actions)
        self.assert_actual_matches_expected_given_arguments(input_command_chain, prose, insert_to_modify_index, expected_command_chain)
    
    def test_handles_subsequent_actions(self):
        actions = [generate_insert_action('simple'), generate_insert_action('test'), generate_press_a_action()]
        input_command_chain = generate_command_chain_with_actions(actions)
        prose = 'simple'
        insert_to_modify_index = 0
        expected_actions = [generate_abstract_prose_action('lower', ''), generate_insert_action('test'), generate_press_a_action()]
        expected_command_chain = generate_command_chain_with_actions(expected_actions)
        self.assert_actual_matches_expected_given_arguments(input_command_chain, prose, insert_to_modify_index, expected_command_chain)
    
    def test_handles_actions_before_and_after(self):
        actions = [generate_insert_action('extra'), generate_insert_action('simple'), generate_insert_action('test'), generate_press_a_action()]
        input_command_chain = generate_command_chain_with_actions(actions)
        prose = 'simple'
        insert_to_modify_index = 1
        expected_actions = [generate_insert_action('extra'), generate_abstract_prose_action('lower', ''), generate_insert_action('test'), generate_press_a_action()]
        expected_command_chain = generate_command_chain_with_actions(expected_actions)
        self.assert_actual_matches_expected_given_arguments(input_command_chain, prose, insert_to_modify_index, expected_command_chain)

    def assert_actual_matches_expected_given_arguments(self, command_chain, prose: str, insert_to_modify_index: int, expected):
        text = command_chain.get_actions()[insert_to_modify_index].get_arguments()[0]
        analyzer = TextSeparationAnalyzer(text)
        analyzer.search_for_prose_in_separated_part(prose)
        new_name = 'name'
        match = ProseMatch(analyzer, new_name)
        actual = make_abstract_representation_for_prose_command(command_chain, match, insert_to_modify_index)
        expected_with_new_name = compute_command_chain_copy_with_new_name_and_actions(expected, new_name, expected.get_actions())
        assert_command_chains_match(self, actual, expected_with_new_name)

class TestObtainInsertsFromCommandChain(unittest.TestCase):
    def test_handles_empty_command_chain(self):
        actions = []
        expected = []
        self.assert_gets_expected_list_given_input(expected, actions)
    
    def test_handles_no_inserts(self):
        actions = [generate_press_a_action(), generate_key_press_action('ctrl-t'), BasicAction('mouse_click', [1])]
        expected = []
        self.assert_gets_expected_list_given_input(expected, actions)
    
    def test_detects_single_insert(self):
        actions = [generate_insert_action('testing')]
        expected = [InsertAction('testing', 0)]
        self.assert_gets_expected_list_given_input(expected, actions)
    
    def test_detects_multiple_inserts(self):
        actions = [generate_insert_action('testing'), generate_insert_action('last')]
        expected = [InsertAction('testing', 0), InsertAction('last', 1)]
        self.assert_gets_expected_list_given_input(expected, actions)

    def test_detects_insert_after_action(self):
        actions = [generate_press_a_action(), generate_insert_action('last')]
        expected = [InsertAction('last', 1)]
        self.assert_gets_expected_list_given_input(expected, actions)
    
    def test_detect_insert_before_action(self):
        actions = [generate_insert_action('testing'), generate_press_a_action()]
        expected = [InsertAction('testing', 0)]
        self.assert_gets_expected_list_given_input(expected, actions)
    
    def test_detect_inserts_between_actions(self):
        actions = [generate_press_a_action(), generate_insert_action('testing'), generate_insert_action('last'), generate_press_a_action()]
        expected = [InsertAction('testing', 1), InsertAction('last', 2)]
        self.assert_gets_expected_list_given_input(expected, actions)

    def assert_gets_expected_list_given_input(self, expected, actions):
        input = generate_command_chain_with_actions(actions)
        actual = obtain_inserts_from_command_chain(input)
        self.assert_insert_lists_match(actual, expected)

    def assert_insert_lists_match(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for index in range(len(actual)): self.assert_inserts_match(actual[index], expected[index])

    def assert_inserts_match(self, actual, expected):
        self.assertEqual(actual.text, expected.text)
        self.assertEqual(actual.index, expected.index)

TEST_MAX_PROSE_SIZE_TO_CONSIDER = 10
class TestFindProseMatchesForCommandGivenInsert(unittest.TestCase):
    def test_no_matching_text_gives_empty_list(self):
        command_chain = CommandChain('chicken', generate_test_insert_action(), 0, 1)
        insert = InsertAction('this is a test', 0)
        expected = []
        actual = find_prose_matches_for_command_given_insert(command_chain, insert, TEST_MAX_PROSE_SIZE_TO_CONSIDER)
        self.assertEqual(actual, expected)
    
    def test_inconsistent_separator_matches_not_included(self):
        insert_text = 'testing_this!here'
        dictation = 'say testing this here'
        command_chain = CommandChain(dictation, generate_insert_action(insert_text))
        insert = InsertAction(insert_text, 0)
        expected_names = ['say <user.text> this here', 'say <user.text> here', 'say testing <user.text> here', 'say testing <user.text>', 'say testing this <user.text>']
        expected_text_before = ['', '', 'testing_', 'testing_', 'testing_this!']
        expected_text_after = ['_this!here', '!here', '!here', '', '']
        actual = find_prose_matches_for_command_given_insert(command_chain, insert, TEST_MAX_PROSE_SIZE_TO_CONSIDER)
        self.assert_actual_matches_expected_names_and_text(actual, expected_names, expected_text_before, expected_text_after)
    
    def assert_actual_matches_expected_names_and_text(self, actual, expected_names, expected_text_before, expected_text_after):
        actual_names = [match.name for match in actual]
        actual_analyzers = [match.analyzer for match in actual]
        actual_text_before = [analyzer.compute_text_before_prose() for analyzer in actual_analyzers]
        actual_text_after = [analyzer.compute_text_after_prose() for analyzer in actual_analyzers]
        self.assertEqual(actual_names, expected_names)
        self.assertEqual(actual_text_before, expected_text_before)
        self.assertEqual(actual_text_after, expected_text_after)

class TestMakeAbstractProseRepresentationsForCommandGivenInserts(unittest.TestCase):
    def test_handles_no_inserts(self):
        no_insert_command_chain = generate_no_insert_command_chain()
        expected = []
        actual = make_abstract_prose_representations_for_command_given_inserts(no_insert_command_chain, [], TEST_MAX_PROSE_SIZE_TO_CONSIDER)
        self.assertEqual(actual, expected)
    
    def test_handles_one_insert(self):
        one_insert_command_chain = generate_single_insert_command_chain()
        expected_command_chain = generate_single_insert_command_chain_abstract_prose_representation()
        expected_number_of_commands = 1
        actual = make_abstract_prose_representations_for_command_given_inserts(one_insert_command_chain, [InsertAction('test', 0)], TEST_MAX_PROSE_SIZE_TO_CONSIDER)
        self.assertEqual(len(actual), expected_number_of_commands)
        assert_command_chains_match(self, actual[0], expected_command_chain)
    
    def test_returns_nothing_with_single_prose_action_only(self):
        single_prose_action_only_command_chain = CommandChain('say test', [generate_insert_action('test')], 0, 1)
        expected = []
        actual = make_abstract_prose_representations_for_command_given_inserts(single_prose_action_only_command_chain, [InsertAction('test', 0)], TEST_MAX_PROSE_SIZE_TO_CONSIDER)
        self.assertEqual(actual, expected)
    
    def test_handles_two_inserts(self):
        two_inserts_command_chain = CommandChain('this is a test', [generate_insert_action('this is'), generate_press_a_action(), generate_insert_action('a test')], 0, 1)
        expected_number_of_commands = 6
        actual = make_abstract_prose_representations_for_command_given_inserts(two_inserts_command_chain, [InsertAction('this is', 0), InsertAction('a test', 2)], TEST_MAX_PROSE_SIZE_TO_CONSIDER)
        self.assertEqual(len(actual), expected_number_of_commands)
        expected_commands = generate_two_inserts_command_chain_abstract_prose_representations()
        for index, expected in enumerate(expected_commands): assert_command_chains_match(self, actual[index], expected)

class MakeAbstractProseRepresentationsForCommand(unittest.TestCase):
    def test_handles_no_inserts(self):
        no_insert_command_chain = generate_no_insert_command_chain()
        expected = []
        actual = make_abstract_prose_representations_for_command(no_insert_command_chain, TEST_MAX_PROSE_SIZE_TO_CONSIDER)
        self.assertEqual(actual, expected)
    
    def test_handles_one_insert(self):
        one_insert_command_chain = generate_single_insert_command_chain()
        expected_command_chain = generate_single_insert_command_chain_abstract_prose_representation()
        expected_number_of_commands = 1
        actual = make_abstract_prose_representations_for_command(one_insert_command_chain, TEST_MAX_PROSE_SIZE_TO_CONSIDER)
        self.assertEqual(len(actual), expected_number_of_commands)
        assert_command_chains_match(self, actual[0], expected_command_chain)

    def test_handles_two_inserts(self):
        two_inserts_command_chain = generate_two_inserts_command_chain()
        expected_number_of_commands = 6
        actual = make_abstract_prose_representations_for_command(two_inserts_command_chain, TEST_MAX_PROSE_SIZE_TO_CONSIDER)
        self.assertEqual(len(actual), expected_number_of_commands)
        expected_commands = generate_two_inserts_command_chain_abstract_prose_representations()
        for index, expected in enumerate(expected_commands): assert_command_chains_match(self, actual[index], expected)

def generate_rain_potential_command_information():
    return generate_potential_command_information_with_uses(generate_rain_as_down_command().get_actions(), ['rain'])

def generate_copy_all_potential_command_information():
    return generate_potential_command_information_with_uses(generate_copy_all_command().get_actions(), ['copy all'])

def generate_air_potential_command_information():
    return generate_potential_command_information_with_uses(generate_press_a_command().get_actions(), ['air'])

def generate_rain_copy_all_potential_command_information():
    return generate_potential_command_information_with_uses(generate_multiple_key_pressing_actions(['down', 'ctrl-a', 'ctrl-c']), ['rain copy all'])

def generate_rain_copy_all_air_potential_command_information():
    return generate_potential_command_information_with_uses(generate_multiple_key_pressing_actions(['down', 'ctrl-a', 'ctrl-c', 'a']), ['rain copy all air'])

def generate_copy_all_air_potential_command_information():
    return generate_potential_command_information_with_uses(generate_multiple_key_pressing_actions(['ctrl-a', 'ctrl-c', 'a']), ['copy all air'])


def generate_no_insert_command_chain():
    no_insert_command_chain = CommandChain('this is a test ctrl-a ctrl-c', generate_copy_all_action_list(), 0, 1)
    return no_insert_command_chain

def generate_single_insert_command_chain():
    one_insert_command_chain = CommandChain('say test', [generate_insert_action('test'), generate_press_a_action()], 0, 1)
    return one_insert_command_chain

def generate_single_insert_command_chain_abstract_prose_representation():
    one_insert_abstract_prose_representation = CommandChain('say <user.text>', [generate_abstract_prose_action('lower', ''), generate_press_a_action()], 0, 1)
    return one_insert_abstract_prose_representation

def generate_two_inserts_command_chain():
    two_inserts_command_chain = CommandChain('this is a test', [generate_insert_action('this is'), generate_press_a_action(), generate_insert_action('a test')], 0, 1)
    return two_inserts_command_chain

def generate_two_inserts_command_chain_abstract_prose_representations():
    first_expected = CommandChain('<user.text> is a test', [generate_abstract_prose_action('lower', ''), generate_insert_action(' is'), generate_press_a_action(), generate_insert_action('a test')], 0, 1)
    second_expected = CommandChain('<user.text> a test', [generate_abstract_prose_action('lower', ' '), generate_press_a_action(), generate_insert_action('a test')], 0, 1)
    third_expected = CommandChain('this <user.text> a test', [generate_insert_action('th'), generate_abstract_prose_action('lower', ''), generate_insert_action(' is'), generate_press_a_action(), generate_insert_action('a test')], 0, 1)
    #This third one technically behaves as it should, but I might want to improve this functionality later
    fourth_expected = CommandChain('this is <user.text> test', [generate_insert_action('this is'), generate_press_a_action(), generate_abstract_prose_action('lower', ''), generate_insert_action(' test')], 0, 1)
    fifth_expected = CommandChain('this is <user.text>', [generate_insert_action('this is'), generate_press_a_action(), generate_abstract_prose_action('lower', ' ')], 0, 1)
    sixth_expected = CommandChain('this is a <user.text>', [generate_insert_action('this is'), generate_press_a_action(), generate_insert_action('a '), generate_abstract_prose_action('lower', '')], 0, 1)
    representations = [first_expected, second_expected, third_expected, fourth_expected, fifth_expected, sixth_expected]
    return representations

def generate_test_insert_action():
    action = generate_insert_action('this is a test')
    return action

def generate_abstract_prose_action(case_string: str, first_prose_separator: str):
    prose_argument = TalonCapture('user.text', 1)
    action = BasicAction('user.fire_chicken_auto_generated_command_action_insert_formatted_text', [prose_argument, case_string, first_prose_separator])
    return action

def generate_command_chain_with_name_and_actions(name, actions):
    result = CommandChain(name, actions, 0, 1)
    return result

def generate_command_chain_with_actions(actions):
    result = CommandChain('name', actions, 0, 1)
    return result

def assert_command_chains_match(test_object, actual, expected):
    test_object.assertEqual(actual.get_actions(), expected.get_actions())
    test_object.assertEqual(actual.get_name(), expected.get_name())
    test_object.assertEqual(actual.get_chain_number(), expected.get_chain_number())
    test_object.assertEqual(actual.get_chain_ending_index(), expected.get_chain_ending_index())

def command_set_matches_expected_potential_command_information(command_set, expected):
    if command_set.get_size() != len(expected):
        print(f'Incorrect size! Expected {len(expected)} but received {command_set.get_size()}')
        print('Received: ', str(command_set))
        return False
    for command_information in expected:
        matching_command_information = get_command_set_information_matching_actions(command_set, command_information.get_actions())
        if not potential_command_informations_match(matching_command_information, command_information):
            print(f'Potential commands do not match: Received: {matching_command_information} Expected: {command_information}')
            return False
    return True

def generate_named_press_a_command_chain(name: str, number: int = 0):
    return CommandChain(name, generate_press_a_action_list(), number, 1)

def generate_simple_command_record():
    record = [generate_rain_as_down_command(), generate_copy_all_command(), generate_press_a_command()]
    return record

def generate_command_record_with_many_seconds_before_middle_command():
    record = [generate_rain_as_down_command(), generate_copy_all_command(90000000000), generate_press_a_command()]
    return record

def generate_insert_action(text: str):
    return BasicAction('insert', [text])

def get_command_set_information_matching_actions(command_set, actions):
    def search_condition(command):
        return command.get_actions() == actions
    matching_actions = command_set.get_commands_meeting_condition(search_condition)
    return matching_actions[0]

def generate_potential_command_information_with_uses(actions, invocations):        
    information = PotentialCommandInformation(actions)
    for index, invocation in enumerate(invocations):
        information.process_usage(CommandChain(invocation, actions, index))
    return information
        
def potential_command_informations_match(original, other):
    return original.get_average_words_dictated() == other.get_average_words_dictated() and original.get_number_of_actions() == other.get_number_of_actions() and \
            original.get_number_of_times_used() == other.get_number_of_times_used() and original.get_actions() == other.get_actions()

def return_true(value):
    return True

def generate_potential_command_information_on_press_a():
    return PotentialCommandInformation(generate_press_a_action_list())

def generate_press_a_command_chain(chain: int = 0, chain_ending_index: int = 0):
    return CommandChain('air', generate_press_a_action_list(), chain, chain_ending_index)

def generate_press_a_command():
    return Command('air', generate_press_a_action_list())

def generate_press_a_action_list():
    return [generate_press_a_action()]

def generate_press_a_action():
    return generate_key_press_action('a')

def generate_copy_all_command_chain(chain, chain_ending_index):
    copy_all_command = generate_copy_all_command()
    return CommandChain(copy_all_command.get_name(), copy_all_command.get_actions(), chain, chain_ending_index)

def generate_copy_all_command(seconds_since_last_action: int = None):
    return generate_multiple_key_pressing_command('copy all', generate_copy_all_keystroke_list(), seconds_since_last_action)

def generate_copy_all_action_list():
    return generate_multiple_key_pressing_actions(generate_copy_all_keystroke_list())

def generate_copy_all_keystroke_list():
    return ['ctrl-a', 'ctrl-c']

def generate_rain_as_down_command():
    return generate_key_pressing_command('rain', 'down')

def generate_multiple_key_pressing_command(name: str, keystrokes, seconds_since_last_action: int = None):
    actions = generate_multiple_key_pressing_actions(keystrokes)
    command = Command(name, actions, seconds_since_last_action)
    return command

def generate_multiple_key_pressing_actions(keystrokes):
    actions = [generate_key_press_action(keystroke) for keystroke in keystrokes]
    return actions

def generate_key_pressing_command(name: str, keystroke: str):
    action = generate_key_press_action(keystroke)
    command = Command(name, [action])
    return command

def generate_key_press_action(keystroke: str):
    return BasicAction('key', [keystroke])

if __name__ == '__main__':
    unittest.main()
