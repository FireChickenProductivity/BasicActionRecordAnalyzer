import os

DEFAULT_MAX_CHAIN_LENGTH = 20
NO_NUMBER_OF_RECOMMENDATIONS_LIMIT = 0

class InputParameter:
    def __init__(self, description, is_valid, explain_error, convert_value=lambda x: x, default_value=None):
        self.description = description
        self.is_valid = is_valid
        self.explain_error = explain_error
        self.default_value = default_value
        self.convert_value = convert_value

class InputParameters:
    def __init__(self):
        self.input_path = ""
        self.max_chain_length = DEFAULT_MAX_CHAIN_LENGTH
        self.max_number_of_recommendations = NO_NUMBER_OF_RECOMMENDATIONS_LIMIT

def compute_input_text(parameter: InputParameter) -> str:
    text = f"Input {parameter.description}"
    if parameter.default_value is not None:
        text += f". Press enter with no input to take default of {parameter.default_value}"
    text += ": "
    return text

def get_input_parameter_from_user(parameter: InputParameter):
    needs_valid_input = True
    prompt = compute_input_text(parameter)
    while needs_valid_input:
        text_input = input(prompt)
        if len(text_input) == 0 and parameter.default_value is not None:
            needs_valid_input = False
            value = parameter.default_value
        elif parameter.is_valid(text_input):
            needs_valid_input = False
            value = parameter.convert_value(text_input)
        else:
            print(parameter.explain_error(text_input))
    return value

def get_file_input_path_from_user() -> str:
    input_path_parameter = InputParameter(
        description="the file path to the command record",
        is_valid=os.path.exists,
        explain_error=lambda _: 'Please input a valid path!',
    )
    return get_input_parameter_from_user(input_path_parameter)

def _is_string_integer(text: str) -> bool:
    return text.isdigit()

def _create_message_asking_for_a_positive_integer_value(_):
    return 'Please enter a positive integer value.'

def get_max_chain_length_from_user():
    max_chain_length_parameter = InputParameter(
        description="the maximum number of consecutive commands to consider as a single potential command.\nMaking this bigger can allow finding longer patterns but it takes longer",
        is_valid=_is_string_integer,
        explain_error=_create_message_asking_for_a_positive_integer_value,
        default_value=DEFAULT_MAX_CHAIN_LENGTH,
        convert_value=int,
    )
    return get_input_parameter_from_user(max_chain_length_parameter)

def get_max_recommendations_from_user():
    parameter = InputParameter(
        description="the maximum number of command recommendations to output",
        is_valid=_is_string_integer,
        explain_error=_create_message_asking_for_a_positive_integer_value,
        default_value=NO_NUMBER_OF_RECOMMENDATIONS_LIMIT,
        convert_value=int,
    )
    return get_input_parameter_from_user(parameter)

def get_input_parameters_from_user() -> InputParameters:
    input_parameters = InputParameters()
    input_parameters.input_path = get_file_input_path_from_user()
    input_parameters.max_chain_length = get_max_chain_length_from_user()
    input_parameters.max_number_of_recommendations = get_max_recommendations_from_user()
    return input_parameters



