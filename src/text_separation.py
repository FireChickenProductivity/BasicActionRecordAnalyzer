from typing import List

def is_character_alpha(character: str):
    return character.isalpha()

class TextSeparation:
    def __init__(self, string: str, character_filter):
        self.separated_parts = []
        self.separators = []
        self.current_separated_part = ''
        self.current_separator = ''
        self.text_prefix = ''
        for character in string: self._process_character(character, character_filter)
        if not self.current_separated_part: self._handle_separator()
        if not self.current_separator: self._add_separated_part()
    
    def _process_character(self, character, character_filter):
        if character_filter(character): 
            if not self.current_separated_part: self._handle_separator()
            self.current_separated_part += character
        else:
            if not self.current_separator and self.current_separated_part: self._add_separated_part()
            self.current_separator += character

    def _handle_separator(self):
        if len(self.separated_parts) > 0:
            self.separators.append(self.current_separator)
        else:
            self.text_prefix = self.current_separator
        self.current_separator = ''
    
    def _add_separated_part(self):
        self.separated_parts.append(self.current_separated_part)
        self.current_separated_part = ''
    
    def get_separated_parts(self):
        return self.separated_parts

    def get_separators(self):
        return self.separators
    
    def get_prefix(self):
        return self.text_prefix

class TextSeparationAnalyzer:
    def __init__(self, text: str, character_filter = is_character_alpha):
        self.text_separation = TextSeparation(text, character_filter)
        self.prose_index = None
        self.final_prose_index_into_separated_parts = None
        self.prose_beginning_index = None
        self.prose_ending_index = None
        self.number_of_prose_words = None
        self.found_prose: bool = False
        self.prose: str = None

    def search_for_prose_beginning_at_separated_part_index(self, words, separated_parts, index):
        initial_separated_part = separated_parts[index].lower()
        first_word = words[0]
        if initial_separated_part.endswith(first_word): self.prose_beginning_index = initial_separated_part.rfind(first_word)
    
    def search_for_prose_at_separated_part_index_beginning(self, prose_without_spaces, separated_parts, index):
        if prose_without_spaces in separated_parts[index].lower(): 
            self.prose_beginning_index = separated_parts[index].lower().find(prose_without_spaces)
            self.prose_ending_index = len(prose_without_spaces) + self.prose_beginning_index
            self.found_prose = True
            self.final_prose_index_into_separated_parts = index
    
    def is_prose_middle_different_from_separated_parts_at_index(self, words, separated_parts, index):
        for prose_index in range(1, len(words) - 1):
            word = words[prose_index]
            separated_part: str = separated_parts[prose_index + index].lower()
            if separated_part != word: return True
        return False
    
    def perform_final_prose_search_at_index(self, words, separated_parts, index):
        if len(words) == 1:
            self.found_prose = True
            self.final_prose_index_into_separated_parts = self.prose_index
        else:
            self.final_prose_index_into_separated_parts = index + len(words) - 1
            final_separated_part = separated_parts[self.final_prose_index_into_separated_parts].lower()
            last_word = words[-1]
            if final_separated_part.startswith(last_word): 
                self.prose_ending_index = len(last_word)
                self.found_prose = True
    
    def reset_indices(self):
        self.prose_beginning_index = None
        self.prose_index = None
        self.prose_ending_index = None
        self.final_prose_index_into_separated_parts = None

    def search_for_prose_at_separated_part_index(self, prose_without_spaces: str, words, index: int):
        self.reset_indices()
        
        separated_parts = self.text_separation.get_separated_parts()

        self.search_for_prose_at_separated_part_index_beginning(prose_without_spaces, separated_parts, index)
        if self.found_prose: return
        if len(words) + index > len(separated_parts): return

        self.search_for_prose_beginning_at_separated_part_index(words, separated_parts, index)
        if self.prose_beginning_index is None: return

        if self.is_prose_middle_different_from_separated_parts_at_index(words, separated_parts, index): return 

        self.perform_final_prose_search_at_index(words, separated_parts, index)
    
    def search_for_prose_in_separated_part(self, prose: str):
        lowercase_prose = prose.lower()
        prose_without_spaces = lowercase_prose.replace(' ', '')
        words = lowercase_prose.split(' ')
        self.number_of_prose_words = len(words)
        self.prose = prose
        for index in range(len(self.text_separation.get_separated_parts())):
            self.search_for_prose_at_separated_part_index(prose_without_spaces, words, index)
            self.prose_index = index
            if self.found_prose: return
        self.found_prose = False
        return
    
    def is_separator_consistent(self, starting_index: int = 0, ending_index: int = -1):
        separators = self.text_separation.get_separators()[starting_index:ending_index]
        if len(separators) <= 1: return True
        initial_separator = separators[0]
        for index in range(1, len(separators)):
            if separators[index] != initial_separator: return False
        return True

    def get_prose_index(self):
        return self.prose_index
    
    def get_prose_beginning_index(self):
        return self.prose_beginning_index
    
    def get_prose_ending_index(self):
        return self.prose_ending_index

    def is_prose_separator_consistent(self):
        return self.is_separator_consistent(self.prose_index, self.final_prose_index_into_separated_parts)

    def get_first_prose_separator(self) -> str:
        separators = self.text_separation.get_separators()
        if self.prose_index < len(separators) and self.prose_index != self.final_prose_index_into_separated_parts: return separators[self.prose_index]
        else: return ''

    def has_found_prose(self):
        return self.found_prose
    
    def compute_text_before_prose(self) -> str:
        text: str = self.text_separation.get_prefix()
        separated_parts = self.text_separation.get_separated_parts()
        separators = self.text_separation.get_separators()
        for index in range(self.prose_index):
            text += separated_parts[index]
            text += separators[index]
        text += separated_parts[self.prose_index][0:self.prose_beginning_index]
        return text
    
    def compute_text_after_prose(self) -> str:
        separated_parts = self.text_separation.get_separated_parts()
        separators = self.text_separation.get_separators()
        text: str = ''
        first_word: str = separated_parts[self.final_prose_index_into_separated_parts]
        if self.prose_ending_index < len(first_word): text += first_word[self.prose_ending_index:]
        if self.final_prose_index_into_separated_parts < len(separators): text += separators[self.final_prose_index_into_separated_parts]
        for index in range(self.final_prose_index_into_separated_parts + 1, len(separated_parts)):
            text += separated_parts[index]
            if index < len(separators): text += separators[index]
        return text
    
    def _compute_prose_portion_of_nonseparated_text(self):
        words = self.prose.split(' ')
        prose_portion_of_text_as_string = self.text_separation.get_separated_parts()[self.prose_index][self.prose_beginning_index:self.prose_ending_index]
        word_starting_index = 0
        words_from_text = []
        for word in words:
            word_ending_index = word_starting_index + len(word)
            word_from_text = prose_portion_of_text_as_string[word_starting_index:word_ending_index]
            words_from_text.append(word_from_text)
            word_starting_index = word_ending_index
        return words_from_text

    def _compute_prose_portion_of_separated_text(self, prose_final_index):
        prose_words = []
        separated_parts = self.text_separation.get_separated_parts()
        prose_words.append(separated_parts[self.prose_index][self.prose_beginning_index:])
        prose_words.extend([separated_parts[index] for index in range(self.prose_index + 1, prose_final_index)])
        prose_words.append(separated_parts[prose_final_index][:self.prose_ending_index])
        return prose_words

    def compute_prose_portion_of_text(self) -> List[str]:
        if self.prose_index == self.final_prose_index_into_separated_parts: return self._compute_prose_portion_of_nonseparated_text()
        else: return self._compute_prose_portion_of_separated_text(self.final_prose_index_into_separated_parts)