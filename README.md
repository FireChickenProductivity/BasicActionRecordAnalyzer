# Dependencies
For generating the histories it works with: https://github.com/FireChickenProductivity/BAR

Provides actions used by some of the commands it generates: https://github.com/FireChickenProductivity/ActionsForGeneratedCommands

# Usage
Execute the python script: basic_action_record_analysis.py in the src folder.

You give it a path to the history to analyze after the program starts running.

You give it the maximum command chain size to consider or press enter to use the default. This is the number of consecutive commands in the history to consider merging into a single command during analysis. Making this bigger can find longer patterns but takes longer.

The program generates a Recommendations directory outputting each set of recommendations in a text file. It will output some statistics proceeded by a # and the actions for every recommended command. 

# State of the Project
This is a prototype with significant room for improvement. I personally find this useful, but it currently generates excessive recommendations.
