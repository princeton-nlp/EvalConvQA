"""
Module to handle universal/general constants used across files.
"""

################################################################################
# Constants #
################################################################################

# GENERAL CONSTANTS:
VERY_SMALL_NUMBER = 1e-12

_UNK_POS = 'unk_pos'
_UNK_NER = 'unk_ner'

_UNK_TOKEN = '<<unk>>'
_QUESTION_SYMBOL = '<q>'
_ANSWER_SYMBOL = '<a>'


# CoQA CONSTANTS:
CoQA_UNK_ANSWER = 'unknown'
CoQA_YES_ANSWER = 'yes'
CoQA_NO_ANSWER = 'no'

CoQA_UNK_ANSWER_LABEL = 0
CoQA_ANSWER_YES_LABEL = 1
CoQA_ANSWER_NO_LABEL = 2
CoQA_ANSWER_SPAN_LABEL = 3
CoQA_ANSWER_CLASS_NUM = 4


# QuAC
QuAC_UNK_ANSWER = 'cannotanswer'

QuAC_YESNO_YES = 'y'
QuAC_YESNO_NO = 'n'
QuAC_YESNO_OTHER = 'x'

QuAC_YESNO_YES_LABEL = 0
QuAC_YESNO_NO_LABEL = 1
QuAC_YESNO_OTHER_LABEL = 2
QuAC_YESNO_CLASS_NUM = 3

QuAC_FOLLOWUP_YES = 'y'
QuAC_FOLLOWUP_NO = 'n'
QuAC_FOLLOWUP_OTHER = 'm'

QuAC_FOLLOWUP_YES_LABEL = 0
QuAC_FOLLOWUP_NO_LABEL = 1
QuAC_FOLLOWUP_OTHER_LABEL = 2
QuAC_FOLLOWUP_CLASS_NUM = 3

# LOG FILES ##

_CONFIG_FILE = "config.json"
_SAVED_WEIGHTS_FILE = "params.saved"
_GOLD_PREDICTION_FILE = "gold_pred.json"
_PRED_PREDICTION_FILE = "pred_pred.json"
_TEST_PREDICTION_FILE = "test_pred.json"
