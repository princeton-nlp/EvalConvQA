# import transformers dependencies

# import model dependencies

class QAModel():
    def __init__(self, args):
        self.args = args

        # load model
        self.model = None

        # Initialize conversation history
        self.QA_history = []
    
    def tokenizer(self):

        # load tokenizer
        self.tokenizer = None
        return self.tokenizer
    
    def load_partial_examples(self, file_name):

        # Pre-load partially filled examples from train/dev file
        # Loaded to a list of passages, each with a list of QAs
        # We will construct the augmented question later
        partial_examples = []
        return partial_examples

    def predict_one_automatic_turn(self, partial_example, unique_id, example_idx):
        
        # Construct the augmented question here
        question = partial_example.question_text
        
        # Run prediction here. Your model might predict these fields.
        prediction_string = ""
        prediction_start = 0
        prediction_end = -1
        
        # Append predictions to QA history as your model will use it
        self.QA_history.append((example_idx, question, (prediction_string, prediction_start, prediction_end)))
        return prediction_string, unique_id + 1
