import os
import generate_data as gd
import utils

DATA_DIR = 'data'
utils.ensure_dir(DATA_DIR)

def process_examples(file_name: str, examples: dict) -> None:
    utils.print_an_example(examples)
    out_path = os.path.join(DATA_DIR, file_name)
    utils.save_pickle(out_path, examples)

# Testing the data in Thousands without a user opinion

out_name = 'math_evals_thousands_no_opinion.tsv'

process_examples(file_name=out_name, 
                 examples=gd.generate_math_eval_data(use_opinion=False, 
                                    num_of_examples=50,
                                    min_num=100,
                                    max_num=1000,
                                    answer_multiplier_min=0.9,
                                    answer_multiplier_max=1.1))

# Testing the data in Thousands with a user opinion

out_name = 'math_evals_thousands_with_opinion.tsv'

process_examples(file_name=out_name, 
                 examples=gd.generate_math_eval_data(use_opinion=True, 
                                    num_of_examples=50,
                                    min_num=100,
                                    max_num=1000,
                                    answer_multiplier_min=0.9,
                                    answer_multiplier_max=1.1))

# Testing the data in Thousands with a user opinion and a big multiplier (originally used in the article)

out_name = 'math_evals_thousands_with_opinion_big_multiplier.tsv'

process_examples(file_name=out_name, 
                 examples=gd.generate_math_eval_data(use_opinion=False, 
                                    num_of_examples=50,
                                    min_num=100,
                                    max_num=1000,
                                    answer_multiplier_min=1e5,
                                    answer_multiplier_max=1e6))

# Testing the data in below hundred with a user opinion

out_name = 'math_evals_below_hundred_with_opinion.tsv'

process_examples(file_name=out_name, 
                 examples=gd.generate_math_eval_data(use_opinion=False, 
                                    num_of_examples=50,
                                    min_num=1,
                                    max_num=100,
                                    answer_multiplier_min=0.9,
                                    answer_multiplier_max=1.1))

# Testing the data in Thousands without a user opinion

out_name = 'math_evals_tens_of_thousands_no_opinion.tsv'

process_examples(file_name=out_name, 
                 examples=gd.generate_math_eval_data(use_opinion=False, 
                                    num_of_examples=50,
                                    min_num=100,
                                    max_num=1000,
                                    answer_multiplier_min=0.9,
                                    answer_multiplier_max=1.1))

# Testing the data in Thousands with a user opinion

out_name = 'math_evals_tens_of_thousands_with_opinion.tsv'

process_examples(file_name=out_name, 
                 examples=gd.generate_math_eval_data(use_opinion=True, 
                                    num_of_examples=50,
                                    min_num=100,
                                    max_num=1000,
                                    answer_multiplier_min=0.9,
                                    answer_multiplier_max=1.1))