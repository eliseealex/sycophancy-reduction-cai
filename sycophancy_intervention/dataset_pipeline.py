"""Pipeline for generating synthetic tuning/evaluation data.

Initial code from Google LLC, modified by the authors of the research.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import generate_data as gd
import utils

DATA_DIR = 'data'
utils.ensure_dir(DATA_DIR)

generate_train = True
max_train_ex = 50
max_synthetic_eval_examples = 50
is_not_enabled = False
add_user_opinion = False
is_linguistic = False
is_cleanup_not = True
# None if all, otherwise a number of a dataset to use
data_set_to_use = None

generate_synthetic_eval = False
synthetic_min_num = 100
synthetic_max_num = 1000
synthetic_answer_multiplier_min = 0.9
synthetic_answer_multiplier_max = 1.1
################################################################################
#################                    MAIN                   ####################
################################################################################
if generate_train:
  generated_examples = gd.generate_nlp_data(use_opinion=add_user_opinion, 
                                            is_not_enabled=is_not_enabled, 
                                            is_linguistic=is_linguistic,
                                            is_cleanup_not=is_cleanup_not,
                                            data_set_to_use=data_set_to_use,
                                            num_examples=max_train_ex)
  utils.print_an_example(generated_examples)
  opinion_suffix = '_no_opinion' if not add_user_opinion else ''
  is_not_suffix = '_is_not' if is_not_enabled else ''
  is_linguistic_suffix = '_not_linguistic' if not is_linguistic else ''
  is_cleanup_not_suffix = '_cleanup_not' if is_cleanup_not else ''
  dataset_suffix = '_{data_set_to_use}' if data_set_to_use is not None else ''
  out_name = f'synthetic_train_{len(generated_examples)}{opinion_suffix}{is_not_suffix}{is_linguistic_suffix}{is_cleanup_not_suffix}{dataset_suffix}.tsv'
  out_path = os.path.join(DATA_DIR, out_name)
  utils.save_pickle(out_path, generated_examples)

if generate_synthetic_eval:
  examples = gd.generate_math_eval_data(use_opinion=add_user_opinion, 
                                        num_of_examples=max_synthetic_eval_examples,
                                        min_num=synthetic_min_num,
                                        max_num=synthetic_max_num,
                                        answer_multiplier_min=synthetic_answer_multiplier_min,
                                        answer_multiplier_max=synthetic_answer_multiplier_max)
  utils.print_an_example(examples)
  out_name = f'synthetic_eval_opinion{add_user_opinion}_{len(examples)}.tsv'
  out_path = os.path.join(DATA_DIR, out_name)
  utils.save_pickle(out_path, examples)
