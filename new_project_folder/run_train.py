
import sys, os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, r'C:\Users\macau\AppData\Roaming\Python\Python310\site-packages')
import transformers
from transformers import XLMRobertaTokenizer, TFXLMRobertaModel
from rasa.nlu.utils.hugging_face import registry
from rasa.nlu.featurizers.dense_featurizer import lm_featurizer
from rasa.nlu.utils.hugging_face.transformers_pre_post_processors import (
    roberta_tokens_pre_processor,
    gpt2_tokens_cleaner,
    roberta_embeddings_post_processor,
)
registry.model_class_dict['xlmroberta'] = TFXLMRobertaModel
registry.model_tokenizer_dict['xlmroberta'] = XLMRobertaTokenizer
registry.model_weights_defaults['xlmroberta'] = 'Davlan/afro-xlmr-base'
registry.model_special_tokens_pre_processors['xlmroberta'] = roberta_tokens_pre_processor
registry.model_tokens_cleaners['xlmroberta'] = gpt2_tokens_cleaner
registry.model_embeddings_post_processors['xlmroberta'] = roberta_embeddings_post_processor
lm_featurizer.MAX_SEQUENCE_LENGTHS['xlmroberta'] = 512

from rasa.__main__ import main
sys.argv = ['rasa', 'train', 'nlu', '--config', r'c:\Users\macau\Downloads\rasa_project\new_project_folder\config.yml', '--nlu', r'c:\Users\macau\Downloads\rasa_project\new_project_folder\data\train_nlu.yml', '--out', r'c:\Users\macau\Downloads\rasa_project\new_project_folder\models']
main()
