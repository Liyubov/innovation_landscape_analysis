import streamlit as st
from typing import Dict, List, Tuple
import tensorflow as tf
import numpy as np
import pandas as pd
import faiss
from bert.tokenization import FullTokenizer
import sys
from absl import flags
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean
#from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

##SQL DB Feedback Collection##
Base = declarative_base()
DATABASE_URL = "mssql+pyodbc://sqladmin:password@database?driver=ODBC Driver 17 for SQL Server&timeout=60"

# Create the database engine
engine = create_engine(DATABASE_URL)

# Create tables
Base.metadata.create_all(bind=engine)

# Create a session to interact with the database
DBSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True, index=True)
    feedback_type = Column(Boolean)  # True for thumbs up, False for thumbs down
    comment = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


def save_feedback(feedback_type, comment):
    db_feedback = Feedback(feedback_type=feedback_type, comment=comment)

    # Create a new session and add the feedback to the database
    db = DBSessionLocal()
    db.add(db_feedback)
    db.commit()
    db.refresh(db_feedback)
    db.close()
####


st.title("BERT Semantic Search")
tooltext = """
        This tool is designed for exploring the semantic closeness between the entry point of a user (e.g. text or idea of a new patent) and the existing patents database. For the construction of the embedding of the initial patent database we used the the BERT for Patents, which is a model trained by Google on 100M+ patents (not just US patents). It is based on BERTLARGE model. We propose two metrics to be used for the semantic search: L2 and cosine similarity measures to be able to explore and compare the search results.For any questions please reach out to Vasanth vasantha.balu@nokia.com or Liubov liubov.tupikina@nokia-bell-labs.com
        """
st.markdown(tooltext)
## Load embeddings and data set
embeddingdf=pd.read_csv("embeddingoutput.csv")
abstractembeddings_df = np.array(embeddingdf)
df = pd.read_csv("results.csv")


## DEFINE BERT PREDICTOR

def get_tokenized_input(
        texts: List[str], tokenizer: FullTokenizer) -> List[List[int]]:
    """Returns list of tokenized text segments."""

    return [tokenizer.tokenize(text) for text in texts]


class BertPredictor():

    def __init__(
            self,
            model_name: str,
            text_tokenizer: FullTokenizer,
            max_seq_length: int,
            max_preds_per_seq: int,
            has_context: bool = False):
        """Initializes a BertPredictor object."""

        self.tokenizer = text_tokenizer
        self.max_seq_length = max_seq_length
        self.max_preds_per_seq = max_preds_per_seq
        self.mask_token_id = 4
        # If you want to add context tokens to the input, set value to True.
        self.context = has_context

        model = tf.compat.v2.saved_model.load(export_dir=model_name, tags=['serve'])
        self.model = model.signatures['serving_default']

    def get_features_from_texts(self, texts: List[str]) -> Dict[str, int]:
        """Uses tokenizer to convert raw text into features for prediction."""

        # examples = [run_classifier.InputExample(0, t, label='') for t in texts]
        # features = run_classifier.convert_examples_to_features(
        #    examples, [''], self.max_seq_length, self.tokenizer)
        examples = [InputExample(0, t, label='') for t in texts]
        features = convert_examples_to_features(
            examples, [''], self.max_seq_length, self.tokenizer)
        return dict(
            input_ids=[f.input_ids for f in features],
            input_mask=[f.input_mask for f in features],
            segment_ids=[f.segment_ids for f in features]
        )

    def insert_token(self, input: List[int], token: int) -> List[int]:
        """Adds token to input."""

        return input[:1] + [token] + input[1:-1]

    def add_input_context(
            self, inputs: Dict[str, int], context_tokens: List[str]
    ) -> Dict[str, int]:
        """Adds context token to input features."""

        context_token_ids = self.tokenizer.convert_tokens_to_ids(context_tokens)
        segment_token_id = 0
        mask_token_id = 1

        for i, context_token_id in enumerate(context_token_ids):
            inputs['input_ids'][i] = self.insert_token(
                inputs['input_ids'][i], context_token_id)

            inputs['segment_ids'][i] = self.insert_token(
                inputs['segment_ids'][i], segment_token_id)

            inputs['input_mask'][i] = self.insert_token(
                inputs['input_mask'][i], mask_token_id)
        return inputs

    def create_mlm_mask(
            self, inputs: Dict[str, int], mlm_ids: List[List[int]]
    ) -> Tuple[Dict[str, List[List[int]]], List[List[str]]]:
        """Creates masked language model mask."""

        masked_text_tokens = []
        mlm_positions = []

        if not mlm_ids:
            inputs['mlm_ids'] = mlm_positions
            return inputs, masked_text_tokens

        for i, _ in enumerate(mlm_ids):

            masked_text = []

            # Pad mlm positions to max seqeuence length.
            mlm_positions.append(
                mlm_ids[i] + [0] * (self.max_preds_per_seq - len(mlm_ids[i])))

            for pos in mlm_ids[i]:
                # Retrieve the masked token.
                masked_text.extend(
                    self.tokenizer.convert_ids_to_tokens([inputs['input_ids'][i][pos]]))
                # Replace the mask positions with the mask token.
                inputs['input_ids'][i][pos] = self.mask_token_id

            masked_text_tokens.append(masked_text)

        inputs['mlm_ids'] = mlm_positions
        return inputs, masked_text_tokens

    def predict(
            self, texts: List[str], mlm_ids: List[List[int]] = None,
            context_tokens: List[str] = None
    ) -> Tuple[Dict[str, tf.Tensor], Dict[str, List[List[int]]], List[List[str]]]:
        """Gets BERT predictions for provided text and masks.

        Args:
          texts: List of texts to get BERT predictions.
          mlm_ids: List of lists corresponding to the mask positions for each input
            in `texts`.
          context_token: List of string contexts to prepend to input texts.

        Returns:
          response: BERT model response.
          inputs: Tokenized and modified input to BERT model.
          masked_text: Raw strings of the masked tokens.
        """

        if mlm_ids:
            assert len(mlm_ids) == len(texts), ('If mask ids provided, they must be '
                                                'equal to the length of the input text.')

        if self.context:
            # If model uses context, but none provided, use 'UNK' token for context.
            if not context_tokens:
                context_tokens = ['[UNK]' for _ in range(len(texts))]
            assert len(context_tokens) == len(texts), ('If context tokens provided, '
                                                       'they must be equal to the length of the input text.')

        inputs = self.get_features_from_texts(texts)

        # If using a BERT model with context, add corresponding tokens.
        if self.context:
            inputs = self.add_input_context(inputs, context_tokens)

        inputs, masked_text = self.create_mlm_mask(inputs, mlm_ids)

        response = self.model(
            segment_ids=tf.convert_to_tensor(inputs['segment_ids'], dtype=tf.int64),
            input_mask=tf.convert_to_tensor(inputs['input_mask'], dtype=tf.int64),
            input_ids=tf.convert_to_tensor(inputs['input_ids'], dtype=tf.int64),
            mlm_positions=tf.convert_to_tensor(inputs['mlm_ids'], dtype=tf.int64),
        )

        if mlm_ids:
            # Do a reshape of the mlm logits (batch size, num predictions, vocab).
            new_shape = (len(texts), self.max_preds_per_seq, -1)
            response['mlm_logits'] = tf.reshape(
                response['mlm_logits'], shape=new_shape)

        return response, inputs, masked_text

    # The functions in this block are also found in the bert cloned repo in the
# `run_classifier.py` file, however those also have some compatibility issues
# and thus the functions needed are just copied here.

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample."""
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    features = []
    for (ex_index, example) in enumerate(examples):
        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)
        features.append(feature)
    return features


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = label_map[example.label]

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        is_real_example=True)
    return feature


@st.cache_resource()
def load_model():
    filepath = "model/saved_model.pb"
    MAX_PREDS_PER_SEQUENCE = 45
    MAX_SEQ_LENGTH =  512
    MODEL_DIR = "model/"
    VOCAB = "model/vocab.txt"
    tokenizer = FullTokenizer(VOCAB, do_lower_case=True)
    bert_predictor = BertPredictor(
        model_name=MODEL_DIR,
        text_tokenizer=tokenizer,
        max_seq_length=MAX_SEQ_LENGTH,
        max_preds_per_seq=MAX_PREDS_PER_SEQUENCE,
        has_context=False)
    return bert_predictor

bert_predictor = load_model()

def generate_cosine(input_text):
    sys.argv = ['preserve_unused_tokens=False']
    flags.FLAGS(sys.argv)
  

    # Split the DataFrame into batches and make predictions for each batch
    arList = []
    arList.append(input_text)
    response, inputs, masked_text = bert_predictor.predict(arList)  # Make predictions for the batch
    train_inputs = response['cls_token']
    numpy_array = train_inputs.numpy()
    query_vector = numpy_array
    cosine_similarities = cosine_similarity(query_vector, abstractembeddings_df)
    abstractsArray = df.abstract
    group_id = df.group_id
    top5_indices = np.argsort(cosine_similarities[0])[-5:][::-1]
    top5_results = abstractsArray[top5_indices]
    similarity_scores = cosine_similarities[0][top5_indices]
    codes = group_id[top5_indices]

    st.write('\t', 'Top 5 results with Cosine Similarity:')
    for result, code, score in zip(top5_results, codes, similarity_scores):
      
        st.markdown(f"**{code}**. {result}")
        st.write('\t', f"Similarity Score: {score:.4f}")
    
    loadFeedbackControl()


def generate_response(input_text):
    sys.argv = ['preserve_unused_tokens=False']
    flags.FLAGS(sys.argv)
  
    all_predictions = []
    # Split the DataFrame into batches and make predictions for each batch
    arList = []
    arList.append(input_text)
    response, inputs, masked_text = bert_predictor.predict(arList)  # Make predictions for the batch
    train_inputs = response['cls_token']
    numpy_array = train_inputs.numpy()

    abstract = df.abstract
    title = df.title
    group_id = df.group_id
    index = faiss.read_index('patentsearch')
    query_vector = numpy_array

    k = 5
    d, top_k = index.search(query_vector, k)

    results = [{'abstract': abstract[_id], 'classification_code': group_id[_id], 'title':title[_id]} for _id in top_k[0]]
    st.write('\t', 'Top 5 results')
    for i, result in enumerate(results,start=0):
        
        st.markdown(f"**{result['classification_code']}**. {result['abstract']}")
        st.write(f"L2 score: {d[0][i]}")
      
        st.write("")  # Add an empty line for better readability
  
    loadFeedbackControl()



thumbs_up = None
thumbs_down = None
def loadFeedbackControl():
    return ""

with st.form('BERTSEARCH'):
  text = st.text_area('Enter your text for search:', '')

  col1, col2 = st.columns(2)
  with col1:
      submitted = st.form_submit_button('Search')
  with col2:
      cosinesearch = st.form_submit_button('Cosine Similarity Search')

  if submitted:
    generate_response(text)
  if cosinesearch:
    generate_cosine(text)

  col3, col4 = st.columns(2)
  with col3:
      thumbs_up = st.form_submit_button("👍")
  with col4:
      thumbs_down = st.form_submit_button("👎")

  if thumbs_up or thumbs_down:
      feedback_type = True if thumbs_up else False
      #save_feedback(feedback_type, text)
      st.success("Thank you for your feedback!")




 







