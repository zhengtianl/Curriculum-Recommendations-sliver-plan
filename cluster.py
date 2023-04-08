import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from pathlib import Path
from sentence_transformers import SentenceTransformer, models, InputExample, losses
from datasets import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import gc
def read_data():
    class Topic:
        def __init__(self, topic_id):
            self.id = topic_id

        @property
        def parent(self):
            parent_id = topics_df.loc[self.id].parent
            if pd.isna(parent_id):
                return None
            else:
                return Topic(parent_id)

        @property
        def ancestors(self):
            ancestors = []
            parent = self.parent
            while parent is not None:
                ancestors.append(parent)
                parent = parent.parent
            return ancestors

        @property
        def siblings(self):
            if not self.parent:
                return []
            else:
                return [topic for topic in self.parent.children if topic != self]

        @property
        def content(self):
            if self.id in correlations_df.index:
                return [ContentItem(content_id) for content_id in correlations_df.loc[self.id].content_ids.split()]
            else:
                return tuple([]) if self.has_content else []

        def get_breadcrumbs(self, separator=" >> ", include_self=True, include_root=True):
            ancestors = self.ancestors
            if include_self:
                ancestors = [self] + ancestors
            if not include_root:
                ancestors = ancestors[:-1]
            return separator.join(reversed([a.title for a in ancestors]))

        @property
        def children(self):
            return [Topic(child_id) for child_id in topics_df[topics_df.parent == self.id].index]

        def subtree_markdown(self, depth=0):
            markdown = "  " * depth + "- " + self.title + "\n"
            for child in self.children:
                markdown += child.subtree_markdown(depth=depth + 1)
            for content in self.content:
                markdown += ("  " * (depth + 1) + "- " + "[" + content.kind.title() + "] " + content.title) + "\n"
            return markdown

        def __eq__(self, other):
            if not isinstance(other, Topic):
                return False
            return self.id == other.id

        def __getattr__(self, name):
            return topics_df.loc[self.id][name]

        def __str__(self):
            return self.title

        def __repr__(self):
            return f"<Topic(id={self.id}, title=\"{self.title}\")>"


    class ContentItem:
        def __init__(self, content_id):
            self.id = content_id

        @property
        def topics(self):
            return [Topic(topic_id) for topic_id in topics_df.loc[correlations_df[correlations_df.content_ids.str.contains(self.id)].index].index]

        def __getattr__(self, name):
            return content_df.loc[self.id][name]

        def __str__(self):
            return self.title

        def __repr__(self):
            return f"<ContentItem(id={self.id}, title=\"{self.title}\")>"

        def __eq__(self, other):
            if not isinstance(other, ContentItem):
                return False
            return self.id == other.id

        def get_all_breadcrumbs(self, separator=" >> ", include_root=True):
            breadcrumbs = []
            for topic in self.topics:
                new_breadcrumb = topic.get_breadcrumbs(separator=separator, include_root=include_root)
                if new_breadcrumb:
                    new_breadcrumb = new_breadcrumb + separator + self.title
                else:
                    new_breadcrumb = self.title
                breadcrumbs.append(new_breadcrumb)
            return breadcrumbs
        
    data_dir = Path("/content/learning-equality-curriculum-recommendations/")
    topics_df = pd.read_csv(data_dir / "topics.csv").fillna({"title": "", "description": ""})
    content_df = pd.read_csv(data_dir / "content.csv", index_col=0).fillna("")
    #sample_submission = pd.read_csv(data_dir / 'sample_submission.csv')
    # Merge topics with sample submission to only infer test topics
    #topics = topics_df.merge(sample_submission, how = 'inner', left_on = 'id', right_on = 'topic_id').set_index('id')
    topics = topics_df.set_index('id').copy()
    topics_df = topics_df.set_index('id')

    topic_id_texts = []
    content_id_texts = []
    for topic_idx in tqdm(topics.index):
        tmp_topic = Topic(topic_idx)
        children = tmp_topic.children
        child = "" if len(children)==0 else children[0].description

        parent = tmp_topic.parent
        par = "" if parent is None else parent.description
        topic_repre = f"[{tmp_topic.language}, {tmp_topic.level}] {tmp_topic.title} {tmp_topic.description} {tmp_topic.get_breadcrumbs()} {child} {par}"

        topic_id_texts.append((topic_idx, topic_repre))

    for content_idx in tqdm(content_df.index):
        ct = ContentItem(content_idx)
        content_repre = f"{ct.title} {ct.description} {ct.text}"
        content_id_texts.append((content_idx, content_repre))

    topics = pd.DataFrame(data={'id':[item[0] for item in topic_id_texts], 
                             'title':[item[1] for item in topic_id_texts]})
    content = pd.DataFrame(data={'id':[item[0] for item in content_id_texts], 
                                 'title':[item[1] for item in content_id_texts]})

    del topics_df, content_df, topic_id_texts, content_id_texts
    gc.collect()

    print(' ')
    print('-' * 50)
    print(f"topics.shape: {topics.shape}")
    print(f"content.shape: {content.shape}")

    
    return topics, content
topics, content = read_data()

topics.to_pickle('topics.pkl')
content.to_pickle('content.pkl')
from pathlib import Path

import os
import gc
import time
import math
import random
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.checkpoint import checkpoint
import tokenizers
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_cosine_schedule_with_warmup, DataCollatorWithPadding
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================================================================================
# Configurations
# =========================================================================================
class CFG:
    num_workers = 8
    model = '/content/drive/MyDrive/LECR/xlm-roberta-large-exp_fold2_epochs20'
    tokenizer = AutoTokenizer.from_pretrained(model)
    batch_size = 256
    top_n = 20
    seed = 42
    
# =========================================================================================
# Data Loading
# =========================================================================================
def read_data(cfg):
    topics = pd.read_pickle('topics.pkl')
    content = pd.read_pickle('content.pkl')
    correlations = pd.read_csv('/content/exp/kfold_correlations.csv')
    correlations = correlations[correlations.fold == 2]

    topics['title'].fillna("", inplace = True)
    content['title'].fillna("", inplace = True)
    # Sort by title length to make inference faster
    topics['length'] = topics['title'].apply(lambda x: len(x))
    content['length'] = content['title'].apply(lambda x: len(x))
    topics.sort_values('length', inplace = True)
    content.sort_values('length', inplace = True)
    # Drop cols
    # Reset index
    topics.reset_index(drop = True, inplace = True)
    content.reset_index(drop = True, inplace = True)
    print(' ')
    print('-' * 50)
    print(f"topics.shape: {topics.shape}")
    print(f"content.shape: {content.shape}")
    print(f"correlations.shape: {correlations.shape}")
    return topics, content, correlations

# =========================================================================================
# Prepare input, tokenize
# =========================================================================================
def prepare_input(text, cfg):
    inputs = cfg.tokenizer.encode_plus(
        text, 
        max_length = 128,
        truncation=True,
        return_tensors = None, 
        add_special_tokens = True, 
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype = torch.long)
    return inputs

# =========================================================================================
# Unsupervised dataset
# =========================================================================================
class uns_dataset(Dataset):
    def __init__(self, df, cfg):
        self.cfg = cfg
        self.texts = df['title'].values
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, item):
        inputs = prepare_input(self.texts[item], self.cfg)
        return inputs
    
# =========================================================================================
# Mean pooling class
# =========================================================================================
class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

# =========================================================================================
# Unsupervised model
# =========================================================================================
class uns_model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.config = AutoConfig.from_pretrained(cfg.model)
        self.model = AutoModel.from_pretrained(cfg.model, config = self.config)
        self.pool = MeanPooling()
    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        feature = self.pool(last_hidden_state, inputs['attention_mask'])
        return feature
    def forward(self, inputs):
        feature = self.feature(inputs)
        return feature
    
# =========================================================================================
# Get embeddings
# =========================================================================================
def get_embeddings(loader, model, device):
    model.eval()
    preds = []
    for step, inputs in enumerate(tqdm(loader)):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
        preds.append(y_preds.to('cpu').numpy())
    preds = np.concatenate(preds)
    return preds

# =========================================================================================
# Get the amount of positive classes based on the total
# =========================================================================================
def get_pos_score(y_true, y_pred):
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
    int_true = np.array([len(x[0] & x[1]) / len(x[0]) for x in zip(y_true, y_pred)])
    return round(np.mean(int_true), 5)

# =========================================================================================
# F2 Score 
def f2_score(y_true, y_pred):
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
    tp = np.array([len(x[0] & x[1]) for x in zip(y_true, y_pred)])
    fp = np.array([len(x[1] - x[0]) for x in zip(y_true, y_pred)])
    fn = np.array([len(x[0] - x[1]) for x in zip(y_true, y_pred)])
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f2 = tp / (tp + 0.2 * fp + 0.8 * fn)
    return round(f2.mean(), 4)
# ===========================================================================================


# =========================================================================================
# Build our training set
# =========================================================================================
def build_training_set(topics, content, cfg):
    # Create lists for training
    topics_ids = []
    content_ids = []
    title1 = []
    title2 = []
    targets = []
    folds = []
    # Iterate over each topic
    for k in tqdm(range(len(topics))):
        row = topics.iloc[k]
        topics_id = row['id']
        topics_title = row['title']
        predictions = row['predictions'].split(' ')
        ground_truth = row['content_ids'].split(' ')
        fold = row['fold']
        for pred in predictions:
            content_title = content.loc[pred, 'title']
            topics_ids.append(topics_id)
            content_ids.append(pred)
            title1.append(topics_title)
            title2.append(content_title)
            folds.append(fold)
            # If pred is in ground truth, 1 else 0
            if pred in ground_truth:
                targets.append(1)
            else:
                targets.append(0)
    # Build training dataset
    train = pd.DataFrame(
        {'topics_ids': topics_ids, 
         'content_ids': content_ids, 
         'title1': title1, 
         'title2': title2, 
         'target': targets,
         'fold' : folds}
    )
    # Release memory
    del topics_ids, content_ids, title1, title2, targets
    gc.collect()
    return train
    
# =========================================================================================
# Get neighbors
# =========================================================================================
def get_neighbors(topics, content, cfg):
    # Create topics dataset
    topics_dataset = uns_dataset(topics, cfg)
    # Create content dataset
    content_dataset = uns_dataset(content, cfg)
    # Create topics and content dataloaders
    topics_loader = DataLoader(
        topics_dataset, 
        batch_size = cfg.batch_size, 
        shuffle = False, 
        collate_fn = DataCollatorWithPadding(tokenizer = cfg.tokenizer, padding = 'longest'),
        num_workers = cfg.num_workers, 
        pin_memory = False, 
        drop_last = False
    )
    content_loader = DataLoader(
        content_dataset, 
        batch_size = cfg.batch_size, 
        shuffle = False, 
        collate_fn = DataCollatorWithPadding(tokenizer = cfg.tokenizer, padding = 'longest'),
        num_workers = cfg.num_workers, 
        pin_memory = False, 
        drop_last = False
        )
    # Create unsupervised model to extract embeddings
    model = uns_model(cfg)
    model.to(device)
    # Predict topics
    topics_preds = get_embeddings(topics_loader, model, device)
    content_preds = get_embeddings(content_loader, model, device)
    # Transfer predictions to gpu
    topics_preds_gpu = topics_preds#cp.array(topics_preds)
    content_preds_gpu = content_preds#cp.array(content_preds)
    # Release memory
    torch.cuda.empty_cache()
    del topics_dataset, content_dataset, topics_loader, content_loader, topics_preds, content_preds
    gc.collect()
    # KNN model
    print(' ')
    print('Training KNN model...')
    neighbors_model = NearestNeighbors(n_neighbors = cfg.top_n, metric = 'cosine')
    neighbors_model.fit(content_preds_gpu)
    indices = neighbors_model.kneighbors(topics_preds_gpu, return_distance = False)
    predictions = []
    for k in tqdm(range(len(indices))):
        pred = indices[k]
        p = ' '.join([content.loc[ind, 'id'] for ind in pred])
        predictions.append(p)
    topics['predictions'] = predictions
    # Release memory
    del topics_preds_gpu, content_preds_gpu, neighbors_model, predictions, indices, model
    gc.collect()
    return topics, content
     
# Read data
topics, content, correlations = read_data(CFG)
# Run nearest neighbors
topics, content = get_neighbors(topics, content, CFG)
# Merge with target and comput max positive score
topics_test = topics.merge(correlations, how = 'inner', left_on = ['id'], right_on = ['topic_id'])
pos_score = get_pos_score(topics_test['content_ids'], topics_test['predictions'])
print(f'Our max positive score is {pos_score}')

f_score = f2_score(topics_test['content_ids'], topics_test['predictions'])
print(f'Our f2_score is {f_score}')
# We can delete correlations
del correlations
gc.collect()
# Set id as index for content
content.set_index('id', inplace = True)
# Build training set
full_correlations = pd.read_csv('/content/exp/kfold_correlations.csv')
topics_full = topics.merge(full_correlations, how = 'inner', left_on = ['id'], right_on = ['topic_id'])
topics_full['predictions'] = topics_full.apply(lambda x: ' '.join(list(set(x.predictions.split(' ') + x.content_ids.split(' ')))) \
                                               if x.fold != 2 else x.predictions, axis = 1)
train = build_training_set(topics_full, content, CFG)
print(f'Our training set has {len(train)} rows')
# Save train set to disk to train on another notebook
#train.to_csv(f'/content/exp/train_top{CFG.top_n}_fold4_cv_with_groundtruth_final.csv', index = False)
#train.head()
topics = pd.read_csv('/content/learning-equality-curriculum-recommendations/topics.csv', usecols=['id', 'language'])
content = pd.read_csv('/content/learning-equality-curriculum-recommendations/content.csv', usecols=['id', 'language'])
train['topics_lang'] = train['topics_ids'].map(topics.set_index(['id'])['language'])
train['content_lang'] = train['content_ids'].map(content.set_index(['id'])['language'])

filter_train = train[train['topics_lang']==train['content_lang']]
print(f'Our filter training set has {len(filter_train)} rows')
filter_train.drop(columns=['topics_lang', 'content_lang'], axis=1).to_csv(f'/content/exp/train_top{CFG.top_n}_fold2_cv_with_groundtruth_final_filter.csv', index = False)
