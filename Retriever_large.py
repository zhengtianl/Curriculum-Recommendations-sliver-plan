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
#topics = pd.read_pickle('topics.pkl')
#content = pd.read_pickle('content.pkl')
#topics.to_pickle('topics.pkl')
#content.to_pickle('content.pkl')
#
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, models, InputExample, losses
from datasets import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
DATA_PATH = "/content/learning-equality-curriculum-recommendations/"
#topics = pd.read_csv(DATA_PATH + "topics.csv")
#content = pd.read_csv(DATA_PATH + "content.csv")

#topics['title'].fillna("", inplace = True)
#content['title'].fillna("", inplace = True)
# Fillna descriptions
#topics['description'].fillna("", inplace = True)
#content['description'].fillna("", inplace = True)
#content['text'].fillna("", inplace = True)
#topics['title'] = topics['title'] + ' ' + topics['description']
#content['title'] = content['title'] + ' ' + content['description']+ ' ' + content['text']

correlations = pd.read_csv(DATA_PATH + "correlations.csv")

correlations.shape
def cv_split(train, n_folds, seed):
    kfold = KFold(n_splits = n_folds, shuffle = True, random_state = seed)
    for num, (train_index, val_index) in enumerate(kfold.split(train)):
        train.loc[val_index, 'fold'] = int(num)
    train['fold'] = train['fold'].astype(int)
    return train
kfolds = cv_split(correlations, 5, 42)
kfolds.to_csv('/content/exp/kfold_correlations.csv', index=False)


correlations = kfolds[kfolds.fold != 2]
correlations
topics.rename(columns=lambda x: "topic_" + x, inplace=True)
content.rename(columns=lambda x: "content_" + x, inplace=True)

correlations["content_id"] = correlations["content_ids"].str.split(" ")
corr = correlations.explode("content_id").drop(columns=["content_ids"])
corr
corr = corr.merge(topics, how="left", on="topic_id")
corr = corr.merge(content, how="left", on="content_id")
corr.head()
corr["set"] = corr[["topic_title", "content_title"]].values.tolist()
train_df = pd.DataFrame(corr["set"])
dataset = Dataset.from_pandas(train_df)
train_examples = []
train_data = dataset["set"]
n_examples = dataset.num_rows

for i in range(n_examples):
    example = train_data[i]
    if example[0] == None: #remove None
        print(example)
        continue        
    train_examples.append(InputExample(texts=[str(example[0]), str(example[1])]))

model = SentenceTransformer("xlm-roberta-large",)
model.max_seq_length=128
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=80,         num_workers = 8, 
        pin_memory = True,)
train_loss = losses.MultipleNegativesRankingLoss(model=model)
num_epochs = 20
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1) #10% of train data
model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=num_epochs,
          save_best_model = True,
          output_path='/content/drive/MyDrive/LECR/xlm-roberta-large-exp_fold2_epochs20',
          warmup_steps=warmup_steps,
          use_amp=True)
