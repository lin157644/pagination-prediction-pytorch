# %% [markdown]
#  # Pagination prediction

# %%
import json
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import pickle
import json
import re
import sys
from itertools import islice
import multiprocessing

CPU_COUNT = multiprocessing.cpu_count()
from dataclasses import astuple, dataclass

from urllib.parse import parse_qsl, urlsplit, unquote

import parsel
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer

from crf import CRF


# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from torchnlp.encoders.text import CharacterEncoder
from regex_delimiter_encoder import RegexDelimiterEncoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
assert torch.cuda.is_available(), "No GPU/CUDA is detected!"

torch.set_float32_matmul_precision('high')


# %%
# Import autopager
# sys.path.insert(0, "..")
from autopager import AUTOPAGER_LIMITS
from autopager.htmlutils import (
    get_link_href,
    get_link_text,
    get_text_around_selector_list,
)
from autopager.parserutils import (
    MyHTMLParser,
    TagParser,
    get_first_tag,
)
from autopager.storage import Storage
from autopager.utils import (
    get_domain,
    ngrams_wb,
    normalize,
    replace_digits,
)

parser = MyHTMLParser()
tagParser = TagParser()


# %%
from typing import Literal
from argparse import ArgumentParser

# %%
@dataclass(frozen=True)
class PPArgs:
    batch_size: int
    max_epochs: int
    early_stopping_patient: int
    precision: Literal['32-true', '16-mixed']
    sentence_model_name: str
    learning_rate: float
    cls_emb_dim: int
    cls_fc_dim: int
    query_emb_dim: int
    max_query_per_node: int
    ptag_emb_dim: int
    url_char_emb_dim: int
    url_word_emb_dim: int
    conv_filters: int
    filter_sizes: int
    url_fc_dim: int
    lstm_hidden_dim: int
    max_cls_query_per_node: int
    max_url_char_tok_per_node: int
    max_url_word_tok_per_node: int

arg_parser = ArgumentParser()
arg_parser.add_argument("--batch_size", type=int, default=2)
arg_parser.add_argument("--max_epochs", type=int, default=20)
arg_parser.add_argument("--early_stopping_patient", type=int, default=20)
arg_parser.add_argument("--precision", type=str, default='32-true')
arg_parser.add_argument("--sentence_model_name", type=str, default='sentence-transformers/distiluse-base-multilingual-cased-v2')

arg_parser.add_argument("--learning_rate", type=float, default=5e-4) # 0.0005
arg_parser.add_argument("--cls_emb_dim", type=int, default=32)
arg_parser.add_argument("--cls_fc_dim", type=int, default=64)
arg_parser.add_argument("--query_emb_dim", type=int, default=64)
arg_parser.add_argument("--max_query_per_node", type=int, default=32)
arg_parser.add_argument("--ptag_emb_dim", type=int, default=30)
arg_parser.add_argument("--url_char_emb_dim", type=int, default=32)
arg_parser.add_argument("--url_word_emb_dim", type=int, default=32)
arg_parser.add_argument("--conv_filters", type=int, default=64)
arg_parser.add_argument("--filter_sizes", type=json.loads, default="[3, 4, 5, 6]")
arg_parser.add_argument("--url_fc_dim", type=int, default=128)
arg_parser.add_argument("--lstm_hidden_dim", type=int, default=300)

arg_parser.add_argument("--max_cls_query_per_node", type=int, default=256)
arg_parser.add_argument("--max_url_char_tok_per_node", type=int, default=256)
arg_parser.add_argument("--max_url_word_tok_per_node", type=int, default=128)

# This may cause weird behavior!
args = PPArgs(**vars(arg_parser.parse_args(args=[])))

dict_args = vars(args)

# Model candidates
# ('sentence-transformers/paraphrase-multilingual-mpnet-base-v2') # 768 165K X
# ('sentence-transformers/stsb-xlm-r-multilingual') # 768 51K X
# ('sentence-transformers/distiluse-base-multilingual-cased-v2') # 512 40K ~.9
# ('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2') # 384 1.3M ~.83

sentence_model = SentenceTransformer(args.sentence_model_name)


SEED = 42
pl.seed_everything(SEED)

# %%
labels = ["O", "PREV", "PAGE", "NEXT"]
tag2idx = {label: idx for idx, label in enumerate(labels)}
idx2tag = {idx: label for idx, label in enumerate(labels)}
num_tags = len(labels)

# %%
def _as_list(generator, limit=None) -> list:
    """
    Generator to list with limit.
    """
    return list(generator if limit is None else islice(generator, 0, limit))

def link_to_features(link: parsel.Selector):
    # Get text contecnt of the link otherwise alt or img.
    # Normalize multiple white space to one and to lowercase.
    text = normalize(get_link_text(link))
    href = get_link_href(link)
    if href is None:
        href = ""
    p = urlsplit(href)
    parent = link.xpath('..').extract()
    # Retrive the line of first tag opening
    parent = get_first_tag(parser, parent[0])
    query_parsed = parse_qsl(p.query)  # parse query string from path
    query_param_names = [k.lower() for k, v in query_parsed]
    # TODO: change ngrams
    query_param_names_ngrams = _as_list(ngrams_wb(
        " ".join([normalize(name) for name in query_param_names]), 3, 5, True
    ))

    # Classes of link itself and all its children.
    # It is common to have e.g. span elements with fontawesome
    # arrow icon classes inside <a> links.
    self_and_children_classes = ' '.join(link.xpath(".//@class").extract())
    parent_classes = ' '.join(link.xpath('../@class').extract())
    css_classes = normalize(parent_classes + ' ' + self_and_children_classes)

    token_feature = {
        'text-exact': replace_digits(text.strip()[:100].strip()),
        # <scheme>://<netloc>/<path>?<query>#<fragment>
        # 'url': p.path + p.query,
        'url': href if href else "https://",
        'query': query_param_names_ngrams,
        'parent-tag': parent,
        'class': _as_list(ngrams_wb(css_classes, 4, 5),
                          AUTOPAGER_LIMITS.max_css_features),
        'text': _as_list(ngrams_wb(replace_digits(text), 2, 5),
                         AUTOPAGER_LIMITS.max_text_features),
    }
    tag_feature = {
        'isdigit': 1 if text.isdigit() is True else 0,
        'isalpha': 1 if text.isalpha() is True else 0,
        'has-href': 0 if href == "" else 1,
        'path-has-page': 1 if 'page' in p.path.lower() else 0,
        'path-has-pageXX': 1 if re.search(r'[/-](?:p|page\w?)/?\d+', p.path.lower()) is not None else 0,
        'path-has-number': 1 if any(part.isdigit() for part in p.path.split('/')) else 0,
        'href-has-year': 1 if re.search('20\d\d', href) is not None else 0,
        'class-has-disabled': 1 if 'disabled' in css_classes else 0,
    }
    non_token_feature = []
    for k, v in tag_feature.items():
        if type(v) == type([]):
            non_token_feature.extend(v)
        else:
            non_token_feature.append(v)

    return [token_feature, non_token_feature]


def page_to_features(xseq):
    feat_list = [link_to_features(a) for a in xseq]
    around = get_text_around_selector_list(xseq, max_length=15)
    # Append sibling's text-exact to each node's text-full.
    for feat, (before, after) in zip(feat_list, around, strict=True):
        feat[0]['text-full'] = normalize(before) + ',' + feat[0]['text-exact'] + ',' + normalize(after)

    return feat_list



# %%
def get_token_tag_features_from_chunks(chunks):
    token_features = []
    tag_features = []
    for idx, page in enumerate(chunks):
        try:
            feat_list = page_to_features(page)
            token_features.append([node[0] for node in feat_list])
            tag_features.append(torch.tensor([node[1] for node in feat_list]))
        except:
            raise Exception(f"Error occured on {idx}")
    return token_features, tag_features

# %%
def sparse_representation_with_map(tag, data_map):
    # Vector length is the number of tags in the map(30).
    rt_vec = [0] * len(data_map)
    for idx, map_tag in enumerate(data_map):
        # ('tag_name', count)
        if tag == map_tag[0]:
            rt_vec[idx] = 1
            break
    return rt_vec


def get_ptags_vector(token_features, data_map_for_ptag: list[tuple[str, int]]):
    pages_ptag = []
    for page in token_features:
        ptag_page = []
        for node in page:
            p_tag = node['parent-tag']
            ptag_page.append(sparse_representation_with_map(p_tag, data_map_for_ptag))
        pages_ptag.append(torch.tensor(ptag_page, dtype=torch.float32))
    return pages_ptag

# %% [markdown]
#  ## Load data

# %%
class TagTokenizer:
    def __init__(self, tag_name_count=None):
        rt_dict = {}
        rt_dict["[PAD]"] = 0
        rt_dict["[UNK]"] = 1
        # TODO: Sort by count, although embedding layer should compensate this
        if tag_name_count is not None:
            for k in tag_name_count.keys():
                rt_dict[k] = len(rt_dict)
        self.map = rt_dict

    def tokenize(self, word: list[str] | str):
        if isinstance(word, list):
            token_list = []
            for _word in word:
                if _word not in self.map:
                    token_list.append(self.map["[UNK]"])
                else:
                    token_list.append(self.map[_word])
            return token_list
        else:
            if word not in self.map:
                return self.map["[UNK]"]
            else:
                return self.map[word]

    def get_size(self):
        return len(self.map)

# %%
def pages_to_word_vector(token_features) -> list[torch.Tensor]:
    print(f"Transform text-full to word_vector ... ")
    # TODO: Do not use torch.tensor() to fix type hint
    return [sentence_model.encode([node['text-full'] for node in page], convert_to_tensor=True) for page in
            token_features]

# %%
storage = Storage()

# %%
urls = [
    rec["Page URL"]
    for rec in storage.iter_records(
        language=None, contain_button=True, file_type="T"
    )
]
X_raw: list[parsel.SelectorList]
y_raw: list[str]
X_raw, y_raw, page_positions = storage.get_Xy(
    language=None,
    contain_button=True,
    contain_position=True,
    file_type="T",
    scaled_page="normal",
)
print(
    "pages: {}  domains: {}".format(
        len(urls), len({get_domain(url) for url in urls})
    )
)

# %%
token_features: list[list[dict]]
# x_tag: features which only have tag true/false information
# token_features: ['text-exact', 'query', 'parent-tag', 'class', 'text', 'text-full']
token_features, x_tag = get_token_tag_features_from_chunks(X_raw)
token_feature_titles: list[str] = list(token_features[0][0].keys())

# %%
class_token_map = {}
query_token_map = {}

for page in token_features:
    for node in page:
        for class_name in node['class']:
            class_token_map[class_name] = class_token_map.get(class_name, 0) + 1
        for query_name in node['query']:
            query_token_map[query_name] = query_token_map.get(query_name, 0) + 1

class_tokenizer = TagTokenizer(class_token_map)
query_tokenizer = TagTokenizer(query_token_map)
CLS_VOCAB_SIZE = class_tokenizer.get_size()
QUERY_VOCAB_SIZE = query_tokenizer.get_size()


# %%
def get_class_query_ids(page_tokens, max_len):
    pages_class = []
    pages_query = []
    for page in page_tokens:
        class_page = []
        query_page = []
        for node in page:
            # class
            class_ids = class_tokenizer.tokenize(node['class'])
            class_ids = class_ids + [0] * (max_len - len(class_ids))
            class_page.append(class_ids[:max_len])
            # query
            query_ids = query_tokenizer.tokenize(node['query'])
            query_ids = query_ids + [0] * (max_len - len(query_ids))
            query_page.append(query_ids[:max_len])
        pages_class.append(torch.tensor(class_page))
        pages_query.append(torch.tensor(query_page))
    return pages_class, pages_query

# %%
top_parent_tags = {}

for page in token_features:
    for node in page:
        p_tag = node['parent-tag']
        if p_tag not in top_parent_tags:
            top_parent_tags[p_tag] = 1
        else:
            top_parent_tags[p_tag] += 1

sorted_parent_tags = sorted(top_parent_tags.items(), key=lambda x: x[1], reverse=True)

# %%
urls_full = []

for page in token_features:
    for node in page:
        urls_full.append(node['url'])

url_char_tokenizer = CharacterEncoder(urls_full)
url_word_tokenizer = RegexDelimiterEncoder(r"\/|&|\?|#|\.|://|=|-|[\ ]", urls_full)

# %%
@dataclass
class PAGE_X:
    text: torch.Tensor
    ptag: torch.Tensor
    cls: torch.Tensor
    query: torch.Tensor
    url_char: torch.Tensor
    url_word: torch.Tensor
    tag: torch.Tensor

    def __iter__(self):
        return iter(astuple(self))

# %%
def get_input_from_raw(x_raw, y_raw=None, token_features=None, x_tag=None) -> tuple[list[PAGE_X], list[torch.Tensor]]:
    if token_features is None or x_tag is None:
        token_features, x_tag = get_token_tag_features_from_chunks(x_raw)

    x_text: list[torch.Tensor] = pages_to_word_vector(token_features)

    x_ptag: list[torch.Tensor] = get_ptags_vector(token_features, sorted_parent_tags[slice(None, args.ptag_emb_dim)])  # type: ignore

    x_class, x_query = get_class_query_ids(
        token_features, max_len=args.max_cls_query_per_node
    )

    # (pages, nodes)
    x_url_char_list = []
    x_url_word_list = []
    for page in token_features:
        page_url_char = []
        page_url_word = []
        for node in page:
            url_char = url_char_tokenizer.encode(unquote(node["url"]))
            url_char = F.pad(url_char, (0, args.max_url_char_tok_per_node - len(url_char)))
            url_char = url_char[slice(None, args.max_url_char_tok_per_node)].long()
            page_url_char.append(url_char)

            url_word = url_word_tokenizer.encode(unquote(node["url"]))
            url_word = F.pad(url_word, (0, args.max_url_word_tok_per_node - len(url_word)))
            url_word = url_word[slice(None, args.max_url_word_tok_per_node)].long()
            page_url_word.append(url_word)
        x_url_char_list.append(torch.stack(page_url_char))
        x_url_word_list.append(torch.stack(page_url_word))

    x = [PAGE_X(*x) for x in zip(x_text, x_ptag, x_class, x_query, x_url_char_list, x_url_word_list, x_tag, strict=True)]

    if y_raw is not None:
        y: list[torch.Tensor] = [
            torch.tensor([tag2idx.get(l, 0) for l in lab]) for lab in y_raw
        ]
    else:
        y = []

    return x, y


def get_test_data(test_type=None, scaled_page='normal'):
    if test_type is None:
        print("Please assign type of test_data")
        return (None, None, None)
    test_X_one = []
    test_X_two = []
    test_y_one = []
    test_y_two = []
    test_page_positions_one = []
    test_page_positions_two = []
    if test_type != 'EVENT_SOURCE':
        storage.test_file = 'NORMAL'
        test_urls = [rec['Page URL'] for rec in storage.iter_test_records(exclude_en=None)]
        test_X_one, test_y_one, test_page_positions_one = storage.get_test_Xy(validate=False, contain_position=True,
                                                                              scaled_page=scaled_page, exclude_en=None)
        print("pages: {}  domains: {}".format(len(test_urls), len({get_domain(url) for url in test_urls})))
        if test_type == 'NORMAL':
            return test_X_one, test_y_one, test_page_positions_one
    if test_type != 'NORMAL':
        storage.test_file = 'EVENT_SOURCE'
        test_urls = [rec['Page URL'] for rec in storage.iter_test_records(exclude_en=None)]
        test_X_two, test_y_two, test_page_positions_two = storage.get_test_Xy(validate=False, contain_position=True,
                                                                              scaled_page=scaled_page, exclude_en=None)
        print("pages: {}  domains: {}".format(len(test_urls), len({get_domain(url) for url in test_urls})))
        if test_type == 'EVENT_SOURCE':
            return test_X_two, test_y_two, test_page_positions_two
    test_X_raw = test_X_one + test_X_two
    test_y = test_y_one + test_y_two
    test_positions = test_page_positions_one + test_page_positions_two
    return test_X_raw, test_y, test_positions

# %%
x_train, y_train = get_input_from_raw(X_raw, y_raw, token_features)

# %%
x_val_raw: list[parsel.selector.SelectorList]
x_val_raw, y_val_raw = storage.get_test_Xy_by_language(language='event', contain_button=True)
x_val, y_val = get_input_from_raw(x_val_raw, y_val_raw)

# %%
test_x_raw, test_y_raw, test_page_positions = get_test_data('EVENT_SOURCE')
x_test, y_test = get_input_from_raw(test_x_raw, test_y_raw)

# %%
def get_mask(lens: torch.Tensor, max_len: int) -> torch.Tensor:
    return torch.arange(max_len)[None, :] < lens[:, None]

# %%
class PPDataset(Dataset):
    def __init__(self, x: list[PAGE_X], y: list[torch.Tensor]) -> None:
        self.x = x
        self.y = y

    def __len__(self):
        # assert all([len(feat) == first_len for feat in self.comp_data])
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# %%
def pad_collect(batch: list[tuple]):
    # [ (x, y), (x, y), ...]
    xx: tuple[PAGE_X]
    yy: tuple[torch.Tensor]
    (xx, yy) = zip(*batch, strict=True)

    # xx_pad: tuple[text, ptag, cls, query, url_char, url_word, tag]
    xx_pad = []

    # (text, ptag, cls, query, url_char, url_word, tag) = zip(*xx)
    for feature in zip(*xx, strict=True):
        # feature: tuple[torch.Tensor]
        # xx_pad.append(torch.nn.utils.rnn.pack_sequence(feature))
        xx_pad.append(pad_sequence(list(feature), batch_first=True, padding_value=0))

    x_lens = [x.text.shape[0] for x in xx]
    y_lens = [len(y) for y in yy]

    try:
        yy_pad = pad_sequence(list(yy), batch_first=True, padding_value=0)
    except:
        print(f'Exception when padding y: {yy}')
        raise

    return xx_pad, yy_pad, x_lens, y_lens



# %%
class PPDataModule(pl.LightningDataModule):
    def __init__(self, train_x, train_y, x_val, y_val, x_test, y_test, batch_size: int = 2):
        super().__init__()
        self.train_x = train_x
        self.train_y = train_y
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        return super().prepare_data()

    def setup(self, stage: str):
        if stage == "fit":
            # self.train_dataset, self.val_dataset = random_split(PPDataset(self.train_x, self.train_y), [0.85, 0.15])
            self.train_dataset = PPDataset(self.train_x, self.train_y)
            self.val_dataset = PPDataset(self.x_val, self.y_val)
        if stage == "test":
            self.test_dataset = PPDataset(self.x_test, self.y_test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0,
                          collate_fn=pad_collect, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0,
                          collate_fn=pad_collect, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=pad_collect)

    # def teardown(self, stage: str):
    #     # Used to clean-up when the run is finished
    #     ...

# %%
class PPModule(pl.LightningModule):    
    def __init__(
        self,
        *,
        learning_rate: float,
        max_cls_query_per_node: int,
        max_url_char_tok_per_node: int,
        max_url_word_tok_per_node: int,
        text_emb_dim: int,
        cls_emb_dim: int,
        cls_fc_dim: int,
        query_emb_dim: int,
        ptag_emb_dim: int,
        max_query_per_node: int,
        url_char_emb_dim: int,
        url_word_emb_dim: int,
        conv_filters: int,
        filter_sizes: list[int],
        url_fc_dim: int,
        lstm_hidden_dim: int,
        cls_vocab_size: int,
        query_vocab_size: int,
        url_char_vocab_size: int,
        url_word_vocab_size: int,
        **kwargs,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()
        self.hparams: PPArgs

        # TODO: Embed hyperparameters in module

        self.relu = nn.ReLU()

        # [BATCH, NODES, MAX_CLS_QUERY_INPUT]
        # We use 0 to pad to 256 class/query per node
        self.cls_emb_layer = nn.Embedding(
            num_embeddings=self.hparams.cls_vocab_size,
            embedding_dim=self.hparams.cls_emb_dim,
            padding_idx=0,
        )
        # [BATCH, NODES, MAX_CLS_QUERY_PER_NODE, CLS_EMB_DIM]
        self.cls_emb_pool_layer = nn.MaxPool2d(
            (self.hparams.max_cls_query_per_node, 1), 1, padding=(0, 0)
        )

        self.cls_conv_layer_1 = nn.Conv2d(
            self.hparams.cls_emb_dim,
            self.hparams.cls_emb_dim * 2,
            kernel_size=3,
            stride=1,
            padding="same",
        )
        self.cls_conv_layer_2 = nn.Conv2d(
            self.hparams.cls_emb_dim,
            self.hparams.cls_emb_dim * 2,
            kernel_size=5,
            stride=1,
            padding="same",
        )
        # TODO: Calculate padding according to MAX_CLS_QUERY_PER_NODE and CLS_EMB_DIM
        self.cls_pool_layer_1 = nn.MaxPool2d(3, stride=2, padding=(1, 1))
        self.cls_pool_layer_2 = nn.MaxPool2d(3, stride=2, padding=(1, 1))

        self.cls_linear_layer = nn.Linear(
            self.hparams.max_cls_query_per_node // 2 * self.hparams.cls_emb_dim
            + self.hparams.max_cls_query_per_node // 4 * self.hparams.cls_emb_dim
            + self.hparams.cls_emb_dim,
            self.hparams.cls_fc_dim,
        )

        self.ptag_linear = nn.Linear(30, 30)

        self.query_emb_layer = nn.Embedding(
            self.hparams.query_vocab_size, self.hparams.query_emb_dim, padding_idx=0
        )
        self.query_emb_pool_layer = nn.MaxPool2d(
            (self.hparams.max_cls_query_per_node, 1), stride=1
        )
        # TODO: Add query fc layer

        # in: (BATCH, #LINKS, #TOKENS) out: (BATCH, #LINKS, #TOKENS, EMBEDDING_SIZE)
        self.url_char_emb_layer = nn.Embedding(
            self.hparams.url_char_vocab_size,
            self.hparams.url_char_emb_dim,
            padding_idx=0,
        )
        self.url_word_emb_layer = nn.Embedding(
            self.hparams.url_word_vocab_size,
            self.hparams.url_word_emb_dim,
            padding_idx=0,
        )

        self.num_filters_total = self.hparams.conv_filters * len(self.hparams.filter_sizes)

        self.url_char_convs = nn.ModuleList(
            [
                nn.Conv2d(
                    1,
                    self.hparams.conv_filters,
                    (filter_size, self.hparams.url_char_emb_dim),
                    stride=1,
                    padding="valid",
                    # bias=True,
                )
                for filter_size in self.hparams.filter_sizes
            ]
        )

        self.url_char_pools = nn.ModuleList(
            [
                nn.MaxPool2d(
                    kernel_size=(
                        self.hparams.max_url_char_tok_per_node - filter_size + 1,
                        1,
                    ),
                    stride=(1, 1),
                )
                for filter_size in self.hparams.filter_sizes
            ]
        )

        self.url_char_linear_layer = nn.Linear(self.num_filters_total, 512)

        self.url_word_convs = nn.ModuleList(
            [
                nn.Conv2d(
                    1,
                    self.hparams.conv_filters,
                    (filter_size, self.hparams.url_word_emb_dim),
                    stride=1,
                    padding="valid",
                    # bias=True,
                )
                for filter_size in self.hparams.filter_sizes
            ]
        )
        self.url_word_pools = nn.ModuleList(
            [
                nn.MaxPool2d(
                    kernel_size=(
                        self.hparams.max_url_word_tok_per_node - filter_size + 1,
                        1,
                    ),
                    stride=(1, 1),
                )
                for filter_size in self.hparams.filter_sizes
            ]
        )

        self.url_word_linear_layer = nn.Linear(self.num_filters_total, 512)

        # url_char_linear_layer+url_word_linear_layer
        self.url_linear_layer_1 = nn.Linear(1024, 512)
        self.url_linear_layer_2 = nn.Linear(512, 256)
        self.url_linear_layer_3 = nn.Linear(256, self.hparams.url_fc_dim)

        # BiLSTM-CRF
        # Ptag embedding: 30
        self.embedding_dim = (
            self.hparams.text_emb_dim
            + self.hparams.ptag_emb_dim
            + self.hparams.cls_fc_dim
            + self.hparams.max_cls_query_per_node // 4 * self.hparams.cls_emb_dim
            + self.hparams.query_emb_dim
            + self.hparams.url_fc_dim
            + 8
        )

        self.tag2idx = tag2idx
        self.num_tags = len(tag2idx)

        self.lstm = nn.LSTM(
            self.embedding_dim,
            self.hparams.lstm_hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        # O PREV PAGE NEXT
        self.crf: CRF = CRF(tagset_size=len(tag2idx), gpu=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(self.hparams.lstm_hidden_dim, self.num_tags + 2)

        # TODO: Test random or zero yield better result
        self.hidden = self.init_hidden(2)

        # self.hparams.some_layer_dim
        self.test_predictions = []
        self.test_label = []

    def init_hidden(self, batch_size):
        return (
            torch.randn(2, batch_size, self.hparams.lstm_hidden_dim // 2).to(device),
            torch.randn(2, batch_size, self.hparams.lstm_hidden_dim // 2).to(device),
        )

    def _get_lstm_features(self, x, x_lens: list[int]):
        x_text, x_ptag, x_class, x_query, x_url_char, x_url_word, x_tag = x

        del self.hidden
        self.hidden = self.init_hidden(x_text.shape[0])
        # lstm_hidden_h_0 = torch.randn(2, B, LSTM_HIDDEN_DIM // 2).to(device)
        # lstm_hidden_c_0 = torch.randn(2, B, LSTM_HIDDEN_DIM // 2).to(device)

        # batch_size = x_text.shape[0]
        # assert all(batch_size ==
        #            feat.shape[0] for feat in x)

        # x_class:
        # (B, NODES, MAX_CLS_QUERY_PER_NODE)
        class_emb = self.cls_emb_layer(x_class)
        # (B, NODES, MAX_CLS_QUERY_PER_NODE, CLS_EMB_DIM)
        class_emb_max_pool = self.cls_emb_pool_layer(class_emb)
        # (B, NODES, 1, CLS_EMB_DIM)
        class_emb_max_pool = class_emb_max_pool.squeeze(dim=2)
        # (B, NODES, CLS_EMB_DIM)

        # N, Cin, H, W = class_emb.shape

        # class_emb.shape # (B, NODES, MAX_CLS_QUERY_PER_NODE, CLS_EMB_DIM)
        class_emb = torch.permute(class_emb, (0, 3, 1, 2))
        # class_emb.shape # (B, CLS_EMB_DIM, NODES, MAX_CLS_QUERY_PER_NODE)

        class_conv_1 = self.cls_conv_layer_1(class_emb)

        # class_conv_1.shape # (B, CLS_EMB_DIM*2, NODES, MAX_CLS_QUERY_PER_NODE)
        class_conv_1 = torch.permute(class_conv_1, (0, 2, 3, 1))
        # class_conv_1.shape # (B, NODES, MAX_CLS_QUERY_PER_NODE, CLS_EMB_DIM*2)
        class_conv_1 = self.cls_pool_layer_1(class_conv_1)
        # class_conv_1.shape # (B, NODES, MAX_CLS_QUERY_PER_NODE/2, CLS_EMB_DIM)

        class_conv_2 = torch.permute(class_conv_1, (0, 3, 1, 2))
        # class_conv_2.shape # (B, CLS_EMB_DIM, NODES, MAX_CLS_QUERY_PER_NODE/2)
        class_conv_2 = self.cls_conv_layer_2(class_conv_2)
        # class_conv_2.shape # (B, CLS_EMB_DIM*2, NODES, MAX_CLS_QUERY_PER_NODE/2)
        class_conv_2 = torch.permute(class_conv_2, (0, 2, 3, 1))
        # class_conv_2.shape # (B, NODES, MAX_CLS_QUERY_PER_NODE/2, CLS_EMB_DIM*2)
        class_conv_2 = self.cls_pool_layer_2(class_conv_2)
        # (B, NODES, MAX_CLS_QUERY_PER_NODE/4, CLS_EMB_DIM)

        class_conv_2_flat = torch.flatten(class_conv_2, start_dim=2, end_dim=3)

        class_concat = torch.cat(
            (
                class_emb_max_pool,
                torch.flatten(class_conv_1, 2),
                torch.flatten(class_conv_2, 2),
            ),
            dim=2,
        )
        # class_concat.shape
        cls_emb = self.cls_linear_layer(class_concat)
        cls_emb = self.relu(cls_emb)

        # ptag_feat = x_ptag
        ptag_feat = self.ptag_linear(x_ptag)
        ptag_feat = self.relu(ptag_feat)

        # (B, NODES, MAX_CLS_QUERY_PER_NODE)
        query_emb = self.query_emb_layer(x_query)
        # (B, NODES, MAX_CLS_QUERY_PER_NODE, QUERY_EMB_DIM)
        # (B, NODES, 256, 64)
        query_emb = self.query_emb_pool_layer(query_emb)
        # (B, NODES, 1, QUERY_EMB_DIM)
        query_emb = query_emb.squeeze(2)

        #############################

        # (BATCH, NODES, MAX_URL_CHAR_LEN)
        url_char_emb = self.url_char_emb_layer(x_url_char)
        # (BATCH, NODES, MAX_URL_CHAR_LEN, URL_CHAR_EMBEDDING_SIZE)
        url_char_emb = torch.unsqueeze(url_char_emb, 2)
        # (BATCH, NODES, 1, MAX_URL_CHAR_LEN, URL_CHAR_EMBEDDING_SIZE)

        B, F, C, H, W = url_char_emb.shape
        url_char_emb = url_char_emb.view(-1, C, H, W)
        # (BATCH*NODES, 1, MAX_URL_CHAR_LEN, URL_CHAR_EMBEDDING_SIZE)

        pooled_char_x = []
        for conv, pool in zip(
            self.url_char_convs, self.url_char_pools, strict=True
        ):
            convolved = conv(url_char_emb)
            # (BATCH*NODES, CONV_FILTERS, MAX_URL_CHAR_LEN-filter_size+1, 1)
            convolved = self.relu(convolved)
            pooled = pool(convolved)
            # (BATCH*NODES, CONV_FILTERS, 1, 1)
            pooled_char_x.append(pooled)

        url_char_emb = torch.cat(pooled_char_x, dim=1)
        # (BATCH*NODES, num_filters_total, 1, 1)

        # Since torch.cat creates a copy we won't need it anymore
        # TODO: Check if this prevention of memory leak work
        del pooled_char_x

        url_char_emb = torch.squeeze(url_char_emb, dim=(-1, -2))
        # (BATCH*NODES, num_filters_total)

        url_char_emb = url_char_emb.reshape(B, F, -1)
        # (BATCH, NODES, num_filters_total)

        char_output = self.url_char_linear_layer(url_char_emb)
        # (BATCH, NODES, 512)
        char_output = self.relu(char_output)

        del B, F, C, H, W

        #############################

        url_word_emb = self.url_word_emb_layer(x_url_word)
        url_word_emb = url_word_emb.unsqueeze(2)

        B, F, C, H, W = url_word_emb.shape
        url_word_emb = url_word_emb.view(-1, C, H, W)

        pooled_word_x = []
        for conv, pool in zip(
            self.url_word_convs, self.url_word_pools, strict=True
        ):
            convolved = conv(url_word_emb)
            convolved = self.relu(convolved)
            pooled = pool(convolved)
            pooled_word_x.append(pooled)

        url_word_emb = torch.cat(pooled_word_x, dim=1)
        del pooled_word_x
        url_word_emb = url_word_emb.squeeze(dim=(-1, -2))

        url_word_emb = url_word_emb.reshape(B, F, -1)

        word_output = self.url_word_linear_layer(url_word_emb)
        word_output = self.relu(word_output)

        #############################

        conv_output = torch.cat((char_output, word_output), dim=2)
        url_emb = self.url_linear_layer_1(conv_output)
        url_emb = self.relu(url_emb)
        url_emb = self.url_linear_layer_2(url_emb)
        url_emb = self.relu(url_emb)
        url_emb = self.url_linear_layer_3(url_emb)
        url_emb = self.relu(url_emb)

        del B, F, C, H, W

        #############################

        # (BATCH, NODES, very_long)
        merged = torch.cat(
            (
                x_text,
                ptag_feat,
                class_conv_2_flat,
                cls_emb,
                query_emb,
                url_emb,
                x_tag,
            ),
            dim=2,
        )

        # print(f"{merged.shape}")

        packed_merged = torch.nn.utils.rnn.pack_padded_sequence(
            merged, x_lens, batch_first=True, enforce_sorted=False
        )

        # By default, PyTorch’s nn.LSTM module assumes the input to be sorted as [seq_len, batch_size, input_size].
        # TODO: Find better way to get # of nodes other then merged.shape[1]
        # merged = merged.view(merged.shape[1], batch_size, -1)
        lstm_out, self.hidden = self.lstm(packed_merged, self.hidden)
        # batch_first=True: NLDH (Batch_size, Sequence_length, 2, Hidden_size) (LSTM_HIDDEN_DIM//2)
        # lstm_out = lstm_out.view(merged.shape[0], merged.shape[1], LSTM_HIDDEN_DIM)

        seq_unpacked, lens_unpacked = torch.nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True, padding_value=0
        )

        lstm_feats = self.hidden2tag(seq_unpacked)

        return lstm_feats

    def loss(
        self,
        lstm_feats: torch.Tensor,
        y: torch.Tensor,
        x_lens: list[int],
        y_lens: list[int],
    ):
        # (B, NODES, LSTM_HIDDEN_DIM)
        # padded_lstm_feats = torch.nn.utils.rnn.pad_packed_sequence(lstm_feats, batch_first=True, padding_value=0)
        mask: torch.Tensor = get_mask(
            torch.Tensor(x_lens), lstm_feats.shape[1]
        ).to(device)
        loss = self.crf.neg_log_likelihood_loss(lstm_feats, mask, y)
        del mask

        return loss

    def forward(self, x, x_lens: list[int]):
        ## LSTM-CRF
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(x, x_lens)
        # padded_lstm_feats = torch.nn.utils.rnn.pad_packed_sequence(lstm_feats, batch_first=True, padding_value=0)

        # (BATCH, NODES, 300)
        mask: torch.Tensor = get_mask(
            torch.Tensor(x_lens), lstm_feats.shape[1]
        ).to(device)

        path_score, best_path = self.crf(lstm_feats, mask)
        # Transition Score
        del mask

        return best_path

    def training_step(self, batch, batch_idx):
        x, y, x_len, y_len = batch
        lstm_feats = self._get_lstm_features(x, x_len)
        loss = self.loss(lstm_feats, y, x_len, y_len)
        self.log("train_loss", loss, batch_size=x[0].shape[0], prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, x_len, y_len = batch
        lstm_feats = self._get_lstm_features(x, x_len)
        loss = self.loss(lstm_feats, y, x_len, y_len)
        self.log("val_loss", loss, batch_size=x[0].shape[0], prog_bar=True)

    def test_step(self, batch, batch_idx):
        # Note: Input and result are batched
        x, y, x_len, y_len = batch
        lstm_feats = self._get_lstm_features(x, x_len)
        loss = self.loss(lstm_feats, y, x_len, y_len)

        best_path = self(x, x_len)

        self.log("test_loss", loss, batch_size=x[0].shape[0])

        self.test_predictions.append(best_path)
        self.test_label.append(y)
        
    def on_test_epoch_start(self) -> None:
        del self.test_predictions
        del self.test_label
        self.test_predictions = []
        self.test_label = []

    def on_test_epoch_end(self) -> None:
        # Flatten the list to nodes
        test_predictions_flat = [
            node.cpu()
            for batch in self.test_predictions
            for page_list in batch
            for node in page_list
        ]

        test_label_flat = []
        for batch in self.test_label:
            test_label_flat.extend(batch.flatten().tolist())

        reports = classification_report(
            test_label_flat,
            test_predictions_flat,
            labels=[0, 1, 2, 3],
            target_names=["O", "PREV", "PAGE", "NEXT"],
            digits=3,
            output_dict=False,
            zero_division=1,
        )
        print(reports)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate
        )
        return optimizer
        # Maybe try a scheduler?
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, gamma=0.9)
        # return [optimizer], [scheduler]

# %%
prefix: str = 'pp_data/'
try:
    os.makedirs(prefix, exist_ok=True)
    with open(prefix + 'url_char_tokenizer.pickle', 'wb') as f:
        pickle.dump(url_char_tokenizer, f)
    with open(prefix + 'url_word_tokenizer.pickle', 'wb') as f:
        pickle.dump(url_word_tokenizer, f)
    with open(prefix + 'class_token_map.json', 'w') as f:
        f.write(json.dumps(class_tokenizer.map))
    with open(prefix + 'query_token_map.json', 'w') as f:
        f.write(json.dumps(query_tokenizer.map))
    with open(prefix + 'sorted_parent_tags.json', 'w') as f:
        f.write(json.dumps(sorted_parent_tags))
except:
    print(f"Export fail")

# %% [markdown]
# ## Run

# %% [markdown]
#  ### Get Class, Query

# %%
logger_path = 'tb_logs'
os.makedirs(logger_path, exist_ok=True)
logger = TensorBoardLogger(logger_path, name="pp_model")

# %%
# pl.seed_everything(SEED)

module: PPModule = PPModule(
    text_emb_dim=sentence_model.get_sentence_embedding_dimension(),
    cls_vocab_size=class_tokenizer.get_size(),
    query_vocab_size=query_tokenizer.get_size(),
    url_char_vocab_size=url_char_tokenizer.vocab_size,
    url_word_vocab_size=url_word_tokenizer.vocab_size,
    **dict_args
)



dm = PPDataModule(
    x_train, y_train, x_val, y_val, x_test, y_test, batch_size=args.batch_size
)
# 'epoch' or 'step'
lr_monitor = LearningRateMonitor(logging_interval="step")

# The EarlyStopping callback runs at the end of every validation epoch by default.
# Frequency set by check_val_every_n_epoch and val_check_interval of Trainer
early_stop_callback = EarlyStopping(
    monitor="val_loss",
    min_delta=0.00,
    patience=args.early_stopping_patient,
    verbose=False,
    mode="min",
    check_finite=True,  # Stops training when loss becomes NaN or infinite
)

checkpoint_callback = ModelCheckpoint(
    dirpath="ckpt/",
    filename="{epoch}-{step}-{val_loss:.2f}",
    save_top_k=5,
    monitor="val_loss",
)

# By default, Lightning logs every 50 training steps. log_every_n_steps
trainer = pl.Trainer(
    precision=args.precision,
    logger=logger,
    callbacks=[early_stop_callback, checkpoint_callback],
    max_epochs=args.max_epochs,
)  # type: ignore
# trainer = pl.Trainer(precision=PRECISION, logger=logger, callbacks=[early_stop_callback, checkpoint_callback], max_epochs=25)  # type: ignore
print(module.hparams)
trainer.fit(module, datamodule=dm)

# %%
# torch.cuda.empty_cache()

# %%
# If need to resume training from checkpoint
# trainer.fit(module, datamodule=dm, ckpt_path="/content/model_end_of_training.ckpt")

# %%
# trainer.save_checkpoint("model_end_of_training.ckpt")

# %% [markdown]
#  ## Evaluation

# %%
for path in os.listdir("ckpt"):
    print(f"{path}")
    module_loaded = PPModule.load_from_checkpoint(f"ckpt/{path}")
    module_loaded.to(device)
    module_loaded.eval()
    trainer.test(module_loaded, datamodule=dm)


# %% [markdown]
# ## Inference

# %%
page_path: str = 'autopager/data/html_all/1.html'

# %%
def get_model_output(module: PPModule, html: str) -> list[int]:
    module.eval()
    module.to(device)

    x_raw = storage.get_single_page_X_from_html(html)
    x, y = get_input_from_raw([x_raw])

    with torch.no_grad():
        x = tuple(x[0])
        x = [feat.unsqueeze(0).to(device) for feat in x]
        x_len = [len(x[0][0])]
        print(len(x[0][0]))
        y_pred = module(x, x_lens=x_len)

    return y_pred[0]

# %%
def get_page_links(module: PPModule, html: str) -> list[str]:
    urls = []

    x_raw = storage.get_single_page_X_from_html(html)
    x, y = get_input_from_raw([x_raw])
    y_pred = get_model_output(module, html)

    for x, y in zip(x_raw, y_pred, strict=True):
        if y == tag2idx["PAGE"]:
            urls.append(x.xpath('@href').extract_first())

    return urls

# %%
with open(page_path, 'r') as f:
    page_html = f.read()
    print(get_model_output(module_loaded, page_html))
    print(get_page_links(module_loaded, page_path))

# %% [markdown]
# 
# ## Export
# 

# %% [markdown]
# ## Debug purpose codes

# %%
# import gc
# counter = {}
# for obj in gc.get_objects():
#     try:
#         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
#             # print(type(obj), obj.size())
#             if obj.size() in counter:
#                 counter[obj.size()] += 1
#             else:
#                 counter[obj.size()] = 1
#     except:
#         pass

# print(dict(sorted(counter.items(), key=lambda item: item[1], reverse=True)))


# %%
# class PrintingCallback(pl.Callback):
#     def on_train_start(self, trainer, pl_module):
#         ...

#     def on_train_end(self, trainer, pl_module):
#         ...


# %%
# print(module.model.modules())
# print(module.model.state_dict())

# from tensorboard import notebook
# notebook.list()


