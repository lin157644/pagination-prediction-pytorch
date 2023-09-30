# %% [markdown]
#  # Pagination prediction

# %%
import json
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    filename="train.log",
    encoding="utf-8",
    level=logging.DEBUG,
)

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert torch.cuda.is_available(), "No GPU/CUDA is detected!"

torch.set_float32_matmul_precision("high")


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
    precision: Literal["32-true", "16-mixed"]
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
    logging: bool


arg_parser = ArgumentParser()
arg_parser.add_argument("--batch_size", type=int, default=2)
arg_parser.add_argument("--max_epochs", type=int, default=20)
arg_parser.add_argument("--early_stopping_patient", type=int, default=20)
arg_parser.add_argument("--precision", type=str, default="32-true")
arg_parser.add_argument(
    "--sentence_model_name",
    type=str,
    default="sentence-transformers/distiluse-base-multilingual-cased-v2",
)

arg_parser.add_argument("--learning_rate", type=float, default=5e-4)  # 0.0005
arg_parser.add_argument("--cls_emb_dim", type=int, default=32)
arg_parser.add_argument("--cls_fc_dim", type=int, default=64)
arg_parser.add_argument("--query_emb_dim", type=int, default=64)
arg_parser.add_argument("--max_query_per_node", type=int, default=32)
arg_parser.add_argument("--ptag_emb_dim", type=int, default=30)
arg_parser.add_argument("--url_char_emb_dim", type=int, default=32)
arg_parser.add_argument("--url_word_emb_dim", type=int, default=32)
arg_parser.add_argument("--conv_filters", type=int, default=64)
arg_parser.add_argument(
    "--filter_sizes", type=json.loads, default="[3, 4, 5, 6]"
)
arg_parser.add_argument("--url_fc_dim", type=int, default=128)
arg_parser.add_argument("--lstm_hidden_dim", type=int, default=300)

arg_parser.add_argument("--max_cls_query_per_node", type=int, default=256)
arg_parser.add_argument("--max_url_char_tok_per_node", type=int, default=256)
arg_parser.add_argument("--max_url_word_tok_per_node", type=int, default=128)

arg_parser.add_argument("--logging", type=bool, default=True)

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
logging.info(f"SEED: {SEED}")

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
    # Get text content of the link otherwise alt or img.
    # Normalize multiple white space to one and to lowercase.
    text = normalize(get_link_text(link))
    href = get_link_href(link)
    if href is None:
        href = ""
    p = urlsplit(href)
    parent = link.xpath("..").extract()
    # Retrieve the line of first tag opening
    parent = get_first_tag(parser, parent[0])
    query_parsed = parse_qsl(p.query)  # parse query string from path
    query_param_names = [k.lower() for k, v in query_parsed]
    # TODO: change ngrams
    query_param_names_ngrams = _as_list(
        ngrams_wb(
            " ".join([normalize(name) for name in query_param_names]),
            3,
            5,
            True,
        )
    )

    # Classes of link itself and all its children.
    # It is common to have e.g. span elements with fontawesome
    # arrow icon classes inside <a> links.
    self_and_children_classes = " ".join(link.xpath(".//@class").extract())
    parent_classes = " ".join(link.xpath("../@class").extract())
    css_classes = normalize(parent_classes + " " + self_and_children_classes)

    token_feature = {
        "text-exact": replace_digits(text.strip()[:100].strip()),
        # <scheme>://<netloc>/<path>?<query>#<fragment>
        # 'url': p.path + p.query,
        "url": href if href else "https://",
        "query": query_param_names_ngrams,
        "parent-tag": parent,
        "class": _as_list(
            ngrams_wb(css_classes, 4, 5), AUTOPAGER_LIMITS.max_css_features
        ),
        "text": _as_list(
            ngrams_wb(replace_digits(text), 2, 5),
            AUTOPAGER_LIMITS.max_text_features,
        ),
    }
    tag_feature = {
        "isdigit": 1 if text.isdigit() is True else 0,
        "isalpha": 1 if text.isalpha() is True else 0,
        "has-href": 0 if href == "" else 1,
        "path-has-page": 1 if "page" in p.path.lower() else 0,
        "path-has-pageXX": 1
        if re.search(r"[/-](?:p|page\w?)/?\d+", p.path.lower()) is not None
        else 0,
        "path-has-number": 1
        if any(part.isdigit() for part in p.path.split("/"))
        else 0,
        "href-has-year": 1 if re.search("20\d\d", href) is not None else 0,
        "class-has-disabled": 1 if "disabled" in css_classes else 0,
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
        feat[0]["text-full"] = (
            normalize(before)
            + ","
            + feat[0]["text-exact"]
            + ","
            + normalize(after)
        )

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
            raise Exception(f"Error occurred on {idx}")
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
            p_tag = node["parent-tag"]
            ptag_page.append(
                sparse_representation_with_map(p_tag, data_map_for_ptag)
            )
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
    return [
        sentence_model.encode(
            [node["text-full"] for node in page], convert_to_tensor=True
        )
        for page in token_features
    ]


# %%
storage = Storage()

# %%
logging.debug("Loading raw training data...")
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
logging.debug("Loaded raw training data.")

# %%
logging.debug("Creating token/tag features...")
token_features: list[list[dict]]
# x_tag: features which only have tag true/false information
# token_features: ['text-exact', 'query', 'parent-tag', 'class', 'text', 'text-full']
token_features, x_tag = get_token_tag_features_from_chunks(X_raw)
token_feature_titles: list[str] = list(token_features[0][0].keys())
logging.debug("Created token/tag features.")

# %%
logging.debug("Creating class/query tokenizer...")
class_token_map = {}
query_token_map = {}

for page in token_features:
    for node in page:
        for class_name in node["class"]:
            class_token_map[class_name] = class_token_map.get(class_name, 0) + 1
        for query_name in node["query"]:
            query_token_map[query_name] = query_token_map.get(query_name, 0) + 1

class_tokenizer = TagTokenizer(class_token_map)
query_tokenizer = TagTokenizer(query_token_map)
CLS_VOCAB_SIZE = class_tokenizer.get_size()
QUERY_VOCAB_SIZE = query_tokenizer.get_size()
logging.debug(f"Created class/query tokenizer: {CLS_VOCAB_SIZE=} {QUERY_VOCAB_SIZE=}")

# %%
def get_class_query_ids(page_tokens, max_len):
    pages_class = []
    pages_query = []
    for page in page_tokens:
        class_page = []
        query_page = []
        for node in page:
            # class
            class_ids = class_tokenizer.tokenize(node["class"])
            class_ids = class_ids + [0] * (max_len - len(class_ids))
            class_page.append(class_ids[:max_len])
            # query
            query_ids = query_tokenizer.tokenize(node["query"])
            query_ids = query_ids + [0] * (max_len - len(query_ids))
            query_page.append(query_ids[:max_len])
        pages_class.append(torch.tensor(class_page))
        pages_query.append(torch.tensor(query_page))
    return pages_class, pages_query


# %%
top_parent_tags = {}

for page in token_features:
    for node in page:
        p_tag = node["parent-tag"]
        top_parent_tags[p_tag] = top_parent_tags.get(p_tag, 0) + 1

sorted_parent_tags = sorted(
    top_parent_tags.items(), key=lambda x: x[1], reverse=True
)

# %%
urls_full = []

for page in token_features:
    for node in page:
        urls_full.append(node["url"])

url_char_tokenizer = CharacterEncoder(urls_full)
url_word_tokenizer = RegexDelimiterEncoder(
    r"\/|&|\?|#|\.|://|=|-|[\ ]", urls_full
)


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
def get_input_from_raw(
    x_raw, y_raw=None, token_features=None, x_tag=None
) -> tuple[list[PAGE_X], list[torch.Tensor]]:
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
            url_char = F.pad(
                url_char, (0, args.max_url_char_tok_per_node - len(url_char))
            )
            url_char = url_char[
                slice(None, args.max_url_char_tok_per_node)
            ].long()
            page_url_char.append(url_char)

            url_word = url_word_tokenizer.encode(unquote(node["url"]))
            url_word = F.pad(
                url_word, (0, args.max_url_word_tok_per_node - len(url_word))
            )
            url_word = url_word[
                slice(None, args.max_url_word_tok_per_node)
            ].long()
            page_url_word.append(url_word)
        x_url_char_list.append(torch.stack(page_url_char))
        x_url_word_list.append(torch.stack(page_url_word))

    x = [
        PAGE_X(*x)
        for x in zip(
            x_text,
            x_ptag,
            x_class,
            x_query,
            x_url_char_list,
            x_url_word_list,
            x_tag,
            strict=True,
        )
    ]

    total_page  = 0
    total_next = 0
    total_none = 0
    for lab in y_raw:
        for l in lab:
            match l:
                case "NEXT":
                    total_next += 1
                case "PAGE":
                    total_page += 1
                case _:
                    total_none += 1

    logging.info(f"{total_page=} {total_next=} {total_none=} {total_page+total_next+total_none=}")

    if y_raw is not None:
        y: list[torch.Tensor] = [
            torch.tensor([tag2idx.get(l, 0) for l in lab]) for lab in y_raw
        ]
    else:
        y = []

    return x, y


def get_test_data(test_type=None, scaled_page="normal", language="en") -> tuple[list[PAGE_X], list[torch.Tensor], list[torch.Tensor]]:
    logging.debug(f"Getting test data: {test_type=}")
    match test_type:
        case "MULTILINGUAL":
            test_urls = [
                rec["Page URL"]
                for rec in storage.iter_test_records_by_language(language=language)
            ]

            test_X, test_y = storage.get_test_Xy_by_language(
                language=language
            )
            print(
                f"pages: {len(test_urls)}  domains: {len({get_domain(url) for url in test_urls})}"
            )
            return test_X, test_y, test_page_positions
        case "NORMAL":
            storage.test_file = "NORMAL"
            test_urls = [
                rec["Page URL"]
                for rec in storage.iter_test_records(exclude_en=None)
            ]
            test_X, test_y, test_page_positions = storage.get_test_Xy(
                validate=False,
                contain_position=True,
                scaled_page=scaled_page,
                exclude_en=None,
            )
            print(
                "pages: {}  domains: {}".format(
                    len(test_urls), len({get_domain(url) for url in test_urls})
                )
            )
            return test_X, test_y, test_page_positions
        case "EVENT_SOURCE":
            storage.test_file = "EVENT_SOURCE"
            test_urls = [
                rec["Page URL"]
                for rec in storage.iter_test_records(exclude_en=None)
            ]
            test_X, test_y, test_page_positions = storage.get_test_Xy(
                validate=False,
                contain_position=True,
                scaled_page=scaled_page,
                exclude_en=None,
            )
            print(
                "pages: {}  domains: {}".format(
                    len(test_urls), len({get_domain(url) for url in test_urls})
                )
            )
            return test_X, test_y, test_page_positions
        case None:
            raise ValueError("Please assign type of test_data")
        case _:
            raise ValueError("Unexpected test_type")


# %%
logging.debug("Creating training data...")
x_train, y_train = get_input_from_raw(X_raw, y_raw, token_features)
logging.debug("Created training data.")

# %%
logging.debug("Creating validation data...")
x_val_raw: list[parsel.selector.SelectorList]
x_val_raw, y_val_raw = storage.get_test_Xy_by_language(
    language="event", contain_button=True
)
x_val, y_val = get_input_from_raw(x_val_raw, y_val_raw)
logging.debug("Created validation data.")

# %%
logging.debug("Creating test data...")
test_x_raw, test_y_raw, test_page_positions = get_test_data("EVENT_SOURCE")
x_test, y_test = get_input_from_raw(test_x_raw, test_y_raw)
logging.debug("Created test data.")

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
        xx_pad.append(
            pad_sequence(list(feature), batch_first=True, padding_value=0)
        )

    x_lens = [x.text.shape[0] for x in xx]
    y_lens = [len(y) for y in yy]

    try:
        yy_pad = pad_sequence(list(yy), batch_first=True, padding_value=0)
    except:
        print(f"Exception when padding y: {yy}")
        raise

    return xx_pad, yy_pad, x_lens, y_lens


# %%
class PPDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_x,
        train_y,
        x_val,
        y_val,
        x_test,
        y_test,
        batch_size: int = 2,
    ):
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
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=pad_collect,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=pad_collect,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=pad_collect,
        )

    # def teardown(self, stage: str):
    #     # Used to clean-up when the run is finished
    #     ...

# %%
logging.debug("Exporting tokenizers...")
prefix: str = "pp_data/"
try:
    os.makedirs(prefix, exist_ok=True)
    with open(prefix + "url_char_tokenizer.pickle", "wb") as f:
        logging.info(f"{url_char_tokenizer.vocab_size=}")
        pickle.dump(url_char_tokenizer, f)
    with open(prefix + "url_word_tokenizer.pickle", "wb") as f:
        logging.info(f"{url_word_tokenizer.vocab_size=}")
        pickle.dump(url_word_tokenizer, f)
    with open(prefix + "class_token_map.json", "w") as f:
        f.write(json.dumps(class_tokenizer.map))
    with open(prefix + "query_token_map.json", "w") as f:
        f.write(json.dumps(query_tokenizer.map))
    with open(prefix + "sorted_parent_tags.json", "w") as f:
        f.write(json.dumps(sorted_parent_tags))
except:
    logging.warning(f"Export fail")
logging.debug("Exported tokenizers.")

# %% [markdown]
# ## Run

# %% [markdown]
#  ### Get Class, Query

# %%
logger_path = "tb_logs"
os.makedirs(logger_path, exist_ok=True)
logger = TensorBoardLogger(logger_path, name="pp_model")

# %%
# pl.seed_everything(SEED)
from pp_module import PPModule

module: PPModule = PPModule(
    text_emb_dim=sentence_model.get_sentence_embedding_dimension(),
    cls_vocab_size=class_tokenizer.get_size(),
    query_vocab_size=query_tokenizer.get_size(),
    url_char_vocab_size=url_char_tokenizer.vocab_size,
    url_word_vocab_size=url_word_tokenizer.vocab_size,
    **dict_args,
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
page_path: str = "autopager/data/html_all/1.html"


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
            urls.append(x.xpath("@href").extract_first())

    return urls


# %%
with open(page_path, "r") as f:
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
