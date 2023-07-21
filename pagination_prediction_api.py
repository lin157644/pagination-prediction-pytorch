# %%
from regex_delimiter_encoder import RegexDelimiterEncoder
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report
from torchnlp.encoders.text import CharacterEncoder
import json
import os
import pickle
import re
import sys
from dataclasses import astuple, dataclass
from itertools import islice
from typing import List, Tuple
from urllib.parse import parse_qsl, unquote, urlsplit

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from autopager import AUTOPAGER_LIMITS
from autopager.htmlutils import (
    get_link_href,
    get_link_text,
    get_text_around_selector_list,
)
from autopager.parserutils import MyHTMLParser, get_first_tag
from autopager.storage import Storage
from autopager.utils import get_domain, ngrams_wb, normalize, replace_digits
from crf import CRF

from pp_module import PPModule

_pp_prefix: str = "pp_data/"

# %%
class TagTokenizer:
    def __init__(self, tag_name_count=None):
        rt_dict = {"[PAD]": 0, "[UNK]": 1}
        # TODO: Sort by count, although embedding layer should compensate this
        if tag_name_count is not None:
            for k in tag_name_count.keys():
                rt_dict[k] = len(rt_dict)
        self.map = rt_dict

    def tokenize(self, word):
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


class PaginationPredictionAPI:
    def __init__(
        self,
        model_path: str = "ckpt/epoch=14-step=2385-val_loss=14.19.ckpt",
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        self.device = torch.device("cuda")

        if not os.path.exists(model_path):
            raise RuntimeError(f"Model path {model_path} does not exist")

        self.module: PPModule = PPModule.load_from_checkpoint(model_path)
        self.module.eval()
        self.module.to(self.device)

        self.storage = Storage()
        self.sentence_model = SentenceTransformer(
            self.module.hparams.sentence_model_name
        )
        self.parser = MyHTMLParser()

        self.labels = ["O", "PREV", "PAGE", "NEXT"]
        self.tag2idx = {label: idx for idx, label in enumerate(self.labels)}

        with open(_pp_prefix + "class_token_map.json") as cls_tok_file:
            self.class_token_map: dict = json.load(cls_tok_file)
        with open(_pp_prefix + "query_token_map.json") as query_tok_file:
            self.query_token_map: dict = json.load(query_tok_file)

        self.class_tokenizer = TagTokenizer()
        self.class_tokenizer.map = self.class_token_map
        self.query_tokenizer = TagTokenizer()
        self.query_tokenizer.map = self.query_token_map

        with open(_pp_prefix + "url_char_tokenizer.pickle", "rb") as f:
            self.url_char_tokenizer: CharacterEncoder = pickle.load(f)
        with open(_pp_prefix + "url_word_tokenizer.pickle", "rb") as f:
            self.url_word_tokenizer: RegexDelimiterEncoder = pickle.load(f)

        with open(_pp_prefix + "sorted_parent_tags.json") as f:
            self.sorted_parent_tags = json.load(f)

    def get_model_output(self, module: PPModule, html: str) -> list[int]:
        x_raw = self.storage.get_single_page_X_from_html(html)
        x, y = self.get_input_from_raw([x_raw])

        with torch.no_grad():
            x = tuple(x[0])
            x = [feat.unsqueeze(0).to(self.device) for feat in x]
            x_len = [len(x[0][0])]
            print(f"Number of page links: {len(x[0][0])}")
            y_pred = module.forward(x, x_lens=x_len)

        return y_pred[0]

    def get_page_links(self, html: str, page_url: str = None) -> list[str]:
        urls = []

        netloc = ""
        if page_url:
            netloc = urlsplit(page_url).netloc

        x_raw = self.storage.get_single_page_X_from_html(html)
        x, y = self.get_input_from_raw([x_raw])
        y_pred = self.get_model_output(self.module, html)

        for x, y in zip(x_raw, y_pred, strict=True):
            if y == self.tag2idx["PAGE"]:
                url: str = x.xpath("@href").extract_first()
                # Add the host part if:
                # page_url is provided
                # url is not empty
                # url does not have netloc part
                if page_url and url and not urlsplit(url).netloc:
                    if not url.startswith("/"):
                        url = "/" + url
                    url = netloc + url
                urls.append(url)

        return [url for url in urls if url]  # Remove empty string

    def pages_to_word_vector(self, token_features) -> list[torch.Tensor]:
        return [
            self.sentence_model.encode(
                [node["text-full"] for node in page], convert_to_tensor=True
            )
            for page in token_features
        ]

    def get_class_query_ids(self, page_tokens, max_len):
        pages_class = []
        pages_query = []
        for page in page_tokens:
            class_page = []
            query_page = []
            for node in page:
                # class
                class_ids = self.class_tokenizer.tokenize(node["class"])
                class_ids = class_ids + [0] * (max_len - len(class_ids))
                class_page.append(class_ids[:max_len])
                # query
                query_ids = self.query_tokenizer.tokenize(node["query"])
                query_ids = query_ids + [0] * (max_len - len(query_ids))
                query_page.append(query_ids[:max_len])
            pages_class.append(torch.tensor(class_page))
            pages_query.append(torch.tensor(query_page))
        return pages_class, pages_query

    def _as_list(self, generator, limit=None) -> list:
        """
        Generator to list with limit.
        """
        return list(generator if limit is None else islice(generator, 0, limit))

    def link_to_features(self, link):
        # Get text contecnt of the link otherwise alt or img.
        # Normalize multiple white space to one and to lowercase.
        text = normalize(get_link_text(link))
        href = get_link_href(link)
        if href is None:
            href = ""
        p = urlsplit(href)
        parent = link.xpath("..").extract()
        # Retrive the line of first tag opening
        parent = get_first_tag(self.parser, parent[0])
        query_parsed = parse_qsl(p.query)  # parse query string from path
        query_param_names = [k.lower() for k, v in query_parsed]
        query_param_names_ngrams = self._as_list(
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
        css_classes = normalize(
            parent_classes + " " + self_and_children_classes
        )

        token_feature = {
            "text-exact": replace_digits(text.strip()[:100].strip()),
            # <scheme>://<netloc>/<path>?<query>#<fragment>
            # 'url': p.path + p.query,
            "url": href if href else "https://",
            "query": query_param_names_ngrams,
            "parent-tag": parent,
            "class": self._as_list(
                ngrams_wb(css_classes, 4, 5), AUTOPAGER_LIMITS.max_css_features
            ),
            "text": self._as_list(
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

    def page_to_features(self, xseq):
        feat_list = [self.link_to_features(a) for a in xseq]
        around = get_text_around_selector_list(xseq, max_length=15)
        # Append sibling's text-exact to each node's text-full.
        for feat, (before, after) in zip(feat_list, around):
            feat[0]["text-full"] = (
                normalize(before)
                + ","
                + feat[0]["text-exact"]
                + ","
                + normalize(after)
            )

        return feat_list

    def get_token_tag_features_from_chunks(self, chunks):
        token_features = []
        tag_features = []
        for idx, page in enumerate(chunks):
            try:
                feat_list = self.page_to_features(page)
                token_features.append([node[0] for node in feat_list])
                tag_features.append(
                    torch.tensor([node[1] for node in feat_list])
                )
            except:
                raise Exception(f"Error occured on {idx}")
        return token_features, tag_features

    def sparse_representation_with_map(self, tag, data_map):
        # Vector length is the number of tags in the map(30).
        rt_vec = [0] * len(data_map)
        for idx, map_tag in enumerate(data_map):
            # ('tag_name', count)
            if tag == map_tag[0]:
                rt_vec[idx] = 1
                break
        return rt_vec

    def get_ptags_vector(
        self, token_features, data_map_for_ptag: list[Tuple[str, int]]
    ):
        pages_ptag = []
        for page in token_features:
            ptag_page = []
            for node in page:
                p_tag = node["parent-tag"]
                ptag_page.append(
                    self.sparse_representation_with_map(
                        p_tag, data_map_for_ptag
                    )
                )
            pages_ptag.append(torch.tensor(ptag_page, dtype=torch.float32))
        return pages_ptag

    def get_input_from_raw(
        self, x_raw, y_raw=None, token_features=None, x_tag=None
    ) -> Tuple[list[PAGE_X], list[torch.Tensor]]:
        if token_features is None or x_tag is None:
            token_features, x_tag = self.get_token_tag_features_from_chunks(
                x_raw
            )

        x_text: list[torch.Tensor] = self.pages_to_word_vector(token_features)

        x_ptag: list[torch.Tensor] = self.get_ptags_vector(token_features, self.sorted_parent_tags[:30])  # type: ignore

        x_class, x_query = self.get_class_query_ids(
            token_features, max_len=self.module.hparams.max_cls_query_per_node
        )

        # (pages, nodes)
        x_url_char_list = []
        x_url_word_list = []
        for page in token_features:
            page_url_char = []
            page_url_word = []
            for node in page:
                url_char = self.url_char_tokenizer.encode(unquote(node["url"]))
                url_char = F.pad(
                    url_char,
                    (
                        0,
                        self.module.hparams.max_url_char_tok_per_node
                        - len(url_char),
                    ),
                )
                url_char = url_char[
                    slice(None, self.module.hparams.max_url_char_tok_per_node)
                ].long()
                page_url_char.append(url_char)

                url_word = self.url_word_tokenizer.encode(unquote(node["url"]))
                url_word = F.pad(
                    url_word,
                    (
                        0,
                        self.module.hparams.max_url_word_tok_per_node
                        - len(url_word),
                    ),
                )
                url_word = url_word[
                    slice(None, self.module.hparams.max_url_word_tok_per_node)
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
            )
        ]

        if y_raw is not None:
            y: list[torch.Tensor] = [
                torch.tensor([self.tag2idx.get(l, 0) for l in lab])
                for lab in y_raw
            ]
        else:
            y = []

        return x, y

    def get_test_data(self, test_type=None, scaled_page="normal"):
        if test_type is None:
            print("Please assign type of test_data")
            return None, None, None
        test_X_one = []
        test_X_two = []
        test_y_one = []
        test_y_two = []
        test_page_positions_one = []
        test_page_positions_two = []
        if test_type != "EVENT_SOURCE":
            self.storage.test_file = "NORMAL"
            test_urls = [
                rec["Page URL"]
                for rec in self.storage.iter_test_records(exclude_en=None)
            ]
            (
                test_X_one,
                test_y_one,
                test_page_positions_one,
            ) = self.storage.get_test_Xy(
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
            if test_type == "NORMAL":
                return test_X_one, test_y_one, test_page_positions_one
        if test_type != "NORMAL":
            self.storage.test_file = "EVENT_SOURCE"
            test_urls = [
                rec["Page URL"]
                for rec in self.storage.iter_test_records(exclude_en=None)
            ]
            (
                test_X_two,
                test_y_two,
                test_page_positions_two,
            ) = self.storage.get_test_Xy(
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
            if test_type == "EVENT_SOURCE":
                return test_X_two, test_y_two, test_page_positions_two
        test_X_raw = test_X_one + test_X_two
        test_y = test_y_one + test_y_two
        test_positions = test_page_positions_one + test_page_positions_two
        return test_X_raw, test_y, test_positions

# %%

if __name__ == "__main__":
    with open("test/test.html", "r") as f:
        page_html = f.read()
    api = PaginationPredictionAPI()
    urls = api.get_page_links(
        page_html, "https://forums.oneplus.net/threads/marsh-cm13.405700/"
    )
    print(urls)
