import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import classification_report
from crf import CRF


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
        # PPArgs for type hint purpose only
        # self.hparams: PPArgs

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
            + self.hparams.max_cls_query_per_node
            // 4
            * self.hparams.cls_emb_dim
            + self.hparams.cls_emb_dim,
            self.hparams.cls_fc_dim,
        )

        self.ptag_linear = nn.Linear(30, 30)

        self.query_emb_layer = nn.Embedding(
            self.hparams.query_vocab_size,
            self.hparams.query_emb_dim,
            padding_idx=0,
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

        self.num_filters_total = self.hparams.conv_filters * len(
            self.hparams.filter_sizes
        )

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
                        self.hparams.max_url_char_tok_per_node
                        - filter_size
                        + 1,
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
                        self.hparams.max_url_word_tok_per_node
                        - filter_size
                        + 1,
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
            + self.hparams.max_cls_query_per_node
            // 4
            * self.hparams.cls_emb_dim
            + self.hparams.query_emb_dim
            + self.hparams.url_fc_dim
            + 8
        )

        self.lstm = nn.LSTM(
            self.embedding_dim,
            self.hparams.lstm_hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        labels = ["O", "PREV", "PAGE", "NEXT"]
        self.tag2idx = {label: idx for idx, label in enumerate(labels)}
        self.num_tags = len(self.tag2idx)

        # O PREV PAGE NEXT
        self.crf: CRF = CRF(tagset_size=len(self.tag2idx), gpu=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(
            self.hparams.lstm_hidden_dim, self.num_tags + 2
        )

        # TODO: Test random or zero yield better result
        self.hidden = self.init_hidden(2)

        # self.hparams.some_layer_dim
        self.test_predictions = []
        self.test_label = []

    def init_hidden(self, batch_size):
        return (
            torch.randn(2, batch_size, self.hparams.lstm_hidden_dim // 2).to(
                self.device
            ),
            torch.randn(2, batch_size, self.hparams.lstm_hidden_dim // 2).to(
                self.device
            ),
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
        mask: torch.Tensor = self.get_mask(
            torch.Tensor(x_lens), lstm_feats.shape[1]
        ).to(self.device)
        loss = self.crf.neg_log_likelihood_loss(lstm_feats, mask, y)
        del mask

        return loss

    def forward(self, x, x_lens: list[int]):
        ## LSTM-CRF
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(x, x_lens)
        # padded_lstm_feats = torch.nn.utils.rnn.pad_packed_sequence(lstm_feats, batch_first=True, padding_value=0)

        # (BATCH, NODES, 300)
        mask: torch.Tensor = self.get_mask(
            torch.Tensor(x_lens), lstm_feats.shape[1]
        ).to(self.device)

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

    def get_mask(self, lens: torch.Tensor, max_len: int) -> torch.Tensor:
        return torch.arange(max_len)[None, :] < lens[:, None]