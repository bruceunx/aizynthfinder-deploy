from typign import Dict

from torch import nn
from torch.nn import functional as F
import torch.optim as optim
import lightning as L


class Expansion(L.LightningModule):

    def __init__(self, config: Dict[str, int | float]) -> None:
        super().__init__()
        self.hidden_nodes = config["hidden_nodes"]
        self.input_dim = config["input_dim"]
        self.output_dim = config["output_dim"]
        self.drop_out = config["drop_out"]

        self.lr = config["lr"]

        self.linear1 = nn.Linear(self.input_dim, self.hidden_nodes)
        self.linear2 = nn.Linear(self.hidden_nodes, self.output_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = F.elu(x)
        x = F.dropout(x, self.drop_out)
        x = self.linear2(x)
        return F.log_softmax(x, dim=-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.nll_loss(y_hat, y)
        loss += self.l2_reg(self.dense1.weight) + self.l2_reg(
            self.dense2.weight)
        return loss

    def validation_step(self, batch, batch_idx):
        pass
        # Similar to training_step, but without regularization

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)


class Filter(L.LightningModule):

    def __init__(self, config: Dict[str, int | float]):
        super().__init__()
        self.hidden_nodes = config["hidden_nodes"]
        self.fingerprint_len = config["fingerprint_len"]
        self.drop_out = config["drop_out"]
        self.lr = config["lr"]

        self.product_linear = nn.Linear(self.fingerprint_len,
                                        self.hidden_nodes)

        self.reaction_linear = nn.Linear(self.fingerprint_len,
                                         self.hidden_nodes)

        self.cosine_layer = nn.CosineSimilarity(dim=-1)
        self.output_layer = nn.Linear(1, 1)

    def forward(self, product_input, reaction_input):
        product_features = self.product_linear(product_input)
        product_features = F.elu(product_features)
        product_features = F.dropout(product_features, self.drop_out)

        reaction_features = self.reaction_linear(reaction_input)
        reaction_features = F.elu(reaction_features)

        similarity = self.cosine_layer(product_features, reaction_features)
        output = self.output_layer(similarity.unsqueeze(-1))
        return F.sigmoid(output)

    def training_step(self, batch, batch_idx):
        product_features, reaction_features, labels = batch
        outputs = self(product_features, reaction_features)

        loss = F.binary_cross_entropy(outputs.squeeze(), labels.float())
        accuracy = F.binary_acc(outputs.squeeze(), labels)
        return {"loss": loss, "accuracy": accuracy}

    def validation_step(self, batch, batch_idx):
        pass
        # Similar to training_step, but without regularization

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)


class Recommend(L.LightningModule):

    def __init__(self, config: Dict[str, int | float]):
        super().__init__()
        self.hidden_nodes = config["hidden_nodes"]
        self.fingerprint_len = config["fingerprint_len"]
        self.output_dim = config["output_dim"]
        self.drop_out = config["drop_out"]
        self.lr = config["lr"]

        self.linear1 = nn.Linear(self.fingerprint_len, self.hidden_nodes)
        self.linear2 = nn.Linear(self.hidden_nodes, self.output_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = F.elu(x)
        x = F.dropout(x, self.drop_out)
        return F.log_softmax(self.dense2(x), dim=-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.nll_loss(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def test_step(self, batch, batch_idx):
        # Calculate top-k accuracy metrics (optional)
        pass

    def _top_k_acc(self, preds, y, top_k):
        # Implement top-k accuracy calculation (optional)
        pass
