import torch

from models.model import EmbedderEncoderDecoder
from modules.decoders.joint import JointDecoder
from modules.decoders.ner import IobesNERDecoder, SpanNERDecoder
from modules.decoders.re import REDecoder
from modules.embedders.embedder import Embedder, StackedEmbedder
from modules.encoders.encoders import BiLSTMEncoder
from utils.embeddings import trim_embeddings


class ERE(EmbedderEncoderDecoder):
    "Specific Class for Joint NER and RE"

    def forward(self, data, task="re"):
        if self.encoder is not None:
            data.update({"embeddings": self.embedder(data)["embeddings"]})
            data.update({"encoded": self.encoder(data)["output"]})
        else:
            data.update({"encoded": self.embedder(data)["embeddings"]})

        if task == "ner":
            self.decoder(data, "ner")
        else:
            self.decoder(data, "ner")
            self.decoder(data, "re")

        return data

    @classmethod
    def from_config(cls, config, vocab):
        # Embedder
        if "word" in config.embedder and config.word_embedding_path is not None:
            w_embeddings = torch.tensor(
                trim_embeddings(vocab, config.word_embedding_path, embedding_dim=config.word_embedding_dim)).to(
                torch.float)
        else:
            w_embeddings = None

        embedders = [Embedder(mode, config, vocab, w_embeddings=w_embeddings) for mode in config.embedder]
        embedder = StackedEmbedder(embedders)

        # Encoder
        if config.encoder == "bilstm":
            encoder = BiLSTMEncoder(embedder.w_embed_dim, config.bilstm_hidden, dropout=config.dropout,
                                    n_layers=config.bilstm_layers)
            encoded_dim = encoder.output_dim

        else:
            encoder = None
            encoded_dim = embedder.w_embed_dim

        # NER Decoder
        if config.ner_decoder == "iobes":
            ner_decoder = IobesNERDecoder(encoded_dim, vocab, dropout=config.dropout)
            entity_dim = encoded_dim
        else:
            ner_decoder = SpanNERDecoder(encoded_dim, vocab, neg_sampling=config.ner_negative_sampling,
                                         max_span_len=config.ner_max_span, span_len_embedding_dim=config.ner_span_emb,
                                         pooling_fn=config.pool_fn, dropout=config.dropout)
            entity_dim = encoded_dim + config.ner_span_emb

        decoders = [ner_decoder]

        # RE Decoder
        if "re" in config.tasks:
            if config.re_context:
                re_decoder = REDecoder(entity_dim, vocab, neg_sampling=config.re_negative_sampling,
                                       context_dim=encoded_dim, biaffine=config.re_biaffine, dropout=config.dropout,
                                       pooling_fn=config.pool_fn)
            else:
                re_decoder = REDecoder(entity_dim, vocab, neg_sampling=config.re_negative_sampling,
                                       context_dim=0, biaffine=config.re_biaffine, dropout=config.dropout,
                                       pooling_fn=config.pool_fn)

            decoders.append(re_decoder)

        # Joint Decoder
        decoder = JointDecoder(decoders)

        return cls(embedder, encoder, decoder)
