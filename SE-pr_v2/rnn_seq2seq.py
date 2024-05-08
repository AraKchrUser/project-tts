# TODO: =====> 
# https://github.com/bentrevett/pytorch-seq2seq/blob/main/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb
# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

import time
from pathlib import Path

import torch 
from torch import optim
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

from cutils import asMinutes, timeSince
from dataset import Text2PseudoPhonemes, CoquiTTSTokenizer, Text2SemanticCode

from IPython import display

DEVICE = "cpu"


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden
    

###  wtf
class DecoderRNN(nn.Module):
    def __init__(
            self, hidden_size, output_size,
            # extra:
            sos_token, device, max_len,
            ):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

        self.sos_token = sos_token
        self.device = device
        self.max_len = max_len

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None, max_len=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=self.device)
        decoder_input = decoder_input.fill_(self.sos_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        _ = max_len if max_len is not None else self.max_len
        for i in range(_):
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)
            # print(f"{decoder_output.shape=}")
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden


class RNNTrainer:
    #TODO: save  ckpt torch.save(model.state_dict(), "ckpts/seq2seq_v4.pkl")
    # TODO: Add eval
    def __init__(
            self, texts_path, contents_path, clusters_path, sr, labels_path,
            ckpt_save_to, batch_size=320, lr=0.001, n_epochs=500, print_every=50, 
            plot_every=50, device="cpu", hidden_size=128, gen_max_len=300,
            ):
        self.lr = lr
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = device
        self.print_every = print_every
        self.plot_every = plot_every
        save_dir = Path(ckpt_save_to) / "rnn_seq2seq"
        save_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir = save_dir

        dataset, train, val = self.get_train_dataloader(
            texts_path, contents_path, 
            clusters_path, sr, labels_path,
            )
        self.dataset = dataset
        self.train_loader = train
        self.val_loader = val
        #TODO
        self.pad = dataset.semantic_codes_clusters.pad_id
        self.eos = dataset.semantic_codes_clusters.eos_id
        self.bos = dataset.semantic_codes_clusters.bos_id
        src_vocab_size = dataset.text_tokenizer.size
        tgt_vocab_size = dataset.semantic_codes_clusters.vocab_size
        self.encoder = EncoderRNN(src_vocab_size, hidden_size).to(device)
        self.decoder = DecoderRNN(hidden_size, tgt_vocab_size, self.bos, device, gen_max_len).to(device)
        # self.max_len = gen_max_len
    
    def get_train_dataloader(self, texts_path, contents_path, clusters_path, sr, labels_path):
        dataset = Text2SemanticCode(
            texts_path=texts_path, contents_path=contents_path, 
            clusters_path=clusters_path, tokenizer_conf=None, dsrate=sr,
            pre_calc_labels=True, labels_path=labels_path,
        )
        train_set, val_set = torch.utils.data.random_split(dataset, [len(dataset)-10, 10])
        train_loader = DataLoader(
            train_set, batch_size=self.batch_size, num_workers=3,
            collate_fn=dataset.collate_fn, pin_memory=True,
            )
        val_loader = DataLoader(
            val_set, batch_size=1, pin_memory=True,
            collate_fn=dataset.collate_fn, num_workers=3,
            )
        return dataset, train_loader, val_loader
       

    def _init_train_params(self):
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=self.lr)
        self.criterion = nn.NLLLoss()
    
    def train_epoch(self):
        total_loss = 0
        for batch in self.train_loader:
            tokens_padded = batch["tokens_padded"].to(self.device)
            lables = batch['lables'].to(self.device)

            # print(f"{tokens_padded.shape=}, {lables.shape=}")

            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()

            encoder_outputs, encoder_hidden = self.encoder(tokens_padded)
            logits, _, _ = self.decoder(
                encoder_outputs, encoder_hidden, 
                lables, lables.shape[-1],
                ) #TODO

            # print(f"{encoder_outputs.shape=}, {encoder_hidden.shape=}, {logits.shape=}")
            
            logits = logits.view(-1, logits.size(-1))
            lables = lables.view(-1)

            # print(f"{lables.shape=}, {logits.shape=}")

            loss = self.criterion(logits, lables)
            loss.backward()

            self.encoder_optimizer.step()
            self.decoder_optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)
    
    def train(self, plotting=False):
        start = time.time()
        self.plot_losses = []
        print_loss_total = 0
        plot_loss_total = 0

        self._init_train_params()

        for epoch in range(1, self.n_epochs + 1):
            loss = self.train_epoch()
            print_loss_total += loss
            plot_loss_total += loss

            if epoch % self.print_every == 0:
                print_loss_avg = print_loss_total / self.print_every
                print_loss_total = 0
                t1, t2 = timeSince(start, epoch / self.n_epochs)
                print(f"time: {t1} ({t2}),  epoch: {epoch} ({epoch / self.n_epochs * 100}%), loss: {print_loss_avg}")

                self.evaluate(epoch, True)
            
            if epoch % self.plot_every == 0:
                plot_loss_avg = plot_loss_total / self.plot_every
                self.plot_losses.append(plot_loss_avg)
                plot_loss_total = 0
        
            if plotting: self._plot(self.plot_losses)
            torch.save(self.encoder.state_dict(),  self.save_dir / "encoder.pkl")
            torch.save(self.decoder.state_dict(),  self.save_dir / "decoder.pkl")
    

    def evaluate(self, epoch, save): #TODO: #self.save_dir
        res = [] 
        for item in self.val_loader:
            with torch.no_grad():

                tokens_padded = item["tokens_padded"].to(self.device)
                lables = item['lables']
                text = item['text']

                enc_out, enc_hidden = self.encoder(tokens_padded)
                dec_out, _, _ = self.decoder(enc_out, enc_hidden)

                # loss = self.criterion(dec_out.view(-1, dec_out.size(-1)), lables.view(-1))

                _, topi = dec_out.topk(1)
                decoded_ids = topi.squeeze()
                decoded_words = []
                for idx in decoded_ids:
                    if idx.item() == self.eos:
                        break
                    decoded_words.append(idx.item())

                decoder = self.dataset.semantic_codes_clusters
                decoded_words = decoder.decode(decoded_words)
                res.append({
                    "text": text,
                    "lables": np.array(decoder.decode(lables.numpy()[0])),
                    "decoded_words": np.array(decoded_words),
                    "loss": np.mean(self.plot_losses),
                })
        
        if save: 
            saved = self.save_dir / f"evals/epoch_{epoch}" 
            saved.parent.mkdir(parents=True, exist_ok=True)
            print(saved)
            with saved.open("wb") as f:
                torch.save({"res": res}, f)
        
        return res

    
    def inference(self, ckpt_paths, text):
        ckpt_paths = Path(ckpt_paths)
        self.encoder.load_state_dict(torch.load(ckpt_paths/'encoder.pkl'))
        self.decoder.load_state_dict(torch.load(ckpt_paths/'decoder.pkl'))
        
        token_ids = self.dataset.tokenizer_encode(text)
        token_ids = torch.LongTensor([token_ids]).to(self.device)
        with torch.no_grad():
            enc_out, enc_hidden = self.encoder(token_ids)
            dec_out, _, _ = self.decoder(enc_out, enc_hidden)

            _, topi = dec_out.topk(1)
            decoded_ids = topi.squeeze()
            decoded_words = []
            for idx in decoded_ids:
                if idx.item() == self.eos:
                    break
                decoded_words.append(idx.item())
            
            decoder = self.dataset.semantic_codes_clusters
            decoded_words = decoder.decode(decoded_words)

        return decoded_words


    def _plot(self, points):
        # in Jupyter use `%matplotlib inline`
        display.clear_output(wait=True)
        plt.figure()
        fig, ax = plt.subplots()
        # this locator puts ticks at regular intervals
        loc = ticker.MultipleLocator(base=0.2)
        ax.yaxis.set_major_locator(loc)
        plt.plot(points)
        plt.show()


if __name__ == "__main__":
    bp = "./examples/"
    texts_path = bp + "rudevices_chunk/"
    contents_path = bp + "extracted_contents/"
    labels_path = bp + "build_labels_v2/"
    clusters_path = "../../NIR/ruslan_content_clusters/clusters_250.pt"
    sr=16_000
    trainer = RNNTrainer(
        texts_path=texts_path, contents_path=contents_path, 
        clusters_path=clusters_path, sr=sr, labels_path=labels_path,
        )
    trainer.train()