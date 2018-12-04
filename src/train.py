from src.models import BiLSTM_Segmentor_Classifier


def main():
    model = BiLSTM_Segmentor_Classifier(d_word_embd, d_pos_embd,
                 d_h1, n_lstm_layers, ac_tagset_size,
                 len_seq, pretraind_embd_layer=None)


if __name__ = "__main__":
    main()
