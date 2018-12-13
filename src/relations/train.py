import sys
import time
from argparse import ArgumentParser
from enum import Enum

import os

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import date
from src.utils import HyperParams, mode_dict

def main():
    # manual random seed

    torch.manual_seed(h_params.rand_seed)
    training_files, _ = get_train_test_split(os.path.abspath(os.path.join("..", "data", "train-test-split.csv")))
    training_data, _ = prepare_rel_data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SegmentorClassifier = BiLSTM_Segmentor_Classifier if h_params.use_pos else BiLSTM_Segmentor_Classifier_no_pos
    model = SegmentorClassifier(h_params.d_word_embd, h_params.d_pos_embd, h_params.d_h1,
                                h_params.n_lstm_layers, h_params.word_voc_size, h_params.pos_voc_size,
                                h_params.ac_tagset_size, h_params.batch_size, device,
                                h_params.pretraind_embd_layer_path)

    # set loss function and adam optimizer (using negative log likelihood with adam optimizer
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=h_params.learning_rate, weight_decay=h_params.weight_decay)

    ## set CUDA if available
    if torch.cuda.is_available():
        model.cuda()
        loss_function.cuda()

    # display parameters in model
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # display optimizers paramersparamete
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

    # set train mode
    model.train()
    for epoch in range(h_params.n_epochs):
        start_time = time.time()
        acc_loss = 0.0  # accumalating loss per epoch for display
        for (indexed_tokens, indexed_POSs, indexed_AC_tags) in tqdm(training_data):
            # reset accumalated gradients and lstm's hidden state between iterations
            model.zero_grad()
            model.hidden1 = model.init_hidden(model.h1dimension)

            # make a forward pass
            tag_scores = model((indexed_tokens.to(device), indexed_POSs.to(device)))

            # backprop
            loss = loss_function(tag_scores, indexed_AC_tags.to(device))
            acc_loss += loss.item()
            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), h_params.clip_threshold)

            # call optimizer step
            optimizer.step()
        end_time = time.time()
        # output stats
        sys.stdout.write("===> Epoch[{}/{}]: Loss: {:.4f} , time = {:d}[s]\n".format(epoch + 1, h_params.n_epochs,
                                                                                     acc_loss,
                                                                                     int(end_time - start_time)))

        if epoch in [25, 50, 75]:
            try:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss}, os.path.abspath(os.path.join(
                    h_params.models_dir, "{}_SegClass_mode-{}_ep-{}_{}.pt".format(str(date.today()), mode, epoch,
                                                                                  "no_POS" if not h_params.use_pos else ""))))
            except:
                sys.stdout.write('failed to save model in epoch {}\n'.format(epoch))

    # save model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss}, os.path.abspath(os.path.join(h_params.models_dir,
                                                    "{}_SegClass_{}_ep-{}_{}.pt".format(str(date.today()), mode, epoch,
                                                                                        "no_POS" if not h_params.use_pos else ""))))

    # announce end
    sys.stdout.write("finished training")

if __name__ == '__main__':
    main()