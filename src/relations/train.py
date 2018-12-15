import os
import sys
import time
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim

from src.models import BiLSTMRelationClassifier, BlandRelationClassifier
from src.utils import HyperParams, prepare_relations_data
from src.utils.preprocess import get_train_test_split


def main(config_file_path):
    # manual random seed
    h_params: HyperParams = HyperParams(config_file_path)
    torch.manual_seed(h_params.rand_seed)

    training_files, _ = get_train_test_split(os.path.abspath(os.path.join("..", "data", "train-test-split.csv")))
    training_data = prepare_relations_data(training_files, h_params.data_dir, h_params.vocab_dir) # list of (ac_dict, [(ac_id,ac_id),type]) tupels for each essay

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define use of constructed features aided model or bland lstm - basically a 2 layered consequtive lstm with further 2 linear layers and ReLU activations
    RelationClassifier = BiLSTMRelationClassifier if h_params.d_distance_embd else BlandRelationClassifier

    model = RelationClassifier(h_params.d_word_embd, h_params.d_pos_embd, h_params.d_h1,h_params.n_lstm_layers,
                               h_params.word_voc_size, h_params.pos_voc_size,h_params.ac_tagset_size,
                               h_params.batch_size, device, h_params.pretraind_embd_layer_path,
                               h_params.rel_tagset_size,h_params.d_tag_embd,h_params.d_small_embd,
                               h_params.d_distance_embd, h_params.d_h2, h_params.d_h3)

    model.to(device)
    # if using previous trained model weights for transfer learning (weights)
    if h_params.pretrained_segmentor_path:
        checkpoint = torch.load(h_params.pretrained_segmentor_path)
        pre_trained_state_dict = dict(checkpoint['model_state_dict'])
        model_dict = dict(model.state_dict())
        # filter unused keys
        pre_trained_state_dict = {param:value for param, value in pre_trained_state_dict.items() if param in model_dict}
        # overwrite new parametes in the model dictionary
        for param, value in pre_trained_state_dict.items():
            model_dict[param] = value
        # update state dict in the model
        model.load_state_dict(model_dict)



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
    save_name = "{}{}".format(model.__class__.__name__,"_transfer" if h_params.pretrained_segmentor_path else "")

    for epoch in range(h_params.n_epochs):
        start_time = time.time()
        acc_loss = 0.0  # accumalating loss per epoch for display
        done = 1
        ess_total_time = 0
        for (ac_dict, ac_pairs, rel_tags) in training_data:
            ess_start_time = time.time()
            for i_rel in range(len(ac_pairs)):
                a_id, b_id = ac_pairs[i_rel][0],ac_pairs[i_rel][1]
                ac_a, ac_b = ac_dict[a_id], ac_dict[b_id]
                # reset accumalated gradients and lstm's hidden state between iterations
                model.zero_grad()
                model.hidden1 = model.init_hidden(model.h1dimension)
                model.hidden2 = model.init_hidden(model.h2dimension)

                # make a forward pass
                tag_scores = model((ac_a,ac_b))

                # backprop
                loss = loss_function(tag_scores, rel_tags[i_rel].view(1).to(device))
                acc_loss += loss.item()
                loss.backward()

                # gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), h_params.clip_threshold)

                # call optimizer step
                optimizer.step()
            ess_total_time += time.time() - ess_start_time
            sys.stdout.write("epoch:{}\tfinished {}/322 essays in {:.2f}[s] per essay\n".format(epoch,done, ess_total_time/done))
            done += 1
        end_time = time.time()
        # output stats
        sys.stdout.write("===> Epoch[{}/{}]: Loss: {:.4f} , time = {:d}[s]\n".format(epoch + 1, h_params.n_epochs,
                                                                                     acc_loss,
                                                                                     int(end_time - start_time)))

        if (epoch+1)%25 ==0:
            try:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss}, os.path.abspath(os.path.join(h_params.models_dir,"{}_ep-{}.pt".format(save_name,epoch))))
            except:
                sys.stdout.write('failed to save model in epoch {}\n'.format(epoch))

    # save model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss}, os.path.abspath(os.path.join(h_params.models_dir,"{}_ep-{}.pt".format(save_name,epoch))))

    # announce end
    sys.stdout.write("finished training")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-cp','--config_path',help= " path to learning parameters file")
    args = parser.parse_args(sys.argv[1:])
    main(os.path.abspath(args.config_path))