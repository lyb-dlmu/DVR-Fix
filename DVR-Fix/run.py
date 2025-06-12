from __future__ import absolute_import, division, print_function
import argparse
import math
import scipy
import multiprocessing
from ast import And
import logging
from sklearn.cluster import KMeans
import os
import random
import pickle
import multiprocessing
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from transformers import get_linear_schedule_with_warmup, RobertaTokenizer, T5ForConditionalGeneration, RobertaModel
from tqdm import tqdm
import pandas as pd
from AutoVF import AutoVF
from loc_bert import LocModel
from torch.distributions.beta import Beta
from scipy import special
import scipy
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 input_ids,
                 vul_query_label,
                 repair_input_ids,
                 commit_index):
        self.input_ids = input_ids
        self.vul_query_label = vul_query_label
        self.repair_input_ids = repair_input_ids
        self.commit_index = commit_index


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_type="train"):
        if file_type == "train":
            file_path = args.train_data_file
        elif file_type == "eval":
            file_path = args.eval_data_file
        elif file_type == "test":
            file_path = args.test_data_file
        self.examples = []
        df = pd.read_csv(file_path)
        print(df.columns)
        indexx = df.index.tolist()
        source = df["source"].tolist()
        repair_target = df["target"].tolist()
        for i in tqdm(range(len(source))):
            self.examples.append(convert_examples_to_features(indexx[i], source[i], repair_target[i], tokenizer, args))
        if file_type == "train":
            for example in self.examples[:3]:
                logger.info("*** Example ***")
                logger.info("index: {}".format(example.commit_index))
                logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))
                logger.info("vul_query_label: {}".format(' '.join(map(str, example.vul_query_label))))
                logger.info("repair_input_ids: {}".format(' '.join(map(str, example.repair_input_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].vul_query_label), torch.tensor(
            self.examples[i].repair_input_ids), torch.tensor(self.examples[i].commit_index)


def cal_Prob(one_commit_data):
    idx, score, phais, mean, std = one_commit_data
    return idx, scipy.stats.norm(mean, std).pdf(score) * phais


def EM(data, k=2, max_step=200, threhold=0.00001):
    phais = torch.tensor([[1.0 / k] for i in range(k)], device=data.device)
    mean = torch.tensor([[i] for i in range(k)], device=data.device)
    std = torch.tensor([[1] for i in range(k)], device=data.device)
    pi_times_2 = torch.tensor(2 * math.pi)
    for i in range(max_step):
        # Qs = e_step(data,phais,mean,std)
        data_k = data.repeat(k).reshape(k, data.shape[0])
        exponent = torch.pow((data_k - mean), 2) * (-1 / (2 * std))
        Qs = (torch.exp(exponent) / torch.sqrt(pi_times_2 * std) * phais)
        Qs = Qs / torch.sum(Qs, dim=0, keepdim=True)
        # phais, mean, std= m_step(data,phais,mean,std,Qs)
        gama_j = torch.sum(Qs, dim=1)
        new_phais = (gama_j / data.shape[0]).reshape(k, 1)
        new_mean = (torch.sum(data * Qs, dim=1) / gama_j).reshape(k, 1)
        X_i_mu_j = torch.pow((data_k - mean), 2)
        # new_std = (torch.sum((X_i_mu_j*Qs).transpose(0,1), axis=1) / gama_j).reshape(k, 1)
        new_std = (torch.sum(X_i_mu_j * Qs, axis=1) / gama_j).reshape(k, 1)
        if i > 0 and False not in (torch.abs(new_mean - mean) < threhold):
            break
        phais, mean, std = new_phais, new_mean, new_std
    return phais[:, 0].tolist(), mean[:, 0].tolist(), std[:, 0].tolist()


def convert_examples_to_features(commit_index, source, repair_target, tokenizer, args):
    # encode - subword tokenize
    input_ids = tokenizer.encode(source, truncation=True, max_length=args.encoder_block_size, padding='max_length')
    repair_input_ids = tokenizer.encode(repair_target, truncation=True, max_length=args.vul_repair_block_size,
                                        padding='max_length')

    vul_query = []
    is_vul = False
    for n in range(512):
        if input_ids[n] == tokenizer.start_bug_id:
            is_vul = True
            vul_query.append(1)
        elif input_ids[n] == tokenizer.end_bug_id:
            is_vul = False
            vul_query.append(1)
        elif is_vul:
            vul_query.append(1)
        else:
            vul_query.append(0)
    return InputFeatures(input_ids, vul_query, repair_input_ids, commit_index)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def mixup_criterion(pred, y_a, y_b, lam):
    return (lam * F.cross_entropy(pred, y_a, reduce=False) + (1 - lam) * F.cross_entropy(pred, y_b,
                                                                                         reduce=False)).mean()


def mixup_data(x, y, alpha):
    lam = Beta(torch.tensor(alpha), torch.tensor(alpha)).sample() if alpha > 0 else 1
    index = torch.randperm(x.size()[0]).cuda()
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def train(args, train_dataset, model, loc_model, eval_dataset, tokenizer, train_with_mask):
    """ Train the model """
    # build dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=0)

    args.max_steps = args.epochs * len(train_dataloader)
    print(len(train_dataloader) * args.epochs * 0.1)
    # evaluate model per epoch
    args.save_steps = len(train_dataloader) * 1

    args.warmup_steps = args.max_steps // 5
    model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_dataloader) * args.epochs * 0.1,
                                                num_training_steps=len(train_dataloader) * args.epochs)

    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size // max(args.n_gpu, 1))
    logger.info("  Total train batch size = %d", args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    global_step = 0
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_loss = 100000

    model.zero_grad()
    early_stop = 0
    for idx in range(args.epochs):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        model.train()
        loss_list = list()
        commit_index_list = list()
        confidence_list2 = list()
        for step, batch in enumerate(bar):
            (input_ids, _, repair_input_ids, commit_index) = [x.to(args.device) for x in batch]
            if train_with_mask:
                vul_query_mask = loc_model(input_ids)
                outputs = model(input_ids=input_ids, vul_query_mask=vul_query_mask, repair_input_ids=repair_input_ids)
            else:
                outputs = model(input_ids=input_ids, vul_query_mask=None, repair_input_ids=repair_input_ids)

            # loss = outputs.loss
            loss_con = outputs.loss
            inputs, targets_a, targets_b, lam = mixup_data(outputs.logits, repair_input_ids, alpha=5.0)
            logits = outputs.logits
            logits = torch.tensor(logits, requires_grad=True, dtype=torch.float32).to(args.device)
            logits = logits.reshape(-1, logits.size(-1))
            targets_a = targets_a.to(args.device, dtype=torch.long)
            targets_b = targets_b.to(args.device, dtype=torch.long)
            targets_a = targets_a.flatten()
            targets_b = targets_b.flatten()
            loss_sup = mixup_criterion(logits, targets_a, targets_b, lam)
            loss = 0.5 * loss_sup + loss_con

            lm_logits = outputs.logits
            loss_fct = CrossEntropyLoss(ignore_index=0, reduction='none')
            # logger.info("lm_logits.size:{}".format(lm_logits.size))
            loss_batch = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), repair_input_ids.view(-1)).view(-1,
                                                                                                          lm_logits.size(
                                                                                                              1))
            loss_batch = torch.nanmean(loss_batch.masked_fill(repair_input_ids == tokenizer.pad_token_id, torch.nan),
                                       dim=1)
            loss_list += loss_batch.tolist()
            commit_index_list += commit_index.tolist()
            if idx > 3:
                loss_batch = [loss * confidence_list[commit_index[idxa]] for idxa, loss in enumerate(loss_batch)]
                loss_batch = torch.stack(loss_batch, dim=0)
                loss = torch.mean(loss_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss
            avg_loss = round(train_loss / tr_num, 5)
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                avg_loss = round(np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)
                if global_step % args.save_steps == 0:
                    # placeholder of evaluation
                    if idx >= 3:
                        confidence_list2 = prevaluate(args, model, loc_model, eval_dataset, tokenizer, idx,
                                                      eval_when_training=True, eval_with_mask=train_with_mask)
                    eval_loss = evaluate(args, model, loc_model, eval_dataset, tokenizer, confidence_list2, idx,
                                         eval_when_training=True, eval_with_mask=train_with_mask)
                    # Save model checkpoint
                    if eval_loss < best_loss and idx > 3:
                        best_loss = eval_loss
                        logger.info("  " + "*" * 20)
                        logger.info("  Best Loss:%s", round(best_loss, 4))
                        logger.info("  " + "*" * 20)
                        checkpoint_prefix = 'checkpoint-best-loss'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format(args.model_name))
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)
                    else:
                        early_stop += 1
                        if not train_with_mask and early_stop >= 5:
                            print("Early stopping for warm-up training without mask.")
                            break

        pathh = "sss"
        os.makedirs(pathh, exist_ok=True)
        with open(os.path.join(pathh, "loss_list_cur_epoch_{}.pickle".format(idx)), "wb") as f:
            pickle.dump(loss_list, f)
        with open(os.path.join(pathh, "commit_index_list_cur_epoch_{}.pickle".format(idx)), "wb") as f:
            pickle.dump(commit_index_list, f)
        if idx >= 3:
            score_file_path = os.path.join(pathh, "loss_list_cur_epoch_{}.pickle".format(idx))
            commit_index_file = os.path.join(pathh, "commit_index_list_cur_epoch_{}.pickle".format(idx))
            if score_file_path.split(".")[-1] == "loss":
                with open(score_file_path) as score_f:
                    score_list = score_f.read().split("\n")
            elif score_file_path.split(".")[-1] == "pickle":
                with open(score_file_path, "rb") as score_f:
                    score_list = pickle.load(score_f)
            with open(commit_index_file, "rb") as f:
                commit_index_list = pickle.load(f)
            logger.info("There are {} commit messages are calculated in the probability".format(len(commit_index_list)))
            if set(commit_index_list) != set(range(len(commit_index_list))):
                logger.info("Some indexes are missing in commit_index_list")
            unsorted_score_list = [float(item) for item in score_list]
            score_list = []
            for i in np.argsort(commit_index_list):
                score_list.append(unsorted_score_list[i])

            # Get distribution by EM algorithm
            ratio, avg, std = EM(torch.tensor(score_list, device=args.device, dtype=torch.float64))
            if avg[0] < avg[1]:
                clean_idx = 0
            else:
                clean_idx = 1
            # Get Confidence of loss of each commit
            # NOT IN ORDER
            logger.info("Calculating the clean_weight ...")
            with multiprocessing.Pool(min(multiprocessing.cpu_count(), 24)) as pool:
                commit_data = [(idxb, score, ratio[clean_idx], avg[clean_idx], std[clean_idx] * 9) for idxb, score in
                               enumerate(score_list)]
                idx_clean_weight_list = pool.map( cal_Prob, commit_data)
            logger.info("Calculating the noisy_weight ...")
            with multiprocessing.Pool(min(multiprocessing.cpu_count(), 24)) as pool:
                commit_data = [(idxb, score, ratio[1 - clean_idx], avg[1 - clean_idx], std[1 - clean_idx]) for
                               idxb, score in enumerate(score_list)]
                idx_noisy_weight_list = pool.map(cal_Prob, commit_data)
            # IN ORDER
            sorted_idx_clean_weight_list = sorted(idx_clean_weight_list, key=lambda x: x[0])
            sorted_idx_noisy_weight_list = sorted(idx_noisy_weight_list, key=lambda x: x[0])
            clean_weight_list = [idx_weight[1] for idx_weight in sorted_idx_clean_weight_list]
            noisy_weight_list = [idx_weight[1] for idx_weight in sorted_idx_noisy_weight_list]

            # clean_weight_list = [cal_Prob(each_data, ratio[clean_idx], avg[clean_idx], std[clean_idx]*9) for each_data in score_list]
            # noisy_weight_list = [cal_Prob(each_data, ratio[1-clean_idx], avg[1-clean_idx], std[1-clean_idx]) for each_data in score_list]
            logger.info("Calculating the confidence ...")
            confidence_list = []
            fenzi = []
            fenmu = []
            cleann = []
            noisyy = []
            for idxc in range(len(score_list)):
                denom = clean_weight_list[idxc] + noisy_weight_list[idxc]
                if denom == 0:
                    con = 1
                else:
                    con = (clean_weight_list[idxc] + (np.power(1.6, -score_list[idxc])) * noisy_weight_list[idxc]) / (
                            clean_weight_list[idxc] + noisy_weight_list[idxc])
                confidence_list.append(con)
                cleann.append(clean_weight_list[idxc])
                noisyy.append(noisy_weight_list[idxc])
                fenzi.append(clean_weight_list[idxc] + (np.power(1.6, -score_list[idxc])) * noisy_weight_list[idxc])
                fenmu.append(clean_weight_list[idxc] + noisy_weight_list[idxc])
            logger.info("Calculate over ..")
            os.makedirs(pathh, exist_ok=True)
            with open(os.path.join(pathh, "cleann_cur_epoch_{}.pickle".format(idx)), "wb") as f:
                pickle.dump(cleann, f)
            with open(os.path.join(pathh, "noisyy_cur_epoch_{}.pickle".format(idx)), "wb") as f:
                pickle.dump(noisyy, f)
            with open(os.path.join(pathh, "confidence_list_cur_epoch_{}.pickle".format(idx)), "wb") as f:
                pickle.dump(confidence_list, f)
            with open(os.path.join(pathh, "fenzi_cur_epoch_{}.pickle".format(idx)), "wb") as f:
                pickle.dump(fenzi, f)
            with open(os.path.join(pathh, "fenmu_cur_epoch_{}.pickle".format(idx)), "wb") as f:
                pickle.dump(fenmu, f)


def clean_tokens(tokens):
    tokens = tokens.replace("<pad>", "")
    tokens = tokens.replace("<s>", "")
    tokens = tokens.replace("</s>", "")
    tokens = tokens.strip("\n")
    tokens = tokens.strip()
    return tokens


def prevaluate(args, model, loc_model, eval_dataset, tokenizer, idx, eval_when_training=False, eval_with_mask=False):
    # build dataloader
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0)
    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)
    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()
    eval_loss, num = 0, 0
    loss_list = list()
    commit_index_list = list()
    bar = tqdm(eval_dataloader, total=len(eval_dataloader))
    for batch in bar:
        (input_ids, _, repair_input_ids, commit_index) = [x.to(args.device) for x in batch]
        if eval_with_mask:
            vul_query_mask = loc_model(input_ids)
            outputs = model(input_ids=input_ids, vul_query_mask=vul_query_mask, repair_input_ids=repair_input_ids)
        else:
            outputs = model(input_ids=input_ids, vul_query_mask=None, repair_input_ids=repair_input_ids)
        # loss = outputs.loss
        loss_con = outputs.loss
        inputs, targets_a, targets_b, lam = mixup_data(outputs.logits, repair_input_ids, alpha=5.0)
        logits = outputs.logits
        logits = torch.tensor(logits, requires_grad=True, dtype=torch.float32).to(args.device)
        logits = logits.reshape(-1, logits.size(-1))
        targets_a = targets_a.to(args.device, dtype=torch.long)
        targets_b = targets_b.to(args.device, dtype=torch.long)
        targets_a = targets_a.flatten()
        targets_b = targets_b.flatten()
        loss_sup = mixup_criterion(logits, targets_a, targets_b, lam)
        loss = 0.5 * loss_sup + loss_con

        lm_logits = outputs.logits
        loss_fct = CrossEntropyLoss(ignore_index=0, reduction='none')
        # logger.info("lm_logits.size:{}".format(lm_logits.size))
        loss_batch = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), repair_input_ids.view(-1)).view(-1,
                                                                                                      lm_logits.size(1))
        loss_batch = torch.nanmean(loss_batch.masked_fill(repair_input_ids == tokenizer.pad_token_id, torch.nan), dim=1)
        loss_list += loss_batch.tolist()
        commit_index_list += commit_index.tolist()

        eval_loss += loss.item()
        num += 1
    pathh = "bbb"
    os.makedirs(pathh, exist_ok=True)
    with open(os.path.join(pathh, "loss_list_cur_epoch_{}.pickle".format(idx)), "wb") as f:
        pickle.dump(loss_list, f)
    with open(os.path.join(pathh, "commit_index_list_cur_epoch_{}.pickle".format(idx)), "wb") as f:
        pickle.dump(commit_index_list, f)
    if idx >= 3:
        score_file_path = os.path.join(pathh, "loss_list_cur_epoch_{}.pickle".format(idx))
        commit_index_file = os.path.join(pathh, "commit_index_list_cur_epoch_{}.pickle".format(idx))
        if score_file_path.split(".")[-1] == "loss":
            with open(score_file_path) as score_f:
                score_list = score_f.read().split("\n")
        elif score_file_path.split(".")[-1] == "pickle":
            with open(score_file_path, "rb") as score_f:
                score_list = pickle.load(score_f)
        with open(commit_index_file, "rb") as f:
            commit_index_list = pickle.load(f)
        logger.info("There are {} commit messages are calculated in the probability".format(len(commit_index_list)))
        if set(commit_index_list) != set(range(len(commit_index_list))):
            logger.info("Some indexes are missing in commit_index_list")
        unsorted_score_list = [float(item) for item in score_list]
        score_list = []
        for i in np.argsort(commit_index_list):
            score_list.append(unsorted_score_list[i])

        # Get distribution by EM algorithm
        ratio, avg, std = EM(torch.tensor(score_list, device=args.device, dtype=torch.float64))
        if avg[0] < avg[1]:
            clean_idx = 0
        else:
            clean_idx = 1
        # Get Confidence of loss of each commit
        # NOT IN ORDER
        logger.info("Calculating the clean_weight ...")
        with multiprocessing.Pool(min(multiprocessing.cpu_count(), 24)) as pool:
            commit_data = [(idxb, score, ratio[clean_idx], avg[clean_idx], std[clean_idx] * 9) for idxb, score in
                           enumerate(score_list)]
            idx_clean_weight_list = pool.map(cal_Prob, commit_data)
        logger.info("Calculating the noisy_weight ...")
        with multiprocessing.Pool(min(multiprocessing.cpu_count(), 24)) as pool:
            commit_data = [(idxb, score, ratio[1 - clean_idx], avg[1 - clean_idx], std[1 - clean_idx]) for idxb, score
                           in enumerate(score_list)]
            idx_noisy_weight_list = pool.map(cal_Prob, commit_data)
        # IN ORDER
        sorted_idx_clean_weight_list = sorted(idx_clean_weight_list, key=lambda x: x[0])
        sorted_idx_noisy_weight_list = sorted(idx_noisy_weight_list, key=lambda x: x[0])
        clean_weight_list = [idx_weight[1] for idx_weight in sorted_idx_clean_weight_list]
        noisy_weight_list = [idx_weight[1] for idx_weight in sorted_idx_noisy_weight_list]

        # clean_weight_list = [cal_Prob(each_data, ratio[clean_idx], avg[clean_idx], std[clean_idx]*9) for each_data in score_list]
        # noisy_weight_list = [cal_Prob(each_data, ratio[1-clean_idx], avg[1-clean_idx], std[1-clean_idx]) for each_data in score_list]
        logger.info("Calculating the confidence ...")
        confidence_list = []
        fenzi = []
        fenmu = []
        cleann = []
        noisyy = []
        for idxc in range(len(score_list)):
            denom = clean_weight_list[idxc] + noisy_weight_list[idxc]
            if denom == 0:
                con = 1
            else:
                con = (clean_weight_list[idxc] + (np.power(1.6, -score_list[idxc])) * noisy_weight_list[idxc]) / (
                        clean_weight_list[idxc] + noisy_weight_list[idxc])
            confidence_list.append(con)
            cleann.append(clean_weight_list[idxc])
            noisyy.append(noisy_weight_list[idxc])
            fenzi.append(clean_weight_list[idxc] + (np.power(1.6, -score_list[idxc])) * noisy_weight_list[idxc])
            fenmu.append(clean_weight_list[idxc] + noisy_weight_list[idxc])
        logger.info("Calculate over ..")
        os.makedirs(pathh, exist_ok=True)
        with open(os.path.join(pathh, "cleann_cur_epoch_{}.pickle".format(idx)), "wb") as f:
            pickle.dump(cleann, f)
        with open(os.path.join(pathh, "noisyy_cur_epoch_{}.pickle".format(idx)), "wb") as f:
            pickle.dump(noisyy, f)
        with open(os.path.join(pathh, "confidence_list_cur_epoch_{}.pickle".format(idx)), "wb") as f:
            pickle.dump(confidence_list, f)
        with open(os.path.join(pathh, "fenzi_cur_epoch_{}.pickle".format(idx)), "wb") as f:
            pickle.dump(fenzi, f)
        with open(os.path.join(pathh, "fenmu_cur_epoch_{}.pickle".format(idx)), "wb") as f:
            pickle.dump(fenmu, f)
    eval_loss = round(eval_loss / num, 5)
    model.train()
    logger.info("***** Eval results *****")
    # logger.info(f"Evaluation Loss: {str(eval_loss)}")
    return confidence_list


def evaluate(args, model, loc_model, eval_dataset, tokenizer, confidence_list, idx, eval_when_training=False,
             eval_with_mask=False):
    # build dataloader
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0)
    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)
    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()
    eval_loss, num = 0, 0
    loss_list = list()
    commit_index_list = list()
    bar = tqdm(eval_dataloader, total=len(eval_dataloader))
    for batch in bar:
        (input_ids, _, repair_input_ids, commit_index) = [x.to(args.device) for x in batch]
        if eval_with_mask:
            vul_query_mask = loc_model(input_ids)
            outputs = model(input_ids=input_ids, vul_query_mask=vul_query_mask, repair_input_ids=repair_input_ids)
        else:
            outputs = model(input_ids=input_ids, vul_query_mask=None, repair_input_ids=repair_input_ids)
        # loss = outputs.loss

        loss_con = outputs.loss
        inputs, targets_a, targets_b, lam = mixup_data(outputs.logits, repair_input_ids, alpha=5.0)
        logits = outputs.logits
        logits = torch.tensor(logits, requires_grad=True, dtype=torch.float32).to(args.device)
        logits = logits.reshape(-1, logits.size(-1))
        targets_a = targets_a.to(args.device, dtype=torch.long)
        targets_b = targets_b.to(args.device, dtype=torch.long)
        targets_a = targets_a.flatten()
        targets_b = targets_b.flatten()
        loss_sup = mixup_criterion(logits, targets_a, targets_b, lam)
        loss = 0.5 * loss_sup + loss_con

        lm_logits = outputs.logits
        loss_fct = CrossEntropyLoss(ignore_index=0, reduction='none')
        # logger.info("lm_logits.size:{}".format(lm_logits.size))
        loss_batch = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), repair_input_ids.view(-1)).view(-1,
                                                                                                      lm_logits.size(1))
        loss_batch = torch.nanmean(loss_batch.masked_fill(repair_input_ids == tokenizer.pad_token_id, torch.nan), dim=1)
        loss_list += loss_batch.tolist()
        commit_index_list += commit_index.tolist()
        if idx > 3:
            loss_batch = [loss * confidence_list[commit_index[idxa]] for idxa, loss in enumerate(loss_batch)]
            loss_batch = torch.stack(loss_batch, dim=0)
            loss = torch.mean(loss_batch)

        eval_loss += loss.item()
        num += 1
    eval_loss = round(eval_loss / num, 5)
    model.train()
    logger.info("***** Eval results *****")
    logger.info(f"Evaluation Loss: {str(eval_loss)}")
    return eval_loss


def test(args, model, loc_model, tokenizer, test_dataset):
    # build dataloader
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size, num_workers=0)
    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # Test!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    nb_eval_steps = 0
    model.eval()
    accuracy = []
    raw_predictions = []
    correct_prediction = ""
    bar = tqdm(test_dataloader, total=len(test_dataloader))
    for batch in bar:
        correct_pred = False
        (input_ids, vq, repair_input_ids, _) = [x.to(args.device) for x in batch]
        vul_query_mask = loc_model(input_ids)
        with torch.no_grad():
            beam_outputs = model(input_ids=input_ids, repair_input_ids=repair_input_ids, vul_query_mask=vul_query_mask,
                                 generate_repair=True)
        beam_outputs = beam_outputs.detach().cpu().tolist()
        repair_input_ids = repair_input_ids.detach().cpu().tolist()
        for single_output in beam_outputs:
            # pred
            prediction = tokenizer.decode(single_output, skip_special_tokens=False)
            prediction = clean_tokens(prediction)
            # truth
            ground_truth = tokenizer.decode(repair_input_ids[0], skip_special_tokens=False)
            ground_truth = clean_tokens(ground_truth)
            if prediction == ground_truth:
                correct_prediction = prediction
                correct_pred = True
                break
        if correct_pred:
            raw_predictions.append(correct_prediction)
            accuracy.append(1)
        else:
            # if not correct, use the first output in the beam as the raw prediction
            raw_pred = tokenizer.decode(beam_outputs[0], skip_special_tokens=False)
            raw_pred = clean_tokens(raw_pred)
            raw_predictions.append(raw_pred)
            accuracy.append(0)
        nb_eval_steps += 1
        t = str(round(sum(accuracy) / len(accuracy), 4))
        bar.set_description(f"test acc: {t}")
    # calculate accuracy
    test_result = round(sum(accuracy) / len(accuracy), 4)
    logger.info("***** Test results *****")
    logger.info(f"Test Accuracy: {str(test_result)}")


def main():
    parser = argparse.ArgumentParser()
    # Params
    parser.add_argument("--train_data_file", default=None, type=str, required=False,
                        help="The input training data file (a csv file).")
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    ## Other parameters
    parser.add_argument("--encoder_block_size", default=512, type=int,
                        help="")
    parser.add_argument("--vul_repair_block_size", default=256, type=int,
                        help="")

    parser.add_argument("--num_beams", default=50, type=int,
                        help="Beam size to use when decoding.")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--model_name", default="model.bin", type=str,
                        help="Saved model name.")
    parser.add_argument("--checkpoint_model_name", default="non_domain_model.bin", type=str,
                        help="Checkpoint model name.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--use_non_pretrained_model", action='store_true', default=False,
                        help="Whether to use non-pretrained model.")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--load_pretrained_t5", default=False, action='store_true',
                        help="Whether to load model from checkpoint.")
    parser.add_argument("--load_pretrained_model", default=False, action='store_true',
                        help="Whether to load model from checkpoint.")
    parser.add_argument("--pretrained_model_name", default="pretrained_model.bin", type=str,
                        help="")

    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--warmup", action='store_true',
                        help="")
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for AdamW.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epochs', type=int, default=1,
                        help="training epochs")

    args = parser.parse_args()
    # Setup CUDA, GPU
    args.n_gpu = 1
    args.device = "cuda:0"

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s", args.device, args.n_gpu)
    # Set seed
    set_seed(args)

    tok = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    tok.add_tokens(["<S2SV_StartBug>", "<S2SV_EndBug>", "<S2SV_blank>", "<S2SV_ModStart>", "<S2SV_ModEnd>"])
    encoder = RobertaModel.from_pretrained("microsoft/codebert-base")
    encoder.resize_token_embeddings(len(tok))
    loc_model = LocModel(encoder=encoder, config=encoder.config, tokenizer=tok, args=args, num_labels=2)
    loc_model.load_state_dict(
        torch.load("./saved_models/checkpoint-best-loss/loc_fine_tuned_model.bin", map_location=args.device))
    loc_model.to(args.device)

    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.add_tokens(["<S2SV_StartBug>", "<S2SV_EndBug>", "<S2SV_blank>", "<S2SV_ModStart>", "<S2SV_ModEnd>"])
    start_bug_id = tokenizer.encode("<S2SV_StartBug>", add_special_tokens=False)[0]
    end_bug_id = tokenizer.encode("<S2SV_EndBug>", add_special_tokens=False)[0]
    tokenizer.start_bug_id = start_bug_id
    tokenizer.end_bug_id = end_bug_id

    t5 = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    t5.resize_token_embeddings(len(tokenizer))
    t5.config.use_encoder_vul_mask = True
    t5.config.use_decoder_vul_mask = True

    if args.load_pretrained_t5:
        t5.load_state_dict(
            torch.load(f"saved_models/checkpoint-best-loss/{args.pretrained_model_name}", map_location=args.device))

    model = AutoVF(t5=t5, tokenizer=tokenizer, args=args)

    if args.load_pretrained_model:
        checkpoint_prefix = f'checkpoint-best-loss/{args.pretrained_model_name}'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir, map_location=args.device))

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = TextDataset(tokenizer, args, file_type='train')
        eval_dataset = TextDataset(tokenizer, args, file_type='eval')
        if args.warmup:
            train(args, train_dataset, model, loc_model, eval_dataset, tokenizer, train_with_mask=False)
        train(args, train_dataset, model, loc_model, eval_dataset, tokenizer, train_with_mask=True)

    if args.do_test:
        checkpoint_prefix = f'checkpoint-best-loss/{args.model_name}'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir, map_location=args.device))
        model.to(args.device)
        test_dataset = TextDataset(tokenizer, args, file_type='test')
        test(args, model, loc_model, tokenizer, test_dataset)


if __name__ == "__main__":
    main()
