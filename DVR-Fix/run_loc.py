from __future__ import absolute_import, division, print_function
import argparse
import logging
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from transformers import get_linear_schedule_with_warmup, RobertaTokenizer, RobertaModel
from tqdm import tqdm
import pandas as pd
from loc_bert import LocModel
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, cdist, squareform
import copy

cpu_cont = 16
logger = logging.getLogger(__name__)


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 input_ids,
                 vul_query_label,
                 repair_input_ids):
        self.input_ids = input_ids
        self.vul_query_label = vul_query_label
        self.repair_input_ids = repair_input_ids


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
        source = df["source"].tolist()
        repair_target = df["target"].tolist()
        for i in tqdm(range(len(source))):
            self.examples.append(convert_examples_to_features(source[i], repair_target[i], tokenizer, args))
        if file_type == "train":
            for example in self.examples[:3]:
                logger.info("*** Example ***")
                logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))
                logger.info("vul_query_label: {}".format(' '.join(map(str, example.vul_query_label))))
                logger.info("repair_input_ids: {}".format(' '.join(map(str, example.repair_input_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(
            self.examples[i].vul_query_label).float(), torch.tensor(self.examples[i].repair_input_ids)

    def __setitem__(self, idx, new_example):
        self.examples[idx] = new_example



def convert_examples_to_features(source, repair_target, tokenizer, args):
    # encode - subword tokenize
    input_ids = tokenizer.encode(source, truncation=True, max_length=args.encoder_block_size, padding='max_length')
    repair_input_ids = tokenizer.encode(repair_target, truncation=True, max_length=args.vul_repair_block_size,
                                        padding='max_length')
    # get first start bug token loc
    """
    if tokenizer.start_bug_id in input_ids and tokenizer.end_bug_id in input_ids:
        start_vul_query = (torch.tensor(input_ids) == tokenizer.start_bug_id).nonzero(as_tuple=False)[0].item() / args.encoder_block_size
        end_vul_query = (torch.tensor(input_ids) == tokenizer.end_bug_id).nonzero(as_tuple=False)[-1].item() / args.encoder_block_size
        vul_query = [start_vul_query, end_vul_query]
    else:
        vul_query = [0, 0]
    """
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
    return InputFeatures(input_ids, vul_query, repair_input_ids)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


# 对输入 X 中的每一个样本，基于 k 近邻估计它的局部本征维数 (LID)。
def get_lids_random_batch(model, X, k=20, batch_size=128):
    """
    Get the local intrinsic dimensionality of each Xi in X_adv
    estimated by k close neighbours in the random batch it lies in.
    :param model: if None: lid of raw inputs, otherwise LID of deep representations
    :param X: normal images (NumPy array)
    :param k: the number of nearest neighbours for LID estimation
    :param batch_size: default 128
    :return: lids: LID of normal images of shape (num_examples, lid_dim)
    """
    # 如果没有提供模型，则直接计算 LID
    if model is None:
        lids = []
        # 样本总数量/batch_size   得到一个轮次有几个batch
        n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
        for i_batch in range(n_batches):
            start = i_batch * batch_size
            end = min(len(X), (i_batch + 1) * batch_size)
            X_batch = torch.tensor(X[start:end].reshape((end - start, -1)), dtype=torch.float32)

            # Maximum likelihood estimation of local intrinsic dimensionality (LID)
            lid_batch = mle_batch(X_batch, X_batch, k=k)
            lids.extend(lid_batch)

        lids = np.asarray(lids, dtype=np.float32)
        return lids

    # 如果提供了模型，则提取 "lid" 层的输出
    else:
        # 设置模型为评估模式
        model.eval()

        # 定义一个函数来提取 "lid" 层的输出（基于 encoder 的最后一层隐藏状态）
        def extract_lid_output(input_ids):
            with torch.no_grad():  # 不需要梯度
                # 获取 encoder 的最后一层隐藏状态
                outputs = model.encoder(
                    input_ids,
                    attention_mask=input_ids.ne(model.tokenizer.pad_token_id)
                ).last_hidden_state

                # 对序列维度取最大值（或平均值）以获得固定长度的特征表示
                outputs = outputs.amax(dim=1)  # 或者使用 .mean(dim=1) 取平均值
                return outputs

        # 分批处理数据
        def estimate(i_batch):
            start = i_batch * batch_size
            end = min(len(X), (i_batch + 1) * batch_size)
            n_feed = end - start

            X_batch = X[start:end]

            # 转换为张量并移动到适当的设备
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            X_batch = torch.tensor(X_batch, dtype=torch.long).to(device)

            # 提取 "lid" 层的输出
            X_act = extract_lid_output(X_batch).cpu().numpy()
            X_act = X_act.reshape((n_feed, -1))  # 展平为二维数组

            # 使用 mle_batch 计算 LID
            lid_batch = mle_batch(X_act, X_act, k=k)
            return lid_batch

        lids = []
        n_batches = int(np.ceil(len(X) / float(batch_size)))
        for i_batch in range(n_batches):
            lid_batch = estimate(i_batch)
            lids.extend(lid_batch)

        lids = np.asarray(lids, dtype=np.float32)
        return lids


def mle_batch(data, batch, k):
    """
    lid of a batch of query points X.
    numpy implementation.

    :param data:
    :param batch:
    :param k:
    :return:
    """
    data = np.asarray(data, dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)

    k = min(k, len(data) - 1)
    f = lambda v: - k / np.sum(np.log(v / v[-1] + 1e-8))
    a = cdist(batch, data)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:, 1:k + 1]
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a


def add_label_noise(dataset, noise_ratio):
    noisy_dataset = copy.deepcopy(dataset)
    n_samples = len(noisy_dataset)
    n_noisy = int(noise_ratio * n_samples)
    noisy_indices = np.random.choice(n_samples, n_noisy, replace=False)

    for idx in noisy_indices:
        original_features = noisy_dataset[idx]

        # 方式1：如果 original_features 是 (input_ids, vul_query, repair_input_ids)
        if isinstance(original_features, tuple):
            input_ids, vul_query, repair_input_ids = original_features
            flipped_vul_query = 1 - vul_query
            new_features = (input_ids, flipped_vul_query, repair_input_ids)
        # 方式2：如果是 InputFeatures 对象
        else:
            flipped_vul_query = 1 - original_features.vul_query
            new_features = InputFeatures(
                input_ids=original_features.input_ids,
                vul_query=flipped_vul_query,
                repair_input_ids=original_features.repair_input_ids
            )

        noisy_dataset[idx] = new_features
    return noisy_dataset


def run_lid_experiment(original_dataset, args, model, tokenizer, eval_dataset,
                       noise_levels=[0.1, 0.2, 0.3, 0.4], save_dir="lid_results", plot_name="lid_vs_noise.png"):
    """ 运行不同噪声比例下的 LID 训练实验并绘图保存 """

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    all_lids = {}

    # 逐个噪声比例执行训练并记录 LID 曲线
    for noise_ratio in noise_levels:
        print(f"\n>>> Running for noise ratio: {noise_ratio:.1%}")
        noisy_dataset = add_label_noise(original_dataset, noise_ratio)
        model_copy = copy.deepcopy(model)  # 避免参数污染
        lids = train(args, noisy_dataset, model_copy, tokenizer, eval_dataset)
        all_lids[noise_ratio] = lids

        # 保存单个噪声水平下的 LID 值
        np.save(os.path.join(save_dir, f"lid_noise_{int(noise_ratio * 100)}.npy"), np.array(lids))

    # 绘图
    plt.figure(figsize=(10, 6))
    for noise_ratio, lids in all_lids.items():
        plt.plot(lids, label=f"Noise {int(noise_ratio * 100)}%")

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Average LID", fontsize=12)
    plt.title("LID vs Epoch under Different Noise Levels", fontsize=14)
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(save_dir, plot_name)
    plt.savefig(plot_path)
    print(f"\n✅ Plot saved to: {plot_path}")
    plt.close()


def train(args, train_dataset, model, tokenizer, eval_dataset):
    """ Train the model """
    # build dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=0)

    args.max_steps = args.epochs * len(train_dataloader)

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
    # optimizer = Adafactor(optimizer_grouped_parameters, scale_parameter=False, relative_step=False, warmup_init=False, lr=args.learning_rate)
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

    # D2L 参数初始化
    lids = []  # 存储每个 epoch 的 LID 值
    alpha = 1.0  # 初始 alpha 值
    turning_epoch = -1  # 转折点的 epoch

    # 可调整的参数
    init_epoch = 5  # 初始轮数，在此之后才开始检测转折点
    epoch_win = 5  # 用于判断转折点时使用的窗口大小
    lid_subset_size = 1280  # 计算 LID 使用的子集大小
    lid_k = 20  # 计算 LID 使用的参数

    model.zero_grad()

    for idx in range(args.epochs):
        # 在每个 epoch 开始时计算 LID 值
        rand_idxes = np.random.choice(len(train_dataloader.dataset), lid_subset_size, replace=False)
        # X_subset = train_dataloader.dataset.tensors[0][rand_idxes]
        # 提取 input_ids 子集
        if isinstance(train_dataloader.dataset, TextDataset):
            # 从 TextDataset 中提取 input_ids
            X_subset = torch.stack([torch.tensor(train_dataloader.dataset[i][0]) for i in rand_idxes])
        else:
            raise ValueError("Unsupported dataset type")
        lid = np.mean(get_lids_random_batch(model, X_subset, k=lid_k, batch_size=128))
        lids.append(lid)
        # 检测转折点
        if len(lids) > init_epoch + epoch_win:
            smooth_lids = lids[-epoch_win - 1:-1]
            if lids[-1] - np.mean(smooth_lids) > 2 * np.std(smooth_lids):
                turning_epoch = len(lids) - 2
                expansion = lids[-1] / np.min(lids)
                alpha = np.exp(-idx / args.epochs * expansion)
                print(f"Turning epoch: {turning_epoch}, Alpha updated to: {alpha:.2f}")

        # 更新模型的 alpha 值
        model.alpha = alpha

        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            model.train()
            (input_ids, vul_query_label, _) = [x.to(args.device) for x in batch]
            # the forward function automatically creates the correct decoder_input_ids
            loss = model(input_ids=input_ids, vul_query_label=vul_query_label)
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

    return lids


def clean_tokens(tokens):
    tokens = tokens.replace("<pad>", "")
    tokens = tokens.replace("<s>", "")
    tokens = tokens.replace("</s>", "")
    tokens = tokens.strip("\n")
    tokens = tokens.strip()
    return tokens


def evaluate(args, model, tokenizer, eval_dataset, eval_when_training=False):
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
    bar = tqdm(eval_dataloader, total=len(eval_dataloader))
    for batch in bar:
        (input_ids, vul_query_label, _) = [x.to(args.device) for x in batch]
        loss = model(input_ids=input_ids, vul_query_label=vul_query_label)
        eval_loss += loss.item()
        num += 1
    eval_loss = round(eval_loss / num, 5)
    model.train()
    logger.info("***** Eval results *****")
    logger.info(f"Evaluation Loss: {str(eval_loss)}")
    return eval_loss


def test(args, model, eval_dataset, eval_when_training=False):
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
    bar = tqdm(eval_dataloader, total=len(eval_dataloader))
    for batch in bar:
        (input_ids, vul_query_label, _) = [x.to(args.device) for x in batch]
        loss = model(input_ids=input_ids, vul_query_label=vul_query_label, test_loc=True)
        eval_loss += loss.item()
        num += 1
    eval_loss = round(eval_loss / num, 5)
    model.train()
    logger.info("***** Test results *****")
    logger.info(f"Test Loss: {str(eval_loss)}")
    return eval_loss


def main():
    parser = argparse.ArgumentParser()
    # Params
    parser.add_argument("--train_data_file", default=None, type=str, required=False,
                        help="The input training data file (a csv file).")
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    ## Other parameters
    parser.add_argument("--model_type", default="roberta", type=str,
                        help="The model architecture to be fine-tuned.")

    parser.add_argument("--encoder_block_size", default=512, type=int,
                        help="")
    parser.add_argument("--vul_query_block_size", default=256, type=int,
                        help="")
    parser.add_argument("--vul_repair_block_size", default=256, type=int,
                        help="")

    parser.add_argument("--max_stat_length", default=-1, type=int,
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
    parser.add_argument("--load_pretrained_model", default=False, action='store_true',
                        help="Whether to load model from checkpoint.")
    parser.add_argument("--pretrained_model_name", default="pretrained_model.bin", type=str,
                        help="")

    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")

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

    parser.add_argument("--hidden_size", default=768, type=int,
                        help="hidden size.")
    parser.add_argument("--feature_dim_size", default=768, type=int,
                        help="feature dim size.")
    parser.add_argument("--num_GNN_layers", default=2, type=int,
                        help="num GNN layers.")
    parser.add_argument("--gnn", default="ReGCN", type=str, help="ReGCN or ReGGNN")
    parser.add_argument("--format", default="non-uni", type=str,
                        help="idx for index-focused method, uni for unique token-focused method")
    parser.add_argument("--window_size", default=3, type=int, help="window_size to build graph")
    parser.add_argument("--remove_residual", default=False, action='store_true', help="remove_residual")
    parser.add_argument("--att_op", default='mul', type=str,
                        help="using attention operation for attention: mul, sum, concat")

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
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.add_tokens(["<S2SV_StartBug>", "<S2SV_EndBug>", "<S2SV_blank>", "<S2SV_ModStart>", "<S2SV_ModEnd>"])
    start_bug_id = tokenizer.encode("<S2SV_StartBug>", add_special_tokens=False)[0]
    end_bug_id = tokenizer.encode("<S2SV_EndBug>", add_special_tokens=False)[0]
    tokenizer.start_bug_id = start_bug_id
    tokenizer.end_bug_id = end_bug_id

    encoder = RobertaModel.from_pretrained(args.model_name_or_path)
    encoder.resize_token_embeddings(len(tokenizer))
    model = LocModel(encoder=encoder, config=encoder.config, tokenizer=tokenizer, args=args, num_labels=512)

    model.to(args.device)

    if args.load_pretrained_model:
        checkpoint_prefix = f'checkpoint-best-loss/{args.pretrained_model_name}'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir, map_location=args.device))
    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = TextDataset(tokenizer, args, file_type='train')
        eval_dataset = TextDataset(tokenizer, args, file_type='eval')
        # train(args, train_dataset, model, tokenizer, eval_dataset)
        run_lid_experiment(original_dataset=train_dataset,
                           args=args,
                           model=model,
                           tokenizer=tokenizer,
                           eval_dataset=eval_dataset,
                           noise_levels=[0.0, 0.1, 0.2, 0.3, 0.4])

    if args.do_test:
        test_dataset = TextDataset(tokenizer, args, file_type='test')
        test(args, model, test_dataset, False)


if __name__ == "__main__":
    main()
