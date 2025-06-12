import torch
import torch.nn as nn

# 生成漏洞掩码进行漏洞定位的模型
class LocModel(nn.Module):
    def __init__(self, encoder, config, tokenizer, args, num_labels):
        super(LocModel, self).__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.args = args
        self.alpha = 1.0  # 初始 alpha 值

    def forward(self, input_ids, vul_query_label=None):
        # 获取模型输出
        outputs = self.encoder(input_ids, attention_mask=input_ids.ne(self.tokenizer.pad_token_id)).last_hidden_state
        outputs = torch.amax(outputs, dim=-1)
        vul_query_prob = torch.sigmoid(outputs)

        if vul_query_label is not None:
            # 使用 lid_paced_loss 计算损失
            loss_fn = self.lid_paced_loss(alpha=self.alpha)
            vq_loss = loss_fn(vul_query_label, vul_query_prob)
            if vq_loss.dim() > 0:  # 如果 vq_loss 不是标量
                vq_loss = vq_loss.mean()  # 对 vq_loss 进行聚合
            return vq_loss
        else:
            # 生成漏洞掩码
            vul_query_mask = self.mask_activation(vul_query_prob)
            vul_query_mask = vul_query_mask.unsqueeze(-1).expand(vul_query_mask.shape[0], vul_query_mask.shape[1], 768)
            return vul_query_mask

    def mask_activation(self, prob, beta=0.1, alpha=1000):
        x = beta / (1 + torch.exp(-alpha * (prob - 0.5))).float()
        x = torch.where(x == torch.tensor(beta, device=self.args.device).float(), x,
                        torch.tensor(0, device=self.args.device).float())
        return x

    @staticmethod
    def symmetric_cross_entropy(alpha, beta):
        """
        对称交叉熵损失函数：
        ICCV2019 "Symmetric Cross Entropy for Robust Learning with Noisy Labels"
        https://arxiv.org/abs/1908.06112
        """

        def loss(y_true, y_pred):
            # 确保预测值和真实值在有效范围内，避免数值不稳定
            y_pred_1 = torch.clamp(y_pred, min=1e-7, max=1.0)  # 防止 log(0)
            y_true_2 = torch.clamp(y_true, min=1e-4, max=1.0)  # 防止 log(0)

            # 计算第一部分：标准交叉熵损失
            ce_loss = -torch.sum(y_true * torch.log(y_pred_1), dim=-1)

            # 计算第二部分：反向交叉熵损失
            reverse_ce_loss = -torch.sum(y_pred * torch.log(y_true_2), dim=-1)

            # 返回加权和
            return alpha * torch.mean(ce_loss) + beta * torch.mean(reverse_ce_loss)

        return loss

    @staticmethod
    def lid_paced_loss(alpha=1.0, beta1=0.1, beta2=1.0):
        """
        LID 调整的损失函数：
        根据 alpha 动态调整原始标签和预测标签的权重。
        """

        if alpha == 1.0:
            # 如果 alpha == 1.0，使用对称交叉熵损失
            return LocModel.symmetric_cross_entropy(alpha=beta1, beta=beta2)
        else:
            def loss(y_true, y_pred):
                # 将预测概率转换为独热编码
                pred_labels = torch.zeros_like(y_pred)
                pred_labels.scatter_(1, torch.argmax(y_pred, dim=1, keepdim=True), 1)

                # 混合原始标签和预测标签
                y_new = alpha * y_true + (1. - alpha) * pred_labels

                # 归一化预测概率
                y_pred = y_pred / torch.sum(y_pred, dim=-1, keepdim=True)
                y_pred = torch.clamp(y_pred, min=1e-7, max=1.0 - 1e-7)

                # 计算交叉熵损失
                return -torch.sum(y_new * torch.log(y_pred), dim=-1)

            return loss