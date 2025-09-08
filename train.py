import os
import torch
import numpy as np
import random
from model.Config import Config
from dataset import create_dataset, collate
from torch.utils.data import DataLoader
from model.Unet import Unet
from torch.utils.tensorboard import SummaryWriter
from model.BLAST import BLAST
from model.loss import GeneralizedDiceLoss
import math
import sys
import torch.nn.functional as F
from metric_caculate import calculate_precision, calculate_recall, calculate_f1_score, calculate_number, calculate_iou_percent
from sklearn.model_selection import train_test_split
from early_stopping import EarlyStopping

def merge_segments(segments, min_len):
    if len(segments) == 0:
        return segments
    # 将线段按照起始点坐标升序排序
    segments = segments[np.argsort(segments[:, 0])]
    # 初始化合并后的线段列表
    merged_segments = [segments[0]]
    # 遍历每个线段
    for i in range(1, segments.shape[0]):
        # 如果当前线段与前一个线段之间的间隔小于等于min_len，则合并它们
        if segments[i, 0] - merged_segments[-1][1] <= min_len:
            merged_segments[-1][1] = segments[i, 1]
        # 否则，将当前线段加入合并后的线段列表
        else:
            merged_segments.append(segments[i])
    return np.array(merged_segments)



def binary_to_array(x, len_threshold):
    """ Return [start, duration] from binary array

    binary_to_array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1])
    [[4, 8], [11, 13]]
    """

    tmp = np.array([0] + list(x) + [0])
    all_events = np.where((tmp[1:] - tmp[:-1]) != 0)[0].reshape((-1, 2))
    # # # 如果两个事件之间间隔小于0.1s，那么就合并连个事件
    # all_events = merge_segments(all_events, int(0.1 * 100))
    valid_event = (all_events[:, 1] - all_events[:, 0]) >= len_threshold
    return all_events[valid_event, :].tolist()

def moving_average(logits, moving_avg_size=42):
    """
    The previously calculated logits are averaged using a sliding window with step size one and
    the configured window width.

    Parameters
    ----------
    logits : torch.Tensor
        The values as calculated by `dense_logits` in format [N,C,K] where N is the batch size, C the number of
        classes and K the number of elements in each observation.

    Returns
    -------
    averaged_logits : torch.Tensor
        The averaged results in format [N,C,K] where N is the batch size, C the number of classes and K the number
        of elements in each observation.
    """

    # zero padding before moving average
    s = moving_avg_size - 1
    logits = F.pad(logits, (s // 2, s // 2 + s % 2), mode='constant', value=0)

    return F.avg_pool1d(logits, moving_avg_size, stride=1)

@torch.no_grad()
def predict_segment_night(model, inference_dataloader, config:Config2, moving_avg_size:int):
    # Set network to eval mode
    model.eval()
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for signal, events in inference_dataloader:
        batch_predictions = model(signal)
        batch_predictions = moving_average(batch_predictions, moving_avg_size)
        batch_prob = F.softmax(batch_predictions, dim=1)
        for prob, event in zip(batch_prob, events):
            # 选择概率值较大的类别作为标签
            predicted_labels = torch.argmax(prob, dim=0)
            predicted_events = binary_to_array(predicted_labels.cpu(), int(0.3 * config.signal_fs))
            predicted_events = np.array(predicted_events)
            if event.size()[0] == 0:
                # 没有事件但是预测出来事件，都是false_positive
                false_positive += predicted_events.shape[0]
            elif predicted_events.shape[0] == 0:
                # 有事件但是没有预测出来，都是false_negative
                false_negative += event.size()[0]
            else:
                true_event = event[:, :2].cpu() * signal.size()[-1]
                res = calculate_number(predicted_events, true_event, cfg.min_iou)
                true_positive += res[0]
                false_positive += res[1]
                false_negative += res[2]

    if true_positive + false_positive == 0:
        precision = 0
    else:
        precision = true_positive / (true_positive + false_positive)

    if true_positive + false_positive == 0:
        recall = 0
    else:
        recall = true_positive / (true_positive + false_negative)
    if precision == 0 or recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * precision * recall / (precision + recall)

    metrics_test = { metric: [] for metric in ["precision", "recall", "f1"] }
    metrics_test["precision"] = precision
    metrics_test["recall"] = recall
    metrics_test["f1"] = f1_score
    print(metrics_test)
    return metrics_test

@torch.no_grad()
def predict_whole_night(model, inference_dataset, batch_size, config:Config2, moving_avg_size:int):
    """
    将所有预测映射到整晚尺度然后再计算TP\FP\FN，然后计算指标
    """
    # Set network to eval mode
    model.eval()

    # Set network prediction parameters
    window_size = inference_dataset.window_size

    # List of dicts, to save predictions of each class per record
    predictions = {}
    for record in inference_dataset.records:
        predictions[record] = []
        result = np.zeros(inference_dataset.signals[record]["size"])
        for signals, times in inference_dataset.get_record_batch(record, batch_size=int(batch_size)):
            # 一个时间序列的logits
            batch_predictions = model(signals)
            batch_predictions = moving_average(batch_predictions, moving_avg_size)
            batch_prob = F.softmax(batch_predictions, dim=1)
            for prob, time in zip(batch_prob, times):
                # 选择概率值较大的类别作为标签
                predicted_labels = torch.argmax(prob, dim=0)
                # # # 只取最中间部分的结果
                # start = int(window_size / 4)
                # end = int(start + window_size / 2)
                # result[start + time[0] : end + time[0]] = predicted_labels[start : end].cpu()
                result[time[0] : window_size + time[0]] = predicted_labels[ : ].cpu()

        predicted_events = binary_to_array(result, int(0.3 * config.signal_fs))
        predictions[record] = predicted_events

    return predictions

@torch.no_grad()
def compute_metrics_whole_night(model, dataset, config, moving_avg_size):
    all_predicted_events = predict_whole_night(model, dataset, 128, config, moving_avg_size)

    metrics_test = { metric: [] for metric in ["precision", "recall", "f1"] }

    found_some_events = False
    for record in dataset.records:
        # Select current event predictions
        predicted_events = all_predicted_events[record]
        # If no predictions skip record, else some_events = 1
        if len(predicted_events) == 0:
            continue

        found_some_events = True
        # Select current true events
        events = dataset.get_record_events(record)[0]

        predicted_events = np.array(predicted_events)
        # Compute_metrics(events, predicted_events, threshold)
        metrics_test["precision"].append(calculate_precision(predicted_events, events, cfg.min_iou))
        metrics_test["recall"].append(calculate_recall(predicted_events, events, cfg.min_iou))
        metrics_test["f1"].append(calculate_f1_score(predicted_events, events, cfg.min_iou))

        percent_list, true_positive = calculate_iou_percent(predicted_events, events)
        # for i in percent_list: print("{:.17f} ".format(i / true_positive), end='')  # 不换行输出
        for i in percent_list: print("{0:>5} ".format(i), end=',')  # 不换行输出
        print()

        event_num.append(len(predicted_events))
        ave_duration.append(np.mean((predicted_events[:, 1] - predicted_events[:, 0]) / 100))
    # If for any event and record the network predicted events, return -1
    if found_some_events is False:
        metrics_test["precision"] = [0]
        metrics_test["recall"] = [0]
        metrics_test["f1"] = [0]

    for metric in ["precision", "recall", "f1"]:
        metrics_test[metric] = np.nanmean(np.array(metrics_test[metric]))
    # print(metrics_test)
    return metrics_test, all_predicted_events



def events_2_mask(batch_events, signal_length, device):
    # 创建一个长度为signal_length的Tensor，初始值为0
    batch_masks = torch.zeros(len(batch_events), signal_length, device=device)
    # 遍历每个事件，将事件范围内的位置标记为1
    for i, events in enumerate(batch_events):
        for event in events:
            start_relative, end_relative = event[:2]
            start_index = int(start_relative * signal_length)
            end_index = int(end_relative * signal_length)
            batch_masks[i, start_index:end_index + 1] = 1
    return batch_masks

def train(model, optimizer, data_loader, epoch, loss_func, writer, lr_scheduler=None):
    model.train()
    loss_list = []

    for signal, events in data_loader:
        results = model(signal)
        batch_masks = events_2_mask(events, signal.size()[-1], signal.device)
        loss = loss_func(results, batch_masks)
        if not math.isfinite(loss):  # 当计算的损失为无穷大时停止训练
            print("Loss is {}, stopping training".format(loss))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        loss_list.append(loss.item())

    # print("loss:{}".format(np.mean(loss_list)))
    if writer is not None:
        writer.add_scalar('train/loss', np.mean(loss_list), epoch)

@torch.no_grad()
def valuate(model, data_loader, loss_func, early_stopping):
    """
    验证集验证
    """
    model.eval()
    loss_list = []
    for signal, events in data_loader:
        results = model(signal)
        batch_masks = events_2_mask(events, signal.size()[-1], signal.device)
        loss = loss_func(results, batch_masks)
        loss_list.append(loss.cpu())

    early_stopping(np.mean(loss_list), model)


def main(cfg: Config2, test_record, weight_path, moving_avg_size):
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    # print("Using {} device training. Test: {}".format(device.type, test_record))

    train_records = [x for x in os.listdir(cfg.file_directory) if x != '.cache' and x not in test_record]
    test_records = test_record

    train_records, val_records = train_test_split(train_records, train_size=0.9, test_size=0.1)

    train_dataset = create_dataset(
        train_records, cfg.file_directory, cfg.chan_list, cfg.event_list, cfg.signal_fs, cfg.window,cfg.slide_window,
        transformations=True,training=True
    )

    val_dataset = create_dataset(
        val_records, cfg.file_directory, cfg.chan_list, cfg.event_list, cfg.signal_fs, cfg.window,cfg.window,
        transformations=False,training=False
    )

    test_dataset = create_dataset(
        test_records, cfg.file_directory, cfg.chan_list, cfg.event_list, cfg.signal_fs, cfg.window,cfg.window,
        transformations=False,training=False
    )

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, collate_fn=collate)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, collate_fn=collate)

    test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, collate_fn=collate)

    model = BLAST()
    model.to(device)

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    one_clr = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, total_steps=len(train_dataloader) * cfg.epochs,
        pct_start=0.3, anneal_strategy="cos",
        div_factor=25, final_div_factor=25
    )
    loss = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2], device=device))

    early_stopping = EarlyStopping(patience=10, verbose=True,
                                   path=weight_path)

    for epoch in range(cfg.epochs):
        print("Epoch:{}".format(epoch))
        train(model, optimizer, train_dataloader, epoch, loss, None, one_clr)
        valuate(model, val_dataloader, loss, early_stopping)
        if early_stopping.early_stop:
            break

    model.load_state_dict(torch.load(weight_path))
    metrics_test_segment = predict_segment_night(model, test_dataloader, cfg, moving_avg_size)
    metrics_test, all_predicted_events = compute_metrics_whole_night(model, test_dataset, cfg, moving_avg_size)
    if cfg.save_event:
        save_all_predicted_events(all_predicted_events, "F:/Spindle/my_code/result/MASS/pred_events", cfg.signal_fs)
    return metrics_test_segment, metrics_test


def seed_everything(seed: int):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def show_res(precision, recall, f1):
    for p, r, f in zip(precision, recall, f1):
        print("{:.17f}, {:.17f}, {:.17f}".format(p, r, f))
    print(np.mean(precision))
    print(np.mean(recall))
    print(np.mean(f1))

def iou_threshold():
    # 设置随机种子, 保证每次训练结果尽量一致
    seed_everything(42)
    torch.backends.cudnn.deterministic = True

    cfg = Config2()
    records = [x for x in os.listdir(cfg.file_directory) if x != '.cache']
    records.sort()
    segment_precision = []
    segment_recall = []
    segment_f1 = []

    numbers = [i * 0.05 for i in range(1, 20)]
    for i in numbers:
        cfg.min_iou = i
        precision = []
        recall = []
        f1 = []
        for record in records:
            test_record = [record]
            weight_name = "{}".format(record)
            weight_path = "F:/Spindle/my_code/weight/one_channel_sigma_band/{}.pth".format(
                weight_name)
            metrics_test_segment, metrics_test = main(cfg, test_record, weight_path, 40)
            # segment_precision.append(metrics_test_segment["precision"])
            # segment_recall.append(metrics_test_segment["recall"])
            # segment_f1.append(metrics_test_segment["f1"])
            precision.append(metrics_test["precision"])
            recall.append(metrics_test["recall"])
            f1.append(metrics_test["f1"])
            torch.cuda.empty_cache()
        # print("Segment Res:##########################################")
        # show_res(segment_precision, segment_recall, segment_f1)
        # print("Whole night Res:######################################")
        # show_res(precision, recall, f1)
        segment_precision.append(np.mean(precision))
        segment_recall.append(np.mean(recall))
        segment_f1.append(np.mean(f1))
    print(segment_precision)
    print(segment_recall)
    print(segment_f1)



def save_all_predicted_events(all_predicted_events, directory, signal_fs):
    for key, value in all_predicted_events.items():
        all_events = np.array(value, dtype=np.float)
        all_events[:, :2] = all_events[:, :2] / signal_fs
        np.savetxt(os.path.join(directory, key.split(".")[0]+".csv"), all_events, delimiter=',', fmt='%f')

event_num = []
ave_duration = []
if __name__ == '__main__':
    # 设置随机种子, 保证每次训练结果尽量一致
    seed_everything(42)
    torch.backends.cudnn.deterministic = True

    cfg = Config()
    records = [x for x in os.listdir(cfg.file_directory) if x != '.cache']
    records.sort()
    segment_precision = []
    segment_recall = []
    segment_f1 = []
    precision = []
    recall = []
    f1 = []
    for record in records:
        test_record = [record]
        weight_name = "{}".format(record)
        weight_path = "F:/Spindle_Work/my_code/weight/BLAST/{}.pth".format(weight_name)
        metrics_test_segment, metrics_test = main(cfg, test_record, weight_path, 42)
        segment_precision.append(metrics_test_segment["precision"])
        segment_recall.append(metrics_test_segment["recall"])
        segment_f1.append(metrics_test_segment["f1"])
        precision.append(metrics_test["precision"])
        recall.append(metrics_test["recall"])
        f1.append(metrics_test["f1"])
        torch.cuda.empty_cache()
        # break
    print("Segment Res:##########################################")
    show_res(segment_precision, segment_recall, segment_f1)
    print("Whole night Res:######################################")
    show_res(precision, recall, f1)
    # print("###################")
    # print(event_num)
    # print(ave_duration)