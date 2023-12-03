import math
from collections import defaultdict
import os


import torch
from torch import nn
from torch.nn import functional as F

from torchdrug import core, layers, tasks, metrics, utils
from torchdrug.core import Registry as R
from torchdrug.layers import functional
from sklearn.metrics import confusion_matrix
import numpy as np

@R.register("tasks.PropertyPrediction")
class PropertyPrediction(tasks.Task, core.Configurable):
    """
    Graph / molecule / protein property prediction task.

    This class is also compatible with semi-supervised learning.

    Parameters:
        model (nn.Module): graph representation model
        task (str, list or dict, optional): training task(s).
            For dict, the keys are tasks and the values are the corresponding weights.
        criterion (str, list or dict, optional): training criterion(s). For dict, the keys are criterions and the values
            are the corresponding weights. Available criterions are ``mse``, ``bce`` and ``ce``.
        metric (str or list of str, optional): metric(s).
            Available metrics are ``mae``, ``rmse``, ``auprc`` and ``auroc``.
        num_mlp_layer (int, optional): number of layers in mlp prediction head
        normalization (bool, optional): whether to normalize the target
        num_class (int, optional): number of classes
        mlp_batch_norm (bool, optional): apply batch normalization in mlp or not
        mlp_dropout (float, optional): dropout in mlp
        graph_construction_model (nn.Module, optional): graph construction model
        verbose (int, optional): output verbose level
    """

    eps = 1e-10
    _option_members = {"task", "criterion", "metric"}
 # fold_idx _to save diffrent fold outputs
    def __init__(self, model, task=(), criterion="mse", metric=("mae", "rmse"), num_mlp_layer=1,
                 normalization=True, num_class=None, mlp_batch_norm=False, mlp_dropout=0,
                 graph_construction_model=None, verbose=0):
        super(PropertyPrediction, self).__init__()
        self.model = model
        self.task = task
        self.criterion = criterion
        self.metric = metric
        self.num_mlp_layer = num_mlp_layer
        # self.fold_idx=fold_idx
        # For classification tasks, we disable normalization tricks.
        self.normalization = normalization and ("ce" not in criterion) and ("bce" not in criterion)
        self.num_class = (num_class,) if isinstance(num_class, int) else num_class
        self.mlp_batch_norm = mlp_batch_norm
        self.mlp_dropout = mlp_dropout
        self.graph_construction_model = graph_construction_model
        self.verbose = verbose
        self.all_name =[]
        

    def preprocess(self, train_set, valid_set, test_set):
        """
        Compute the mean and derivation for each task on the training set.
        """
        values = defaultdict(list)
        for sample in train_set:
            if not sample.get("labeled", True):
                continue
            # print(self.task)
            for task in self.task:
                # print('task')
                # print(self.task)
                # print(sample)
                # print(task)
                # print(sample[task])
                # print(sample['name'])
                if not math.isnan(sample[task]):


                    
                    values[task].append(sample[task])
        mean = []
        std = []
        weight = []
        num_class = []
        for task, w in self.task.items():
            value = torch.tensor(values[task])
            mean.append(value.float().mean())
            std.append(value.float().std())
            weight.append(w)
            if value.ndim > 1:
                num_class.append(value.shape[1])
            elif value.dtype == torch.long:
                task_class = value.max().item()
                if task_class == 1 and "bce" in self.criterion:
                    num_class.append(1)
                else:
                    num_class.append(task_class + 1)
            else:
                num_class.append(1)

        print(f'this is weight {weight}')

        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float))
        self.register_buffer("weight", torch.as_tensor(weight, dtype=torch.float))
        self.num_class = self.num_class or num_class

        hidden_dims = [self.model.output_dim] * (self.num_mlp_layer - 1)
        self.mlp = layers.MLP(self.model.output_dim, hidden_dims + [sum(self.num_class)],
                            batch_norm=self.mlp_batch_norm, dropout=self.mlp_dropout)

    def forward(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred = self.predict(batch, all_loss, metric)
        

        # print(f'inside forward { pred}')
        



        if all([t not in batch for t in self.task]):
            # unlabeled data
            return all_loss, metric

        target = self.target(batch)
        labeled = ~torch.isnan(target)
        target[~labeled] = 0

        for criterion, weight in self.criterion.items():
            if criterion == "mse":
                if self.normalization:
                    loss = F.mse_loss((pred - self.mean) / self.std, (target - self.mean) / self.std, reduction="none")
                else:
                    loss = F.mse_loss(pred, target, reduction="none")
            elif criterion == "bce":
                loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
            elif criterion == "ce":
                loss = F.cross_entropy(pred, target.long().squeeze(-1), reduction="none").unsqueeze(-1)
            else:
                raise ValueError("Unknown criterion `%s`" % criterion)
            loss = functional.masked_mean(loss, labeled, dim=0)

            name = tasks._get_criterion_name(criterion)
            if self.verbose > 0:
                for t, l in zip(self.task, loss):
                    metric["%s [%s]" % (name, t)] = l
            loss = (loss * self.weight).sum() / self.weight.sum()
            metric[name] = loss
            all_loss += loss * weight

        return all_loss, metric

    def predict(self, batch, all_loss=None, metric=None):
        # self.all_name =[]
        graph = batch["graph"]
        name_batch = batch['name']
        self.all_name.append(name_batch)
        # print(f'batch {batch}')
        # base_path = "/ocean/projects/bio230029p/bpokhrel/names"

        # file_prefix = "pp_name"
        # file_number = 1
        # while True:
        #     file_name = f"{file_prefix}{file_number}.npy"
        #     file_path = os.path.join(base_path, file_name)

        #     if not os.path.exists(file_path):
        #         # Files with these names do not exist, so save the data with these names
        #         np.save(file_path, all_name)
        #         # np.save(target_file_path, all_target_values)
        #         break  # Exit the loop

            
            # file_number += 1
        # print('allname')
        # print(self.allname)
        # print('Beforegraph')
        # print(graph)
        # print('Inside the property prediction')
        # print('graph.num_node')
        # print(graph.num_node)
        # print('graph.num_edge')
        # print(graph.num_edge)
        # print('graph.edge_list')
        # print(graph.edge_list)
        # print('graph.edge_weight')
        # print(graph.edge_weight)
        # print('New')
        # print('graph.bond_feature')
        # print(graph.bond_feature)
        if self.graph_construction_model:
            graph = self.graph_construction_model(graph)
            # print('graph')
            # print(graph)
            # print('Inside the property prediction')
            # print('graph.num_node')
            # print(graph.num_node)
            # print('graph.num_edge')
            # print(graph.num_edge)
            # print('graph.edge_list')
            # print(graph.edge_list)
            # print('graph.edge_weight')
            # print(graph.edge_weight)
            # print('New')
            # print('graph.bond_feature')
            # print(graph.bond_feature)

        output = self.model(graph, graph.node_feature.float(), all_loss=all_loss, metric=metric)
        pred = self.mlp(output["graph_feature"])
        if self.normalization:
            pred = pred * self.std + self.mean
        return pred

    def target(self, batch):
        target = torch.stack([batch[t].float() for t in self.task], dim=-1)
        labeled = batch.get("labeled", torch.ones(len(target), dtype=torch.bool, device=target.device))
        target[~labeled] = math.nan
        return target

    def evaluate(self, pred, target):
        labeled = ~torch.isnan(target)
        # base_path = "/ocean/projects/bio230029p/bpokhrel/names"
        # # print('self.all_name')
        
        # file_prefix = "oo1_name"
        # file_number = 1
        # while True:
        #     file_name = f"{file_prefix}{file_number}.npy"
        #     file_path = os.path.join(base_path, file_name)
           

            # if not os.path.exists(file_path):
            #     # Files with these names do not exist, so save the data with these names
            #     np.save(file_path, self.all_name)
            #     # np.save(target_file_path, all_target_values)
            #     break  # Exit the loop


            
            # file_number += 1
        

        metric = {}
        for _metric in self.metric:
            if _metric == "mae":
                score = F.l1_loss(pred, target, reduction="none")
                score = functional.masked_mean(score, labeled, dim=0)
            elif _metric == "rmse":
                score = F.mse_loss(pred, target, reduction="none")
                score = functional.masked_mean(score, labeled, dim=0).sqrt()
            elif _metric == "acc":
                print('inside acc pred')
                print(pred)
                print('inside acc target')
                print(target)
                score = []
                num_class = 0
                correct_predictions = 0
                total_predictions = 0
                all_pred=[]
                all_target=[]
                all_pred_values = []
                all_target_values = []
                print(f'self.num_class {self.num_class}')
                for i, cur_num_class in enumerate(self.num_class):
                    _pred = pred[:, num_class:num_class + cur_num_class]
                    _target = target[:, i]
                    _labeled = labeled[:, i]
                    _score = metrics.accuracy(_pred[_labeled], _target[_labeled].long())
                    score.append(_score)
                    # for CM, total_prediction, total_target
                    print(f'_pred[_labeled] {_pred[_labeled]}')
                    print(f'_target[_labeled].long(){_target[_labeled].long()}')
                    # # Append values to lists
                    # all_pred_values.append(_pred[_labeled])
                    # all_target_values.append(_target[_labeled].long())
                    # Move tensors from GPU to CPU and append values to lists
                    all_pred_values.append(_pred[_labeled].cpu().detach().numpy())
                    all_target_values.append(_target[_labeled].long().cpu().detach().numpy())

                    # score.append(_score)
                    num_class += cur_num_class
                     # For confusion matrix
                    _pred = _pred[_labeled].argmax(dim=1).cpu().numpy()
                    _target = _target[_labeled].long().cpu().numpy()
                    all_pred.extend(_pred)
                    all_target.extend(_target)
                    correct_predictions += (_pred == _target).sum()
                    total_predictions += len(_pred)
                    print('myaccu')
                    print(correct_predictions/total_predictions)
                # Convert lists to NumPy arrays
                # all_pred_array = np.concatenate(all_pred_values)
                # all_target_array = np.concatenate(all_target_values)
                score = torch.stack(score)
                #my_add
                # Save the NumPy arrays to files
                base_path = "/ocean/projects/bio230029p/bpokhrel/kfold_11_24_023"
                file_prefix = "kfold_pred_notrim"
                file_number = 1

                # Keep incrementing the file number until a non-existing file name is found
                while True:
                    pred_file_name = f"{file_prefix}{file_number}.npy"
                    target_file_name = f"kfold_target_no_trim{file_number}.npy"
                    pred_file_path = os.path.join(base_path, pred_file_name)
                    target_file_path = os.path.join(base_path, target_file_name)

                    if not os.path.exists(pred_file_path) and not os.path.exists(target_file_path):
                        # Files with these names do not exist, so save the data with these names
                        np.save(pred_file_path, all_pred_values)
                        np.save(target_file_path, all_target_values)
                        
                        break  # Exit the loop

                    # Increment the file number for the next iteration
                    file_number += 1
                # np.save(f"/ocean/projects/bio230029p/bpokhrel/kfold_pred{1}.npy", all_pred_values)
                # np.save(f"/ocean/projects/bio230029p/bpokhrel/kfold_target{1}.npy", all_target_values)
                # base_path = "/ocean/projects/bio230029p/bpokhrel/names"

                # file_prefix = "oo_name"
                # file_number = 1
                # while True:
                #     file_name = f"{file_prefix}{file_number}.npy"
                #     file_path = os.path.join(base_path, file_name)
                #     print('self.all_name')
                #     print(self.all_name)

                #     if not os.path.exists(file_path):
                #         # Files with these names do not exist, so save the data with these names
                #         np.save(file_path, self.all_name)
                #         # np.save(target_file_path, all_target_values)
                #         break  # Exit the loop


                    
                #     file_number += 1
                
                all_pred_values=np.array(all_pred_values)
                all_target_values=np.array(all_target_values)
                cm = confusion_matrix(all_target, all_pred, labels=list(range(23)))
                print("Confusion matrix:")
                # print(f'all_target: { all_target}')
                # print(f' all_pred: {all_pred}')
                # print(f'all_pred_values {all_pred_values}')
                # print(f'all_target_values {all_target_values}')
                print(cm)
               
              
                
            elif _metric == "mcc":
                score = []
                num_class = 0
                for i, cur_num_class in enumerate(self.num_class):
                    _pred = pred[:, num_class:num_class + cur_num_class]
                    _target = target[:, i]
                    _labeled = labeled[:, i]
                    _score = metrics.matthews_corrcoef(_pred[_labeled], _target[_labeled].long())
                    score.append(_score)
                    num_class += cur_num_class
                score = torch.stack(score)
            elif _metric == "auroc":
                score = []
                for _pred, _target, _labeled in zip(pred.t(), target.long().t(), labeled.t()):
                    _score = metrics.area_under_roc(_pred[_labeled], _target[_labeled])
                    score.append(_score)
                score = torch.stack(score)
            elif _metric == "auprc":
                score = []
                for _pred, _target, _labeled in zip(pred.t(), target.long().t(), labeled.t()):
                    _score = metrics.area_under_prc(_pred[_labeled], _target[_labeled])
                    score.append(_score)
                score = torch.stack(score)
            elif _metric == "r2":
                score = []
                for _pred, _target, _labeled in zip(pred.t(), target.t(), labeled.t()):
                    _score = metrics.r2(_pred[_labeled], _target[_labeled])
                    score.append(_score)
                score = torch.stack(score)
            elif _metric == "spearmanr":
                score = []
                for _pred, _target, _labeled in zip(pred.t(), target.t(), labeled.t()):
                    _score = metrics.spearmanr(_pred[_labeled], _target[_labeled])
                    score.append(_score)
                score = torch.stack(score)
            elif _metric == "pearsonr":
                score = []
                for _pred, _target, _labeled in zip(pred.t(), target.t(), labeled.t()):
                    _score = metrics.pearsonr(_pred[_labeled], _target[_labeled])
                    score.append(_score)
                score = torch.stack(score)
            else:
                raise ValueError("Unknown metric `%s`" % _metric)

            name = tasks._get_metric_name(_metric)
            for t, s in zip(self.task, score):
                metric["%s [%s]" % (name, t)] = s

        return metric


@R.register("tasks.MultipleBinaryClassification")
class MultipleBinaryClassification(tasks.Task, core.Configurable):
    """
    Multiple binary classification task for graphs / molecules / proteins.

    Parameters:
        model (nn.Module): graph representation model
        task (list of int, optional): training task id(s).
        criterion (list or dict, optional): training criterion(s). For dict, the keys are criterions and the values
            are the corresponding weights. Available criterions are ``bce``.
        metric (str or list of str, optional): metric(s).
            Available metrics are ``auroc@macro``, ``auprc@macro``, ``auroc@micro``, ``auprc@micro`` and ``f1_max``.
        num_mlp_layer (int, optional): number of layers in the MLP prediction head
        normalization (bool, optional): whether to normalize the target
        reweight (bool, optional): whether to re-weight tasks according to the number of positive samples
        graph_construction_model (nn.Module, optional): graph construction model
        verbose (int, optional): output verbose level
    """

    eps = 1e-10
    _option_members = {"criterion", "metric"}

    def __init__(self, model, task=(), criterion="bce", metric=("auprc@micro", "f1_max"), num_mlp_layer=1,
                 normalization=True, reweight=False, graph_construction_model=None, verbose=0):
        super(MultipleBinaryClassification, self).__init__()
        self.model = model
        self.task = task
        self.register_buffer("task_indices", torch.LongTensor(task))
        self.criterion = criterion
        self.metric = metric
        self.num_mlp_layer = num_mlp_layer
        self.normalization = normalization
        self.reweight = reweight
        self.graph_construction_model = graph_construction_model
        self.verbose = verbose

        hidden_dims = [self.model.output_dim] * (self.num_mlp_layer - 1)
        self.mlp = layers.MLP(self.model.output_dim, hidden_dims + [len(task)])

    def preprocess(self, train_set, valid_set, test_set):
        """
        Compute the weight for each task on the training set.
        """
        values = []
        for data in train_set:
            values.append(data["targets"][self.task_indices])
        values = torch.stack(values, dim=0)    
        
        if self.reweight:
            num_positive = values.sum(dim=0)
            weight = (num_positive.mean() / num_positive).clamp(1, 10)
        else:
            weight = torch.ones(len(self.task), dtype=torch.float)

        self.register_buffer("weight", torch.as_tensor(weight, dtype=torch.float))

    def forward(self, batch):
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred = self.predict(batch, all_loss, metric)
        target = self.target(batch)

        for criterion, weight in self.criterion.items():
            if criterion == "bce":
                loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
            else:
                raise ValueError("Unknown criterion `%s`" % criterion)
            loss = loss.mean(dim=0)
            loss = (loss * self.weight).sum() / self.weight.sum()
            
            name = tasks._get_criterion_name(criterion)
            metric[name] = loss
            all_loss += loss * weight

        return all_loss, metric

    def predict(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]
        if self.graph_construction_model:
            graph = self.graph_construction_model(graph)
        output = self.model(graph, graph.node_feature.float(), all_loss=all_loss, metric=metric)
        pred = self.mlp(output["graph_feature"])
        return pred

    def target(self, batch):
        target = batch["targets"][:, self.task_indices]
        return target

    def evaluate(self, pred, target):
        metric = {}
        for _metric in self.metric:
            if _metric == "auroc@micro":
                score = metrics.area_under_roc(pred.flatten(), target.long().flatten())
            elif _metric == "auroc@macro":
                score = metrics.variadic_area_under_roc(pred, target.long(), dim=0).mean()
            elif _metric == "auprc@micro":
                score = metrics.area_under_prc(pred.flatten(), target.long().flatten())
            elif _metric == "auprc@macro":
                score = metrics.variadic_area_under_prc(pred, target.long(), dim=0).mean()
            elif _metric == "f1_max":
                score = metrics.f1_max(pred, target)
            else:
                raise ValueError("Unknown criterion `%s`" % _metric)

            name = tasks._get_metric_name(_metric)
            metric[name] = score

        return metric


@R.register("tasks.NodePropertyPrediction")
class NodePropertyPrediction(tasks.Task, core.Configurable):
    """
    Node / atom / residue property prediction task.

    Parameters:
        model (nn.Module): graph representation model
        criterion (str, list or dict, optional): training criterion(s). For dict, the keys are criterions and the values
            are the corresponding weights. Available criterions are ``mse``, ``bce`` and ``ce``.
        metric (str or list of str, optional): metric(s).
            Available metrics are ``mae``, ``rmse``, ``auprc`` and ``auroc``.
        num_mlp_layer (int, optional): number of layers in mlp prediction head
        normalization (bool, optional): whether to normalize the target
            Available entities are ``node``, ``atom`` and ``residue``.
        num_class (int, optional): number of classes
        verbose (int, optional): output verbose level
    """

    _option_members = {"criterion", "metric"}

    def __init__(self, model, criterion="bce", metric=("macro_auprc", "macro_auroc"), num_mlp_layer=1,
                 normalization=True, num_class=None, verbose=0):
        super(NodePropertyPrediction, self).__init__()
        self.model = model
        self.criterion = criterion
        self.metric = metric
        # For classification tasks, we disable normalization tricks.
        self.normalization = normalization and ("ce" not in criterion) and ("bce" not in criterion)
        self.num_mlp_layer = num_mlp_layer
        self.num_class = num_class
        self.verbose = verbose

    def preprocess(self, train_set, valid_set, test_set):
        """
        Compute the mean and derivation on the training set.
        """
        self.view = getattr(train_set[0]["graph"], "view", "atom")
        values = torch.cat([data["graph"].target for data in train_set])
        mean = values.float().mean()
        std = values.float().std()
        if values.dtype == torch.long:
            num_class = values.max().item()
            if num_class > 1 or "bce" not in self.criterion:
                num_class += 1
        else:
            num_class = 1

        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float))
        self.num_class = self.num_class or num_class

        if hasattr(self.model, "node_output_dim"):
            model_output_dim = self.model.node_output_dim
        else:
            model_output_dim = self.model.output_dim
        hidden_dims = [model_output_dim] * (self.num_mlp_layer - 1)
        self.mlp = layers.MLP(model_output_dim, hidden_dims + [self.num_class])

    def predict(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]
        output = self.model(graph, graph.node_feature.float(), all_loss=all_loss, metric=metric)
        if self.view in ["node", "atom"]:
            output_feature = output["node_feature"]
        else:
            output_feature = output.get("residue_feature", output.get("node_feature"))
        pred = self.mlp(output_feature)
        if self.normalization:
            pred = pred * self.std + self.mean
        return pred

    def target(self, batch):
        size = batch["graph"].num_nodes if self.view in ["node", "atom"] else batch["graph"].num_residues
        return {
            "label": batch["graph"].target,
            "mask": batch["graph"].mask,
            "size": size
        }

    def forward(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred, target = self.predict_and_target(batch, all_loss, metric)
        labeled = ~torch.isnan(target["label"]) & target["mask"]

        for criterion, weight in self.criterion.items():
            if criterion == "mse":
                if self.normalization:
                    loss = F.mse_loss((pred - self.mean) / self.std, (target - self.mean) / self.std, reduction="none")
                else:
                    loss = F.mse_loss(pred, target, reduction="none")
            elif criterion == "bce":
                loss = F.binary_cross_entropy_with_logits(pred, target["label"].float(), reduction="none")
            elif criterion == "ce":
                loss = F.cross_entropy(pred, target["label"], reduction="none")
            else:
                raise ValueError("Unknown criterion `%s`" % criterion)
            loss = functional.masked_mean(loss, labeled, dim=0)

            name = tasks._get_criterion_name(criterion)
            metric[name] = loss
            all_loss += loss * weight

        all_loss += loss

        return all_loss, metric

    def evaluate(self, pred, target):
        metric = {}
        _target = target["label"]
        _labeled = ~torch.isnan(_target) & target["mask"]
        _size = functional.variadic_sum(_labeled.long(), target["size"]) 
        for _metric in self.metric:
            if _metric == "micro_acc":
                score = metrics.accuracy(pred[_labeled], _target[_labeled].long())
            elif metric == "micro_auroc":
                score = metrics.area_under_roc(pred[_labeled], _target[_labeled])
            elif metric == "micro_auprc":
                score = metrics.area_under_prc(pred[_labeled], _target[_labeled])
            elif _metric == "macro_auroc":
                score = metrics.variadic_area_under_roc(pred[_labeled], _target[_labeled], _size).mean()
            elif _metric == "macro_auprc":
                score = metrics.variadic_area_under_prc(pred[_labeled], _target[_labeled], _size).mean()
            elif _metric == "macro_acc":
                score = pred[_labeled].argmax(-1) == _target[_labeled]
                score = functional.variadic_mean(score.float(), _size).mean()
            else:
                raise ValueError("Unknown criterion `%s`" % _metric)

            name = tasks._get_metric_name(_metric)
            metric[name] = score

        return metric


@R.register("tasks.InteractionPrediction")
@utils.copy_args(PropertyPrediction, ignore=("graph_construction_model",))
class InteractionPrediction(PropertyPrediction):
    """
    Predict the interaction property of graph pairs.

    Parameters:
        model (nn.Module): graph representation model
        model2 (nn.Module, optional): graph representation model for the second item. If ``None``, use tied-weight
            model for the second item.
        **kwargs
    """

    def __init__(self, model, model2=None, **kwargs):
        super(InteractionPrediction, self).__init__(model, **kwargs)
        self.model2 = model2 or model

    def preprocess(self, train_set, valid_set, test_set):
        """
        Compute the mean and derivation for each task on the training set.
        """
        values = defaultdict(list)
        for sample in train_set:
            if not sample.get("labeled", True):
                continue
            for task in self.task:
                if not math.isnan(sample[task]):
                    values[task].append(sample[task])
        mean = []
        std = []
        weight = []
        num_class = []
        for task, w in self.task.items():
            value = torch.tensor(values[task])
            mean.append(value.float().mean())
            std.append(value.float().std())
            weight.append(w)
            if value.ndim > 1:
                num_class.append(value.shape[1])
            elif value.dtype == torch.long:
                task_class = value.max().item()
                if task_class == 1 and "bce" in self.criterion:
                    num_class.append(1)
                else:
                    num_class.append(task_class + 1)
            else:
                num_class.append(1)

        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float))
        self.register_buffer("weight", torch.as_tensor(weight, dtype=torch.float))
        self.num_class = self.num_class or num_class

        hidden_dims = [self.model.output_dim] * (self.num_mlp_layer - 1)
        self.mlp = layers.MLP(self.model.output_dim + self.model2.output_dim, hidden_dims + [sum(self.num_class)])

    def predict(self, batch, all_loss=None, metric=None):
        graph1 = batch["graph1"]
        output1 = self.model(graph1, graph1.node_feature.float(), all_loss=all_loss, metric=metric)
        graph2 = batch["graph2"]
        output2 = self.model2(graph2, graph2.node_feature.float(), all_loss=all_loss, metric=metric)
        pred = self.mlp(torch.cat([output1["graph_feature"], output2["graph_feature"]], dim=-1))
        if self.normalization:
            pred = pred * self.std + self.mean
        return pred


@R.register("tasks.Unsupervised")
class Unsupervised(nn.Module, core.Configurable):
    """
    Wrapper task for unsupervised learning.

    The unsupervised loss should be computed by the model.

    Parameters:
        model (nn.Module): any model
    """

    def __init__(self, model, graph_construction_model=None):
        super(Unsupervised, self).__init__()
        self.model = model
        self.graph_construction_model = graph_construction_model

    def forward(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred = self.predict(batch, all_loss, metric)

        return all_loss, metric

    def predict(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]
        if self.graph_construction_model:
            graph = self.graph_construction_model(graph)
        pred = self.model(graph, graph.node_feature.float(), all_loss=all_loss, metric=metric)
        return pred
