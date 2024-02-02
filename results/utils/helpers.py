import csv
import numpy as np
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import torch as th
import pickle
import seaborn as sns


class Summarizer:

    def __init__(self, filename:str) -> None:
        self.true_saliency_known = "hmm" in filename
        self.stats = self.get_stats(filename)

    def get_stats(self, filename:str) -> dict[tuple[str], dict[str, float]]:
        data_dict = self.read_csv_to_dict(filename)
        stats_dict = dict()
        for key, metrics_dict in data_dict.items():
            stats_dict[key] = dict()
            for metric, values in metrics_dict.items():
                stats_dict[key][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values, ddof=1)
                }
        return stats_dict

    def read_csv_to_dict(self, filename:str) -> dict[tuple[str], dict[str, float]]:
        data_dict = {}

        with open(f"{filename}.csv", 'r') as file:
            csvreader = csv.reader(file)
            next(csvreader)  # skip header

            if self.true_saliency_known:
                for row in csvreader:
                    if not row: continue
                    if "our_loss_study" in filename:
                        _ ,_, exp, l1, l2, aup, aur, I, S, auroc, auprc = row
                        key = (exp, l1, l2)
                    elif "loss_study" in filename:
                        _,pres,delet,_, exp, l1, l2, aup, aur, I, S, auroc, auprc = row
                        key = (exp, l1, l2, pres, delet)
                    else:
                        _, _, exp, l1, l2, aup, aur, I, S, auroc, auprc = row
                        key = (exp, l1, l2)
                    
                    if key not in data_dict:
                        data_dict[key] = {
                            'aup': [],
                            'aur': [],
                            'I': [],
                            'S': [],
                            'auroc': [],
                            'auprc': []
                        }
                    
                    data_dict[key]['aup'].append(float(aup))
                    data_dict[key]['aur'].append(float(aur))
                    data_dict[key]['I'].append(float(I))
                    data_dict[key]['S'].append(float(S))
                    data_dict[key]['auroc'].append(float(auroc))
                    data_dict[key]['auprc'].append(float(auprc))
            else:
                for row in csvreader:
                    if len(row) == 0: continue
                    if filename[-10:] == "loss_study":
                        _,_,deletion,_,baseline,topk, exp, l1, l2, acc, comp, CE, logodds, suff = row
                        key = (baseline, topk, exp, l1, l2, deletion)
                    elif "extremal_mask_params" in filename:
                        _, _, baseline, topk, exp, acc, comp, CE, logodds, suff = row
                        key = (baseline, topk, exp, "1.0", "1.0")
                    else:
                        _, _, baseline, topk, exp, l1, l2, acc, comp, CE, logodds, suff = row
                        key = (baseline, topk, exp, l1, l2)
                    
                    if key not in data_dict:
                        data_dict[key] = {
                            'acc': [],
                            'comp': [],
                            'logodds': [],
                            'CE': [],
                            'suff': [],
                        }
                    
                    data_dict[key]['acc'].append(float(acc))
                    data_dict[key]['comp'].append(float(comp))
                    data_dict[key]['logodds'].append(float(logodds))
                    data_dict[key]['CE'].append(float(CE))
                    data_dict[key]['suff'].append(float(suff))
                
        return data_dict

    def display_table(self, lambdas:bool=False, only_02:bool=False, zeros:bool=False) -> 'Summarizer':
        result_table = PrettyTable()
        if lambdas:
            metric1, metric2 = ("aup", "aur") if self.true_saliency_known else ("acc", "CE")
            result_table.field_names = ["", "0.01", "0.1", "1.0", "10.0", "100.0"]
            table = [[None]*5 for _ in range(5)]
            for key, metrics_dict in self.stats.items():
                if only_02 and (key[1] != "0.2" or (zeros and key[0] != "Zeros") or (not zeros and key[0] == "Zeros")): continue
                x, y = int(np.log10(float(key[-2]))+2), int(np.log10(float(key[-1]))+2)
                table[y][x] = (metrics_dict[metric1]["mean"], metrics_dict[metric2]["mean"])
            for idx, row in enumerate(table):
                result_table.add_row([str(10**(idx-2))] + [f"{m1:.5f}-{m2:.5f}" for (m1, m2) in row])
        else:
            metrics = ['aup', 'aur', 'I', 'S', 'auroc', 'auprc'] if self.true_saliency_known else ['acc', 'comp', 'logodds', 'CE', 'suff']
            result_table.field_names = ['Key'] + metrics
            for key, metrics_dict in self.stats.items():
                if only_02 and key[1] != "0.2": continue
                result_table.add_row([str(key)] + [
                    f"{metrics_dict[metric]['mean']:.5f} ({metrics_dict[metric]['std']:.5f})"
                    for metric in result_table.field_names[1:]
                ])

        print(result_table, "\n")
        return self

    def make_plots(self, explainers: list[str], metrics:list[str]=["acc"], baselines:list[str]=["Zeros"]) -> 'Summarizer':
        i20 = {"deep_lift": "DeepLift", "dyna_mask": "DynaMask", "occlusion": "Occlusion", "extremal_mask": "ExtrMask",
               "fit": "Fit", "augmented_occlusion": "Aug Occlusion", "integrated_gradients": "IG", "retain": "Retain",
               "extremal_mask_mse_deletion": "ExtrMask MSE Del", "extremal_mask_mse": "ExtrMask MSE", "gradient_shap": "GradientShap"}
        i30 = {"CE": "Cross Entropy", "comp": "Comprehensiveness", "acc": "Accuracy", "suff": "Sufficiency"}
               
        for metric in metrics:
            for baseline in baselines:
                print(f"Metric {metric}, Baseline: {baseline}")
                for exp in explainers:
                    means = [self.stats[baseline,topk,exp,"1.0","1.0"][metric]["mean"] for topk in ("0.1","0.2","0.3","0.4","0.5","0.6")]
                    stds = [self.stats[baseline,topk,exp,"1.0","1.0"][metric]["std"] for topk in ("0.1","0.2","0.3","0.4","0.5","0.6")]
                    lows = [m-s for m,s in zip(means, stds)]
                    highs = [m+s for m,s in zip(means, stds)]
                    plt.plot([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], means, label=i20.get(exp, exp))
                    plt.fill_between([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], lows, highs, alpha=0.2)
                
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.ylabel(i30.get(metric, metric))
                plt.xlabel("% Masked")
                plt.show()
        
        return self


class AttributionVisualizer:

    die_idx = [27,29,30,41,43,48,53,54,61,65,66,67,87,89,97,100,103,110,123,132,146,158,167,170,175,179,196,198,200,208,213,226,236,239,241,247,258,265,267,272,275,276,304,314,315,317,329,336,356,357,359,370,376,391,395,404,412,416,419,422,427,431,432,435,440,450,462,464,468,469,473,484,491,501,510,511,522,544,546,551,562,567,574,575,585,600,603,606,610,621,630,632,637,650,664,672,679,684,692,707,723,729,731,744,765,773,776,777,784,788,793,794,802,807,808,834,835,839,843,875,876,882,891,896,898,910,943,951,954,963,969,983,993,994,995,996,999,1006,1009,1010,1018,1026,1031,1039,1048,1054,1064,1068,1085,1089,1090,1093,1102,1108,1109,1124,1126,1129,1130,1140,1151,1153,1154,1158,1160,1170,1175,1182,1195,1196,1200,1201,1203,1207,1208,1212,1235,1239,1250,1257,1261,1262,1263,1265,1283,1289,1293,1301,1307,1308,1309,1311,1316,1330,1333,1334,1341,1348,1369,1370,1375,1387,1402,1403,1415,1417,1418,1421,1424,1427,1444,1445,1466,1481,1487,1491,1500,1503,1518,1527,1537,1543,1545,1549,1557,1575,1578,1584,1586,1590,1592,1599,1605,1607,1611,1612,1613,1628,1639,1647,1657,1669,1673,1674,1675,1685,1689,1690,1691,1707,1714,1726,1728,1735,1742,1749,1751,1755,1757,1769,1770]
    features = ["ANION GAP", "ALBUMIN", "BICARBONATE", "BiliRubin", "CREATININE", "CHLORIDE", "GLUCOSE", "HEMATOCRIT", "HEMOGLOBINLACTATE", "MAGNESIUM", "PHOSPHATE", "PLATELET", "POTASSIUM", "PTT", "INT", "PT", "SODIUM", "BUN", "WBC", "HeartRate", "SysBP", "DiasBP", "MeanBP", "RespRate", "SpO2", "Glucose", "Temp", "Gender", "Age", "Ethnicity", "First ICU stay"]

    def __init__(self, loss_fctn:str, only_positive:bool=True) -> None:
        self.only_positive = only_positive
        self.data = self.load_data(loss_fctn)
    
    def load_data(self, loss_fctn:str) -> th.Tensor:
        data = []
        for fold in range(5):
            with open(f'attributions/attr_mimic3_{loss_fctn.lower()}_fold_{fold}.pkl', 'rb') as f:
                data.append(pickle.load(f))
        data = th.stack(data)
        if self.only_positive: data[:, self.die_idx,:,:]
        return data

    def make_dist_plot(self, bins:int=50) -> 'Summarizer':
        sns.distplot(self.data, bins=bins)
        plt.show()
        return self

    def plot_mean_attribution_over_time(self, last_t:int=42) -> 'Summarizer':
        average_feature_importance_over_time_perfold = self.data[:, :,-last_t:,:].mean(dim=(1,3))
        std_dev_feature_importance_over_time = average_feature_importance_over_time_perfold.std(0)
        std_err_feature_importance_over_time = 1.96 * std_dev_feature_importance_over_time / th.sqrt(th.tensor(5))
        average_feature_importance_over_time = average_feature_importance_over_time_perfold.mean(0)

        plt.plot(average_feature_importance_over_time, label="Mean attribution")
        plt.fill_between(th.arange(average_feature_importance_over_time.shape[0]), 
                            average_feature_importance_over_time - std_err_feature_importance_over_time, 
                            average_feature_importance_over_time + std_err_feature_importance_over_time, 
                            alpha=0.2)
        plt.legend(loc=2)
        plt.show()
        return self

    def plot_feature_attribution(self, last_t:int=42) -> 'Summarizer':
        average_feature_importance_perfold = self.data[:, :,-last_t:,:].mean(dim=(1,2)) 
        std_dev_feature_importance = average_feature_importance_perfold.std(0)
        std_err_feature_importance = 1.96 * std_dev_feature_importance/ th.sqrt(th.tensor(5))
        average_feature_importance = average_feature_importance_perfold.mean(0)
        average_feature_importance = average_feature_importance

        plt.figure(figsize=(6, 5))
        plt.errorbar(self.features, average_feature_importance, yerr=std_err_feature_importance, fmt='o', capsize=2)
        plt.ylabel('Attribution')
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.tight_layout()
        plt.show()
        
        return self