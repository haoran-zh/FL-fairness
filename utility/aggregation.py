import torch
import numpy as np


def federated_prob(global_weights, models_gradient_dict, local_data_num, p_list, args, chosen_clients, tasks_local_training_loss):

    global_weights_dict = global_weights.state_dict()
    global_keys = list(global_weights_dict.keys())
    # Sum the state_dicts of all client models
    # sum loss power a-1
    alpha = args.alpha
    N = args.num_clients

    L = args.L
    denominator = 0
    # aggregate
    if (args.fairness == 'notfair'):
        denominator = L
        for i, gradient_dict in enumerate(models_gradient_dict):
            d_i = local_data_num[chosen_clients[i]] / np.sum(local_data_num)
            if args.equalP2 is True:
                d_i = 1
            for key in global_keys:
                global_weights_dict[key] -= (d_i / p_list[i]) * gradient_dict[key] / denominator

    elif args.fairness == 'clientfair':
        if len(tasks_local_training_loss) == len(chosen_clients):
            for i, gradient_dict in enumerate(models_gradient_dict):
                norm_2 = sum(torch.norm(diff, p=2) ** 2 for diff in gradient_dict.values()) / (args.lr ** 2)
                a = (alpha - 1) * tasks_local_training_loss[i] ** (alpha - 2) * norm_2
                b = tasks_local_training_loss[i] ** (alpha - 1) * L
                newL = a + b
                denominator += (local_data_num[chosen_clients[i]] / np.sum(local_data_num)) / p_list[i] * newL
            for i, gradient_dict in enumerate(models_gradient_dict):
                for key in global_keys:
                    global_weights_dict[key] -= (local_data_num[chosen_clients[i]] / np.sum(local_data_num) / p_list[
                        i]) * gradient_dict[key] * tasks_local_training_loss[i] ** (
                                                            alpha - 1) / denominator
        else:
            # only difference is chosen_client[i] or [i].
            for i, gradient_dict in enumerate(models_gradient_dict):
                norm_2 = sum(torch.norm(diff, p=2) ** 2 for diff in gradient_dict.values()) / (args.lr**2)
                a = (alpha-1)*tasks_local_training_loss[chosen_clients[i]]**(alpha-2)*norm_2
                b = tasks_local_training_loss[chosen_clients[i]]**(alpha-1)*L
                newL = a + b
                denominator += (local_data_num[chosen_clients[i]]/np.sum(local_data_num)) / p_list[i] * newL

            for i, gradient_dict in enumerate(models_gradient_dict):
                for key in global_keys:
                    global_weights_dict[key] -= (local_data_num[chosen_clients[i]] / np.sum(local_data_num) / p_list[
                        i]) * gradient_dict[key] * tasks_local_training_loss[chosen_clients[i]] ** (alpha - 1) / denominator
    else:
        print("aggregation wrong!")
        exit(1)

    return global_weights_dict