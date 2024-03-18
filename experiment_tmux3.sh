#!/bin/bash
# 1 task experiments
# project: client fairness for FL.
# --L: 1/L is the learning rate.
# --unbalance: the unbalance level of the dataset
# For each client, it will have 400~500 data points by default.
# --unbalance 0.9 0.1 means 90% (0.9) clients will have 40~50 data points (400*0.1~500*0.1).
# --fairness: fairness type. clientfair use objective: sum_i f_i^{a}. notfair use objective: sum_i f_i.
# --alpha: f^alpha.
# --notes: the notes of the experiment. will become part of the experiment folder name.
# --C: class rate, control the non-iid level. Each client will only has data points from C*total_classes classes. Example: C=0.2, total_clases=10, then each client only has 2 classes data.
# --insist: overwrite the existing folder.
seedlist=(14 15 16 17)
for sd in "${seedlist[@]}"; do
# q-Fel. algorithm 1, this command use round_num=300, local_epoch=1. convergence is slow, so recommend round_num=1500
python main.py --L 100 --unbalance 0.9 0.1 --fairness clientfair --alpha 3 --notes u91c0.3_qFel_a3_$sd --C 0.2 --num_clients 40 --class_ratio 0.3 --iid_type noniid --task_type fashion_mnist --algo_type proposed --seed $sd --cpumodel --local_epochs 1 --round_num 300 --insist
# random
python main.py --L 100 --unbalance 0.9 0.1 --fairness notfair --alpha 3 --notes u91c0.3_random_$sd --C 0.2 --num_clients 40 --class_ratio 0.3 --iid_type noniid --task_type fashion_mnist --algo_type random --seed $sd --cpumodel --local_epochs 1 --round_num 300 --insist

# ****** using optimal sampling will perform better, but it is not helpful in accelerating the convergence.
# optimal sampling using loss function, alpha=1
#python main.py --L 100 --unbalance 0.9 0.1 --fairness notfair --alpha 1 --notes u91c0.3_AS_a1_$sd --approx_optimal --C 0.2 --num_clients 40 --class_ratio 0.3 --iid_type noniid --task_type fashion_mnist --algo_type proposed --seed $sd --cpumodel --local_epochs 1 --round_num 300 --insist
# optimal sampling, using gradient norm, alpha=1
#python main.py --L 100 --unbalance 0.9 0.1 --fairness notfair --alpha 1 --notes u91c0.3_OS_a1_$sd --optimal_sampling --C 0.2 --num_clients 40 --class_ratio 0.3 --iid_type noniid --task_type fashion_mnist --algo_type proposed --seed $sd --cpumodel --local_epochs 1 --round_num 300 --insist
# optimal sampling using loss function, alpha=3
#python main.py --L 100 --unbalance 0.9 0.1 --fairness clientfair --alpha 3 --notes u91c0.3_AS_a3_$sd --approx_optimal --alpha_loss --C 0.2 --num_clients 40 --class_ratio 0.3 --iid_type noniid --task_type fashion_mnist --algo_type proposed --seed $sd --cpumodel --local_epochs 1 --round_num 300 --insist # or round_num 1500
# optimal sampling using gradient norm, alpha=3
#python main.py --L 100 --unbalance 0.9 0.1 --fairness clientfair --alpha 3 --notes u91c0.3_OS_a3_$sd --optimal_sampling --alpha_loss --C 0.2 --num_clients 40 --class_ratio 0.3 --iid_type noniid --task_type fashion_mnist --algo_type proposed --seed $sd --cpumodel --local_epochs 1 --round_num 300 --insist # or round_num 1500
# change the probability of sampling for each client, but not consider it in the aggregation.
#python main.py --L 100 --unbalance 0.9 0.1 --fairness notfair --alpha 2 --equalP --approx_optimal --alpha_loss --notes u91c0.3_testfixed_a3_$sd --C 0.2 --num_clients 40 --class_ratio 0.3 0.3 0.3 --iid_type noniid noniid noniid --task_type fashion_mnist --algo_type proposed --seed $sd --cpumodel --local_epochs 1 1 1 --round_num 300 --insist
#python main.py --L 1000 --unbalance 0.9 0.1 --fairness notfair --alpha 2 --equalP2 --approx_optimal --alpha_loss --notes u91c0.3_testfixed2_a3_$sd --C 0.2 --num_clients 40 --class_ratio 0.3 0.3 0.3 --iid_type noniid noniid noniid --task_type fashion_mnist --algo_type proposed --seed $sd --cpumodel --local_epochs 1 1 1 --round_num 300 --insist
done