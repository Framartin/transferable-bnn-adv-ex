import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.close('all')
sns.set_style("whitegrid")

# Figure - tranferability vs nb cycles

df1 = pd.read_csv("X_adv/CIFAR10/PreResNet110/cSGLD_cycles15_savespercycle12_it1__nbcycles/eval_target.csv")
df1['nb_cycles'] = df1['adv_ex_path'].str.extract('_limcy([0-9]*)spc').astype(int)
df1['nb_iters'] = df1['adv_ex_path'].str.extract('_niter([0-9]*)_')#.astype(int)
df1['stop_at_best_iter'] = df1['adv_ex_path'].str.contains('_best.npy')
df1['n_examples'] = df1['adv_ex_path'].str.extract('_n([0-9None]+)_')
df1.sort_values(by=['stop_at_best_iter', 'nb_cycles'], inplace=True)
n_examples = 'None'  # '1000' or 'None': restrict to only 1000 examples or full dataset
df1_ = df1.loc[(df1['n_examples'] == n_examples) & (~df1['stop_at_best_iter']), :]
fig = plt.figure()
plt.xlim([0,16])
plt.ylim([0,0.25])
#sns.lineplot(x='nb_cycles', y='attack_fail_rate', data=df1_, hue='nb_iters', legend='full', marker='o')
sns.lineplot(x='nb_cycles', y='attack_fail_rate', data=df1_, marker='o')
plt.xlabel('Number of sampling cycles')
plt.ylabel('Transfer fail rate')
plt.savefig('plot/PGDens_transfer_vs_nb_cycles.png', dpi=200)
plt.savefig('plot/PGDens_transfer_vs_nb_cycles.pdf', dpi=200)
plt.show()


# Figure - tranferability vs nb samples per cycle

df1 = pd.read_csv("X_adv/CIFAR10/PreResNet110/cSGLD_cycles15_savespercycle12_it1__nbsamplespercycle/eval_target.csv")
# ./X_adv/CIFAR10/PreResNet110/cSGLD_cycles15_savespercycle12_it1__RQ2_nbsamplespercycles/PGD_ensout1in1_niterout6in12att10_nraninitout1in0_shuffleoutTrueinTrue/Xadv_maxL2norm0.5_normiter0.125_n1000_seed0_limoutNonein1.npy
df1['nb_sample_per_cycle'] = df1['adv_ex_path'].str.extract('spc([0-9]*)[\\._m]').astype(int)
df1['stop_at_best_iter'] = df1['adv_ex_path'].str.contains('_best.npy')
df1['limit_samples_method'] = df1['adv_ex_path'].str.extract('method([a-z]*)[\\._m]')
df1.loc[df1['limit_samples_method'].isna(), 'limit_samples_method'] = 'interval'
df1['limit_samples_method'] = df1['limit_samples_method'].str.capitalize().astype('category')
df1['n_examples'] = df1['adv_ex_path'].str.extract('_n([0-9None]+)_')
df1.sort_values(by=['stop_at_best_iter', 'nb_sample_per_cycle'], inplace=True)
fig = plt.figure()
plt.xlim([0,13])
plt.ylim([0,0.15])
n_examples = 'None'  # '1000' or 'None': restrict to only 1000 examples or full dataset
df1_ = df1.loc[(df1['n_examples'] == n_examples) & (~df1['stop_at_best_iter']) & (df1['limit_samples_method'] != 'Interval'), :]
#sns.lineplot(x='nb_sample_per_cycle', y='attack_fail_rate', data=df1.loc[~df1['stop_at_best_iter'],], marker='o')
df1_ = df1_.rename(columns={'limit_samples_method': 'Keep first/last samples'})
g = sns.lineplot(x='nb_sample_per_cycle', y='attack_fail_rate', data=df1_,
                 hue='Keep first/last samples', marker='o', style='Keep first/last samples')
plt.xlabel('Number of samples per cycle')
plt.ylabel('Transfer fail rate')
plt.savefig('plot/PGDens_transfer_vs_nb_samples_per_cycle.png', dpi=200)
plt.savefig('plot/PGDens_transfer_vs_nb_samples_per_cycle.pdf', dpi=200)
plt.show()
