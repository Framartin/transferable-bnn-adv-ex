import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.close('all')
sns.set_style("whitegrid")
# utils
# surrogate name
def surrogate_name(row):
    if row['surrogate_type'] == 'cSGLD':
        return 'cSGLD'
    elif row['surrogate_type'] == 'dnn':
        return '1 DNN' if row['surrogate_size_ensembles'] == 1 else f'{row["surrogate_size_ensembles"]} DNNs'
    return


# Figure - tranferability vs nb cycles

df1 = pd.read_csv("X_adv/CIFAR10/RQ/results_nb_cycles.csv")
#df1 = pd.read_csv("X_adv/CIFAR10/RQ/results_nb_cycles_tmp2.csv")
df1['Attack Norm'] = df1.apply(lambda row: f'L{row["norm_type"]:.0f}', axis=1)
# df1['adv_sucess_rate'] = df1.apply(lambda row: f'{row["adv_sucess_rate"]*100:.2f} %', axis=1)
df1['adv_sucess_rate'] = df1["adv_sucess_rate"]*100
df1['limit_cycles'] = df1["limit_cycles"].astype(int)
def f(row):
    if row['n_random_init'] > 0 and row['n_iter'] > 1:
        return f'PGD ({row["n_random_init"]} restarts)'
    elif row['n_iter'] == 1 and row['norm_max'] == row['norm_step']:
        return 'FG(S)M'
    elif row['n_iter'] > 1 and pd.isna(row['momentum']):
        return 'I-FG(S)M'
    elif row['n_iter'] > 1 and row['momentum'] > 0:
        return 'MI-FG(S)M'
    return
df1['Attack'] = df1.apply(f, axis=1)
df1['Attack'] = pd.Categorical(df1['Attack'], categories=df1['Attack'].unique())  # save custom order

df1_ = df1.query("Attack == 'I-FG(S)M'")
fig = plt.figure()
#plt.xlim([0.5,16.5])
# plt.ylim([0,80])
# sns.lineplot(x='nb_cycles', y='attack_fail_rate', data=df1_, hue='nb_iters', legend='full', marker='o')
sns.lineplot(x='limit_cycles', y='adv_sucess_rate', data=df1_, marker='o', hue='Attack Norm', style="Attack Norm")
## sns.lineplot(x='limit_cycles', y='adv_sucess_rate', data=df1, marker='o', hue='Attack Norm', style="Attack")
# sns.lineplot(data=df1_, x="limit_cycles", y="adv_fail_rate")
plt.xticks(range(0,17))
plt.xlabel('Number of cSGLD cycles')
plt.ylabel('Transfer success rate (%)')
plt.savefig('plot/IFSGM_transfer_vs_nb_cycles.png', dpi=200)
plt.savefig('plot/IFSGM_transfer_vs_nb_cycles.pdf', dpi=200)
plt.show()


# Figure - tranferability vs nb samples per cycle

df1 = pd.read_csv("X_adv/CIFAR10/RQ/results_nb_samples_per_cycle_true.csv")
df1['Attack Norm'] = df1.apply(lambda row: f'L{row["norm_type"]:.0f}', axis=1)
df1['adv_sucess_rate'] = df1["adv_sucess_rate"]*100
df1['n_samples_per_cycle'] = (df1["surrogate_size_ensembles"]/5).astype(int)
df1_ = df1
fig = plt.figure()
# plt.xlim([0,22])
# plt.ylim([0,80])
# sns.lineplot(x='limit_samples_cycle', y='attack_fail_rate', data=df1_, hue='nb_iters', legend='full', marker='o')
# sns.lineplot(x='limit_samples_cycle', y='adv_sucess_rate', data=df1_, marker='o', hue='Attack Norm', style="Attack Norm")
sns.lineplot(x='n_samples_per_cycle', y='adv_sucess_rate', data=df1_, marker='o', hue='Attack Norm', style="Attack Norm")
plt.xticks(range(1,11))
plt.xlabel('Number of samples per cycle')
plt.ylabel('Transfer success rate (%)')
plt.savefig('plot/IFSGM_transfer_vs_nb_samples_per_cycle.png', dpi=200)
plt.savefig('plot/IFSGM_transfer_vs_nb_samples_per_cycle.pdf', dpi=200)
plt.show()



# Figure nb epochs for cSGLD and DNNs
df1 = pd.read_csv("X_adv/CIFAR10/RQ/results_nb_cycles.csv") # nb cycles cSGLD
df1['Number epochs'] = df1['limit_cycles'] * 50
df7 = pd.read_csv("X_adv/CIFAR10/PreResNet110/results_dee.csv") # transfer rate for all DNNs ensemble size [0,15]
df7['Number epochs'] = df7['surrogate_size_ensembles'] * 250
df_ = pd.concat([df1, df7])
df_.reset_index(drop=True, inplace=True)
df_['Attack Norm'] = df_.apply(lambda row: f'L{row["norm_type"]:.0f}', axis=1)
df_['adv_sucess_rate'] = df_["adv_sucess_rate"]*100
def f(row):
    if row['n_random_init'] > 0 and row['n_iter'] > 1:
        return f'PGD ({row["n_random_init"]} restarts)'
    elif row['n_iter'] == 1 and row['norm_max'] == row['norm_step']:
        return 'FG(S)M'
    elif row['n_iter'] > 1 and pd.isna(row['momentum']):
        return 'I-FG(S)M'
    elif row['n_iter'] > 1 and row['momentum'] > 0:
        return 'MI-FG(S)M'
    return
df_['Attack'] = df_.apply(f, axis=1)
df_['Attack'] = pd.Categorical(df_['Attack'], categories=df_['Attack'].unique())  # save custom order
df_['Type of surrogate'] = df_['surrogate_type']
df__ = df_.query("Attack == 'I-FG(S)M'")  # L2 not represented for DNN
df__ = df_.query("Attack == 'MI-FG(S)M'")
fig = plt.figure()
sns.lineplot(x='Number epochs', y='adv_sucess_rate', data=df__, marker='o', hue='Attack Norm', style="surrogate_type")
plt.xlabel('Number of epochs')
plt.ylabel('Transfer success rate (%)')
plt.savefig('plot/IFSGM_transfer_vs_nb_epochs.png', dpi=200)
plt.savefig('plot/IFSGM_transfer_vs_nb_epochs.pdf', dpi=200)
plt.close()
#plt.show()



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



# Figure nb iterations

df3 = pd.read_csv("X_adv/ImageNet/RQ/results_nb_iters.csv")
df3 = df3.query('(surrogate_type == "dnn") & (surrogate_size_ensembles == 1)')
#df3 = df3.query('(norm_step in [0.001568, 0.3])')  # 1/10th step
df3 = df3.query('norm_step in [0.00392, 0.75]')  # 1/4th step
df3['norm_type'] = df3['norm_type'].astype(str)

g = sns.lineplot(x='n_iter', y='adv_sucess_rate', data=df3,
                 hue='norm_type')
plt.xlabel('Number of iterations')
plt.ylabel('Transfer success rate')
plt.savefig('plot/nb_iterations_dnn_resnet50.png', dpi=200)
plt.show()

# idem CIFAR10
df3 = pd.read_csv("X_adv/CIFAR10/RQ/results_nb_iters.csv")
df3['Surrogate'] = df3.apply(surrogate_name, axis=1)
df3['norm_type'] = df3['norm_type'].astype(str)
df3 = df3.query('(norm_type == "2.0")')
#df3 = df3.query('(Surrogate == "15 DNNs")')

g = sns.lineplot(x='n_iter', y='adv_sucess_rate', data=df3,
                 hue='Surrogate')
#g = sns.lineplot(x='n_iter', y='adv_sucess_rate', data=df3,
#                 hue='norm_type')
plt.xlabel('Number of iterations')
plt.ylabel('Transfer success rate')
plt.savefig('plot/nb_iterations_cifar10_dnn.png', dpi=200)
plt.show()


# other figure nb iters
df3 = pd.read_csv("X_adv/CIFAR10/RQ/results_nb_iters.csv")
df3['Surrogate'] = df3.apply(surrogate_name, axis=1)
df3['Attack Norm'] = df3.apply(lambda row: f'{row["norm_type"]:.0f}', axis=1)
df3['Transfer success rate (%)'] = df3['adv_sucess_rate']*100
df3['Number of iterations'] = df3['n_iter'].astype(int)
def f(row):
    if row['n_random_init'] > 0:
        return f'PGD ({row["n_random_init"]} restart)'
    elif row['n_iter'] == 1 and row['norm_max'] == row['norm_step']:
        return 'FG(S)M'
    elif pd.isna(row['momentum']):
        return 'I-FG(S)M'
    elif row['momentum'] > 0:
        return 'MI-FG(S)M'
    return
df3['Attack'] = df3.apply(f, axis=1)
df3['Attack'] = pd.Categorical(df3['Attack'], categories=df3['Attack'].unique())  # save custom order

#kw = {'color': ['pink', 'pink', 'blue', 'red', 'orange'], 'linestyle' : ["-","--","-","-","--"]}
kw = {'linestyle' : ["-",":","-.","--",":"]}
g = sns.FacetGrid(df3, row="Attack Norm", col="Attack", hue="Surrogate", hue_kws=kw)
g.map(plt.plot, "Number of iterations", "Transfer success rate (%)", alpha=.7)
g.add_legend()
plt.savefig('plot/nb_iters_all_attacks_CIFAR.png', dpi=200)
plt.savefig('plot/nb_iters_all_attacks_CIFAR.pdf', dpi=200)
plt.show()


# same for Imagenet
df3 = pd.read_csv("X_adv/ImageNet/RQ/results_nb_iters.csv")
df3['Surrogate'] = df3.apply(surrogate_name, axis=1)
df3['Attack Norm'] = df3.apply(lambda row: f'{row["norm_type"]:.0f}', axis=1)
df3['Transfer success rate (%)'] = df3['adv_sucess_rate']*100
df3['Number of iterations'] = df3['n_iter'].astype(int)
def f(row):
    if row['n_random_init'] > 0:
        return f'PGD ({row["n_random_init"]} restart)'
    elif row['n_iter'] == 1 and row['norm_max'] == row['norm_step']:
        return 'FG(S)M'
    elif pd.isna(row['momentum']):
        return 'I-FG(S)M'
    elif row['momentum'] > 0:
        return 'MI-FG(S)M'
    return
df3['Attack'] = df3.apply(f, axis=1)
df3['Attack'] = pd.Categorical(df3['Attack'], categories=df3['Attack'].unique())  # save custom order

#kw = {'color': ['pink', 'pink', 'blue', 'red', 'orange'], 'linestyle' : ["-","--","-","-","--"]}
kw = {'linestyle' : ["-",":","-.","--",":"]}
g = sns.FacetGrid(df3, row="Attack Norm", col="Attack", hue="Surrogate", hue_kws=kw)
g.map(plt.plot, "Number of iterations", "Transfer success rate (%)", alpha=.7)
g.add_legend()
plt.savefig('plot/nb_iters_all_attacks_ImageNet.png', dpi=200)
plt.savefig('plot/nb_iters_all_attacks_ImageNet.pdf', dpi=200)
plt.show()


# analyse the min fail rate acheieved

df3 = pd.read_csv("X_adv/ImageNet/RQ/results_nb_iters.csv")
df3['adv_fail_rate'] = 100*df3['adv_fail_rate']
df_best_iter = df3.loc[df3.groupby(['surrogate_type', 'surrogate_size_ensembles', 'norm_type', 'norm_step', 'n_ensemble']).adv_fail_rate.idxmin(), ['surrogate_type', 'surrogate_size_ensembles', 'norm_type', 'norm_step', 'n_ensemble', 'n_iter', 'adv_fail_rate']]

df_best_iter_light = df_best_iter.query('n_ensemble == 1')
df_best_iter_full = df_best_iter.query('n_ensemble > 1')



df3_50iter = df3.query('(n_ensemble == 1) & (n_iter == 50)')
df_50_iter = df3_50iter.loc[df3_50iter.groupby(['surrogate_type', 'surrogate_size_ensembles', 'norm_type', 'norm_step', 'n_ensemble']).adv_fail_rate.idxmin(), ['surrogate_type', 'surrogate_size_ensembles', 'norm_type', 'norm_step', 'n_ensemble', 'n_iter', 'adv_fail_rate']]


# Produce LaTeX table test time techniques
df4 = pd.read_csv("X_adv/ImageNet/test_techniques/results_test_techniques.csv")
df4['Dataset'] = "ImageNet"
df_tmp = pd.read_csv("X_adv/CIFAR10/test_techniques/results_test_techniques.csv")
df_tmp['Dataset'] = "CIFAR10"
df4 = pd.concat([df_tmp, df4])

#create columns with name of technique
def f(row):
    if row['ghost']:
        return 'Ghost Networks'
    elif row['input_diversity']:
        return 'Input Diversity'
    elif row['translation_invariant']:
        return 'Translation Invariant\\textasteriskcentered'
    return 'Baseline (None)'
df4['Test-time technique'] = df4.apply(f, axis=1)
df4['Test-time technique'] = pd.Categorical(df4['Test-time technique'], categories=df4['Test-time technique'].unique())  # save custom order
df4['Surrogate'] = df4.apply(surrogate_name, axis=1)
df4['Surrogate'] = pd.Categorical(df4['Surrogate'], categories=df4['Surrogate'].unique())  # save custom order
df4['Attack'] = df4.apply(lambda row: f'L{row["norm_type"]:.0f} Attack', axis=1)
df4['adv_sucess_rate'] = df4.apply(lambda row: f'{row["adv_sucess_rate"]*100:.2f} %', axis=1)
# df4_ = pd.pivot_table(df4, values="adv_sucess_rate", index=["Test-time technique", "Surrogate", "Nb training epochs", "Nb backward passes"], columns=["Attack"])
# df4_ = df4_.reset_index(level=['Nb training epochs', 'Nb backward passes'])
# #df4_ = df4_.sort_index(level=['Surrogate'], ascending=False, sort_remaining=False)
# df4_ = df4_[['L2 Attack', 'Linf Attack', 'Nb training epochs', 'Nb backward passes']]
# df4_.to_latex('X_adv/ImageNet/test_techniques/results_test_techniques.tex')

df4_ = df4.set_index(["Dataset", "Test-time technique", "Surrogate", "Attack"])[['adv_sucess_rate']]
df4_ = df4_.unstack(3)
df4_.columns = df4_.columns.droplevel()  # drop 1st level of columns
#df4_ = df4_.sort_index(level=['Surrogate'], ascending=False, sort_remaining=False)
#df4_ = df4_.sort_index(level=['Test-time technique'], sort_remaining=False)
# define nb epochs
df4_.loc[("ImageNet", slice(None), "cSGLD", slice(None)), "Nb training epochs"] = 5*45
df4_.loc[("ImageNet", slice(None), "2 DNNs", slice(None)), "Nb training epochs"] = 2*130
df4_.loc[("CIFAR10", slice(None), "cSGLD", slice(None)), "Nb training epochs"] = 5*50
df4_.loc[("CIFAR10", slice(None), "1 DNN", slice(None)), "Nb training epochs"] = 1*250
df4_["Nb training epochs"] = df4_["Nb training epochs"].astype('int')
# define nb backward passes
df4_.loc[("CIFAR10", slice(None), slice(None), slice(None)), "Nb backward passes"] = 1*50
df4_.loc[("ImageNet", slice(None), slice(None), slice(None)), "Nb backward passes"] = 2*50
df4_["Nb backward passes"] = df4_["Nb backward passes"].astype('int')
df4_.to_latex('results/results_test_techniques.tex', multirow=True)


# Produce LaTeX table ImageNet holdout archs
df5 = pd.read_csv("X_adv/ImageNet/holdout/results_holdout.csv")
map_names = {
    'resnet50': 'ResNet50',
    'resnext50_32x4d': 'ResNeXt50',
    'densenet121': 'DenseNet121',
    'mnasnet1_0': 'MNASNet',
    'efficientnet_b0': 'EfficientNet B0',
}
df5['Holdout Target'] = '-'+df5['arch_target'].map(map_names)
df5['Holdout Target'] = pd.Categorical(df5['Holdout Target'], categories=['-ResNet50', '-ResNeXt50', '-DenseNet121', '-MNASNet', '-EfficientNet B0'])  # save custom order
df5['Attack'] = df5.apply(lambda row: f'L{row["norm_type"]:.0f}', axis=1)
df5['adv_sucess_rate'] = df5.apply(lambda row: f'{row["adv_sucess_rate"]*100:.2f} %', axis=1)
def f(row):
    if row['surrogate_type'] == 'cSGLD':
        return '1 cSGLD per arch.'
    elif row['surrogate_type'] == 'dnn':
        return '1 DNN per arch.' if row['surrogate_size_ensembles'] == 1 else f'{row["surrogate_size_ensembles"]} DNNs per arch.'
    return
df5['Surrogate'] = df5.apply(f, axis=1)
df5['Surrogate'] = pd.Categorical(df5['Surrogate'], categories=['1 cSGLD per arch.', '1 DNN per arch.'])  # save custom order
df5_ = df5.set_index(["Attack", "Surrogate", "Holdout Target"])[['adv_sucess_rate']]
df5_ = df5_.sort_index(key=lambda x: x.str.lower())
df5_ = df5_.unstack(2)
df5_.to_latex('X_adv/ImageNet/holdout/results_holdout.tex', multirow=True)
df5_.to_csv('X_adv/ImageNet/holdout/results_holdout__latextable.csv')

# Produce LaTeX table CIFAR-10 holdout archs
df5 = pd.read_csv("X_adv/CIFAR10/holdout/results_holdout.csv")
map_names = {
    'PreResNet110': 'PreResNet110',
    'PreResNet164': 'PreResNet164',
    'VGG16BN': 'VGG16bn',
    'VGG19BN': 'VGG19bn',
    'WideResNet28x10': 'WideResNet',
}
df5['Holdout Target'] = '-'+df5['arch_target'].map(map_names)
df5['Holdout Target'] = pd.Categorical(df5['Holdout Target'], categories=df5['Holdout Target'].unique())  # save custom order
df5['Attack'] = df5.apply(lambda row: f'L{row["norm_type"]:.0f}', axis=1)
df5['adv_sucess_rate'] = df5.apply(lambda row: f'{row["adv_sucess_rate"]*100:.2f} %', axis=1)
def f(row):
    if row['surrogate_type'] == 'cSGLD':
        return '1 cSGLD per arch.'
    elif row['surrogate_type'] == 'dnn':
        return '1 DNN per arch.' if row['surrogate_size_ensembles'] == 1 else f'{row["surrogate_size_ensembles"]} DNNs per arch.'
    return
df5['Surrogate'] = df5.apply(f, axis=1)
df5['Surrogate'] = pd.Categorical(df5['Surrogate'], categories=df5['Surrogate'].unique())  # save custom order
df5_ = df5.set_index(["Attack", "Surrogate", "Holdout Target"])[['adv_sucess_rate']]
df5_ = df5_.sort_index(key=lambda x: x.str.lower())
df5_ = df5_.unstack(2)
df5_.columns = df5_.columns.droplevel()  # drop 1st level of columns
df5_.to_latex('X_adv/CIFAR10/holdout/results_holdout.tex', multirow=True)
df5_.to_csv('X_adv/CIFAR10/holdout/results_holdout__tablelatex.csv')


# Produce LaTeX tables intra-arch

df6 = pd.read_csv("X_adv/ImageNet/resnet50/results_same_arch_total.csv")
df6['Dataset'] = "ImageNet"
df_tmp = pd.read_csv("X_adv/CIFAR10/PreResNet110/results_same_arch.csv")
df_tmp['Dataset'] = "CIFAR10"
df6 = pd.concat([df6, df_tmp])
df6.reset_index(drop=True, inplace=True)
df6['Surrogate'] = df6.apply(surrogate_name, axis=1)
df6['Surrogate'] = pd.Categorical(df6['Surrogate'], categories=['cSGLD', '1 DNN', '2 DNNs', '5 DNNs', '15 DNNs'])  # save custom order
df6['Attack Norm'] = df6.apply(lambda row: f'L{row["norm_type"]:.0f}', axis=1)
df6['adv_sucess_rate_'] = df6['adv_sucess_rate'] # save for use in df7 analysis
df6['adv_sucess_rate'] = df6.apply(lambda row: f'{row["adv_sucess_rate"]*100:.2f} %', axis=1)
def f(row):
    if row['n_random_init'] > 0 and row['n_iter'] > 1:
        return f'PGD ({row["n_random_init"]} restarts)'
    elif row['n_iter'] == 1 and row['norm_max'] == row['norm_step']:
        return 'FG(S)M'
    elif row['n_iter'] > 1 and pd.isna(row['momentum']):
        return 'I-FG(S)M'
    elif row['n_iter'] > 1 and row['momentum'] > 0:
        return 'MI-FG(S)M'
    return
df6['Attack'] = df6.apply(f, axis=1)
df6['Attack'] = pd.Categorical(df6['Attack'], categories=df6['Attack'].unique())  # save custom order
def f(row):
    if row['n_ensemble'] == 1:
        return 'Shallow (1 model per iter.)'
    elif row['n_ensemble'] == row['surrogate_size_ensembles']:
        return 'Original'
    return
df6['Attack iteration'] = df6.apply(f, axis=1)
df6.loc[df6.Attack == 'FG(S)M', 'Attack iteration'] = 'Original'  # FSGM on 1 DNN is original
# 2nd shallow attack on 1 DNN -> full attack on 1 DNN
idx_update = df6.loc[(df6.Dataset == 'ImageNet') & (df6.Attack == 'I-FG(S)M') & (df6.surrogate_size_ensembles == 1)].tail(2).index
df6.loc[idx_update, 'Attack iteration'] = 'Original'
idx_update = df6.loc[(df6.Dataset == 'ImageNet') & (df6.Attack == 'MI-FG(S)M') & (df6.surrogate_size_ensembles == 1)].tail(2).index
df6.loc[idx_update, 'Attack iteration'] = 'Original'
idx_update = df6.loc[(df6.Dataset == 'ImageNet') & (df6.Attack == 'PGD (5 restarts)') & (df6.surrogate_size_ensembles == 1)].tail(2).index
df6.loc[idx_update, 'Attack iteration'] = 'Original'
idx_update = df6.loc[(df6.Dataset == 'CIFAR10') & (df6.Attack == 'I-FG(S)M') & (df6.surrogate_size_ensembles == 1)].tail(2).index
df6.loc[idx_update, 'Attack iteration'] = 'Original'
idx_update = df6.loc[(df6.Dataset == 'CIFAR10') & (df6.Attack == 'MI-FG(S)M') & (df6.surrogate_size_ensembles == 1)].tail(2).index
df6.loc[idx_update, 'Attack iteration'] = 'Original'
idx_update = df6.loc[(df6.Dataset == 'CIFAR10') & (df6.Attack == 'PGD (5 restarts)') & (df6.surrogate_size_ensembles == 1)].tail(2).index
df6.loc[idx_update, 'Attack iteration'] = 'Original'
df6['Attack iteration'] = pd.Categorical(df6['Attack iteration'], categories=df6['Attack iteration'].unique())  # save custom order
# remove Original PGD (only executed for 2 and 5 DNNs, too costly for the rest)
#df6.drop(df6.query('n_random_init > 1 & n_ensemble > 1').index, inplace=True)

def f(row):
    if row['Dataset'] == 'ImageNet':
        if row['surrogate_type'] == 'cSGLD':
            return 45/3 * row['surrogate_size_ensembles']
        else:
            return 130 * row['surrogate_size_ensembles']
    if row['Dataset'] == 'CIFAR10':
        if row['surrogate_type'] == 'cSGLD':
            return 50/3 * row['surrogate_size_ensembles']
        else:
            return 250 * row['surrogate_size_ensembles']
    return
df6['Nb training epochs'] = df6.apply(f, axis=1).astype('int')
df6["Nb backward passes"] = (df6["n_ensemble"] * df6["n_iter"] * df6["n_random_init"].apply(lambda x: max(x, 1))).astype('int')  # nb ens x nb iter x nb restarts

# full table
df6_ = df6.set_index(["Dataset", "Attack iteration", "Attack", "Surrogate", "Attack Norm", 'Nb training epochs', 'Nb backward passes'])[['adv_sucess_rate']]
df6_ = df6_.unstack(4)
df6_ = df6_.reset_index(level=['Nb training epochs', 'Nb backward passes'], col_level=1)
df6_.columns = df6_.columns.droplevel()  # drop 1st level of columns
df6_ = df6_[['L2', 'Linf', 'Nb training epochs', 'Nb backward passes']]  # reorder columns
df6_.to_latex('results/results_same_arch_fulltable.tex', multirow=True)


# first table shallow iterative attacks + FSGM
#df6_ = df6.set_index(["Attack iteration", "Attack Norm", "Surrogate", "Attack"])[['adv_sucess_rate']]
df6_table1 = df6.query("`Attack iteration` == 'Shallow (1 model per iter.)' or Attack == 'FG(S)M'")
df6_ = df6_table1.set_index(["Dataset", "Attack", "Surrogate", "Attack Norm", 'Nb training epochs', 'Nb backward passes'])[['adv_sucess_rate']]
df6_ = df6_.unstack(3)
df6_ = df6_.reset_index(level=['Nb training epochs', 'Nb backward passes'], col_level=1)
df6_.columns = df6_.columns.droplevel()  # drop 1st level of columns
df6_ = df6_[['L2', 'Linf', 'Nb training epochs', 'Nb backward passes']]  # reorder columns
df6_.to_latex('results/results_same_arch_table1.tex', multirow=True)

# 2nd table full iterative attacks (except FSGM)
df6_table2 = df6.query("`Attack iteration` == 'Original' and Attack != 'FG(S)M'")
df6_ = df6_table2.set_index(["Dataset", "Attack", "Surrogate", "Attack Norm", 'Nb training epochs', 'Nb backward passes'])[['adv_sucess_rate']]
df6_ = df6_.unstack(3)
df6_ = df6_.reset_index(level=['Nb training epochs', 'Nb backward passes'], col_level=1)
df6_.columns = df6_.columns.droplevel()  # drop 1st level of columns
df6_ = df6_[['L2', 'Linf', 'Nb training epochs', 'Nb backward passes']]  # reorder columns
df6_.to_latex('results/results_same_arch_table2.tex', multirow=True)


# RQ - deep ensemble equivalent in transferability
# transfer rate for all DNNs ensemble size [0,15]
df7 = pd.read_csv("X_adv/CIFAR10/PreResNet110/results_dee.csv")
df7['Dataset'] = "CIFAR10"
df_tmp = pd.read_csv("X_adv/ImageNet/resnet50/results_dee.csv")
df_tmp['Dataset'] = "ImageNet"
df7 = pd.concat([df7, df_tmp])

df7['Attack Norm'] = df7.apply(lambda row: f'L{row["norm_type"]:.0f}', axis=1)
def f(row):
    if row['n_random_init'] > 0 and row['n_iter'] > 1:
        return f'PGD ({row["n_random_init"]} restarts)'
    elif row['n_iter'] == 1 and row['norm_max'] == row['norm_step']:
        return 'FG(S)M'
    elif row['n_iter'] > 1 and pd.isna(row['momentum']):
        return 'I-FG(S)M'
    elif row['n_iter'] > 1 and row['momentum'] > 0:
        return 'MI-FG(S)M'
    return
df7['Attack'] = df7.apply(f, axis=1)
def f(row):
    if row['Dataset'] == 'ImageNet':
        if row['surrogate_type'] == 'cSGLD':
            return 45/3 * row['surrogate_size_ensembles']
        else:
            return 130 * row['surrogate_size_ensembles']
    if row['Dataset'] == 'CIFAR10':
        if row['surrogate_type'] == 'cSGLD':
            return 50/3 * row['surrogate_size_ensembles']
        else:
            return 250 * row['surrogate_size_ensembles']
    return
df7['Nb training epochs'] = df7.apply(f, axis=1).astype('int')

def find_dee(df, query, df6=df6):
    query_end = " and n_ensemble == 1" if not "'FG(S)M'" in query else ""  # restrict to light ensembling only for iteraive attack
    df6_query = df6.query(query + " and surrogate_type == 'cSGLD'" + query_end)
    if df6_query.shape[0] != 1:
        raise ValueError("failed extraction from df6")
    fail_rate_csgld = df6_query.iloc[0]['adv_sucess_rate_']
    df_query = df.query(query)
    for i in range(df_query.shape[0]-1):
        df_query = df_query.sort_values(by=['surrogate_size_ensembles'])
        y0 = df_query.iloc[i,:]['adv_sucess_rate']
        y1 = df_query.iloc[i+1,:]['adv_sucess_rate']
        if y1 < fail_rate_csgld <= y0:
            # equivalence in nb of DNNs
            x0 = df_query.iloc[i,:]['surrogate_size_ensembles']
            x1 = df_query.iloc[i+1,:]['surrogate_size_ensembles']
            dee = (fail_rate_csgld - y0) * (x1-x0) / (y1-y0) + x0
            # equivalence in computational ratio
            if df_query.iloc[0]['Dataset'] == 'CIFAR10':
                dee_nb_epochs = dee * 250
                nb_epochs_csgld = 250
            elif df_query.iloc[0]['Dataset'] == 'ImageNet':
                dee_nb_epochs = dee * 130
                nb_epochs_csgld = 225
            else:
                raise ValueError('unknown dataset')
            computational_ratio = dee_nb_epochs / nb_epochs_csgld
            print(f'DEE: {dee:.3f} ; Comput ratio: {computational_ratio:.3f}')
            return (dee, computational_ratio)
    return None
find_dee(df7, query="Dataset == 'CIFAR10' and Attack == 'I-FG(S)M' and `Attack Norm` == 'Linf'")
find_dee(df7, query="Dataset == 'CIFAR10' and Attack == 'MI-FG(S)M' and `Attack Norm` == 'L2'")
find_dee(df7, query="Dataset == 'CIFAR10' and Attack == 'MI-FG(S)M' and `Attack Norm` == 'Linf'")
find_dee(df7, query="Dataset == 'CIFAR10' and Attack == 'FG(S)M' and `Attack Norm` == 'Linf'")
find_dee(df7, query="Dataset == 'CIFAR10' and Attack == 'PGD (5 restarts)' and `Attack Norm` == 'Linf'")

find_dee(df7, query="Dataset == 'ImageNet' and Attack == 'I-FG(S)M' and `Attack Norm` == 'L2'")
find_dee(df7, query="Dataset == 'ImageNet' and Attack == 'I-FG(S)M' and `Attack Norm` == 'Linf'")
find_dee(df7, query="Dataset == 'ImageNet' and Attack == 'MI-FG(S)M' and `Attack Norm` == 'L2'")
find_dee(df7, query="Dataset == 'ImageNet' and Attack == 'MI-FG(S)M' and `Attack Norm` == 'Linf'")
find_dee(df7, query="Dataset == 'ImageNet' and Attack == 'PGD (5 restarts)' and `Attack Norm` == 'L2'") #
find_dee(df7, query="Dataset == 'ImageNet' and Attack == 'PGD (5 restarts)' and `Attack Norm` == 'Linf'") #
find_dee(df7, query="Dataset == 'ImageNet' and Attack == 'FG(S)M' and `Attack Norm` == 'L2'")
find_dee(df7, query="Dataset == 'ImageNet' and Attack == 'FG(S)M' and `Attack Norm` == 'Linf'")
