import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['bmi'] = df['weight'] / ((df['height']/100)**2)
df['overweight'] = (df['bmi'] > 25).astype(int)
#df['overweight'] = df.apply(lambda x: 1 if x['bmi'] > 25 else 0, axis=1)

# 3


df['cholesterol'] = df['cholesterol'].mask(df['cholesterol'] == 1, 0)
df['cholesterol'] = df['cholesterol'].mask(df['cholesterol'] > 1, 1)
df['gluc'] = df['gluc'].mask(df['gluc'] == 1, 0)
df['gluc'] = df['gluc'].mask(df['gluc'] > 1, 1)

# 4
def draw_cat_plot():
    # 5
    df_cat = df.melt(id_vars=['id','cardio'], value_vars=['cholesterol','gluc', 'smoke',  'alco',  "active",'overweight'])

    # 6

    dff = df_cat.groupby(['cardio','variable','value']).size().reset_index()
    dff.rename(columns={0:'total'}, inplace=True)


    # 7
    plot = sns.catplot(data=dff, x='variable', y='total', kind='bar', hue='value', col='cardio')

    # 8
    fig = plot.figure


    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    mask = (df['ap_lo'] <= df['ap_hi'])
    mask2 = (df['height'] >= df['height'].quantile(0.025))
    mask3 = (df['height'] <= df['height'].quantile(0.975))  
    mask4 = (df['weight'] >= df['weight'].quantile(0.025))    
    mask5 = (df['weight'] <= df['weight'].quantile(0.975))      
    df_heat = df.loc[mask & mask2 & mask3 & mask4 & mask5]
    df_heat.drop(columns=['bmi'], inplace=True)
    # 12
    corr = df_heat.corr()

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14
    fig, ax = plt.subplots(figsize=(12, 12))

    # 15
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt='.1f',
        square=True,
        cmap='coolwarm',
        ax=ax
    )


    # 16
    fig.savefig('heatmap.png')
    return fig
