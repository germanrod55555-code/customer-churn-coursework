"""Рисунки и дополнительные сводки из сохраненных результатов эксперимента."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from churn import load_data, FEATURES
FIG=ROOT/'figures'; FIG.mkdir(exist_ok=True)
R=ROOT/'results'; M=json.loads((R/'metrics.json').read_text(encoding='utf-8'))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                    'axes.spines.right':False,'axes.titleweight':'bold','savefig.dpi':190})
BLUE='#315C84'; ORANGE='#B25F2A'; GRAY='#7A8793'
NAMES={'dummy':'Базовая модель','logistic':'Логистическая регрессия','forest':'Случайный лес','boosting':'Градиентный бустинг'}
RUS=dict(zip(FEATURES,['Сбои звонков','Срок обслуживания','Уровень расходов','Длительность звонков',
    'Число звонков','Число SMS','Различные номера','Жалоба','Тариф']))

def save(name):
    plt.tight_layout();plt.savefig(FIG/name,bbox_inches='tight');plt.close()

def build():
    d,_=load_data(ROOT/'data/customer_churn.csv')
    split=pd.read_csv(R/'split_manifest.csv')
    tr=d[d.source_row.isin(split.loc[split.split=='train','source_row'])]
    counts=tr.Churn.value_counts().reindex([0,1])
    fig,ax=plt.subplots(figsize=(7.4,3.6))
    bars=ax.bar(['Сохранили обслуживание','Ушли'],counts,color=[BLUE,ORANGE],width=.55)
    ax.bar_label(bars,labels=[f'{n} ({n/len(tr):.1%})' for n in counts],padding=4)
    ax.set_ylim(0,counts.max()*1.2);ax.set_ylabel('Число клиентов');ax.set_title('Баланс классов в обучающей выборке')
    save('01_classes.png')
    comparison=pd.DataFrame(M['model_comparison'])
    fig,ax=plt.subplots(figsize=(7.4,4.0))
    ax.barh([NAMES[n] for n in comparison.model],comparison.cv_ap,xerr=comparison.cv_std,color=[GRAY,BLUE,BLUE,ORANGE],capsize=4)
    ax.set_xlim(0,1);ax.set_xlabel('Средняя AP на трех групповых разбиениях')
    ax.set_title('Сравнение моделей на обучающих данных')
    ax.grid(axis='x',alpha=.2);save('02_models.png')
    curve=pd.read_csv(R/'threshold_validation.csv')
    fig,ax=plt.subplots(figsize=(7.4,4.0))
    for key,label,color in [('precision','Точность',BLUE),('recall','Полнота',ORANGE),('f2','F2',GRAY)]:
        ax.plot(curve.threshold,curve[key],label=label,color=color,lw=2)
    ax.axvline(M['threshold'],ls='--',color='black',lw=1,label=f'Выбранный порог {M["threshold"]:.2f}')
    ax.set(xlabel='Порог отнесения к группе риска',ylabel='Значение метрики',ylim=(0,1.05))
    ax.legend(fontsize=9);ax.grid(alpha=.2);save('03_threshold.png')
    pred=pd.read_csv(R/'test_predictions.csv'); y=pred.Churn; p=pred.probability
    fpr,tpr,_=roc_curve(y,p);pr,re,_=precision_recall_curve(y,p)
    fig,axes=plt.subplots(1,2,figsize=(8.5,3.7))
    axes[0].plot(fpr,tpr,color=BLUE,label=f'ROC AUC = {M["test"]["roc_auc"]:.3f}')
    axes[0].plot([0,1],[0,1],ls='--',color=GRAY);axes[0].set(xlabel='Доля ложных тревог',ylabel='Полнота',title='ROC на тесте')
    axes[1].step(re,pr,where='post',color=BLUE,label=f'AP = {M["test"]["ap"]:.3f}')
    axes[1].axhline(y.mean(),ls='--',color=GRAY,label=f'Базовый уровень {y.mean():.3f}')
    axes[1].set(xlabel='Полнота',ylabel='Точность',title='Precision–Recall на тесте')
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.15);ax.set_xlim(0,1);ax.set_ylim(0,1.03)
    save('04_curves.png')
    matrix=np.array(M['test']['confusion'])
    fig,ax=plt.subplots(figsize=(6.2,4.3))
    ax.imshow(matrix,cmap='Blues')
    for i in range(2):
        for j in range(2):ax.text(j,i,str(matrix[i,j]),ha='center',va='center',fontsize=24,color='white' if matrix[i,j]>220 else 'black')
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Останется','Риск ухода'],yticklabels=['Остался','Ушел'],xlabel='Прогноз модели',ylabel='Фактический класс')
    ax.set_title(f'Матрица ошибок при пороге {M["threshold"]:.2f}')
    save('05_confusion.png')
    importance=pd.read_csv(R/'permutation_validation.csv').sort_values('mean_ap_drop')
    fig,ax=plt.subplots(figsize=(8,4.6))
    ax.barh([RUS[x] for x in importance.feature],importance.mean_ap_drop,xerr=importance.std_ap_drop,color=BLUE,capsize=3)
    ax.set_xlabel('Снижение AP после перестановки признака');ax.set_title('Важность признаков на validation')
    ax.grid(axis='x',alpha=.2);save('06_importance.png')
    ordered=pred.sort_values('probability',ascending=False,kind='stable')
    lift=[]
    for share in (.1,.2,.3,.5,1):
        n=int(np.ceil(len(ordered)*share));found=int(ordered.head(n).Churn.sum())
        lift.append({'share':share,'contacts':n,'found_churn':found,'precision':found/n,
                     'recall':found/y.sum(),'lift':(found/n)/y.mean()})
    pd.DataFrame(lift).to_csv(R/'lift.csv',index=False)
    fig,ax=plt.subplots(figsize=(7.4,4.1))
    ax.plot(np.arange(1,len(y)+1)/len(y),ordered.Churn.cumsum()/y.sum(),color=BLUE,lw=2,label='Ранжирование моделью')
    ax.plot([0,1],[0,1],ls='--',color=GRAY,label='Случайный выбор в среднем')
    ax.set(xlabel='Доля клиентов для контакта',ylabel='Доля найденных уходов',xlim=(0,1),ylim=(0,1.02))
    ax.legend();ax.grid(alpha=.2);save('07_gain.png')
    load=pd.read_csv(R/'load_summary.csv')
    for metric,ylabel,name in [('seconds','Медианное время, с','08_load_time.png'),('peak_rss_mib','Медианный пик RSS, МиБ','09_load_memory.png')]:
        fig,ax=plt.subplots(figsize=(7.4,3.9))
        for batch,color,marker in [(5000,BLUE,'o'),(25000,ORANGE,'s')]:
            selected=load[load.batch==batch]
            ax.plot(selected.rows,selected[metric],marker=marker,color=color,label=f'Пакет {batch:,}'.replace(',',' '))
        ax.set(xlabel='Число строк нагрузочного файла',ylabel=ylabel)
        ax.ticklabel_format(style='plain',axis='x');ax.legend();ax.grid(alpha=.2)
        save(name)
    # Условная экономика, а не оценка причинного эффекта удержания.
    econ=[]
    for q in (.1,.2,.3):
        for cost in (100,200,400):
            for label,key in [('Порог 0.05','test'),('Порог 0.50','test_at_05')]:
                cm=M[key]['confusion'];tp=cm[1][1];contacts=cm[0][1]+tp
                econ.append({'policy':label,'success_probability':q,'contact_cost':cost,'margin':3000,
                             'contacts':contacts,'true_positives':tp,'scenario_effect':tp*q*3000-contacts*cost})
    pd.DataFrame(econ).to_csv(R/'economics_scenarios.csv',index=False)
    print('Создано рисунков:',len(list(FIG.glob('*.png'))))

if __name__=='__main__':build()
