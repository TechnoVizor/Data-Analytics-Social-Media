#%%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

df = pd.read_csv('social_media_mental_health.csv')

# %%
platform_colors = {
    'Instagram': '#E1306C',
    'TikTok': '#000000',
    'YouTube': '#FF0000',
    'Facebook': '#1877F2',
    'Twitter': '#1DA1F2',
    'Snapchat': '#FFFC00',
    'Reddit': '#FF4500'
}

def auto_scale_y(data, padding=0.1):
    if hasattr(data, "values"):
        low = data.values.min()
        high = data.values.max()
    else:
        low = data.min()
        high = data.max()
        
    diff = high - low
    if diff == 0:
        plt.ylim(low - 1, low + 1)
        return
    plt.ylim(low - diff * padding, high + diff * padding)
    
    ticks = np.linspace(low - diff * padding, high + diff * padding, 10)
    plt.yticks(ticks)



#%%
comparsionGender = df.groupby('Gender')['Late_Night_Usage'].mean()

comparsionGender.plot(kind='bar' , color = ['orchid' , 'royalblue'] ,figsize=(8,8) ,edgecolor='black' )

auto_scale_y(comparsionGender)

plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--' , alpha=0.1)
plt.title('Использование Соц сетей Ночью')

plt.savefig('chart_1.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
platformCounts = df['Primary_Platform'].value_counts()

randomColors = plt.get_cmap('tab10')(range(len(platformCounts)))

colors_plat = [platform_colors.get(x, 'gray') for x in platformCounts.index]


platformCounts.plot(
    kind='bar' , color = colors_plat , figsize=(10,8) , edgecolor='black' 
)

plt.title('Какая платформа самая популярная')
plt.xlabel('Социальная сеть')
plt.ylabel('Кол-во человек')
plt.xticks(rotation=45)

plt.ylim(800 , 1220)
plt.yticks(np.arange(800,1221 , 25))
plt.grid(axis='y', linestyle='-' , alpha= 0.1)

plt.savefig('chart_2.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
platformGender = df.groupby(['Primary_Platform' , 'Gender']).size().unstack()

platformGender.plot(kind='bar' , color = ['orchid' , 'royalblue'] ,figsize=(12,6) ,edgecolor='black' )


plt.title('Предпочтения платформ по полам')
plt.ylabel('Количество пользователей')
plt.legend(title='Пол')

plt.xticks(rotation=45)


plt.grid(axis='y', linestyle='-' , alpha= 0.2)

auto_scale_y(platformGender)
plt.savefig('chart_3.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

data_gad = df.groupby('Primary_Platform')['GAD_7_Score'].mean().sort_values()

colors_gad = [platform_colors.get(x, 'gray') for x in data_gad.index]

data_gad.plot(kind='barh', color=colors_gad, ax=ax1, edgecolor='black')
ax1.set_title('Средний уровень тревоги (GAD-7)')
ax1.set_xlim(5, 9)
ax1.set_xticks(np.arange(5, 9.1, 0.5))
ax1.grid(axis='x', alpha=0.3)

data_phq = df.groupby('Primary_Platform')['PHQ_9_Score'].mean().sort_values()
colors_phq = [platform_colors.get(x, 'gray') for x in data_phq.index]

data_phq.plot(kind='barh', color=colors_phq, ax=ax2, edgecolor='black')
ax2.set_title('Средний уровень депрессивности (PHQ-9)')
ax2.set_xlim(4, 6)
ax2.set_xticks(np.arange(4, 6.1, 0.5))
ax2.grid(axis='x', alpha=0.3)

plt.savefig('chart_4.png', dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()


# %%
df['Screen_Time_Rounded'] = df['Daily_Screen_Time_Hours'].round()

avg_data = df.groupby('Screen_Time_Rounded')['Sleep_Duration_Hours'].mean()

plt.figure(figsize=(10, 6))

plt.xlim(1 , 10)
plt.xticks(np.arange(1, 10.5 , 1))

plt.ylim(3, 10)
plt.yticks(np.arange(3, 10.5 , 1))

avg_data.plot(kind='line', marker='o', color='red', linewidth=2, label='Средний сон')

plt.scatter(df['Daily_Screen_Time_Hours'], df['Sleep_Duration_Hours'], 
            alpha=0.1, color='gray', label='Все данные')

plt.title('Средняя зависимость сна от экранного времени')
plt.xlabel('Часов в день (округленно)')
plt.ylabel('Среднее количество сна')
plt.legend()
plt.grid(True, alpha=0.2)
plt.savefig('chart_5.png', dpi=300, bbox_inches='tight')
plt.show()



# %%
Dominant = df.value_counts('Dominant_Content_Type')

colors = plt.get_cmap('plasma')(np.linspace(0, 0.8, len(Dominant)))

Dominant.plot(kind='bar', color= colors, figsize=(10,6) , edgecolor='black')

auto_scale_y(Dominant)

plt.grid(axis='y', linestyle='-' , alpha= 0.2)

plt.xticks(rotation=45)
plt.title('Самый популярный контет в соц сетях')
plt.savefig('chart_6.png', dpi=300, bbox_inches='tight')
plt.show()


# %% 
content_comparison = df.groupby(['Dominant_Content_Type', 'Gender']).size().unstack()
content_comparison['Total'] = content_comparison.sum(axis=1)
content_comparison = content_comparison.sort_values('Total', ascending=False).drop(columns='Total')

ax = content_comparison.plot(
    kind='bar', 
    figsize=(14, 8), 
    color=['orchid', 'royalblue'],
    edgecolor='black',
    width=0.8
)

auto_scale_y(content_comparison)


plt.title('Популярность контента: Сравнение Мужчин и Женщин')
plt.xlabel('Тип контента')
plt.ylabel('Количество человек')

plt.xticks(rotation=45)


plt.legend(title='Пол', labels=['Female', 'Male'])
plt.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('chart_7.png', dpi=300, bbox_inches='tight')
plt.show()
