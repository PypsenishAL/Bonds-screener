"""Presentation of saved investing results; no scoring on UI filter changes."""
import json
import numpy as np
import pandas as pd
from moex_bonds.analysis.spread_investing import STATUSES

GROUPS=['AAA','AA','A','BBB','BB']
ORDER=['relative_advantage','no_advantage','provisional','screen_incomplete',
       'insufficient_peers','requires_review','invalid_data','not_saved']
LABELS={'investing__score':'Оценка в категории', 'investing__rank':'Место в категории',
    'investing__cohort_size':'Оценено в категории', 'investing__premium_bps':'Устойчивая премия, б.п.',
    'investing__ordinary_premium_bps':'Премия к взвешенному ориентиру, б.п.',
    'investing_status_label':'Инвестиционный статус','investing_reason_label':'Причина ограничения',
    'investing__n_issuers':'Эмитентов: 1,5 года','investing__n_eff':'Эффективно эмитентов: 1,5 года',
    'investing__narrow_n_issuers':'Эмитентов: 1 год','investing__narrow_n_eff':'Эффективно эмитентов: 1 год'}
REASONS={'unconfirmed_trade':'Нет подтверждённых сделок','structured_product':'Структурный продукт',
    'comparison_sensitive':'Преимущество зависит от сравнения',
    'missing_z_bps':'Нет подтверждённого Z-спреда','missing_YTM_WAPRICE':'Нет YTM',
    'missing_duration':'Нет положительной дюрации','missing_adv20':'Нет положительного оборота',
    'missing_rating_step':'Нет рейтинга','missing_issuer_key':'Нет эмитента','missing_sector_key':'Нет сектора',
    'unsupported_rating':'Рейтинг вне доступных категорий',
    'broad_coverage':'В базовой корзине мало эмитентов или высока концентрация весов',
    'narrow_coverage':'Не подтверждено сравнение в диапазоне дюраций ±1 год',
    'z_bps':'Экстремальный Z-спред','YTM_WAPRICE':'Экстремальный YTM',
    'z_bps:few_issuers':'Для проверки спреда мало внешних эмитентов',
    'YTM_WAPRICE:few_issuers':'Для проверки YTM мало внешних эмитентов',
    'z_bps:zero_mad':'Недостаточный разброс спредов для проверки',
    'YTM_WAPRICE:zero_mad':'Недостаточный разброс YTM для проверки'}

def reason_label(value):
    if pd.isna(value): return ''
    return '; '.join(REASONS.get(v,v) for v in str(value).split(';') if v)

def mode_defaults(state,mode):
    if state.get('_active_mode')==mode: return
    state['_active_mode']=mode
    state['sort_field']='investing__premium_bps' if mode=='Инвестирование' else 'peers_delta'
    state['sort_ascending']=False
    state['list_scope']='Выделены анализом'
    for key in ('status','rating'): state['filter_'+key]=[]

def add_investing_labels(frame):
    result=frame.copy()
    if 'investing__status' not in result:
        result['investing__status']='not_saved'
        result['investing__group']=result.RATING.astype(str).str.replace(r'[+−–-]$','',regex=True)
    for field in LABELS:
        if field not in result: result[field]=np.nan
    result['investing_status_label']=result['investing__status'].map(STATUSES).fillna(STATUSES['not_saved'])
    result['investing_reason_label']=result.get('investing__reason',pd.Series('',index=result.index)).map(reason_label)
    return result

def investing_view(frame,day,group='AAA',all_maturities=False,min_ytm=0.):
    result=frame.loc[frame['investing__group'].eq(group)].copy()
    maturity=(pd.to_datetime(result.MATDATE,errors='coerce')-pd.Timestamp(day)).dt.days/365.25
    if not all_maturities: result=result.loc[maturity.between(1,7)]
    if min_ytm>0: result=result.loc[pd.to_numeric(result.YTM_WAPRICE,errors='coerce').ge(min_ytm)]
    return result

def sort_investing(frame,field='investing__premium_bps',ascending=False):
    r=frame.copy()
    r['_group']=r['investing__group'].map({v:i for i,v in enumerate(GROUPS)}).fillna(99)
    r['_status']=r['investing__status'].map({v:i for i,v in enumerate(ORDER)}).fillna(99)
    if field not in r: field='investing__premium_bps'
    fields=['_group','_status',field];order=[True,True,ascending]
    if 'adv20' not in fields: fields.append('adv20');order.append(False)
    if 'SECID' not in fields: fields.append('SECID');order.append(True)
    return r.sort_values(fields,ascending=order,kind='stable',na_position='last').drop(columns=['_group','_status']).reset_index(drop=True)

def render_card(st,row,number):
    st.markdown('#### Инвестиционное сравнение')
    status=row.get('investing__status','not_saved')
    st.write(STATUSES.get(status,STATUSES['not_saved']))
    explanation=reason_label(row.get('investing__reason',''))
    if explanation: st.caption(explanation)
    if status=='not_saved':
        st.info('В этом снимке инвестиционный расчёт не сохранён. Откройте актуальную дату.'); return
    values=[('Устойчивая премия',number(row.get('investing__premium_bps'),signed=True)+' б.п.'),
            ('Оценка в '+str(row.get('investing__group','')),number(row.get('investing__score'),1)),
            ('Место в категории',number(row.get('investing__rank'),1)+' из '+number(row.get('investing__cohort_size'),0)),
            ('Эмитентов: базовая / узкая',number(row.get('investing__n_issuers'),0)+' / '+number(row.get('investing__narrow_n_issuers'),0))]
    for col,(label,value) in zip(st.columns(4),values): col.metric(label,value)
    st.caption('Премия сравнивает выпуск с его аналогами; место и балл сравнивают такие премии внутри категории. Это не оценка надёжности и не вероятность прибыли.')
    with st.expander('Инвестиционные аналоги и проверка результата'):
        st.write('Одинаковая рейтинговая ступень, один выпуск от эмитента. Устойчивая премия сохраняется при медиане, взвешенном среднем и удалении любого одного представителя в двух диапазонах дюрации.')
        try: comparisons=json.loads(row.get('investing__audit','[]'))
        except (ValueError,TypeError): comparisons=[]
        for c in comparisons:
            st.markdown(f"**Разница дюраций до {number(c['window'],1)} года**")
            st.caption(f"Медиана: {number(c['median_bps'])} б.п. · взвешенный ориентир: {number(c['weighted_bps'])} б.п. · максимальный проверочный: {number(c['ceiling_bps'])} б.п.")
            members=pd.DataFrame(c['members'])
            if not members.empty:
                st.dataframe(members.rename(columns={'issuer':'Эмитент','rating':'Рейтинг','sector':'Сектор',
                    'duration':'Дюрация, лет','z_bps':'Z-спред, б.п.','distance':'Расстояние',
                    'weight':'Вес','screen_status':'Проверка экстремумов'}),hide_index=True)
        st.caption('Проверка экстремумов: '+str(row.get('investing__screen_status','—'))+
                   '. Отсутствие флага не доказывает отсутствие кредитного риска. Неполные проверки аналогов указаны в таблице.')
