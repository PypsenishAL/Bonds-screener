"""Presentation of saved investing results; no scoring on UI filter changes."""
import json
import numpy as np
import pandas as pd
from moex_bonds.analysis.spread_model import text_key

STATUSES={'relative_advantage':'Z-спред выше аналогов', 'no_advantage':'Нет преимущества по Z-спреду',
    'provisional':'Недостаточно близких аналогов', 'insufficient_peers':'Мало подходящих аналогов',
    'screen_incomplete':'Не хватает данных для проверки', 'requires_review':'Нужно проверить показатели',
    'invalid_data':'Не хватает исходных данных', 'not_saved':'Сравнение ещё не рассчитано'}

GROUPS=['AAA','AA','A','BBB','BB']
ORDER=['relative_advantage','no_advantage','provisional','screen_incomplete',
       'insufficient_peers','requires_review','invalid_data','not_saved']
LABELS={'investing__score':'Баллы в категории', 'investing__rank':'Место до фильтров',
    'investing_selection_rank':'Место в выборке',
    'investing__cohort_size':'Всего оценено в категории', 'investing__premium_bps':'Разница Z-спреда с аналогами, б.п.',
    'investing__ordinary_premium_bps':'Разница со средним Z-спредом аналогов, б.п.',
    'investing_status_label':'Инвестиционный статус','investing_reason_label':'Причина ограничения',
    'investing__n_issuers':'Компаний в основном сравнении','investing__n_eff':'Эффективное число компаний: основное сравнение',
    'investing__narrow_n_issuers':'Компаний с более близкой дюрацией','investing__narrow_n_eff':'Эффективное число компаний: близкая дюрация'}
REASONS={'unconfirmed_trade':'Нет подтверждённых сделок','structured_product':'Структурный продукт',
    'comparison_sensitive':'Преимущество зависит от сравнения',
    'missing_z_bps':'Нет подтверждённого Z-спреда','missing_YTM_WAPRICE':'Нет YTM',
    'missing_duration':'Нет положительной дюрации','missing_adv20':'Нет положительного оборота',
    'missing_rating_step':'Нет рейтинга','missing_issuer_key':'Нет эмитента','missing_sector_key':'Нет сектора',
    'unsupported_rating':'Рейтинг вне доступных категорий',
    'broad_coverage':'Недостаточно независимых компаний для сравнения',
    'narrow_coverage':'Не хватает аналогов с более близкой дюрацией',
    'z_bps':'Экстремальный Z-спред','YTM_WAPRICE':'Экстремальный YTM',
    'z_bps:few_issuers':'Для проверки Z-спреда мало других компаний',
    'YTM_WAPRICE:few_issuers':'Для проверки YTM мало других компаний',
    'z_bps:zero_mad':'Недостаточный разброс спредов для проверки',
    'YTM_WAPRICE:zero_mad':'Недостаточный разброс YTM для проверки'}

def reason_label(value):
    if pd.isna(value): return ''
    return '; '.join(REASONS.get(v,v) for v in str(value).split(';') if v)

def mode_defaults(state,mode):
    if state.get('_active_mode')==mode and (mode!='Инвестирование' or state.get('_investing_view_version')==3): return
    state['_active_mode']=mode
    state['sort_field']='investing_selection_rank' if mode=='Инвестирование' else 'peers_delta'
    state['sort_ascending']=mode=='Инвестирование'
    if mode=='Инвестирование':
        state['_investing_view_version']=3
        state.pop('filter_duration',None)
        state.pop('_duration_default_bounds',None)
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
    result.loc[result.get('investing__reason',pd.Series('',index=result.index)).eq('comparison_sensitive'),
               'investing_status_label']='Зависит от выбора аналогов'
    return result

def investing_view(frame,day,group='AAA',all_maturities=False,min_ytm=0.):
    result=frame.loc[frame['investing__group'].eq(group)].copy()
    maturity=(pd.to_datetime(result.MATDATE,errors='coerce')-pd.Timestamp(day)).dt.days/365.25
    if not all_maturities: result=result.loc[maturity.between(1,7)]
    if min_ytm>0: result=result.loc[pd.to_numeric(result.YTM_WAPRICE,errors='coerce').ge(min_ytm)]
    return result

def selection_ranks(frame):
    """Places among filtered scored rows; saved category ranks remain unchanged."""
    r=frame.copy()
    r['investing_selection_rank']=r.groupby('investing__group')['investing__rank'].rank(method='dense')
    return r

def sync_duration_filter(state,bounds):
    """A default full range follows the data; a deliberately narrowed range stays."""
    if bounds is None: return
    current=state.get('filter_duration')
    previous=state.get('_duration_default_bounds')
    if current is None or (previous is not None and tuple(current)==tuple(previous)):
        state['filter_duration']=bounds
    state['_duration_default_bounds']=bounds

def sort_investing(frame,field='investing__rank',ascending=None):
    r=frame.copy()
    r['_group']=r['investing__group'].map({v:i for i,v in enumerate(GROUPS)}).fillna(99)
    main=r['investing__status'].isin(['relative_advantage','no_advantage'])
    r['_status']=np.where(main,0,r['investing__status'].map({v:i+1 for i,v in enumerate(ORDER)}).fillna(99))
    if field not in r: field='investing__premium_bps'
    if ascending is None: ascending=field in ('investing__rank','investing_selection_rank','MATDATE','DURATION_YEARS')
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
    values=[('Z-спред выше / ниже аналогов',number(row.get('investing__premium_bps'),signed=True)+' б.п.'),
            ('Баллы в категории '+str(row.get('investing__group','')),number(row.get('investing__score'),1)),
            ('Место в выборке',number(row.get('investing_selection_rank'),0)),
            ('Компаний для сравнения',number(row.get('investing__n_issuers'),0))]
    for col,(label,value) in zip(st.columns(4),values): col.metric(label,value)
    with st.expander('С какими облигациями сравниваем и как считаем'):
        st.write('Подбираем облигации других компаний с таким же точным рейтингом и близкой дюрацией. От каждой компании берём один выпуск. Для результата используем самый высокий Z-спред аналогов, полученный в проверках ниже, и вычитаем его из Z-спреда выбранной бумаги.')
        st.caption('До применения фильтров: место '+number(row.get('investing__rank'),0)+
                   ' из '+number(row.get('investing__cohort_size'),0)+' оценённых выпусков категории '+str(row.get('investing__group',''))+'.')
        try: comparisons=json.loads(row.get('investing__audit','[]'))
        except (ValueError,TypeError): comparisons=[]
        for c in comparisons:
            st.markdown('**'+('Основное сравнение' if c['window']==1.5 else 'Сравнение с более близкой дюрацией')+'**')
            st.caption(f"Компаний: {c['n']}. Разница дюраций с выбранной бумагой — до {number(c['window'],1)} года.")
            st.caption(f"Z-спред аналогов: медиана {number(c['median_bps'])} б.п.; средний с учётом близости {number(c['weighted_bps'])} б.п.; самый высокий результат после исключения каждой компании по очереди {number(c['ceiling_bps'])} б.п.")
            members=pd.DataFrame(c['members'])
            if not members.empty:
                same=members.sector.eq(text_key(row.get('SECTOR',''))).sum()
                st.caption(f"Из сектора выбранной бумаги: {same} из {len(members)} компаний. Z-спреды аналогов: от {number(members.z_bps.min())} до {number(members.z_bps.max())} б.п.")
                if 'screen_status' in members:
                    if members.screen_status.eq('complete').all(): members=members.drop(columns='screen_status')
                    else: members['screen_status']=members.screen_status.map({'complete':'Данных достаточно','incomplete':'Недостаточно данных','flagged':'Показатели требуют проверки'})
                st.dataframe(members.rename(columns={'issuer':'Эмитент','rating':'Рейтинг','sector':'Сектор',
                    'duration':'Дюрация, лет','z_bps':'Z-спред, б.п.','distance':'Расстояние',
                    'weight':'Вес в среднем','screen_status':'Данные для проверки'}),hide_index=True)
