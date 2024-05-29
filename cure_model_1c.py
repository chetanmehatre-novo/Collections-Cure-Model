#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 23:18:29 2024

@author: chetan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config2

from config2 import SQLQuery

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from snowflake.sqlalchemy import URL


q = SQLQuery('snowflake')


  
self_cure_model_base2 = q("""
          SELECT *
          FROM PROD_DB.ADHOC.SELF_CURE_MODEL_TRAINING_BASE4
          """)
          

self_cure_model_base2.rename(columns={col: col.lower() for col in self_cure_model_base2.columns}, inplace=True)

self_cure_model_base2['_fivetran_synced_lag_7days'] = self_cure_model_base2['base_date_lag_7_days'].copy()

         


## Prepare TCE data base

tce_null_base = pd.read_csv('/Users/admin/Downloads/tce_pred_self_cure_base.csv')

tce_null_base['tce_v1_category'] = tce_null_base['predicted_target'] 

tce_null_base = tce_null_base[['business_id', 'transaction_id', 'amount', 'type', 'description',
       'transaction_date', 'tce_v1_category']]
 
tce_txn_base = q("""
                 SELECT DISTINCT t1.business_id,
                   t1.id AS transaction_id,
                   abs(t1.amount) AS amount,
                   t1.type,
                   t1.description,
                   t1.transaction_date,
                   t2.tce_v1_category
            FROM "FIVETRAN_DB"."PROD_NOVO_API_PUBLIC"."TRANSACTIONS" AS t1
            LEFT JOIN (
                SELECT transaction_id, predicted_category as tce_v1_category 
                FROM PROD_DB.DATA.TCE_NOVOTXNS_SCORED
                UNION
                SELECT transaction_id, tce_v1_category 
                FROM prod_db.data.tce_scored_lending_drawn_customers
            ) AS t2
            ON t1.id = t2.transaction_id
            WHERE date(t1.transaction_date) >= date('2023-05-01')
              AND date(t1.transaction_date) < date('2024-04-26') 
              AND t2.tce_v1_category is not null
              AND t1.business_id IN (
                  SELECT business_id 
                  FROM PROD_DB.ADHOC.SELF_CURE_MODEL_TRAINING_BASE4)
             """)
 
tce_txn_base2 = pd.concat([tce_txn_base, tce_null_base], axis=0)

tce_txn_base2['transaction_date'] = pd.to_datetime(tce_txn_base2['transaction_date'])
    
### Write query

from sqlalchemy.types import NVARCHAR
 # from conf.config import SQLQuery
q = SQLQuery('snowflake')

tce_txn_base2.to_sql(name='self_cure_model_txn_tce_base2',
                  con=q.engine, 
                  schema='prod_db.adhoc',
                  if_exists='append', 
                  index=False, 
                  chunksize=8000, 
                  method='multi',
                  dtype={col_name: NVARCHAR for col_name in tce_txn_base2}) 
    

 
##  Transactional variables

credit_l3m = q("""SELECT 
                t1.business_id, 
                t1.days_past_due,
                t1.base_date_lag_7_days,
                SUM(ABS(t2.amount)) AS novo_total_credit_l3m, 
                AVG(ABS(t2.amount)) AS novo_avg_credit_l3m,
                COUNT(t2.transaction_id) AS novo_credit_txn_count_l3m
            FROM 
                PROD_DB.ADHOC.SELF_CURE_MODEL_TRAINING_BASE4 t1
            LEFT JOIN 
                PROD_DB.ADHOC.self_cure_model_txn_tce_base2 t2
                ON t1.business_id = t2.business_id
                AND t2.type = 'credit' 
                AND date(t2.transaction_date) >= DATEADD(DAY, -90, date(t1.base_date_lag_7_days)) and date(t2.transaction_date) <= date( t1.base_date_lag_7_days)
            GROUP BY 
                t1.base_date_lag_7_days, t1.business_id, t1.days_past_due
            order by business_id
                                """)
                
self_cure_model_base2['base_date_lag_7_days'] = pd.to_datetime(self_cure_model_base2['base_date_lag_7_days'])

credit_l3m['base_date_lag_7_days'] = pd.to_datetime(credit_l3m['base_date_lag_7_days'])

self_cure_model_base2['days_past_due'] = self_cure_model_base2['days_past_due'].astype(str)

base = pd.merge(self_cure_model_base2, credit_l3m, how='left', on=['business_id', 'days_past_due', 'base_date_lag_7_days'])

#   
debit_l3m = q("""SELECT 
                 t1.business_id, 
                 t1.days_past_due,
                 t1.base_date_lag_7_days,
                 SUM(ABS(t2.amount)) AS novo_total_debit_l3m, 
                 AVG(ABS(t2.amount)) AS novo_avg_debit_l3m,
                 COUNT(t2.transaction_id) AS novo_debit_txn_count_l3m
             FROM 
                 PROD_DB.ADHOC.SELF_CURE_MODEL_TRAINING_BASE4 t1
             LEFT JOIN 
                 PROD_DB.ADHOC.self_cure_model_txn_tce_base2 t2
                 ON t1.business_id = t2.business_id
                 AND t2.type = 'debit' 
                 AND t2.transaction_date >= DATEADD(DAY, -90, t1.base_date_lag_7_days) and t2.transaction_date <= date( t1.base_date_lag_7_days)
             GROUP BY 
                 t1.base_date_lag_7_days, t1.business_id, t1.days_past_due
             order by business_id
                                 """)
                 
debit_l3m['base_date_lag_7_days'] = pd.to_datetime(debit_l3m['base_date_lag_7_days'])

base = pd.merge(base, debit_l3m, how='left', on=['business_id', 'days_past_due', 'base_date_lag_7_days'])

#
credit_l1m = q("""SELECT 
                t1.business_id, 
                t1.days_past_due,
                t1.base_date_lag_7_days,
                SUM(ABS(t2.amount)) AS novo_total_credit_l1m, 
                AVG(ABS(t2.amount)) AS novo_avg_credit_l1m,
                COUNT(t2.transaction_id) AS novo_credit_txn_count_l1m
            FROM 
                PROD_DB.ADHOC.SELF_CURE_MODEL_TRAINING_BASE4 t1
            LEFT JOIN 
                PROD_DB.ADHOC.self_cure_model_txn_tce_base2 t2
                ON t1.business_id = t2.business_id
                AND t2.type = 'credit' 
                AND t2.transaction_date >= DATEADD(DAY, -30, t1.base_date_lag_7_days) and t2.transaction_date <= date( t1.base_date_lag_7_days)
            GROUP BY 
                t1.base_date_lag_7_days, t1.business_id, t1.days_past_due
            order by business_id
                                """)
credit_l1m['base_date_lag_7_days'] = pd.to_datetime(credit_l1m['base_date_lag_7_days'])                


base = pd.merge(base, credit_l1m, how='left', on=['business_id', 'days_past_due', 'base_date_lag_7_days'])

#   
debit_l1m = q("""SELECT 
                 t1.business_id, 
                 t1.days_past_due,
                 t1.base_date_lag_7_days,
                 SUM(ABS(t2.amount)) AS novo_total_debit_l1m, 
                 AVG(ABS(t2.amount)) AS novo_avg_debit_l1m,
                 COUNT(t2.transaction_id) AS novo_debit_txn_count_l1m
             FROM 
                 PROD_DB.ADHOC.SELF_CURE_MODEL_TRAINING_BASE4 t1
             LEFT JOIN 
                 PROD_DB.ADHOC.self_cure_model_txn_tce_base2 t2
                 ON t1.business_id = t2.business_id
                 AND t2.type = 'debit' 
                 AND t2.transaction_date >= DATEADD(DAY, -30, t1.base_date_lag_7_days) and t2.transaction_date <= date( t1.base_date_lag_7_days)
             GROUP BY 
                 t1.base_date_lag_7_days, t1.business_id, t1.days_past_due
             order by business_id
                                 """)
                 
debit_l1m['base_date_lag_7_days'] = pd.to_datetime(debit_l1m['base_date_lag_7_days'])           

base = pd.merge(base, debit_l1m, how='left', on=['business_id', 'days_past_due', 'base_date_lag_7_days'])

    
# credit and debit variables, l1m l3m, ratio, categorywise

base['ratio_tot_debit_l1m_l3m'] = base['novo_total_debit_l1m'] / base['novo_total_debit_l3m']

base['ratio_tot_credit_l1m_l3m'] = base['novo_total_credit_l1m'] / base['novo_total_credit_l3m']


base['ratio_debit_txn_count_l1m_l3m'] = base['novo_debit_txn_count_l1m'] / base['novo_debit_txn_count_l3m']

base['ratio_credit_txn_count_l1m_l3m'] = base['novo_credit_txn_count_l1m'] / base['novo_credit_txn_count_l3m']



# Novo feature vars

dc_l3m = q("""SELECT t1.business_id,  t1.days_past_due, t1.base_date_lag_7_days, sum(abs(amount)) as dc_total_amount_l3m, avg(abs(amount)) as dc_avg_txn_amount_l3m
              
                FROM PROD_DB.ADHOC.SELF_CURE_MODEL_TRAINING_BASE4 t1
                
                LEFT JOIN "FIVETRAN_DB"."PROD_NOVO_API_PUBLIC"."TRANSACTIONS" t2
                ON t1.business_id = t2.business_id
                
                AND t2.transaction_date >= DATEADD(DAY, -90, t1.base_date_lag_7_days) and t2.transaction_date <= date( t1.base_date_lag_7_days)
                 
                and t2.type = 'debit' and t2.medium = 'POS Withdrawal'
                
                group by t1.base_date_lag_7_days, t1.business_id, t1.days_past_due
                order by business_id""")
                
dc_l3m['base_date_lag_7_days'] = pd.to_datetime(dc_l3m['base_date_lag_7_days'])  

base = pd.merge(base, dc_l3m, how='left', on=['business_id', 'days_past_due', 'base_date_lag_7_days'])


# Reserves

reserves_setup = q("""SELECT 
                        t1.business_id, 
                        t1.days_past_due, 
                        t1.base_date_lag_7_days, 
                        SUM(t2.balance) AS reserves_balance_sum
                    FROM 
                        PROD_DB.ADHOC.SELF_CURE_MODEL_TRAINING_BASE4 t1
                    LEFT JOIN 
                        "METABASE_DB"."PRODUCT"."RESERVES" t2
                    ON 
                        t1.business_id = t2.business_id
                        AND t2.status = 'active' 
                        AND DATE(t2.created_at) <= DATE(t1.base_date_lag_7_days)
                    GROUP BY 
                        t1.business_id, 
                        t1.days_past_due, 
                        t1.base_date_lag_7_days
                    ORDER BY 
                        t1.business_id;

                      
                """)
                
reserves_setup['base_date_lag_7_days'] = pd.to_datetime(reserves_setup['base_date_lag_7_days']) 

base = pd.merge(base, reserves_setup, how='left', on=['business_id', 'days_past_due', 'base_date_lag_7_days'])



# Invoices

invoice_usage_flag = q("""SELECT DISTINCT t1.business_id, t1.days_past_due, t1.base_date_lag_7_days,
               case when t2.business_id is not null then 1 else 0 end as invoice_usage_flag
               
               FROM PROD_DB.ADHOC.SELF_CURE_MODEL_TRAINING_BASE4 t1
               
               LEFT JOIN "METABASE_DB"."PRODUCT"."INVOICES" t2
               ON t1.business_id = t2.business_id
               
               AND t2.status in ('paid', 'sent', 'scheduled') and date(t2.created_at) <= date(t1.base_date_lag_7_days)
                            
               """)
               
invoice_vars_l3m = q("""SELECT t1.business_id, t1.days_past_due, t1.base_date_lag_7_days, sum(t2.invoice_total) as invoice_amount_sum_l3m, avg(t2.invoice_total) as invoice_avg_txn_amt_l3m
               FROM PROD_DB.ADHOC.SELF_CURE_MODEL_TRAINING_BASE4 t1
               
               LEFT JOIN "METABASE_DB"."PRODUCT"."INVOICES" t2
               ON t1.business_id = t2.business_id
               
               AND t2.status in ('paid', 'sent', 'scheduled')
               AND date(t2.created_at) >= DATEADD(DAY, -90, t1.base_date_lag_7_days) and date(t2.created_at) <= date( t1.base_date_lag_7_days)
               group by t1.base_date_lag_7_days, t1.business_id, t1.days_past_due
               order by business_id
                            
               """)


invoice_usage_flag['base_date_lag_7_days'] = pd.to_datetime(invoice_usage_flag['base_date_lag_7_days']) 

invoice_vars_l3m['base_date_lag_7_days'] = pd.to_datetime(invoice_vars_l3m['base_date_lag_7_days']) 

base = pd.merge(base, invoice_usage_flag, how='left', on=['business_id', 'days_past_due', 'base_date_lag_7_days'])


base = pd.merge(base, invoice_vars_l3m, how='left', on=['business_id', 'days_past_due', 'base_date_lag_7_days'])


# 
pf_deposits_l3m = q(""" select t1.business_id, t1.days_past_due, t1.base_date_lag_7_days,  sum(abs(t2.amount)) AS pf_total_deposits_l3m, 
                    avg(abs(t2.amount)) AS pf_avg_deposits_txn_amt_l3m
                
                FROM PROD_DB.ADHOC.SELF_CURE_MODEL_TRAINING_BASE4 t1
                
                LEFT JOIN "METABASE_DB"."TRANSACTION"."TRANSACTIONS_MODEL" t2
                ON t1.business_id = t2.business_id
                
                AND t2.transaction_date >= DATEADD(DAY, -90, t1.base_date_lag_7_days) and t2.transaction_date <= date( t1.base_date_lag_7_days)
                and t2.type='credit' and t2.medium = 'External Deposit' 
        
                and t2.description ILIKE ANY(
                    '%AMAZON%'
                    , '%AMZN%'
                    , '%STRIPE%'
                    , '%SQUARE INC%'
                    , '%SHOPIFY%'
                    , '%SHOPPAY%'
                    , '%AIRBNB%'
                    , '%ETSY%'
                    , '%EBAY%'
                    , '%BILL.COM%'
                    , '%INTUIT%'
                    , '%GUSTO%'
                    , '%WIX%'
                    , '%WAVE%'
                    , '%UPWORK%'
                    , '%MELIO%'
                    , '%PATREON%'
                    , '%VRBO%'
                    , '%UNITEDHEALTHCARE%'
                    , '%FAIRE%'
                    , '%VAGARO%'
                    , '%CIGNA%'
                    , '%AETNA%'
                    , '%WEPAY%'
                    , '%QUICKBOOKS%'
                    )
                    
                    AND NOT(
                    description ILIKE ANY(
                    '%LYFT%'
                    , '%OfferUp%'
                    , '%Gumroad%'
                    , '%FB%Fundrai%'
                    , '%Verify%'
                    , '%CASH%'
                    , '%PAYROLL%'
                    , '%VRFY%'
                    , '%CAPITAL%'
                    , '%REFUND%'
                    ))
                group by t1.base_date_lag_7_days, t1.business_id, t1.days_past_due
                order by t1.business_id
                
                """)
                
pf_deposits_l3m['base_date_lag_7_days'] = pd.to_datetime(pf_deposits_l3m['base_date_lag_7_days'])

base = pd.merge(base, pf_deposits_l3m, how='left', on=['business_id', 'days_past_due', 'base_date_lag_7_days'])


# application level vars, login details

biz_vars = q("""
             select business_id,
                case when email_domain = 'gmail.com' then 1 else 0 end as email_domain_is_gmail,
                state, business_type, core_customer_flag, date(account_funded_at) as account_funded_date, date(business_created_at) as business_created_date
                from Prod_db.data.businesses
                """)
                
base = pd.merge(base, biz_vars, how='left', on=['business_id'])

# mob

base['days_from_account_funded_date'] = (pd.to_datetime(base['base_date_lag_7_days']) - pd.to_datetime(base['account_funded_date'])).dt.days

base['days_from__account_created_date'] = (pd.to_datetime(base['base_date_lag_7_days']) - pd.to_datetime(base['business_created_date'])).dt.days

# repayment

first_draw = q("""
                    with first_draw_dt as (
                    select distinct business_id, lending_business_id, min(created_at) as created_at
                    from "FIVETRAN_DB"."PROD_NOVO_API_PUBLIC"."LENDING_TRANSACTIONS" 
                    where description ilike '%MCA DRAW%'
                    and (_fivetran_deleted is null or _fivetran_deleted = false)
                    group by 1,2
                    ),
                    
                    first_draw as (
                    select distinct business_id, lending_business_id, date(created_at) as first_draw_date, amount/100 as first_draw_amount
                    from "FIVETRAN_DB"."PROD_NOVO_API_PUBLIC"."LENDING_TRANSACTIONS" a 
                    join first_draw_dt b
                    using (business_id, lending_business_id, created_at)
                    where description ilike '%MCA DRAW%'
                    and (_fivetran_deleted is null or _fivetran_deleted = false)
                    ),
                    
                    first_draw_limit as (
                    select * from (
                    select distinct a.business_id, a.lending_business_id, first_draw_date, first_draw_amount, 
                    case when a.lending_business_id = '81e80281-b42c-46b1-aa8d-84b0557d9cd2' then 4200 else new_limit/100 end as credit_limit, 
                    first_draw_amount/credit_limit as first_draw_utilisation,
                    row_number() over(partition by a.lending_business_id order by a.created_at desc) as row_num
                    from "FIVETRAN_DB"."PROD_NOVO_API_PUBLIC"."LENDING_DECISIONING_HISTORY" a
                    join first_draw b
                    on a.lending_business_id = b.lending_business_id
                    and date(a.created_at) <= date(b.first_draw_date)
                    and (_fivetran_deleted is null or _fivetran_deleted = false)
                    and trigger_event not in ('Lending Application Denied')
                    )
                    where row_num=1
                    order by business_id, first_draw_date desc
                    )
                    
                    select * from first_draw_limit
                    """)


base = pd.merge(base, first_draw, how='left', on=['business_id', 'lending_business_id'])


base['days_from__first_draw_date'] = (pd.to_datetime(base['base_date_lag_7_days']) - pd.to_datetime(base['first_draw_date'])).dt.days



#
draws = q("""
          select distinct t2.business_id, t2.lending_business_id, t1._fivetran_synced,
            count(case when t2.created_at <= t1.base_date_lag_7_days then t2.transaction_id end) over(partition by t2.business_id, t1.base_date_lag_7_days order by t1.business_id) as total_draw_count,
            coalesce(sum(case when t2.created_at <= t1.base_date_lag_7_days then amount end) over(partition by t2.business_id, t1.base_date_lag_7_days order by t1.business_id),0) as total_amount_drawn
            from PROD_DB.ADHOC.SELF_CURE_MODEL_TRAINING_BASE3 t1
            left join "FIVETRAN_DB"."PROD_NOVO_API_PUBLIC"."LENDING_TRANSACTIONS" t2
            ON t1.business_id = t2.business_id AND t1.base_date_lag_7_days >= t2.created_at
            AND t2.description ilike '%MCA DRAW%'
            and (t2._fivetran_deleted is null or t2._fivetran_deleted = false)
            """)
            
draws['_fivetran_synced'] = pd.to_datetime(draws['_fivetran_synced'])

base['_fivetran_synced'] = pd.to_datetime(base['_fivetran_synced'])

base = pd.merge(base, draws, how='left', on=['business_id', 'lending_business_id', '_fivetran_synced'])



# fico

credit_profile = q("""
                 WITH ranked_profiles AS (
                    SELECT 
                        *,
                        ROW_NUMBER() OVER (PARTITION BY lending_business_id ORDER BY updated_at DESC) AS row_num
                    FROM 
                        FIVETRAN_DB.PROD_NOVO_API_PUBLIC.CUSTOMER_CREDIT_PROFILES
                    WHERE 
                        credit_profile IS NOT NULL
                        AND fico_score IS NOT NULL
                )
                SELECT 
                    lending_business_id,
                    PARSE_JSON(aggregate_kpis):"fico_score"::VARCHAR AS fico_score,
                    PARSE_JSON(aggregate_kpis):"number_of_60_days_delinquencies"::VARCHAR AS number_of_60_days_delinquencies,
                    PARSE_JSON(aggregate_kpis):"number_of_inquiries_last_6_months"::VARCHAR AS number_of_inquiries_last_6_months,
                    PARSE_JSON(aggregate_kpis):"number_of_open_trades"::VARCHAR AS number_of_open_trades,
                    PARSE_JSON(aggregate_kpis):"number_of_trades_with_derogatory_status"::VARCHAR AS number_of_trades_with_derogatory_status,
                    updated_at
                FROM (
                    SELECT 
                        lending_business_id,
                        aggregate_kpis,
                        updated_at,
                        ROW_NUMBER() OVER (PARTITION BY lending_business_id ORDER BY updated_at DESC) AS row_num
                    FROM 
                        ranked_profiles
                    WHERE 
                        row_num = 1
                ) AS subquery
                WHERE 
                    row_num = 1    
                 """)
    
 
base = pd.merge(base, credit_profile, how='left', on=['lending_business_id'])



## Additional TCE variables

# loan repayment

loan_repay_l3m = q("""SELECT 
                t1.business_id, 
                t1.days_past_due,
                t1.base_date_lag_7_days,
                SUM(ABS(t2.amount)) AS loan_repayment_l3m
            FROM 
                PROD_DB.ADHOC.SELF_CURE_MODEL_TRAINING_BASE4 t1
            LEFT JOIN 
                PROD_DB.ADHOC.self_cure_model_txn_tce_base2 t2
                ON t1.business_id = t2.business_id
                AND t2.type = 'debit' and t2.tce_v1_category = 'Loans & Interest'
                AND date(t2.transaction_date) >= DATEADD(DAY, -90, date(t1.base_date_lag_7_days)) and date(t2.transaction_date) <= date( t1.base_date_lag_7_days)
            GROUP BY 
                t1.base_date_lag_7_days, t1.business_id, t1.days_past_due
            order by business_id
                                """)


loan_repay_l1m = q("""SELECT 
                t1.business_id, 
                t1.days_past_due,
                t1.base_date_lag_7_days,
                SUM(ABS(t2.amount)) AS loan_repayment_l1m
            FROM 
                PROD_DB.ADHOC.SELF_CURE_MODEL_TRAINING_BASE4 t1
            LEFT JOIN 
                PROD_DB.ADHOC.self_cure_model_txn_tce_base2 t2
                ON t1.business_id = t2.business_id
                AND t2.type = 'debit' and t2.tce_v1_category = 'Loans & Interest'
                AND date(t2.transaction_date) >= DATEADD(DAY, -30, date(t1.base_date_lag_7_days)) and date(t2.transaction_date) <= date( t1.base_date_lag_7_days)
            GROUP BY 
                t1.base_date_lag_7_days, t1.business_id, t1.days_past_due
            order by business_id
                                """)

loan_repay_l3m['base_date_lag_7_days'] = pd.to_datetime(loan_repay_l3m['base_date_lag_7_days'])
loan_repay_l1m['base_date_lag_7_days'] = pd.to_datetime(loan_repay_l1m['base_date_lag_7_days'])

base = pd.merge(base, loan_repay_l3m, how='left', on=['business_id', 'days_past_due', 'base_date_lag_7_days'])
base = pd.merge(base, loan_repay_l1m, how='left', on=['business_id', 'days_past_due', 'base_date_lag_7_days'])

base['ratio_loan_repay_l1m_l3m'] = base['loan_repayment_l1m'] / base['loan_repayment_l3m']

base['ratio_loan_repay_l1m_l3m_normalised'] = base['loan_repayment_l1m'] / (base['loan_repayment_l3m'] / 3)

# recurring credit

recurring_credit_l1m = q("""SELECT 
                t1.business_id, 
                t1.days_past_due,
                t1.base_date_lag_7_days,
                SUM(ABS(t2.amount)) AS recurring_credit_l1m
            FROM 
                PROD_DB.ADHOC.SELF_CURE_MODEL_TRAINING_BASE4 t1
            LEFT JOIN 
                PROD_DB.ADHOC.self_cure_model_txn_tce_base2 t2
                ON t1.business_id = t2.business_id
                AND t2.type = 'credit' and t2.tce_v1_category in ('Other Recurring', 'Investment_earnings', 'Rental_income', 'Payroll', 'Payroll & Benefits', 'Sales_revenue', 'Service_revenue')
                AND date(t2.transaction_date) >= DATEADD(DAY, -30, date(t1.base_date_lag_7_days)) and date(t2.transaction_date) <= date( t1.base_date_lag_7_days)
            GROUP BY 
                t1.base_date_lag_7_days, t1.business_id, t1.days_past_due
            order by business_id
                                """)

recurring_credit_l3m = q("""SELECT 
                t1.business_id, 
                t1.days_past_due,
                t1.base_date_lag_7_days,
                SUM(ABS(t2.amount)) AS recurring_credit_l3m
            FROM 
                PROD_DB.ADHOC.SELF_CURE_MODEL_TRAINING_BASE4 t1
            LEFT JOIN 
                PROD_DB.ADHOC.self_cure_model_txn_tce_base2 t2
                ON t1.business_id = t2.business_id
                AND t2.type = 'credit' and t2.tce_v1_category in ('Other Recurring', 'Investment_earnings', 'Rental_income', 'Payroll', 'Payroll & Benefits', 'Sales_revenue', 'Service_revenue')
                AND date(t2.transaction_date) >= DATEADD(DAY, -90, date(t1.base_date_lag_7_days)) and date(t2.transaction_date) <= date( t1.base_date_lag_7_days)
            GROUP BY 
                t1.base_date_lag_7_days, t1.business_id, t1.days_past_due
            order by business_id
                                """)
                                
                                
recurring_credit_l1m['base_date_lag_7_days'] = pd.to_datetime(recurring_credit_l1m['base_date_lag_7_days'])
recurring_credit_l3m['base_date_lag_7_days'] = pd.to_datetime(recurring_credit_l3m['base_date_lag_7_days'])

base = pd.merge(base, recurring_credit_l1m, how='left', on=['business_id', 'days_past_due', 'base_date_lag_7_days'])
base = pd.merge(base, recurring_credit_l3m, how='left', on=['business_id', 'days_past_due', 'base_date_lag_7_days'])

base['ratio_recurring_credit_l1m_l3m'] = base['recurring_credit_l1m'] / base['recurring_credit_l3m']

base['ratio_recurring_credit_l1m_l3m_normalised'] = base['recurring_credit_l1m'] / (base['recurring_credit_l3m']/3)



### App login variables

## step1: prepare login base

app_login1 = q(""" with segment_id as (
                select distinct  a.business_id as business_id, b.user_id, b.segment_user_id , 1 as confirmed_ato, coalesce(DATE("ATO_Concluded_Date"),date('2099-12-31')) as ato_date
                            , rank() over (partition by a.business_id order by  "ATO_Concluded_Date" desc ) as rnk
                from PROD_DB.ADHOC.ATO_SHEET a
                inner join FIVETRAN_DB.PROD_NOVO_API_PUBLIC.SEGMENT_USERS b 
                on a.business_id = b.business_id
                qualify rnk = 1
                union all
                select   b.business_id as business_id, b.user_id, b.segment_user_id , 0 as confirmed_ato, DATE('2099-12-31') as ato_date, 0 as rnk
                from FIVETRAN_DB.PROD_NOVO_API_PUBLIC.SEGMENT_USERS b
                where business_id not in (select business_id from PROD_DB.ADHOC.ATO_SHEET )
                )
                --select split(sdks_client_name,'-')[1]::string  from PROD_DB.ADHOC.CASTLE_LOGIN_EVENTS
                , successful_logins as (
                    select b.business_id
                          ,a.user_id as external_user_id
                          ,b.user_id
                          ,ato_date
                          ,created_at
                          ,CASE WHEN scores_account_abuse_score>0.6 
                                then 1 
                                else 0 
                           end as is_risky_session
                          ,case when device_software_languages ilike ('%en%') then 1 else 0 end as is_english_user
                          ,case when device_software_languages ilike ('%es%') then 1 else 0 end as is_spanish_user
                          ,case when device_software_languages ilike ('%af%') then 1 else 0 end as is_african_user
                          ,policy_action
                          ,split(a.sdks_client_name,'-')[1]::string as platform
                          ,device_hardware_type
                          ,email_disposable
                          ,signals_abuse_ip_triggered
                          ,SIGNALS_CARRIER_IP_COUNTRY_MISMATCH_TRIGGERED
                          ,SIGNALS_IMPOSSIBLE_TRAVEL_TRIGGERED
                          ,SIGNALS_MULTIPLE_ACCOUNTS_PER_DEVICE_TRIGGERED
                          ,SIGNALS_NEW_DEVICE_TRIGGERED
                          ,SIGNALS_TIMEZONE_AREA_MISMATCH_TRIGGERED
                          ,signals_credential_stuffing_triggered
                          ,signals_datacenter_ip_triggered
                          ,signals_new_country_triggered
                          ,signals_new_isp_triggered
                          ,signals_new_language_triggered
                          ,signals_proxy_ip_triggered
                          ,signals_web_crawler_triggered
                           
                  from  PROD_DB.ADHOC.CASTLE_LOGIN_EVENTS a
                inner join segment_id b on  a.user_id = b.segment_user_id 
                    where status ilike ('%succeeded%')
                
                )
                
                , sessions_temp as (
                    select *
                      ,created_at as session_start_time
                      ,LAG(created_at) over (partition by user_id order by created_at desc) as session_end_time_temp
                from successful_logins
                )
                
                , sessions as (
                select  *
                        , coalesce(session_end_time_temp, DATEADD(day,1,session_start_time)) as session_end_time
                from sessions_temp
                )
                select * from sessions
    
              """)

#
app_login2 = ("""with data as (
    select 
    meta:id::string as id,
    EXTERNAL_USER_ID, created_at,
    meta:type::string as event_type,
    meta:status::string as status,
    meta:device:timezone:name::string as device_timezone,
    meta:device:fingerprint::string as fingerprint,
    meta:policy:action::string as policy_action,
    meta:risk::float as risk_score,
    meta:scores:account_abuse:score::float as account_abuse_score,
    meta:scores:account_takeover:score::float as account_takeover_score,
    meta:scores:bot:score::float as bot_score,
    meta:ip:address::string as ip_address,
    meta:ip:isp:name::string as isp_name,
    case when meta:signals ilike ('%new_device%') THEN 1 ELSE 0 END as is_new_device,
    case when meta:signals ilike ('%new_isp%') THEN 1 ELSE 0 END as is_new_isp,
    case when meta:signals ilike ('%new_country%') THEN 1 ELSE 0 END as is_new_country,
    case when meta:signals ilike ('%carrier_ip_country_mismatch%') 
                           THEN 1 ELSE 0 END as is_carrier_ip_country_mismatch,
    case when meta:signals ilike ('%proxy_ip%') THEN 1 ELSE 0 END as is_proxy_ip,
    case when meta:signals ilike ('%multiple_accounts_per_device%') 
                           THEN 1 ELSE 0 END as is_multiple_accounts_per_device,
    case when meta:signals ilike ('%timezone_area_mismatch%') THEN 1 ELSE 0 END as is_timezone_area_mismatch,
    case when meta:signals ilike ('%impossible_travel%') THEN 1 ELSE 0 END as is_impossible_travel,
    case when meta:signals ilike ('%bot_behavior%') THEN 1 ELSE 0 END as is_bot_behavior,
    case when meta:signals ilike ('%Multiple SSNs per Device%') THEN 1 ELSE 0 END as is_Multiple_ssn_per_device,
    case when meta:signals ilike ('%abuse_ip%') THEN 1 ELSE 0 END as is_abuse_ip,
    case when meta:signals ilike ('%new_os%') THEN 1 ELSE 0 END as is_new_os,
    case when meta:signals ilike ('%disposable_email_domain%') THEN 1 ELSE 0 END as is_disposable_email_domain,
    case when meta:signals ilike ('%high_activity_device%') THEN 1 ELSE 0 END as is_high_activity_device,
    case when meta:signals ilike ('%tor_ip%') THEN 1 ELSE 0 END as is_tor_ip,
    case when meta:signals ilike ('%datacenter_ip%') THEN 1 ELSE 0 END as is_datacenter_ip
    
    from FIVETRAN_DB.PROD_NOVO_API_PUBLIC.CASTLE_API_RESPONSES
    where 
    --created_at < date(current_timestamp)
    --and created_at >= date(DATEADD('month',-3, current_timestamp))
    --and 
    meta:type = '$login'
    and meta:status = '$succeeded'

        )
        select * from data
              
              
              """)


#
login_base = q("""
                      WITH segment_id AS (
            SELECT DISTINCT 
                a.business_id AS business_id, 
                b.user_id, 
                b.segment_user_id, 
                1 AS confirmed_ato, 
                COALESCE(DATE(a."ATO_Concluded_Date"), DATE('2099-12-31')) AS ato_date,
                RANK() OVER (PARTITION BY a.business_id ORDER BY a."ATO_Concluded_Date" DESC) AS rnk
            FROM 
                PROD_DB.ADHOC.ATO_SHEET a
            INNER JOIN 
                FIVETRAN_DB.PROD_NOVO_API_PUBLIC.SEGMENT_USERS b 
                ON a.business_id = b.business_id
            QUALIFY 
                rnk = 1
            UNION ALL
            SELECT   
                b.business_id AS business_id, 
                b.user_id, 
                b.segment_user_id, 
                0 AS confirmed_ato, 
                DATE('2099-12-31') AS ato_date, 
                0 AS rnk
            FROM 
                FIVETRAN_DB.PROD_NOVO_API_PUBLIC.SEGMENT_USERS b
            WHERE 
                b.business_id NOT IN (SELECT business_id FROM PROD_DB.ADHOC.ATO_SHEET)
        ),
        successful_logins AS (
            SELECT 
                b.business_id,
                a.user_id AS external_user_id,
                b.user_id,
                ato_date,
                a.created_at,
                CASE WHEN a.scores_account_abuse_score > 0.6 THEN 1 ELSE 0 END AS is_risky_session,
                CASE WHEN a.device_software_languages ILIKE '%en%' THEN 1 ELSE 0 END AS is_english_user,
                CASE WHEN a.device_software_languages ILIKE '%es%' THEN 1 ELSE 0 END AS is_spanish_user,
                CASE WHEN a.device_software_languages ILIKE '%af%' THEN 1 ELSE 0 END AS is_african_user,
                a.policy_action,
                SPLIT_PART(a.sdks_client_name, '-', 2) AS platform,
                a.device_hardware_type,
                a.email_disposable,
                a.signals_abuse_ip_triggered,
                a.SIGNALS_CARRIER_IP_COUNTRY_MISMATCH_TRIGGERED,
                a.SIGNALS_IMPOSSIBLE_TRAVEL_TRIGGERED,
                a.SIGNALS_MULTIPLE_ACCOUNTS_PER_DEVICE_TRIGGERED,
                a.SIGNALS_NEW_DEVICE_TRIGGERED,
                a.SIGNALS_TIMEZONE_AREA_MISMATCH_TRIGGERED,
                a.signals_credential_stuffing_triggered,
                a.signals_datacenter_ip_triggered,
                a.signals_new_country_triggered,
                a.signals_new_isp_triggered,
                a.signals_new_language_triggered,
                a.signals_proxy_ip_triggered,
                a.signals_web_crawler_triggered
            FROM  
                PROD_DB.ADHOC.CASTLE_LOGIN_EVENTS a
            INNER JOIN 
                segment_id b 
                ON a.user_id = b.segment_user_id 
            WHERE 
                a.status ILIKE '%succeeded%'
        ),
        sessions_temp AS (
            SELECT 
                *,
                created_at AS session_start_time,
                LAG(created_at) OVER (PARTITION BY user_id ORDER BY created_at DESC) AS session_end_time_temp
            FROM 
                successful_logins
        ),
        sessions AS (
            SELECT  
                *,
                COALESCE(session_end_time_temp, DATEADD(day, 1, session_start_time)) AS session_end_time
            FROM 
                sessions_temp
        ),
        base1 AS (
            SELECT 
                business_id, 
                external_user_id, 
                created_at
            FROM 
                sessions
            WHERE 
                DATE(created_at) <= DATE('2023-10-26')
        ),
        data AS (
            SELECT 
                meta:id::STRING AS id,
                EXTERNAL_USER_ID, 
                created_at,
                meta:type::STRING AS event_type,
                meta:status::STRING AS status,
                meta:device:timezone:name::STRING AS device_timezone,
                meta:device:fingerprint::STRING AS fingerprint,
                meta:policy:action::STRING AS policy_action,
                meta:risk::FLOAT AS risk_score,
                meta:scores:account_abuse:score::FLOAT AS account_abuse_score,
                meta:scores:account_takeover:score::FLOAT AS account_takeover_score,
                meta:scores:bot:score::FLOAT AS bot_score,
                meta:ip:address::STRING AS ip_address,
                meta:ip:isp:name::STRING AS isp_name,
                CASE WHEN meta:signals ILIKE '%new_device%' THEN 1 ELSE 0 END AS is_new_device,
                CASE WHEN meta:signals ILIKE '%new_isp%' THEN 1 ELSE 0 END AS is_new_isp,
                CASE WHEN meta:signals ILIKE '%new_country%' THEN 1 ELSE 0 END AS is_new_country,
                CASE WHEN meta:signals ILIKE '%carrier_ip_country_mismatch%' THEN 1 ELSE 0 END AS is_carrier_ip_country_mismatch,
                CASE WHEN meta:signals ILIKE '%proxy_ip%' THEN 1 ELSE 0 END AS is_proxy_ip,
                CASE WHEN meta:signals ILIKE '%multiple_accounts_per_device%' THEN 1 ELSE 0 END AS is_multiple_accounts_per_device,
                CASE WHEN meta:signals ILIKE '%timezone_area_mismatch%' THEN 1 ELSE 0 END AS is_timezone_area_mismatch,
                CASE WHEN meta:signals ILIKE '%impossible_travel%' THEN 1 ELSE 0 END AS is_impossible_travel,
                CASE WHEN meta:signals ILIKE '%bot_behavior%' THEN 1 ELSE 0 END AS is_bot_behavior,
                CASE WHEN meta:signals ILIKE '%Multiple SSNs per Device%' THEN 1 ELSE 0 END AS is_Multiple_ssn_per_device,
                CASE WHEN meta:signals ILIKE '%abuse_ip%' THEN 1 ELSE 0 END AS is_abuse_ip,
                CASE WHEN meta:signals ILIKE '%new_os%' THEN 1 ELSE 0 END AS is_new_os,
                CASE WHEN meta:signals ILIKE '%disposable_email_domain%' THEN 1 ELSE 0 END AS is_disposable_email_domain,
                CASE WHEN meta:signals ILIKE '%high_activity_device%' THEN 1 ELSE 0 END AS is_high_activity_device,
                CASE WHEN meta:signals ILIKE '%tor_ip%' THEN 1 ELSE 0 END AS is_tor_ip,
                CASE WHEN meta:signals ILIKE '%datacenter_ip%' THEN 1 ELSE 0 END AS is_datacenter_ip
            FROM 
                FIVETRAN_DB.PROD_NOVO_API_PUBLIC.CASTLE_API_RESPONSES
            WHERE 
                meta:type = '$login' AND meta:status = '$succeeded'
        ),
        base2 AS (
            SELECT 
                su.business_id, 
                data.external_user_id, 
                data.created_at
            FROM 
                data 
            LEFT JOIN 
                FIVETRAN_DB.PROD_NOVO_API_PUBLIC.SEGMENT_USERS su
                ON data.EXTERNAL_USER_ID = su.segment_user_id
        )
        select *
        from (
        select * from base1
        union 
        select * from base2)
        
        """)


## write login base

from sqlalchemy.types import NVARCHAR
 # from conf.config import SQLQuery
q = SQLQuery('snowflake')

login_base.to_sql(name='self_cure_model_app_login_base',
                  con=q.engine, 
                  schema='prod_db.adhoc',
                  if_exists='append', 
                  index=False, 
                  chunksize=8000, 
                  method='multi',
                  dtype={col_name: NVARCHAR for col_name in login_base}) 



### app login variables

logins_last_7d = q("""
                   SELECT 
                        t1.business_id, 
                        t1.days_past_due,
                        date(t1.base_date_lag_7_days) as base_date_lag_7_days,
                        count(t2.business_id) AS count_login_l7d
                    FROM 
                        PROD_DB.ADHOC.SELF_CURE_MODEL_TRAINING_BASE4 t1
                    LEFT JOIN 
                        PROD_DB.ADHOC.self_cure_model_app_login_base t2
                        ON t1.business_id = t2.business_id
                        AND date(t2.created_at) >= DATEADD(DAY, -7, date(t1.base_date_lag_7_days)) and date(t2.created_at) <= date( t1.base_date_lag_7_days)
                    GROUP BY 
                        t1.base_date_lag_7_days, t1.business_id, t1.days_past_due
                    order by business_id
                       
                   
                   """)
     
logins_last_7d['base_date_lag_7_days'] = pd.to_datetime(logins_last_7d['base_date_lag_7_days'])
                   
base = pd.merge(base, logins_last_7d, how='left', on=['business_id', 'days_past_due', 'base_date_lag_7_days'])



logins_last_15d = q("""
                   SELECT 
                        t1.business_id, 
                        t1.days_past_due,
                        date(t1.base_date_lag_7_days) as base_date_lag_7_days,
                        count(t2.business_id) AS count_login_l15d
                    FROM 
                        PROD_DB.ADHOC.SELF_CURE_MODEL_TRAINING_BASE4 t1
                    LEFT JOIN 
                        PROD_DB.ADHOC.self_cure_model_app_login_base t2
                        ON t1.business_id = t2.business_id
                        AND date(t2.created_at) >= DATEADD(DAY, -15, date(t1.base_date_lag_7_days)) and date(t2.created_at) <= date( t1.base_date_lag_7_days)
                    GROUP BY 
                        t1.base_date_lag_7_days, t1.business_id, t1.days_past_due
                    order by business_id
                       
                   
                   """)
                   
logins_last_15d['base_date_lag_7_days'] = pd.to_datetime(logins_last_15d['base_date_lag_7_days'])
                   
base = pd.merge(base, logins_last_15d, how='left', on=['business_id', 'days_past_due', 'base_date_lag_7_days'])


logins_last_30d = q("""
                   SELECT 
                        t1.business_id, 
                        t1.days_past_due,
                        date(t1.base_date_lag_7_days) as base_date_lag_7_days,
                        count(t2.business_id) AS count_login_l30d
                    FROM 
                        PROD_DB.ADHOC.SELF_CURE_MODEL_TRAINING_BASE4 t1
                    LEFT JOIN 
                        PROD_DB.ADHOC.self_cure_model_app_login_base t2
                        ON t1.business_id = t2.business_id
                        AND date(t2.created_at) >= DATEADD(DAY, -30, date(t1.base_date_lag_7_days)) and date(t2.created_at) <= date( t1.base_date_lag_7_days)
                    GROUP BY 
                        t1.base_date_lag_7_days, t1.business_id, t1.days_past_due
                    order by business_id
                       
                   
                   """)


logins_last_30d['base_date_lag_7_days'] = pd.to_datetime(logins_last_30d['base_date_lag_7_days'])

                   
base = pd.merge(base, logins_last_30d, how='left', on=['business_id', 'days_past_due', 'base_date_lag_7_days'])


## 

base['ratio_login_l7d_l15d'] = base['count_login_l7d'] / base['count_login_l15d']

base['ratio_login_l7d_l30d'] = base['count_login_l7d'] / base['count_login_l30d']

base['ratio_login_l15d_l30d'] = base['count_login_l15d'] / base['count_login_l30d']


base['ratio_login_l7d_l15d_normalised'] = (base['count_login_l7d'] / 7) / (base['count_login_l15d']/15)

base['ratio_login_l7d_l30d_normalised'] = (base['count_login_l7d']/7) / (base['count_login_l30d']/30)

base['ratio_login_l15d_l30d_normalised'] = (base['count_login_l15d']/15) / (base['count_login_l30d']/30)


### MODEL PIPELINE ######


from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, classification_report
from scipy.stats import ks_2samp
# from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
# Boruta Feature Selection using Random forest
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
import shap



def calculate_iv(data, target, bins=10):
    iv_list = []  # Initialize an empty list to store the IV results
    
    for column in data.columns:
        # Combine the feature column and target into a single DataFrame to handle missing values
        combined = pd.concat([data[column], target], axis=1)
        combined.columns = ['feature', 'target']
        
        # Drop rows with missing values
        combined.dropna(inplace=True)
        
        # Split combined DataFrame back into feature and target
        feature_col = combined['feature']
        target_col = combined['target']
        
        # Bin the feature column if it's numeric, otherwise use it as is
        if np.issubdtype(feature_col.dtype, np.number):
            # pd.qcut might still raise an error if there are too few unique values, handle this
            try:
                binned_x = pd.qcut(feature_col, bins, duplicates='drop')
            except ValueError:
                # Fallback to pd.cut if pd.qcut fails
                binned_x = pd.cut(feature_col, bins, duplicates='drop')
        else:
            binned_x = feature_col
        
        # Create a DataFrame with binned values and target
        df = pd.DataFrame({'x': binned_x, 'y': target_col})
        
        # Group by the binned values and calculate counts and sums
        iv_table = df.groupby('x')['y'].agg(['count', 'sum'])
        
        # Ensure 'count' and 'sum' columns are numeric
        iv_table['count'] = pd.to_numeric(iv_table['count'], errors='coerce')
        iv_table['sum'] = pd.to_numeric(iv_table['sum'], errors='coerce')
        
        # Calculate non-event counts
        iv_table['non_event'] = iv_table['count'] - iv_table['sum']
        
        # Calculate event rate and non-event rate
        iv_table['event_rate'] = iv_table['sum'] / iv_table['sum'].sum()
        iv_table['non_event_rate'] = iv_table['non_event'] / iv_table['non_event'].sum()
        
        # Calculate WoE (Weight of Evidence)
        iv_table['woe'] = np.log(iv_table['event_rate'] / iv_table['non_event_rate'])
        
        # Calculate IV (Information Value)
        iv_table['iv'] = (iv_table['event_rate'] - iv_table['non_event_rate']) * iv_table['woe']
        iv = iv_table['iv'].sum()
        
        # Append the IV value for the current feature to the iv_list
        iv_list.append({'feature': column, 'iv': iv})
    
    # Convert the list of dictionaries to a DataFrame
    iv_df = pd.DataFrame(iv_list)
    
    return iv_df




def remove_correlated_features(data, threshold=0.9):
    corr_matrix = data.corr()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return data.drop(columns=to_drop), to_drop

## encoding categorical variables

business_type_mapping = {'sole_proprietorship': 1, 'partnership': 2, 'llc':3, 'corporation': 4}

base['business_type_mapped'] = base['business_type'].map(business_type_mapping)


# state

from sklearn.preprocessing import LabelEncoder

# Step 1: Import LabelEncoder
label_encoder = LabelEncoder()

base['state'].replace(to_replace=0, value=100, inplace=True)

# Step 2: Fit and Transform the data
base['state_encoded'] = label_encoder.fit_transform(base['state'])

# Demonstrate the inverse_transform method
encoded_labels = base['state_encoded'].unique()
original_labels = label_encoder.inverse_transform(encoded_labels)

print("\nEncoded labels to original labels using inverse_transform:")
for encoded, original in zip(encoded_labels, original_labels):
    print(f"{encoded} -> {original}")


# Split data
X = base.drop(columns=['updated_at', 'target_15d', 'self_cured_in_30_days', 'business_id', 'lending_business_id', 'canopy_id',
       'external_account_id', 'application_type', 'due_date', 'due_date_month_year',
       'days_past_due', 'delinquency_bucket', 'cure_payment_cents',
       '_fivetran_synced', 'flag1', 'is_cured', 
       'repayment_or_last_synced_date', 
       'cured_with_contact_in_7_days', 'credit_limit',
       'state', 'business_type',  'first_draw_date', 'row_num_x','row_num_y', 
       'account_funded_date', 'business_created_date', 'base_date_lag_7_days',
       'start_date',
       'end_date', 'successful_contacts_last_7_days', 'start_date2',
       'end_date2', 'successful_contacts_last_15_days', 'start_date3',
       'end_date3', 'successful_contacts_last_30_days',
       'successful_contacts_last_7_days_flag',
       'successful_contacts_last_15_days_flag',
       'successful_contacts_last_30_days_flag', 'cured_with_contact_in_7_days',
       'cured_with_contact_in_15_days', 'cured_with_contact_in_30_days',
       'self_cured_in_7_days', 'self_cured_in_15_days',
        'self_cured_in_30_days', 'row_num_x', 'target_15d',
       'target_30d', '_fivetran_synced_lag_7days', 'target_7d', 'fico_score', 'number_of_60_days_delinquencies',
       'number_of_inquiries_last_6_months', 'number_of_open_trades', 'number_of_trades_with_derogatory_status'])


y = base['target_7d'].astype(int)


X.replace([np.inf, -np.inf], np.nan, inplace=True)

X.fillna(0, inplace=True)


def model_pipeline(X, y, seed, corr_thresh, iv_thresh):
    # Drop correlated features
    X, dropped_correlated_features = remove_correlated_features(X, threshold=corr_thresh)
    # Calculate IV and remove features with low IV
    iv_df = calculate_iv(X, y, bins=10)
    low_iv_features = iv_df[iv_df['iv'] < iv_thresh]['feature']
    X = X.drop(columns=low_iv_features)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    # Handle missing values if any
    X_train.fillna(0, inplace=True)
    X_test.fillna(0, inplace=True)

    model_rf = RandomForestClassifier(n_jobs=-1, max_depth=5, random_state=seed)
    boruta_selector = BorutaPy(model_rf, n_estimators='auto', random_state=seed, verbose=2)
    boruta_selector.fit(X_train.values, y_train.values)

    selected_features = X_train.columns[boruta_selector.support_]
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]

    # Initialize CatBoost
    model = CatBoostClassifier(cat_features=[], verbose=100)

    # Define hyperparameter grid
    param_grid = {
            'iterations': [50, 80],
            'learning_rate': [0.01, 0.05, 0.06],
            'depth': [5, 6, 8],
            'l2_leaf_reg': [10, 20]
        }

    # Custom evaluation function to minimize delta and maximize test AUC
    def custom_evaluation(estimator, X, y):
        estimator.fit(X_train, y_train)
        train_auc = roc_auc_score(y_train, estimator.predict_proba(X_train)[:, 1])
        test_auc = roc_auc_score(y_test, estimator.predict_proba(X_test)[:, 1])
        delta_auc = abs(train_auc - test_auc)
        return test_auc - delta_auc  # prioritize higher test_auc and lower delta_auc

    # Grid search with custom scoring
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=custom_evaluation, cv=3)
    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_

    return best_model, selected_features, X_train, X_test, y_train, y_test


corr_thresh = 0.7
iv_thresh = 0.01
best_model, selected_features, X_train, X_test, y_train, y_test = model_pipeline(X, y, 3, corr_thresh, iv_thresh)



best_model.get_params()


selected_features

len(selected_features)


# Predict probabilities
train_probs = best_model.predict_proba(X_train)[:, 1]
test_probs = best_model.predict_proba(X_test)[:, 1]

# Calculate AUC
train_auc = roc_auc_score(y_train, train_probs)
test_auc = roc_auc_score(y_test, test_probs)

print(f'Train AUC: {train_auc}')
print(f'Test AUC: {test_auc}')


# Predict probabilities
y_probs = best_model.predict_proba(X_test)[:, 1]
y_probs_train = best_model.predict_proba(X_train)[:, 1]


# Predict probabilities for training data
y_train_probs = best_model.predict_proba(X_train)[:, 1]

# Calculate AUC for training and test data
auc_train = roc_auc_score(y_train, y_train_probs)
auc_test = roc_auc_score(y_test, y_probs)
print(f'Train AUC: {auc_train}')
print(f'Test AUC: {auc_test}')

# Calculate KS statistic for training data
fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_train_probs)
ks_train = max(tpr_train - fpr_train)
print(f'Train KS Statistic: {ks_train}')

# Calculate KS statistic for test data
fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_probs)
ks_test = max(tpr_test - fpr_test)
print(f'Test KS Statistic: {ks_test}')


# SHAP Values
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, plot_type="bar")
shap.summary_plot(shap_values, X_test)



## Model Evaluation - KS & ROC AUC

def ks(target=None, prob=None):
    data = pd.DataFrame()
    data['y'] = target
    data['y'] = data['y'].astype(float)
    data['p'] = prob
    data['y0'] = 1- data['y']
   
    data['bucket'] = pd.qcut(data['p'], 10)
    grouped = data.groupby('bucket', as_index=False)
    kstable = pd.DataFrame()
    kstable['min_prob'] = grouped.min()['p']
    kstable['max_prob'] = grouped.max()['p']
    kstable['events'] = grouped.sum()['y']
    kstable['nonevents'] = grouped.sum()['y0']
    kstable['pop'] = kstable['events']+kstable['nonevents']
    
    kstable = kstable.sort_values(by='min_prob', ascending=False).reset_index(drop=True)
    kstable['event_rate'] = (kstable.events / data['y'].sum()).apply('{0:.2%}'.format)
    kstable['nonevent_rate'] = (kstable['nonevents'] /  data['y0'].sum()).apply('{0:2%}'.format)
    kstable['cum_eventrate'] = (kstable.events / data['y'].sum()).cumsum()
    kstable['cum_noneventrate'] = (kstable.nonevents / data['y0'].sum()).cumsum()
    kstable['KS'] = np.round(kstable['cum_eventrate'] - kstable['cum_noneventrate'], 3) * 100
    kstable['cure_rate'] = (kstable['events'] / (kstable['events'] + kstable['nonevents']))
    
    # formatting
    kstable['cum_eventrate'] = kstable['cum_eventrate'].apply('{0:.2%}'.format)
    kstable['cum_noneventrate'] = kstable['cum_noneventrate'].apply('{0:.2%}'.format)
    kstable.index = range(1,11)
    kstable.index.rename('Decile', inplace=True)
    
    # Display KS
    print("KS is " + str(max(kstable['KS']))+"%"+ " at decile " + str((kstable.index[kstable['KS']==max(kstable['KS'])][0])))
    return kstable



y_probs = best_model.predict_proba(X_test)[:, 1]
test_ks = ks(y_test, y_probs)

y_probs_train = best_model.predict_proba(X_train)[:, 1]
train_ks = ks(y_train, y_probs_train)

y_probs_overall = best_model.predict_proba(X)[:, 1]
ks_overall = ks(y, y_probs_overall)

# Calculate AUC for training and test data
auc_overall = roc_auc_score(y, y_probs_overall)
print(f'Overall AUC: {auc_overall}')

base['pred_prob'] = y_probs_overall

# Define the conditions
conditions = [
    base['pred_prob'] <=0.097,
    (base['pred_prob'] > 0.097) & (base['pred_prob'] <= 0.133),
    (base['pred_prob'] > 0.133) & (base['pred_prob'] <= 0.173),
    (base['pred_prob'] > 0.173) & (base['pred_prob'] <= 0.211),
    (base['pred_prob'] > 0.211) & (base['pred_prob'] <= 0.250),
    (base['pred_prob'] > 0.250) & (base['pred_prob'] <= 0.296),
    (base['pred_prob'] > 0.296) & (base['pred_prob'] <= 0.344),
    (base['pred_prob'] > 0.344) & (base['pred_prob'] <= 0.403),
    (base['pred_prob'] > 0.403) & (base['pred_prob'] <= 0.480),
    base['pred_prob'] > 0.480
]

# Define the choices corresponding to the conditions
choices = ['d10', 'd9', 'd8', 'd7', 'd6', 'd5', 'd4', 'd3', 'd2', 'd1']

# Apply np.select to create a new column with the categories
base['decile'] = np.select(conditions, choices)

# Retrieve feature importances
feature_importances = best_model.get_feature_importance(Pool(X_train, label=y_train))
feature_names = X_train.columns

# Create a DataFrame for better visualization
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})

# Sort by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

# Display feature importances
print(importance_df)

iv_df = calculate_iv(X, y, bins=10)



##### seeds

def cbm_pipeline(X, y, seeds, corr_thresh, iv_thresh):
    results_list = []

    for seed in seeds:
        # Drop correlated features
        X, dropped_correlated_features = remove_correlated_features(X, threshold=corr_thresh)
        # Calculate IV and remove features with low IV
        iv_df = calculate_iv(X, y, bins=10)
        low_iv_features = iv_df[iv_df['iv'] < iv_thresh]['feature']
        X = X.drop(columns=low_iv_features)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

        # Handle missing values if any
        X_train.fillna(0, inplace=True)
        X_test.fillna(0, inplace=True)

        model_rf = RandomForestClassifier(n_jobs=-1, max_depth=5, random_state=seed)
        boruta_selector = BorutaPy(model_rf, n_estimators='auto', random_state=seed, verbose=2)
        boruta_selector.fit(X_train.values, y_train.values)

        selected_features = X_train.columns[boruta_selector.support_]
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]

        # Initialize CatBoost
        model = CatBoostClassifier(cat_features=[], verbose=100)

        # Define hyperparameter grid
        param_grid = {
            'iterations': [50, 100],
            'learning_rate': [0.01, 0.05],
            'depth': [5, 6, 8],
            'l2_leaf_reg': [10, 20]
        }

        # Custom evaluation function to minimize delta and maximize test AUC
        def custom_evaluation(estimator, X, y):
            estimator.fit(X_train, y_train)
            train_auc = roc_auc_score(y_train, estimator.predict_proba(X_train)[:, 1])
            test_auc = roc_auc_score(y_test, estimator.predict_proba(X_test)[:, 1])
            delta_auc = abs(train_auc - test_auc)
            return test_auc - delta_auc  # prioritize higher test_auc and lower delta_auc

        # Grid search with custom scoring
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=custom_evaluation, cv=3)
        grid_search.fit(X_train, y_train)

        # Best model
        best_model = grid_search.best_estimator_

        # Evaluate the best model
        train_auc_best = roc_auc_score(y_train, best_model.predict_proba(X_train)[:, 1])
        test_auc_best = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
        delta_auc_best = abs(train_auc_best - test_auc_best)

        # Append results to the list
        results_list.append({
            'seed': seed,
            'train_auc_best': train_auc_best,
            'test_auc_best': test_auc_best,
            'delta_auc_best': delta_auc_best,
            'features': selected_features
        })

    # Convert the list of results to a DataFrame
    results_df = pd.DataFrame(results_list)
    return results_df

seeds = list(range(10))
results_df = cbm_pipeline(X, y, seeds, corr_thresh, iv_thresh)


results_df['count'] = results_df['features'].apply(lambda x: len(x))

master_dict = {}
def create_dict(x):
    for i in range(len(x)):
        if x[i] in master_dict:
            master_dict[x[i]] += 1
        else:
            master_dict[x[i]] = 1
    return master_dict



results_df['features'].apply(lambda x: create_dict(x))


feat_df = pd.DataFrame(master_dict, index=[0])
# feat_df.columns = ['feature_name', 'count']
feat_df




#################### Segmented models #################################


base1 = base[base['cure_payment_cents'] <= 100000]

base2 = base[base['cure_payment_cents'] > 100000]

#####

# Split data
X1 = base1.drop(columns=['updated_at', 'target_15d', 'self_cured_in_30_days', 'business_id', 'lending_business_id', 'canopy_id',
       'external_account_id', 'application_type', 'due_date', 'due_date_month_year',
       'days_past_due', 'delinquency_bucket', 'cure_payment_cents',
       '_fivetran_synced', 'flag1', 'is_cured', 
       'repayment_or_last_synced_date', 
       'cured_with_contact_in_7_days', 'credit_limit',
       'state', 'business_type',  'first_draw_date', 'row_num_x','row_num_y', 
       'account_funded_date', 'business_created_date', 'base_date_lag_7_days',
       'start_date',
       'end_date', 'successful_contacts_last_7_days', 'start_date2',
       'end_date2', 'successful_contacts_last_15_days', 'start_date3',
       'end_date3', 'successful_contacts_last_30_days',
       'successful_contacts_last_7_days_flag',
       'successful_contacts_last_15_days_flag',
       'successful_contacts_last_30_days_flag', 'cured_with_contact_in_7_days',
       'cured_with_contact_in_15_days', 'cured_with_contact_in_30_days',
       'self_cured_in_7_days', 'self_cured_in_15_days',
        'self_cured_in_30_days', 'row_num_x', 'target_15d',
       'target_30d', '_fivetran_synced_lag_7days', 'target_7d', 'fico_score', 'number_of_60_days_delinquencies',
       'number_of_inquiries_last_6_months', 'number_of_open_trades', 'number_of_trades_with_derogatory_status'])


y1= base1['target_7d'].astype(int)


X1.replace([np.inf, -np.inf], np.nan, inplace=True)

X1.fillna(0, inplace=True)



corr_thresh = 0.7
iv_thresh = 0.01
best_model1, selected_features1, X_train1, X_test1, y_train1, y_test1 = model_pipeline(X1, y1, 2, corr_thresh, iv_thresh)



best_model1.get_params()


selected_features1

len(selected_features1)


iv_df1 = calculate_iv(X1, y1, bins=10)


# Predict probabilities
train_probs1 = best_model1.predict_proba(X_train1)[:, 1]
test_probs1 = best_model1.predict_proba(X_test1)[:, 1]

# Calculate AUC
train_auc1 = roc_auc_score(y_train1, train_probs1)
test_auc1 = roc_auc_score(y_test1, test_probs1)

print(f'Train AUC: {train_auc1}')
print(f'Test AUC: {test_auc1}')


# Predict probabilities
y_probs1 = best_model1.predict_proba(X_test1)[:, 1]
y_probs_train1 = best_model1.predict_proba(X_train1)[:, 1]


# Predict probabilities for training data
y_train_probs1 = best_model1.predict_proba(X_train1)[:, 1]

# Calculate AUC for training and test data
auc_train1 = roc_auc_score(y_train1, y_train_probs1)
auc_test1 = roc_auc_score(y_test1, y_probs1)
print(f'Train AUC: {auc_train1}')
print(f'Test AUC: {auc_test1}')

# Calculate KS statistic for training data
fpr_train, tpr_train, thresholds_train = roc_curve(y_train1, y_train_probs1)
ks_train1 = max(tpr_train - fpr_train)
print(f'Train KS Statistic: {ks_train1}')

# Calculate KS statistic for test data
fpr_test, tpr_test, thresholds_test = roc_curve(y_test1, y_probs1)
ks_test1 = max(tpr_test - fpr_test)
print(f'Test KS Statistic: {ks_test1}')

y_probs1 = best_model1.predict_proba(X_test1)[:, 1]
test_ks1 = ks(y_test1, y_probs1)

y_probs_train1 = best_model1.predict_proba(X_train1)[:, 1]
train_ks1 = ks(y_train1, y_probs_train1)

y_probs_overall1 = best_model1.predict_proba(X1)[:, 1]
ks_overall1 = ks(y1, y_probs_overall1)

base1['pred_prob'] = y_probs_overall1

auc_overall1 = roc_auc_score(y1, y_probs_overall1)
print(f'Overall AUC: {auc_overall1}')

# Define the conditions
conditions = [
    base1['pred_prob'] <=0.124,
    (base1['pred_prob'] > 0.124) & (base1['pred_prob'] <= 0.170),
    (base1['pred_prob'] > 0.170) & (base1['pred_prob'] <= 0.215),
    (base1['pred_prob'] > 0.215) & (base1['pred_prob'] <= 0.257),
    (base1['pred_prob'] > 0.257) & (base1['pred_prob'] <= 0.309),
    (base1['pred_prob'] > 0.309) & (base1['pred_prob'] <= 0.362),
    (base1['pred_prob'] > 0.362) & (base1['pred_prob'] <= 0.428),
    (base1['pred_prob'] > 0.428) & (base1['pred_prob'] <= 0.498),
    (base1['pred_prob'] > 0.498) & (base1['pred_prob'] <= 0.600),
    base1['pred_prob'] > 0.600
]

# Define the choices corresponding to the conditions
choices = ['d10', 'd9', 'd8', 'd7', 'd6', 'd5', 'd4', 'd3', 'd2', 'd1']

# Apply np.select to create a new column with the categories
base1['decile'] = np.select(conditions, choices)


# Retrieve feature importances
feature_importances1 = best_model1.get_feature_importance(Pool(X_train1, label=y_train1))
feature_names1 = X_train1.columns

# Create a DataFrame for better visualization
importance_df1 = pd.DataFrame({'Feature': feature_names1, 'Importance': feature_importances1})

# Sort by importance
importance_df1 = importance_df1.sort_values(by='Importance', ascending=False).reset_index(drop=True)

# Display feature importances
print(importance_df1)

# SHAP Values
explainer = shap.TreeExplainer(best_model1)
shap_values = explainer.shap_values(X_test1)

shap.summary_plot(shap_values, X_test1, plot_type="bar")
shap.summary_plot(shap_values, X_test1)





###### segment model 2 ##################


# Split data
X2 = base2.drop(columns=['updated_at', 'target_15d', 'self_cured_in_30_days', 'business_id', 'lending_business_id', 'canopy_id',
       'external_account_id', 'application_type', 'due_date', 'due_date_month_year',
       'days_past_due', 'delinquency_bucket', 'cure_payment_cents',
       '_fivetran_synced', 'flag1', 'is_cured', 
       'repayment_or_last_synced_date', 
       'cured_with_contact_in_7_days', 'credit_limit',
       'state', 'business_type',  'first_draw_date', 'row_num_x','row_num_y', 
       'account_funded_date', 'business_created_date', 'base_date_lag_7_days',
       'start_date',
       'end_date', 'successful_contacts_last_7_days', 'start_date2',
       'end_date2', 'successful_contacts_last_15_days', 'start_date3',
       'end_date3', 'successful_contacts_last_30_days',
       'successful_contacts_last_7_days_flag',
       'successful_contacts_last_15_days_flag',
       'successful_contacts_last_30_days_flag', 'cured_with_contact_in_7_days',
       'cured_with_contact_in_15_days', 'cured_with_contact_in_30_days',
       'self_cured_in_7_days', 'self_cured_in_15_days',
        'self_cured_in_30_days', 'row_num_x', 'target_15d',
       'target_30d', '_fivetran_synced_lag_7days', 'target_7d', 'fico_score', 'number_of_60_days_delinquencies',
       'number_of_inquiries_last_6_months', 'number_of_open_trades', 'number_of_trades_with_derogatory_status'])


y2= base2['target_7d'].astype(int)


X2.replace([np.inf, -np.inf], np.nan, inplace=True)

X2.fillna(0, inplace=True)



corr_thresh = 0.7
iv_thresh = 0.01
best_model2, selected_features2, X_train2, X_test2, y_train2, y_test2 = model_pipeline(X2, y2, 3, corr_thresh, iv_thresh)



best_model2.get_params()


selected_features2

len(selected_features2)

iv_df2 = calculate_iv(X2, y2, bins=10)


# Predict probabilities
train_probs2 = best_model2.predict_proba(X_train2)[:, 1]
test_probs2 = best_model2.predict_proba(X_test2)[:, 1]

# Calculate AUC
train_auc2 = roc_auc_score(y_train2, train_probs2)
test_auc2 = roc_auc_score(y_test2, test_probs2)

print(f'Train AUC: {train_auc2}')
print(f'Test AUC: {test_auc2}')


# Predict probabilities
y_probs2 = best_model2.predict_proba(X_test2)[:, 1]
y_probs_train2 = best_model2.predict_proba(X_train2)[:, 1]


# Predict probabilities for training data
y_train_probs2 = best_model2.predict_proba(X_train2)[:, 1]

# Calculate AUC for training and test data
auc_train2 = roc_auc_score(y_train2, y_train_probs2)
auc_test2 = roc_auc_score(y_test2, y_probs2)
print(f'Train AUC: {auc_train2}')
print(f'Test AUC: {auc_test2}')

# Calculate KS statistic for training data
fpr_train, tpr_train, thresholds_train = roc_curve(y_train2, y_train_probs2)
ks_train2 = max(tpr_train - fpr_train)
print(f'Train KS Statistic: {ks_train2}')

# Calculate KS statistic for test data
fpr_test, tpr_test, thresholds_test = roc_curve(y_test2, y_probs2)
ks_test2 = max(tpr_test - fpr_test)
print(f'Test KS Statistic: {ks_test2}')

y_probs2 = best_model2.predict_proba(X_test2)[:, 1]
test_ks2 = ks(y_test2, y_probs2)

y_probs_train2 = best_model2.predict_proba(X_train2)[:, 1]
train_ks2 = ks(y_train2, y_probs_train2)

y_probs_overall2 = best_model2.predict_proba(X2)[:, 1]
ks_overall2 = ks(y2, y_probs_overall2)

base2['pred_prob'] = y_probs_overall2

auc_overall2 = roc_auc_score(y2, y_probs_overall2)
print(f'Overall AUC: {auc_overall2}')

# Define the conditions
conditions = [
    base2['pred_prob'] < 0.044,
    (base2['pred_prob'] >= 0.044) & (base2['pred_prob'] <= 0.060),
    (base2['pred_prob'] >= 0.060) & (base2['pred_prob'] <= 0.075),
    (base2['pred_prob'] >= 0.075) & (base2['pred_prob'] <= 0.090),
    (base2['pred_prob'] >= 0.090) & (base2['pred_prob'] <= 0.109),
    (base2['pred_prob'] >= 0.109) & (base2['pred_prob'] <= 0.137),
    (base2['pred_prob'] >= 0.137) & (base2['pred_prob'] <= 0.184),
    (base2['pred_prob'] >= 0.184) & (base2['pred_prob'] <= 0.262),
    (base2['pred_prob'] >= 0.262) & (base2['pred_prob'] <= 0.434),
    base2['pred_prob'] > 0.434
]

# Define the choices corresponding to the conditions
choices = ['d10', 'd9', 'd8', 'd7', 'd6', 'd5', 'd4', 'd3', 'd2', 'd1']

# Apply np.select to create a new column with the categories
base2['decile'] = np.select(conditions, choices)

# Retrieve feature importances
feature_importances2 = best_model2.get_feature_importance(Pool(X_train2, label=y_train2))
feature_names2 = X_train2.columns

# Create a DataFrame for better visualization
importance_df2 = pd.DataFrame({'Feature': feature_names2, 'Importance': feature_importances2})

# Sort by importance
importance_df2 = importance_df2.sort_values(by='Importance', ascending=False).reset_index(drop=True)

# Display feature importances
print(importance_df2)


# SHAP Values
explainer = shap.TreeExplainer(best_model2)
shap_values = explainer.shap_values(X_test2)

shap.summary_plot(shap_values, X_test2, plot_type="bar")
shap.summary_plot(shap_values, X_test2)




### Univariates #########

df21 = base.select_dtypes(include=[float, int])

results = []

for col in df21.columns:
    if col == 'target_7d':
        continue
    
    temp = df21[[col]].copy()
    temp.dropna(subset=[col], inplace=True)
    temp = temp[temp[col]>0]
    
    try:
        temp2 = temp.describe(percentiles=[0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 0.8, 0.9, 0.95, 0.99, 1])
    except:
        temp2 = temp.describe(percentiles=[0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 0.8, 0.9, 0.95, 0.99, 1])
    
    temp2 = temp2.T
    results.append(temp2)

grouped = pd.concat(results)
grouped['missing_count'] = len(df21) - grouped['count']
grouped['missing_count%'] = 1 - (grouped['count'] / len(df21))

# Display the result
print(grouped)




#### Single model ########


# Split data
X3 = base.drop(columns=['updated_at', 'target_15d', 'self_cured_in_30_days', 'business_id', 'lending_business_id', 'canopy_id',
       'external_account_id', 'application_type', 'due_date', 'due_date_month_year',
       'days_past_due', 'delinquency_bucket', 'cure_payment_cents',
       '_fivetran_synced', 'flag1', 'flag2', 'is_cured', 
       'repayment_or_last_synced_date', 
       'cured_with_contact_in_7_days', 'credit_limit',
       'state', 'business_type',  'first_draw_date', 'row_num_x','row_num_y', 
       'account_funded_date', 'business_created_date', 'base_date_lag_7_days',
       'start_date',
       'end_date', 'successful_contacts_last_7_days', 'start_date2',
       'end_date2', 'successful_contacts_last_15_days', 'start_date3',
       'end_date3', 'successful_contacts_last_30_days',
       'successful_contacts_last_7_days_flag',
       'successful_contacts_last_15_days_flag',
       'successful_contacts_last_30_days_flag', 'cured_with_contact_in_7_days',
       'cured_with_contact_in_15_days', 'cured_with_contact_in_30_days',
       'self_cured_in_7_days', 'self_cured_in_15_days',
       # 'self_cured_in_30_days', 'row_num_x', 'target_15d',
       'target_30d', '_fivetran_synced_lag_7days', 'target_7d', 'fico_score', 'number_of_60_days_delinquencies',
       'number_of_inquiries_last_6_months', 'number_of_open_trades', 'number_of_trades_with_derogatory_status'])


y3= base['target_7d'].astype(int)


X3.replace([np.inf, -np.inf], np.nan, inplace=True)

X3.fillna(0, inplace=True)



corr_thresh = 0.7
iv_thresh = 0.01
best_model3, selected_features3, X_train3, X_test3, y_train3, y_test3 = model_pipeline(X3, y3, 3, corr_thresh, iv_thresh)



best_model3.get_params()


selected_features3

len(selected_features3)

iv_df3 = calculate_iv(X3, y3, bins=10)


# Predict probabilities
train_probs3 = best_model3.predict_proba(X_train3)[:, 1]
test_probs3 = best_model3.predict_proba(X_test3)[:, 1]

# Calculate AUC
train_auc3 = roc_auc_score(y_train3, train_probs3)
test_auc3 = roc_auc_score(y_test3, test_probs3)

print(f'Train AUC: {train_auc3}')
print(f'Test AUC: {test_auc3}')


# Predict probabilities
y_probs3 = best_model3.predict_proba(X_test3)[:, 1]
y_probs_train3 = best_model3.predict_proba(X_train3)[:, 1]


# Predict probabilities for training data
y_train_probs3 = best_model3.predict_proba(X_train3)[:, 1]

# Calculate AUC for training and test data
auc_train3 = roc_auc_score(y_train3, y_train_probs3)
auc_test3 = roc_auc_score(y_test3, y_probs3)
print(f'Train AUC: {auc_train3}')
print(f'Test AUC: {auc_test3}')

# Calculate KS statistic for training data
fpr_train, tpr_train, thresholds_train = roc_curve(y_train3, y_train_probs3)
ks_train3 = max(tpr_train - fpr_train)
print(f'Train KS Statistic: {ks_train3}')

# Calculate KS statistic for test data
fpr_test, tpr_test, thresholds_test = roc_curve(y_test3, y_probs3)
ks_test3 = max(tpr_test - fpr_test)
print(f'Test KS Statistic: {ks_test3}')

y_probs3 = best_model3.predict_proba(X_test3)[:, 1]
test_ks3 = ks(y_test3, y_probs3)

y_probs_train3 = best_model3.predict_proba(X_train3)[:, 1]
train_ks3 = ks(y_train3, y_probs_train3)


# Retrieve feature importances
feature_importances3 = best_model3.get_feature_importance(Pool(X_train3, label=y_train3))
feature_names3 = X_train3.columns

# Create a DataFrame for better visualization
importance_df3 = pd.DataFrame({'Feature': feature_names3, 'Importance': feature_importances3})

# Sort by importance
importance_df3 = importance_df3.sort_values(by='Importance', ascending=False).reset_index(drop=True)

# Display feature importances
print(importance_df3)



# SHAP Values
explainer3= shap.TreeExplainer(best_model3)
shap_values3 = explainer3.shap_values(X_test3)

shap.summary_plot(shap_values3, X_test3, plot_type="bar")
shap.summary_plot(shap_values3, X_test3)


# SHAP Values
explainer1= shap.TreeExplainer(best_model1)
shap_values1 = explainer1.shap_values(X_test1)

shap.summary_plot(shap_values1, X_test1, plot_type="bar")
shap.summary_plot(shap_values1, X_test1)



# SHAP Values
explainer2 = shap.TreeExplainer(best_model2)
shap_values2 = explainer2.shap_values(X_test2)

shap.summary_plot(shap_values2, X_test2, plot_type="bar")
shap.summary_plot(shap_values2, X_test2)


















         
                 